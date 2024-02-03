# coding=utf-8
# Copyright 2022 The Pax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Decorator to wrap any layer to add support for Ghost Norm."""

import functools
import inspect
from typing import Any, Callable, Iterable, Mapping

from flax.core import frozen_dict
import jax
import jax.numpy as jnp
from paxml.ghostnorm import base
from paxml.ghostnorm import embedding
from paxml.ghostnorm import linears
from praxis import base_layer
from praxis import pax_fiddle
from praxis import pytypes
from praxis.layers import attentions as praxis_attentions
from praxis.layers import embedding_softmax as praxis_embedding
from praxis.layers import linears as praxis_linears
from praxis.layers import normalizations as praxis_normalizations
from praxis.layers import transformers as praxis_transformers


template_field = base_layer.template_field
instantiate = base_layer.instantiate
LayerTpl = pax_fiddle.Config[base_layer.BaseLayer]
PARAMS = base_layer.PARAMS
LAYER = 'ghostnorm_wrapped_layer'
JTensor = pytypes.JTensor
NestedJTensor = pytypes.NestedJTensor
ResType = tuple[
    Callable[..., tuple[JTensor, ...]],
    NestedJTensor,
    JTensor,
    list[Any],
]


def _create_ghostnorm_fn(
    fn: Callable[..., JTensor], signature: inspect.Signature
) -> Callable[..., JTensor]:
  """Adds a custom_vjp to a function to output per example gradient norms.

  Args:
    fn: A function that accepts input in the format (params, *args). The added
      custom_vjp will add the per example gradient norms for params when using
      jax.grad.
    signature: A signature for the function being wrapped.

  Returns:
    A function with the custom_vjp added.
  """

  @jax.custom_vjp
  def f(params: NestedJTensor, *args: Any) -> JTensor:
    return fn(base.get_param(params), *args)

  def fwd(params: NestedJTensor, *args: Any) -> tuple[JTensor, ResType]:
    params, aux = base.get_param(params), base.get_aux(params)
    out, vjp_fun = jax.vjp(fn, params, *args)
    return out, (vjp_fun, params, aux, args)

  def bwd(
      res: ResType,
      g: JTensor,
  ) -> tuple[JTensor, ...]:
    vjp_fun, params, aux, args = res

    if aux is None:
      return vjp_fun(g)

    # When aux is not None, it contains per-example scaling, and the
    # back-propagation also returns per-example gradient square norms.

    # Per-example scaling coefficient. Normally this is all ones. When
    # computing the average of scaled (i.e. L2-norm clipped) per-example
    # gradients, this contains a scaling coefficient for each example in the
    # batch. Shape is (batch_size,).
    scales = aux

    # scaled gradients for parameters to achieve per-eg grad clipping
    # scaled_g: (batch_size, ..., output_dim)
    scaled_g = jax.tree_map(
        lambda g_: jnp.einsum('i, i... -> i...', scales, g_), g
    )
    vjp_params, *_ = vjp_fun(scaled_g)
    _, *vjp_args = vjp_fun(g)

    def vmappable_vjp(g_, *args_):
      _, vjp_fun = jax.vjp(fn, params, *args_)
      return vjp_fun(g_)[0]

    per_example_grad = jax.vmap(vmappable_vjp)(scaled_g, *args)

    # -----------------------------------------------------------------------
    # Compute per-example gradient square norms.
    # The batch_size factor is needed when the loss is *averaged* over the
    # mini-batch of examples (instead of summed over).
    batch_size = args[0].shape[0]
    batch_scaled_per_example_grad = jax.tree_map(
        lambda x: x * batch_size, per_example_grad
    )
    per_example_grad_sq_norms = jax.tree_map(
        jax.vmap(lambda x: (x**2).sum()), batch_scaled_per_example_grad
    )
    vjp_params = jax.tree_map(
        base.ParamWithAux, vjp_params, per_example_grad_sq_norms
    )
    return vjp_params, *vjp_args

  f.defvjp(fwd, bwd)

  def f_with_kwargs(params, *args, **kwargs):
    bound_args = signature.bind(*args, **kwargs)
    return f(params, *bound_args.args)

  return f_with_kwargs


class WrappedGhostNorm(base_layer.BaseLayer):
  """Wraps a pax layer to be compatible with ghost clipping.

  Attributes:
    layer_tpl: A PaxConfig defining the layer that should be wrapped.
  """

  layer_tpl: LayerTpl | None = template_field(None)

  def setup(self):
    super().setup()
    if self.layer_tpl is not None:
      self.create_child(LAYER, self.layer_tpl.clone())
    self.layer_fn = _create_ghostnorm_fn(
        functools.partial(
            self.ghostnorm_wrapped_layer.apply, method='__call__'
        ),
        signature=inspect.signature(self.ghostnorm_wrapped_layer.__call__),
    )

  def __call__(self, *args: Any, **kwargs: Any) -> JTensor:
    # This is a special case that is used when the layer is being initialized.
    if PARAMS not in self.variables:
      return self.ghostnorm_wrapped_layer(*args, **kwargs)

    return self.layer_fn(
        {varname: var[LAYER] for varname, var in self.variables.items()},
        *args,
        **kwargs,
    )


def _monkey_patch_ghostnorm_fn(layer: WrappedGhostNorm, method_name: str):
  """Monkey patches func_name as a ghostnorm function into wrapped layer.

  Args:
    layer: A WrappedGhostNorm layer.
    method_name: A method name that exists in the wrapped layer.
  """

  def ghostnorm_fn_for_class(self, *args: Any, **kwargs: Any) -> JTensor:
    wrapped_layer: base_layer.BaseLayer = self.ghostnorm_wrapped_layer
    current_fn = functools.partial(wrapped_layer.apply, method=method_name)
    method = wrapped_layer.__getattribute__(method_name)
    ghostnorm_fn = _create_ghostnorm_fn(
        current_fn,
        signature=inspect.signature(method),
    )
    return ghostnorm_fn(
        {varname: var[LAYER] for varname, var in self.variables.items()},
        *args,
        **kwargs,
    )

  layer.__setattr__(method_name, ghostnorm_fn_for_class)


# Add layers to this dict that should be wrapped with WrappedGhostNorm wrapper.
# For each entry (layer_cls, tuple[fn_name,...]), the layer_cls is wrapped with
# WrappedGhostNorm when a wrapped template compatible with ghost clipping is
# generated using generate_wrapped_template. By default, only the __call__
# is wrapped and made compatible to be used as part of the ghost clipping model
# but any additional methods that should also be made compatible from the layer
# can be added here.
_WRAPPABLE_LAYERS_TO_EXTRA_FNS: Mapping[type[Any], tuple[str, ...]] = (
    frozen_dict.FrozenDict({
        praxis_attentions.AttentionProjection: (),
        praxis_attentions.CausalDepthwiseConv1D: (),
        praxis_attentions.CombinedQKVProjectionLayer: (),
        praxis_attentions.PerDimScale: (),
        praxis_embedding.Embedding: ('emb_lookup',),
        praxis_normalizations.GroupNorm: (),
        praxis_normalizations.LayerNorm: (),
        praxis_normalizations.RmsNorm: (),
        praxis_transformers.TransformerFeedForwardMoe: (),
    })
)


def _get_method_names(layer_p: pax_fiddle.Config) -> Iterable[str]:
  if layer_p.cls in _WRAPPABLE_LAYERS_TO_EXTRA_FNS.keys():
    return _WRAPPABLE_LAYERS_TO_EXTRA_FNS[layer_p.cls]
  elif issubclass(layer_p.cls, tuple(_WRAPPABLE_LAYERS_TO_EXTRA_FNS.keys())):
    for wrappable_layer_cls in _WRAPPABLE_LAYERS_TO_EXTRA_FNS:
      if issubclass(layer_p.cls, wrappable_layer_cls):
        return _WRAPPABLE_LAYERS_TO_EXTRA_FNS[wrappable_layer_cls]
  return []


# Add a mapping to replace layer with a custom implementation.
_REPLACE_MAP: Mapping[type[Any], type[Any]] = frozen_dict.FrozenDict({
    praxis_embedding.Embedding: embedding.EmbeddingGhostNorm,
    praxis_linears.Linear: linears.LinearGhostNorm,
    praxis_linears.Bias: linears.BiasGhostNorm,
})


class GhostNormPaxConfig(pax_fiddle.Config):
  """A special PaxFiddle config which wraps with WrappedGhostNorm when built."""

  def __build__(self, *args, **kwargs):
    """Builds this ``Config`` for the given ``args`` and ``kwargs``.

    This method is called during `build` to get the output for this `Config`.

    Args:
      *args: Not supported.
      **kwargs: Keyword arguments to pass to ``self.__fn_or_cls__`.

    Returns:
      A `WrappedGhostNorm` class containing the template of the layer with the
        kwargs applied.
    Raises:
      ValueError: Incorrect args were provided for building `WrappedGhostNorm`
        layer.
    """
    if args:
      raise ValueError(
          'Positional args are not supported for initializing wrapped layers.'
      )

    pax_config = self.to_pax_config()
    for argname, argvalue in kwargs.items():
      pax_config.__setattr__(argname, argvalue)
    module = WrappedGhostNorm(layer_tpl=pax_config)
    for method_name in _get_method_names(pax_config):
      _monkey_patch_ghostnorm_fn(module, method_name)
    return module

  @staticmethod
  def from_pax_config(pax_config: pax_fiddle.Config) -> 'GhostNormPaxConfig':
    values, metadata = pax_config.__flatten__()
    wrapped_pax_config = GhostNormPaxConfig.__unflatten__(values, metadata)
    wrapped_pax_config: GhostNormPaxConfig
    return wrapped_pax_config

  def to_pax_config(self) -> pax_fiddle.Config:
    values, metadata = self.__flatten__()
    pax_config = pax_fiddle.Config.__unflatten__(values, metadata)
    pax_config: pax_fiddle.Config
    return pax_config


def _is_wrappable(model_or_layer_p: pax_fiddle.Config) -> bool:
  wrappable_layers = tuple(_WRAPPABLE_LAYERS_TO_EXTRA_FNS.keys())
  return issubclass(model_or_layer_p.cls, base_layer.BaseLayer) and issubclass(
      model_or_layer_p.cls, wrappable_layers
  )


def _wrap(model_or_layer_p: pax_fiddle.Config) -> GhostNormPaxConfig:
  return GhostNormPaxConfig.from_pax_config(
      model_or_layer_p,
  )


def _is_replaceable(model_or_layer_p: pax_fiddle.Config) -> bool:
  return model_or_layer_p.cls in _REPLACE_MAP.keys()


def _replace(model_or_layer_p: pax_fiddle.Config) -> pax_fiddle.Config:
  model_or_layer_p.cls = _REPLACE_MAP[model_or_layer_p.cls]
  return model_or_layer_p


def generate_wrapped_template(
    model_or_layer_p: pax_fiddle.Config,
    force_no_replace: bool = False,
    force_wrap: bool = False,
) -> pax_fiddle.Config:
  """Wraps a Pax Layer PaxFiddle template to be compatible with ghost clipping.

  Note that it only replaces or wraps layers that we know are compatible. To be
  compatible, all the parameteric layers (layers with trainable parameters) in
  the model/layer need to either wrappable or replacable. Furthermore, weight
  sharing is not allowed and will cause an error even if all the others can be
  wrapped or replaced.

  Args:
    model_or_layer_p: A PaxConfig describing the model or layer to wrap.
    force_no_replace: If set, will not replace any layer even if a replacement
      exists.
    force_wrap: If set, will force wrap model_or_layer_p even if it is not in
      the wrappable list.

  Returns:
    A PaxConfig describing the wrapped model.
  """
  assert isinstance(model_or_layer_p, pax_fiddle.Config)

  if not force_no_replace and _is_replaceable(model_or_layer_p):
    return _replace(model_or_layer_p)
  elif force_wrap or _is_wrappable(model_or_layer_p):
    return _wrap(model_or_layer_p)
  else:
    # Go through attributes of layer to wrap sublayers.
    for attr_name in model_or_layer_p.__dir__():
      attr = model_or_layer_p.__getattr__(attr_name)
      if isinstance(attr, pax_fiddle.Config) and issubclass(
          attr.cls, base_layer.BaseLayer
      ):
        layer_p = attr
        wrapped_layer_p = generate_wrapped_template(layer_p)
        model_or_layer_p.__setattr__(attr_name, wrapped_layer_p)

      if isinstance(attr, Iterable):
        wrapped_sublayers_p = []
        for sub_attr in attr:
          if isinstance(sub_attr, pax_fiddle.Config) and issubclass(
              sub_attr.cls, base_layer.BaseLayer
          ):
            layer_p = sub_attr
            wrapped_layer_p = generate_wrapped_template(layer_p)
            wrapped_sublayers_p.append(wrapped_layer_p)
        if wrapped_sublayers_p:
          model_or_layer_p.__setattr__(attr_name, wrapped_sublayers_p)

      if isinstance(attr, Mapping):
        wrapped_sublayers_p = {}
        for sub_attr_name, sub_attr in attr.items():
          if isinstance(sub_attr, pax_fiddle.Config) and issubclass(
              sub_attr.cls, base_layer.BaseLayer
          ):
            layer_p = sub_attr
            wrapped_layer_p = generate_wrapped_template(layer_p)
            wrapped_sublayers_p[sub_attr_name] = wrapped_layer_p
        if wrapped_sublayers_p:
          model_or_layer_p.__setattr__(attr_name, wrapped_sublayers_p)

  return model_or_layer_p
