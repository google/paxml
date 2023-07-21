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

from typing import Any, Callable

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


def _create_ghostnorm_fn(fn: Callable[..., JTensor]) -> Callable[..., JTensor]:
  """Adds a custom_vjp to a function to output per example gradient norms.

  Args:
    fn: A function that accepts input in the format (params, *args). The added
      custom_vjp will add the per example gradient norms for params when using
      jax.grad.

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
    vjp_params, *vjp_args = vjp_fun(scaled_g)

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
  return f


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
    self.layer_fn = _create_ghostnorm_fn(self.ghostnorm_wrapped_layer.apply)

  def __call__(self, *args: Any) -> JTensor:
    # This is a special case that is used when the layer is being initialized.
    if PARAMS not in self.variables:
      return self.ghostnorm_wrapped_layer(*args)

    return self.layer_fn({PARAMS: self.variables[PARAMS][LAYER]}, *args)


_SPECIAL_ATTRS = {'layer_tpl'}


class GhostNormPaxConfig(pax_fiddle.Config):
  """A special PaxFiddle config which applies attributes to its child."""

  def set(self, *args, **kwargs):
    self.layer_tpl.set(*args, **kwargs)
    return self

  def __getattr__(self, name: str) -> Any:
    if name in _SPECIAL_ATTRS:
      return super().__getattr__(name)
    return self.layer_tpl.__getattr__(name)

  def __setattr__(self, name: str, value: Any):
    if name in _SPECIAL_ATTRS:
      super().__setattr__(name, value)
    else:
      self.layer_tpl.__setattr__(name, value)


# Add layers to this list that should be wrapped with WrappedGhostNorm wrapper.
# Note that this list should mututally exclusive with _REPLACE_MAP.
_WRAPPABLE_LAYERS = {
    praxis_normalizations.LayerNorm,
    praxis_normalizations.RmsNorm,
    praxis_attentions.PerDimScale,
    praxis_attentions.CausalDepthwiseConv1D,
    praxis_attentions.AttentionProjection,  # optimize most likely.
    praxis_attentions.CombinedQKVProjectionLayer,  # optimize most likely.
    praxis_transformers.TransformerFeedForwardMoe,  # optimize maybe.
}

# Add a mapping to replace layer with a custom implementation.
_REPLACE_MAP = {
    praxis_embedding.Embedding: embedding.EmbeddingGhostNorm,
    praxis_linears.Linear: linears.LinearGhostNorm,
    praxis_linears.Bias: linears.BiasGhostNorm,
}


def _is_wrappable(model_or_layer_p: pax_fiddle.Config) -> bool:
  return (
      issubclass(model_or_layer_p.cls, base_layer.BaseLayer)
      and model_or_layer_p.cls in _WRAPPABLE_LAYERS
  )


def _is_replaceable(model_or_layer_p: pax_fiddle.Config) -> bool:
  return model_or_layer_p.cls in _REPLACE_MAP.keys()


def _replace(model_or_layer_p: pax_fiddle.Config) -> pax_fiddle.Config:
  model_or_layer_p.cls = _REPLACE_MAP[model_or_layer_p.cls]
  return model_or_layer_p


def generate_wrapped_template(
    model_or_layer_p: pax_fiddle.Config,
) -> pax_fiddle.Config:
  """Wraps a Pax Layer PaxFiddle template to be compatible with ghost clipping.

  Note that it only replaces or wraps layers that we know are compatible. To be
  compatible, all the parameteric layers (layers with trainable parameters) in
  the model/layer need to either wrappable or replacable. Furthermore, weight
  sharing is not allowed and will cause an error even if all the others can be
  wrapped or replaced.

  Args:
    model_or_layer_p: A PaxConfig describing the model or layer to wrap.

  Returns:
    A PaxConfig describing the wrapped model.
  """
  assert isinstance(model_or_layer_p, pax_fiddle.Config)
  for attr_name in model_or_layer_p.__dir__():
    layer_p = model_or_layer_p.__getattr__(attr_name)
    if isinstance(layer_p, pax_fiddle.Config) and issubclass(
        layer_p.cls, base_layer.BaseLayer
    ):
      wrapped_layer_p = generate_wrapped_template(layer_p)
      model_or_layer_p.__setattr__(attr_name, wrapped_layer_p)
  if _is_replaceable(model_or_layer_p):
    model_or_layer_p = _replace(model_or_layer_p)
  if _is_wrappable(model_or_layer_p):
    model_or_layer_p = GhostNormPaxConfig(
        WrappedGhostNorm, layer_tpl=model_or_layer_p
    )
  return model_or_layer_p
