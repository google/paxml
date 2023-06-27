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
from praxis import base_layer
from praxis import pax_fiddle
from praxis import pytypes

template_field = base_layer.template_field
instantiate = base_layer.instantiate
LayerTpl = pax_fiddle.Config[base_layer.BaseLayer]
PARAMS = base_layer.PARAMS
LAYER = 'layer'
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
    scaled_g = jnp.einsum('i, i... -> i...', scales, g)
    vjp_params, *vjp_args = vjp_fun(scaled_g)

    def vmappable_vjp(g_, *args_):
      _, vjp_fun = jax.vjp(fn, params, *args_)
      return vjp_fun(g_)[0]

    per_example_grad = jax.vmap(vmappable_vjp)(scaled_g, *args)

    # -----------------------------------------------------------------------
    # Compute per-example gradient square norms.
    # The batch_size factor is needed when the loss is *averaged* over the
    # mini-batch of examples (instead of summed over).
    batch_size = g.shape[0]
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
    self.layer_fn = _create_ghostnorm_fn(self.layer.apply)

  def __call__(self, *args: Any) -> JTensor:
    # This is a special case that is used when the layer is being initialized.
    if PARAMS not in self.variables:
      return self.layer(*args)

    return self.layer_fn({PARAMS: self.variables[PARAMS][LAYER]}, *args)
