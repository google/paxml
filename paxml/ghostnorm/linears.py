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

"""Linear layers with ghost norm support."""

import jax
import jax.numpy as jnp
from paxml.ghostnorm import base
from praxis import layers
from praxis import pytypes

JTensor = pytypes.JTensor


@jax.custom_vjp
def matmul(inputs, weights):
  """Forward matmul."""
  return jnp.matmul(inputs, base.get_param(weights))


def matmul_fwd(inputs, weights):
  """Forward matmul for custom vjp."""
  y, vjp_fun = jax.vjp(jnp.matmul, inputs, base.get_param(weights))
  return y, (vjp_fun, inputs, base.get_aux(weights))


def matmul_bwd(res, g):
  """Backward matmul for custom vjp, computing per-example gradient norms."""
  vjp_fun, inputs, aux = res
  if aux is None:  # When aux is None, just do standard back-propagation
    u_inputs, u_weights = vjp_fun(g)
  else:
    # When aux is not None, it contains per-example scaling, and the
    # back-propagation also returns per-example gradient square norms.

    # Per-example scaling coefficient. Normally this is all ones. When computing
    # the average of scaled (i.e. L2-norm clipped) per-example gradients, this
    # contains a scaling coefficient for each example in the batch.
    # (batch_size,) => (batch_size, 1)
    scales = jnp.expand_dims(aux, axis=1)
    # scaled_g: (batch_size, output_dim)
    scaled_g = scales * g

    # scaled gradients for parameters to achieve per-eg grad clipping
    # u_inputs_scaled: (batch_size, input_dim)
    # u_weights: (input_dim, output_dim)
    u_inputs_scaled, u_weights = vjp_fun(scaled_g)
    # Revert the effect of per-example scaling. This derivative with
    # respect to the inputs will be back-propagated to lower layers. Each layer
    # handles per-example scaling independently. So we revert scaling here.
    u_inputs = u_inputs_scaled / jnp.maximum(scales, 1e-7)

    # compute per-example gradient square norms for matmul
    batch_size = g.shape[0]
    scaled_g *= batch_size  # this assumes the loss averages over examples
    zsum = jnp.square(scaled_g).sum(axis=1)
    hsum = jnp.square(inputs).sum(axis=1)
    per_example_grad_sq_norms = zsum*hsum
    u_weights = base.ParamWithAux(u_weights, per_example_grad_sq_norms)
  return u_inputs, u_weights


matmul.defvjp(matmul_fwd, matmul_bwd)


class LinearGhostNorm(layers.Linear):
  """Linear layer with ghost norm support using matmul with custom vjp rule."""

  def __call__(self, inputs: JTensor) -> JTensor:
    # TODO(chiyuan): handle higher dimension input tensors as in base class
    return matmul(inputs, self.theta.w)
