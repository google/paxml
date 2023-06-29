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
from praxis.layers import base_ops

JTensor = pytypes.JTensor


def _should_use_efficient_rank3_algorithm(inputs, weights):
  if inputs.ndim == 3:
    efficient_algo_space = 2 * (inputs.shape[1] ** 2)
    base_algo_space = weights.shape[0] * weights.shape[1]
    if efficient_algo_space <= base_algo_space:
      return True
  return False


def make_last_dim_projector(einsum: base_ops.EinsumOp):
  """Constructs an operator that does last-dim projection given an einsum."""

  project_with_einsum = lambda x, w: einsum('...y, yz -> ...z', x, w)

  @jax.custom_vjp
  def project_last_dim(inputs, weights):
    """Linear projection on the last dim of the input JTensor."""
    return project_with_einsum(inputs, base.get_param(weights))

  def project_last_dim_fwd(inputs, weights):
    """Forward linear projection for custom vjp."""
    y, vjp_fun = jax.vjp(project_with_einsum, inputs, base.get_param(weights))
    return y, (vjp_fun, inputs, base.get_param(weights), base.get_aux(weights))

  def project_last_dim_bwd(res, g):
    """Backward linear projection, computing per-example gradient norms."""
    vjp_fun, inputs, weights, aux = res
    if aux is None:  # When aux is None, just do standard back-propagation
      u_inputs, u_weights = vjp_fun(g)
    else:
      # When aux is not None, it contains per-example scaling, and the
      # back-propagation also returns per-example gradient square norms.

      # Per-example scaling coefficient. Normally this is all ones. When
      # computing the average of scaled (i.e. L2-norm clipped) per-example
      # gradients, this contains a scaling coefficient for each example in the
      # batch. Shape is (batch_size,).
      scales = aux

      # u_inputs: (batch_size, ..., input_dim)
      u_inputs = einsum('ij, ...j -> ...i', weights, g)

      # scaled gradients for parameters to achieve per-eg grad clipping
      # scaled_g: (batch_size, ..., output_dim)
      scaled_g = einsum('i, i... -> i...', scales, g)

      # -----------------------------------------------------------------------
      # Compute per-example gradient square norms.
      # The batch_size factor is needed when the loss is *averaged* over the
      # mini-batch of examples (instead of summed over).
      batch_size = g.shape[0]
      if inputs.ndim == 2:
        # There is a more memory efficient implementation for the rank-2 case
        zsum = jnp.square(scaled_g * batch_size).sum(axis=1)
        hsum = jnp.square(inputs).sum(axis=1)
        per_example_grad_sq_norms = zsum * hsum
        u_weights = jnp.matmul(inputs.T, scaled_g)
      elif _should_use_efficient_rank3_algorithm(inputs, weights):

        def rank3_algorithm(single_input, single_grad):
          aat = jnp.matmul(single_input, single_input.T).reshape(-1)
          ggt = jnp.matmul(single_grad, single_grad.T).reshape(-1)
          return jnp.dot(aat, ggt)

        per_example_grad_sq_norms = jax.vmap(rank3_algorithm)(
            inputs, scaled_g * batch_size
        )
        _, u_weights = vjp_fun(scaled_g)
      else:  # generic case
        # shape: (batch_size, input_dim, output_dim)
        per_example_grad = einsum(
            'k...i, k...j -> kij', inputs, scaled_g * batch_size
        )
        per_example_grad_sq_norms = (per_example_grad**2).sum(axis=(1, 2))
        u_weights = jnp.mean(per_example_grad, axis=0)

      u_weights = base.ParamWithAux(u_weights, per_example_grad_sq_norms)
    return u_inputs, u_weights

  project_last_dim.defvjp(project_last_dim_fwd, project_last_dim_bwd)
  return project_last_dim


class LinearGhostNorm(layers.Linear):
  """Linear layer with ghost norm support with custom vjp rule."""

  def setup(self) -> None:
    super().setup()
    self._projector = make_last_dim_projector(self.einsum)

  def __call__(self, inputs: JTensor) -> JTensor:
    return self._projector(inputs, self.theta.w)


@jax.custom_vjp
def bias_add(inputs, bias):
  return inputs + base.get_param(bias)


def bias_add_fwd(inputs, bias):
  """Forward pass for bias_add."""
  outputs = inputs + base.get_param(bias)
  return outputs, base.get_aux(bias)


def bias_add_bwd(res, g):
  """Backward pass for bias_add, computing per-example gradient norms."""
  aux = res
  u_inputs = g
  if aux is None:  # When aux is None, just do standard back-propagation
    u_bias = jnp.sum(g, axis=range(g.ndim-1))
  else:
    scales = aux[:, jnp.newaxis]
    # The batch_size factor is needed when the loss is *averaged* over the
    # mini-batch of examples (instead of summed over).
    batch_size = scales.shape[0]
    # shape: (batch_size, output_dim)
    per_example_grad = jnp.sum(g, axis=range(1, g.ndim-1)) * scales * batch_size
    per_example_grad_sq_norms = (per_example_grad**2).sum(axis=1)
    u_bias = jnp.mean(per_example_grad, axis=0)
    u_bias = base.ParamWithAux(u_bias, per_example_grad_sq_norms)

  return u_inputs, u_bias


bias_add.defvjp(bias_add_fwd, bias_add_bwd)


class BiasGhostNorm(layers.Bias):
  """Bias layer with ghost norm support."""

  def __call__(self, inputs: JTensor) -> JTensor:
    return bias_add(inputs, self.theta.b)
