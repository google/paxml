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

"""Embedding layers with ghost norm support."""
from typing import Callable

import jax
import jax.experimental.sparse as js
import jax.numpy as jnp
from paxml.ghostnorm import base
from paxml.ghostnorm import linears as ghostnorm_linears
from praxis import base_layer
from praxis import layers
from praxis import pytypes
from praxis.layers import base_ops

JTensor = pytypes.JTensor


def make_index_lookup(
    array_lookup: base_ops.ArrayLookup, einsum: base_ops.EinsumOp
) -> Callable[[JTensor, JTensor], JTensor]:
  """Constructs an operator that does index lookup."""

  @jax.custom_vjp
  def index_lookup(weights, idx):
    """Lookup index idx in array weights."""
    return array_lookup(base.get_param(weights), idx)

  def index_lookup_fwd(weights, idx):
    """Forward index lookup for custom vjp."""
    params = base.get_param(weights)
    assert isinstance(params, JTensor), 'Expected JTensor. Got: {}'.format(
        type(params)
    )
    y, vjp_fun = jax.vjp(array_lookup, params, idx)
    return y, (vjp_fun, idx, params.shape, base.get_aux(weights))

  def index_lookup_bwd(res, g):
    """Backward index lookup for custom vjp with per-example gradient norms."""
    vjp_fun, idx, params_shape, aux = res
    if aux is None:
      u_weights, u_idx = vjp_fun(g)
    else:
      # Per-example scaling coefficient. Normally this is all ones. When
      # computing the average of scaled (i.e. L2-norm clipped) per-example
      # gradients, this contains a scaling coefficient for each example in the
      # batch. Shape is (batch_size,).
      scales = aux

      # scaled gradients for parameters to achieve per-eg grad clipping
      # scaled_g: (batch_size, ..., output_dim)
      scaled_g = einsum('i, i... -> i...', scales, g)
      u_weights, u_idx = vjp_fun(scaled_g)

      # Compute per-example gradient square norms.
      # The batch_size factor is needed when the loss is *averaged* over the
      # mini-batch of examples (instead of summed over).
      batch_size = g.shape[0]

      # Gradient for each input index. If input is (batch_size, i, j) then this
      # will be (batch_size, i, j, params_shape[1])
      per_coordinate_grad = batch_size * scaled_g

      in_idx = jnp.expand_dims(idx < params_shape[0], -1)
      # If the index is out of bounds, it produces a zero gradient.
      per_coordinate_grad = jnp.where(
          in_idx,
          per_coordinate_grad,
          0.0,
      )

      def _calculate_per_example_grad_sq_norm(per_coord_grad, idx):
        data = per_coord_grad.reshape(-1, params_shape[1])
        indices = idx.reshape(-1, 1)
        # Use sparse tensor to sum gradients for the same rows without
        # materializing the full gradient.
        sparse_grad = js.BCOO((data, indices), shape=params_shape)
        sparse_grad = js.bcoo_sum_duplicates(sparse_grad, nse=indices.size)
        return (sparse_grad.data**2).sum()

      calculate_per_example_grad_sq_norm = jax.vmap(
          _calculate_per_example_grad_sq_norm
      )

      per_example_sq_grad_norms = calculate_per_example_grad_sq_norm(
          per_coordinate_grad, idx
      )

      u_weights = base.ParamWithAux(u_weights, per_example_sq_grad_norms)
    return u_weights, u_idx

  index_lookup.defvjp(index_lookup_fwd, index_lookup_bwd)
  return index_lookup


class EmbeddingGhostNorm(layers.Embedding):
  """Embedding layer with ghost norm support with custom vjp rule."""

  def setup(self) -> None:
    super().setup()
    if self.lookup_style == 'index':
      self._index_lookup = make_index_lookup(self.array_lookup, self.einsum)
    elif self.lookup_style == 'matmul':
      self._projector = ghostnorm_linears.make_last_dim_projector(self.einsum)

  def emb_lookup(self, ids: JTensor) -> JTensor:
    ap = self.activation_split_dims_mapping

    if self.lookup_style == 'index':
      embs = self._index_lookup(self.theta.emb_var, ids)
    elif self.lookup_style == 'matmul':
      # Explicit casting to fprop_dtype needed for bf16.
      one_hot_ids = jax.nn.one_hot(
          ids, self.num_classes, dtype=self.fprop_dtype
      )
      embs = self._projector(one_hot_ids, self.theta.emb_var)
    else:
      raise ValueError('Unknown lookup style.')

    # Scale with sqrt(embedding dims)
    if self.scale_sqrt_depth:
      embs *= self.input_dims**0.5

    # map out-of-boundary ids to nan for easier debug
    if self.set_nan_for_oob_id:
      embs = jnp.where(ids[..., jnp.newaxis] < self.num_classes, embs, jnp.nan)

    embs = base_layer.maybe_shard(
        embs, ap.emb_out_split_dims_mapping, self.mesh_axis_names
    )
    return embs
