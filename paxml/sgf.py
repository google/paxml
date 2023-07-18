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

"""Module with the stochastic gradient function classes."""

from __future__ import annotations

import abc
from collections.abc import Callable
from typing import Any, Optional

from flax import struct
import jax
from jax import numpy as jnp
import optax
from paxml.ghostnorm import base as ghostnorm_base
from praxis import base_hyperparams
from praxis import base_layer
from praxis import py_utils
from praxis import pytypes

JTensor = pytypes.JTensor
NestedJTensor = pytypes.NestedJTensor
NestedMap = py_utils.NestedMap
PRNGKey = pytypes.PRNGKey
PARAMS = base_layer.PARAMS


@struct.dataclass
class GradAuxInfo:
  aux_info: Any
  loss_weight: float = 1.0


@struct.dataclass
class DPGradAuxInfo(GradAuxInfo):
  dp_aux_info: Any = None


class BaseStochasticGradient(
    base_hyperparams.FiddleBaseParameterizable, metaclass=abc.ABCMeta
):
  """Stochastic gradient function."""

  def process_aux_info(self, aux_info: GradAuxInfo) -> GradAuxInfo:
    """Processes auxiliary info returned by `grad_fn`.

    Args:
      aux_info: Auxiliary info to be processed.

    Returns:
      Processed version of auxiliary info.
    """
    return aux_info

  @abc.abstractmethod
  def grad_fn(
      self,
      loss_fn: Callable[..., Any],
      mdl_vars_grad: NestedJTensor,
      mdl_vars_nograd_and_inputs: tuple[NestedJTensor, NestedMap],
      prng_key: PRNGKey,
  ) -> tuple[tuple[JTensor, GradAuxInfo], NestedJTensor]:
    """Main gradients function.

    Intended to accept a loss function, model parameters, input data, and
    a pseudorandom key, and return the loss (possibly with auxiliary info)
    and the gradient of the loss. Based on `jax.value_and_grad`.

    Args:
      loss_fn: Loss function.
      mdl_vars_grad: Model variables for which to compute gradient.
      mdl_vars_nograd_and_inputs: Tuple containing model variables for which
        gradients should not be computed, and input examples on which to call
        `loss_fn`.
      prng_key: A pseudorandom key.

    Returns:
      A tuple ((loss, auxiliary info), gradients).
    """


class StandardGradient(BaseStochasticGradient):
  """Standard gradient function."""

  def grad_fn(
      self,
      loss_fn: Callable[..., tuple[JTensor, GradAuxInfo]],
      mdl_vars_grad: NestedJTensor,
      mdl_vars_nograd_and_inputs: tuple[NestedJTensor, NestedMap],
      prng_key: PRNGKey,
  ) -> tuple[tuple[JTensor, GradAuxInfo], NestedJTensor]:
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True, allow_int=True)
    (values, aux), grads = grad_fn(
        mdl_vars_grad, mdl_vars_nograd_and_inputs, prng_key
    )
    aux = self.process_aux_info(aux)
    return (values, aux), grads


class PercoreClippedDpSgdGradient(BaseStochasticGradient):
  """DP-SGD stochastic gradient function using per-core clipping.

  Differentially private stochastic gradient using per-core clipping, whose
  running time matches non-private baseline.

  Experimental results with zero noise multiplier:
    Non-private baseline: http://tb/4569190445426541226
    PercoreClippedGradient: http://tb/1885364525575923265
    MicrobatchDpSgdStochasticGradient: http://tb/2683981470965501622
  """

  l2_norm_clip: float = 0.0
  noise_multiplier: float = 0.0

  def _clip_gradients(
      self, grads: NestedMap, l2_norm_clip: float = 1.0
  ) -> tuple[NestedMap, jax.Array, Any]:
    assert (
        self.l2_norm_clip > 0.0
    ), f'Clipping bound must be positive. {l2_norm_clip} is provided.'

    # Clip the per-core mean gradient.
    grads_flat, grads_treedef = jax.tree_flatten(grads)
    global_grad_norm = optax.global_norm(grads_flat)
    divisor = jnp.maximum(global_grad_norm / l2_norm_clip, 1.0)
    num_clipped = jnp.greater(divisor, 1.0)
    clipped_flat = [g / divisor for g in grads_flat]
    clipped = jax.tree_unflatten(grads_treedef, clipped_flat)

    return clipped, num_clipped, global_grad_norm

  def _add_noise(  # pytype: disable=annotation-type-mismatch  # jax-ndarray
      self,
      grads: NestedMap,
      noise_stddev: float,
      loss_weight: float,
      prng_key: PRNGKey = None,
  ) -> NestedMap:
    prng_keys = jax.random.split(
        prng_key, len(jax.tree_util.tree_leaves(grads))
    )
    prng_tree = jax.tree_unflatten(jax.tree_structure(grads), prng_keys)

    if base_layer.is_running_under_pmap():
      # Note: when running under pmap, loss_weight is set to 1/num_devices.
      # In this case, the *global* batch size is batch_size / loss_weight.
      # Moreover, each device adds independent Gaussian noises, and then the
      # noisy gradients are added with `psum``. Because the sum of num_devices
      # copies of independent Gaussian noises is equivalent to a single Gaussian
      # with std scaled by `sqrt(num_devices)``, we need to further scale the
      # noise_std on each device to correct this.
      noise_stddev *= loss_weight * jnp.sqrt(loss_weight)

    def _add_noise_to_array(x, prng):
      return x + noise_stddev * jax.random.normal(prng, shape=x.shape)

    final_grads = jax.tree_map(_add_noise_to_array, grads, prng_tree)
    return final_grads

  def grad_fn(
      self,
      loss_fn: Callable[..., tuple[JTensor, GradAuxInfo]],
      mdl_vars_grad: NestedJTensor,
      mdl_vars_nograd_and_inputs: tuple[NestedJTensor, NestedMap],
      prng_key: PRNGKey,
  ) -> tuple[tuple[JTensor, GradAuxInfo], NestedJTensor]:
    # Obtain the per-core mean gradient.
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True, allow_int=True)
    (values, aux), grads = grad_fn(
        mdl_vars_grad, mdl_vars_nograd_and_inputs, prng_key
    )
    aux = self.process_aux_info(aux)

    clipped, num_clipped, grad_norm = self._clip_gradients(
        grads, aux.loss_weight * self.l2_norm_clip
    )

    noise_stddev = self.noise_multiplier * self.l2_norm_clip
    grads = self._add_noise(clipped, noise_stddev, aux.loss_weight, prng_key)

    return (
        values,
        DPGradAuxInfo(
            dp_aux_info={
                'frac_clipped': num_clipped,
                'per_core_grad_norm': grad_norm,
            },
            aux_info=aux.aux_info,
            loss_weight=aux.loss_weight,
        ),
    ), grads


class DpSgdStochasticGradient(BaseStochasticGradient):
  """DP-SGD stochastic gradient function."""

  # Standard DP-SGD hyperparameters.
  l2_norm_clip: float = 0.0
  noise_multiplier: float = 0.0

  # Whether to apply Gradient Normalization as implemented in eqn 3 of
  # https://arxiv.org/abs/2204.13650 to reduce the dependence between
  # clipping value and learning rate. Note that normalization is only applied
  # post-clipping.
  normalize_gradients: bool = False

  # Number of examples to process at one time. If set to `None`,
  # this will be set to batch size as determined by the 0th element of
  # the shape of the input.
  #
  # This may be useful if a large batch size is desired (for example, for
  # better privacy-utility tradeoffs), but we cannot fit all of the
  # per-example gradients in memory. When set, the code computes
  # `inner_batch_size` per-example gradients at a time, accumulating
  # the total clipped gradient as it goes. Note that the setting of
  # `inner_batch_size` has no effect on the value of the final gradients--
  # it affects only the feasibility and speed of the computation.
  inner_batch_size: Optional[int] = None

  def _clip_and_mean_gradients(
      self, grads: NestedMap, l2_norm_clip: float = 1.0
  ) -> tuple[NestedMap, float, int]:
    grads_flat, grads_treedef = jax.tree_flatten(grads)
    sum_clipped, num_clipped = optax.per_example_global_norm_clip(
        grads=grads_flat, l2_norm_clip=l2_norm_clip
    )
    sum_grads = jax.tree_unflatten(grads_treedef, sum_clipped)

    # Normalize gradients across all examples.
    batch_size = grads_flat[0].shape[0]
    clipped_grads_mean = jax.tree_map(lambda x: x / batch_size, sum_grads)
    frac_clipped = num_clipped / batch_size

    return clipped_grads_mean, frac_clipped, batch_size  # pytype: disable=bad-return-type  # jax-types

  def _add_noise(  # pytype: disable=annotation-type-mismatch  # jax-ndarray
      self,
      grads: NestedMap,
      noise_stddev: float,
      loss_weight: float,
      prng_key: PRNGKey = None,
  ) -> NestedMap:
    prng_keys = jax.random.split(
        prng_key, len(jax.tree_util.tree_leaves(grads))
    )
    prng_tree = jax.tree_unflatten(jax.tree_structure(grads), prng_keys)

    if base_layer.is_running_under_pmap():
      # Note: when running under pmap, loss_weight is set to 1/num_devices.
      # In this case, the *global* batch size is batch_size / loss_weight.
      # Moreover, each device adds independent Gaussian noises, and then the
      # noisy gradients are added with `psum``. Because the sum of num_devices
      # copies of independent Gaussian noises is equivalent to a single Gaussian
      # with std scaled by `sqrt(num_devices)``, we need to further scale the
      # noise_std on each device to correct this.
      noise_stddev *= loss_weight * jnp.sqrt(loss_weight)

    def _add_noise_to_array(x, prng):
      return x + noise_stddev * jax.random.normal(prng, shape=x.shape)

    final_grads = jax.tree_map(_add_noise_to_array, grads, prng_tree)
    return final_grads

  def _prepare_inputs(self, inputs):
    """Reshape inputs to prepare for vmap to find per-example gradients."""
    return jax.tree_map(jax.tree_util.Partial(jnp.expand_dims, axis=1), inputs)

  def process_aux_info(self, aux_info: GradAuxInfo) -> GradAuxInfo:
    aux_info = jax.tree_map(jax.tree_util.Partial(jnp.mean, axis=0), aux_info)
    return aux_info

  def grad_fn(
      self,
      loss_fn: Callable[..., Any],
      mdl_vars_grad: NestedJTensor,
      mdl_vars_nograd_and_inputs: tuple[NestedJTensor, NestedMap],
      prng_key: PRNGKey,
  ) -> tuple[tuple[JTensor, DPGradAuxInfo], NestedJTensor]:
    assert (
        self.l2_norm_clip > 0.0
    ), f'Clipping bound must be positive. {self.l2_norm_clip} is provided.'

    mdl_vars_nograd, inputs = mdl_vars_nograd_and_inputs
    inputs = self._prepare_inputs(inputs)

    # Get batch size.
    inputs_flat, _ = jax.tree_flatten(inputs)
    batch_size = inputs_flat[0].shape[0]

    if self.inner_batch_size is None:
      inner_batch_size = batch_size
    else:
      inner_batch_size = self.inner_batch_size
    if batch_size % inner_batch_size != 0:
      raise ValueError('`batch_size` must be divisible by `inner_batch_size`.')
    num_iters = batch_size // inner_batch_size
    inner_prng_keys = jax.random.split(prng_key, num_iters)

    grad_fn = jax.vmap(
        jax.value_and_grad(loss_fn, has_aux=True, allow_int=True),
        in_axes=(None, (None, 0), None),
        out_axes=0,
    )

    def reshape_batch(x):
      return jnp.reshape(x, [-1, inner_batch_size, *x.shape[1:]])

    # Reshape input so that inner batches are stacked on axis 0.
    inputs = jax.tree_map(reshape_batch, inputs)

    def _process_inner_batch(index: int) -> Any:
      """Computes mean clipped gradient for inner batch specified by index."""
      new_inputs = jax.tree_map(lambda x: x[index], inputs)

      # Compute loss and gradients.
      (values, aux), grads = grad_fn(
          mdl_vars_grad, (mdl_vars_nograd, new_inputs), inner_prng_keys[index]
      )

      # Clip and aggregate gradients.
      grads, frac_clipped, _ = self._clip_and_mean_gradients(
          grads, aux.loss_weight * self.l2_norm_clip
      )
      # Aggregate values and aux.
      values = jax.tree_map(jax.tree_util.Partial(jnp.mean, axis=0), values)
      aux = self.process_aux_info(aux)
      return (
          values,
          DPGradAuxInfo(
              dp_aux_info={'frac_clipped': frac_clipped},
              aux_info=aux.aux_info,
              loss_weight=aux.loss_weight,
          ),
          grads,
      )

    def _loop_process_inner_batch(index: int, val: Any) -> Any:
      """Wrapper for _process_inner_batch suitable for fori_loop."""
      cur_values, cur_aux, cur_grads = val
      values, aux, grads = _process_inner_batch(index)

      new_values = jax.tree_map(jnp.add, cur_values, values)
      new_aux = jax.tree_map(jnp.add, cur_aux, aux)
      new_grads = jax.tree_map(jnp.add, cur_grads, grads)
      return (new_values, new_aux, new_grads)

    # Loop over inner batches, summing the results together.
    # We have to do one iteration first to get the correct shape of the return
    # values.
    values, aux, grads = jax.lax.fori_loop(
        1, num_iters, _loop_process_inner_batch, _process_inner_batch(0)
    )

    # Normalize results by number of inner batches.
    values, aux, grads = jax.tree_map(
        jax.tree_util.Partial(jnp.multiply, 1.0 / num_iters),
        (values, aux, grads),
    )

    # Add noise to normalized gradients.
    if self.normalize_gradients:
      grads = jax.tree_map(lambda x: x / self.l2_norm_clip, grads)
      noise_stddev = self.noise_multiplier / batch_size
    else:
      noise_stddev = self.noise_multiplier * self.l2_norm_clip / batch_size
    grads = self._add_noise(grads, noise_stddev, aux.loss_weight, prng_key)
    return (values, aux), grads


class MicrobatchDpSgdStochasticGradient(DpSgdStochasticGradient):
  """DP-SGD stochastic gradient function with microbatch."""

  microbatch_size: int = 1

  def _prepare_inputs(self, inputs):
    return jax.tree_map(self._prepare_for_microbatching, inputs)

  def _prepare_for_microbatching(self, tensor: JTensor) -> JTensor:
    """Reshapes tensor for vmap with microbatch size support.

    Args:
      tensor: the input tensor, of shape `(batch_size, ...)`, where the
        batch_size should be dividable by the microbatch_size.

    Returns:
      The input tensor reshaped into shape `(batch_size//microbatch_size,
      microbatch_size, ...)`.
    """
    batch_size = tensor.shape[0]
    microbatch_size = self.microbatch_size
    return tensor.reshape(
        (batch_size // microbatch_size, microbatch_size, *tensor.shape[1:])
    )


class AugMulDpSgdStochasticGradient(MicrobatchDpSgdStochasticGradient):
  """DP-SGD with Augmentation Multiplicity.

  Augmentation multiplicity generates multiple different augmentations for each
  training example, and do the l2-norm clipping on the average gradient for
  all the augmentations of each training example.

  If the augmentation happens at the data pipeline, the
  MicrobatchDpSgdStochasticGradient can be used directly. This subclass is for
  the special case where the augmentation happens inside the model call (e.g.
  the current Bert implementation). This class simply makes multiple identical
  copies of each input example, and let the model call handle the augmentation.
  """

  def _prepare_for_microbatching(self, tensor: JTensor) -> JTensor:
    shape = tensor.shape
    num_repeat = self.microbatch_size
    return jnp.repeat(tensor, num_repeat, axis=0).reshape(
        (shape[0], num_repeat, *shape[1:])
    )


class GhostClippingDpSgdStochasticGradient(DpSgdStochasticGradient):
  """DP-SGD stochastic gradient function with Ghost Norm Clipping.

  This class implements DP-SGD without materializing the per-example gradients.
  This reduces memory cost for DP-SGD training and allows large batch training
  without needing to do (sequential) gradient accumulation.

  To use this method, all the parametric layers (layers with trainable
  parameters) in the model need to implement the ghost norm protocol. Please
  see `paxml.ghostnorm` for more details.

  This class computes the clipped gradients in two passes. In the first pass,
  the ghost norm protocol is used to estimate the per-example gradient norms
  from each layers. The norms are aggregated, and then used to calculate
  per-example scaling coefficients. The ghost norm protocol is used again to
  compute the weighted average gradients according to the coefficients. The cost
  of each ghost norm protocol pass should be approximately equal to the cost
  of a standard back-propagation.
  """

  def grad_fn(
      self,
      loss_fn: Callable[..., Any],
      mdl_vars_grad: NestedJTensor,
      mdl_vars_nograd_and_inputs: tuple[NestedJTensor, NestedMap],
      prng_key: PRNGKey,
  ) -> tuple[tuple[JTensor, DPGradAuxInfo], NestedJTensor]:
    assert (
        self.inner_batch_size is None
    ), 'inner_batch_size is not supported yet by GhostClipping.'

    # Gradient Normalization needs to be implemented differently here and is
    # not a supported operation yet.
    if self.normalize_gradients:
      # TODO(b/291615231): Raising error is necessary till normalization
      # implemented in GhostClipping
      raise ValueError(
          'Expected a "False" value for normalize_gradients but got "True" as'
          ' gradient normalization is currently not supported for ghost'
          ' clipping.'
      )

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    batch_size = jax.tree_util.tree_flatten(mdl_vars_nograd_and_inputs[1])[0][
        0
    ].shape[0]

    # Pass 1: get per-example gradient norms
    scales = jnp.ones(batch_size)
    params_with_sq_norms = jax.tree_map(
        lambda x: ghostnorm_base.ParamWithAux(x, scales), mdl_vars_grad[PARAMS]
    )
    (loss, aux), grad_with_sq_norms = grad_fn(
        {**mdl_vars_grad, PARAMS: params_with_sq_norms},
        mdl_vars_nograd_and_inputs,
        prng_key,
    )

    is_leaf = lambda node: isinstance(node, ghostnorm_base.ParamWithAux)
    grad_norms = jnp.sqrt(
        sum(
            x.aux
            for x in jax.tree_util.tree_flatten(
                grad_with_sq_norms[PARAMS], is_leaf=is_leaf
            )[0]
        )
    )

    # PAX scales the loss by global batch size under pmap, specifically:
    # - under pmap:
    #   - loss = local_loss_sum / (local_batch_size * num_devices)
    #   - loss_weight = 1 / num_devices
    # - not under pmap:
    #   - loss = local_loss_sum / local_batch_size
    #   - loss_weight depends on specific models, sometimes it's
    #     local_batch_size, sometimes it's just 1
    if base_layer.is_running_under_pmap():
      # correct grad norm calculation
      num_devices = 1 / aux.loss_weight
      grad_norms *= num_devices

    frac_clipped = 0.0
    if self.l2_norm_clip is not None:
      scales = jnp.minimum(1.0, self.l2_norm_clip / grad_norms)
      frac_clipped = jnp.mean(scales < 1.0)

    # Pass 2: get average of clipped gradients
    params_with_sq_norms = jax.tree_map(
        lambda x: ghostnorm_base.ParamWithAux(x, scales), mdl_vars_grad[PARAMS]
    )
    (loss, aux), clipped_grads = grad_fn(
        {**mdl_vars_grad, PARAMS: params_with_sq_norms},
        mdl_vars_nograd_and_inputs,
        prng_key,
    )
    clipped_grads[PARAMS] = jax.tree_map(
        lambda x: x.param, clipped_grads[PARAMS], is_leaf=is_leaf
    )

    # Note here noise stddev is divided by num_devices because in PAX the loss
    # is scaled by global batch size when pmap is used (see above)
    noise_stddev = self.noise_multiplier * self.l2_norm_clip / batch_size
    noised_grads = self._add_noise(
        clipped_grads, noise_stddev, aux.loss_weight, prng_key
    )

    aux = DPGradAuxInfo(
        dp_aux_info={'frac_clipped': frac_clipped},
        aux_info=aux.aux_info,
        loss_weight=aux.loss_weight,
    )
    return (loss, aux), noised_grads
