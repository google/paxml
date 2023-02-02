# coding=utf-8
# Copyright 2022 Google LLC.
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
from typing import Any, Callable, Optional, Tuple

from flax import struct
import jax
from jax import numpy as jnp
import optax
from praxis import base_hyperparams
from praxis import py_utils
from praxis import pytypes

JTensor = pytypes.JTensor
NestedMap = py_utils.NestedMap
PRNGKey = pytypes.PRNGKey


@struct.dataclass
class GradAuxInfo:
  aux_info: Any
  loss_weight: JTensor = 1.0


class BaseStochasticGradient(
    base_hyperparams.BaseParameterizable, metaclass=abc.ABCMeta
):
  """Stochastic gradient function."""

  def process_aux_info(self, aux_info: Any) -> Any:
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
      mdl_vars: Any,
      inputs: NestedMap,
      prng_key: PRNGKey,
  ) -> Tuple[Any, Any]:
    """Main gradients function.

    Intended to accept a loss function, model parameters, input data, and
    a pseudorandom key, and return the loss (possibly with auxiliary info)
    and the gradient of the loss. Based on `jax.value_and_grad`.

    Args:
      loss_fn: Loss function.
      mdl_vars: Model variables.
      inputs: Input examples on which to call `loss_fn`.
      prng_key: A pseudorandom key.

    Returns:
      A tuple ((loss, auxiliary info), gradients).
    """


class StandardGradient(BaseStochasticGradient):
  """Standard gradient function."""

  def grad_fn(
      self,
      loss_fn: Callable[..., Any],
      mdl_vars: Any,
      inputs: NestedMap,
      prng_key: PRNGKey,
  ) -> Tuple[Any, Any]:
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True, allow_int=True)
    (values, aux), grads = grad_fn(mdl_vars, inputs, prng_key)
    aux = self.process_aux_info(aux)
    return (values, aux), grads


class DpSgdStochasticGradient(BaseStochasticGradient):
  """DP-SGD stochastic gradient function.

  **NOTE** on setting `noise_multiplier` when training with multiple devices:
  when `pmap` is used, the `grad_fn` method is run *inside* `pmap`. As a result,
  the Gaussian noise std is scaled with `1 / local_batch_size`. The correct
  scaling should be done with the *global* batch size instead. Currently the
  library does **not** make the adjustment automatically, and the users should
  make the adjustment according to the training setup.

  For data parallelism training with multiple devices, the user should divide
  the `noise_multiplier` by `sqrt(num_devices)`.

  Rationale: assume the total batch size is `K*B`. Let
  `s = noise_multiplier * l2_norm_clip`. Then when training on a single device
  (without pmap), the amount of noise added to the gradient of each parameter
  is:

    `s / (K*B) * G`

  where `G` represents a standard Gaussian random variable. In comparison, when
  training with K devices using `pmap` with per-device batch size B, the amount
  of noise added to the gradient of each parameter by each device is
  `s / B * G`. After taking `pmean` across K devices, the total noise is:

    `1/K (s/B * G_1 + ... + s/B * G_K) = s / (K*B) (G_1 + ... + G_K)`

  where `G_1, ..., G_K` are K independent standard Gaussian random variables.

  Note the summation of `K` independent standard Gaussian random variables is
  Gaussian with zero mean and standard deviation `sqrt(K)`. So in order for the
  multi-device case to add the correct amount of noise, we can pass in a
  `noise_multiplier` parameter that is pre-divided by the square root of the
  number of devices when training with multiple devices. And the privacy
  accounting should be carried out with the original (unscaled) noise
  multiplier.
  """

  class HParams(BaseStochasticGradient.HParams):
    """Returns the PrivateGradient params."""
    # Standard DP-SGD hyperparameters.
    l2_norm_clip: Optional[float] = None
    noise_multiplier: float = 0.0

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
  ) -> Tuple[NestedMap, float, int]:
    grads_flat, grads_treedef = jax.tree_flatten(grads)
    sum_clipped, num_clipped = optax.per_example_global_norm_clip(
        grads=grads_flat, l2_norm_clip=l2_norm_clip)
    sum_grads = jax.tree_unflatten(grads_treedef, sum_clipped)

    # Normalize gradients across all examples.
    batch_size = grads_flat[0].shape[0]
    clipped_grads_mean = jax.tree_map(lambda x: x / batch_size, sum_grads)
    frac_clipped = num_clipped / batch_size

    return clipped_grads_mean, frac_clipped, batch_size

  def _add_noise(
      self,
      grads: NestedMap,
      noise_stddev: float,
      prng_key: PRNGKey = None) -> NestedMap:
    prng_keys = jax.random.split(prng_key, len(jax.tree_leaves(grads)))
    prng_tree = jax.tree_unflatten(jax.tree_structure(grads), prng_keys)
    final_grads = jax.tree_map(
        lambda x, prng: x + noise_stddev * jax.random.normal(
            prng, shape=x.shape), grads, prng_tree)
    return final_grads

  def _prepare_inputs(self, inputs):
    """Reshape inputs to prepare for vmap to find per-example gradients."""
    return jax.tree_map(jax.tree_util.Partial(jnp.expand_dims, axis=1), inputs)

  def process_aux_info(self, aux_info):
    aux_info = jax.tree_map(jax.tree_util.Partial(jnp.mean, axis=0), aux_info)
    return aux_info

  def grad_fn(self, loss_fn: Callable[..., Any], mdl_vars: Any,
              inputs: NestedMap, prng_key: PRNGKey) -> Tuple[Any, Any]:
    p = self.hparams

    inputs = self._prepare_inputs(inputs)

    # Get batch size.
    inputs_flat, _ = jax.tree_flatten(inputs)
    batch_size = inputs_flat[0].shape[0]

    if self.hparams.inner_batch_size is None:
      inner_batch_size = batch_size
    else:
      inner_batch_size = self.hparams.inner_batch_size
    if batch_size % inner_batch_size != 0:
      raise ValueError('`batch_size` must be divisible by `inner_batch_size`.')
    num_iters = batch_size // inner_batch_size
    inner_prng_keys = jax.random.split(prng_key, num_iters)

    grad_fn = jax.vmap(
        jax.value_and_grad(loss_fn, has_aux=True, allow_int=True),
        in_axes=(None, 0, None),
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
          mdl_vars, new_inputs, inner_prng_keys[index]
      )

      # Clip and aggregate gradients.
      grads, _, _ = self._clip_and_mean_gradients(
          grads, aux.loss_weight * p.l2_norm_clip
      )
      # Aggregate values and aux.
      values = jax.tree_map(jax.tree_util.Partial(jnp.mean, axis=0), values)
      aux = self.process_aux_info(aux)
      return (values, aux, grads)

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
    grads = self._add_noise(
        grads,
        p.noise_multiplier * aux.loss_weight * p.l2_norm_clip / batch_size,
        prng_key,
    )
    return (values, aux), grads


class MicrobatchDpSgdStochasticGradient(DpSgdStochasticGradient):
  """DP-SGD stochastic gradient function with microbatch.

  **NOTE** on setting `noise_multiplier` when training with multiple devices:
  the `noise_multiplier` parameter should be divided by `sqrt(num_devices)`.
  See the docstring of `DpSgdStochasticGradient` for more details.
  """

  class HParams(DpSgdStochasticGradient.HParams):
    """Returns the PrivateGradient params."""
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
    microbatch_size = self.hparams.microbatch_size
    return tensor.reshape((batch_size // microbatch_size, microbatch_size,
                           *tensor.shape[1:]))


class AugMulDpSgdStochasticGradient(MicrobatchDpSgdStochasticGradient):
  """DP-SGD with Augmentation Multiplicity.

  **NOTE** on setting `noise_multiplier` when training with multiple devices:
  the `noise_multiplier` parameter should be divided by `sqrt(num_devices)`.
  See the docstring of `DpSgdStochasticGradient` for more details.

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
    num_repeat = self.hparams.microbatch_size
    return jnp.repeat(tensor, num_repeat, axis=0).reshape(
        (shape[0], num_repeat, *shape[1:]))
