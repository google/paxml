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
from typing import Any, Callable, Tuple

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

    l2_norm_clip: float = None
    noise_multiplier: float = 0.0

  def _clip_and_noise(
      self, grads: NestedMap, prng_key: PRNGKey = None, loss_weight: float = 1.0
  ) -> Tuple[NestedMap, float]:
    p = self.hparams

    grads_flat, grads_treedef = jax.tree_flatten(grads)
    sum_clipped, num_clipped = optax.per_example_global_norm_clip(
        grads=grads_flat, l2_norm_clip=loss_weight * p.l2_norm_clip
    )
    sum_grads = jax.tree_unflatten(grads_treedef, sum_clipped)
    # Average gradients across all examples
    batch_size = grads_flat[0].shape[0]
    clipped_grads_mean = jax.tree_map(lambda x: x / batch_size, sum_grads)
    frac_clipped = num_clipped / batch_size

    if p.noise_multiplier == 0.0:
      final_grads = clipped_grads_mean
    else:
      noise_stddev = (
          p.noise_multiplier * loss_weight * p.l2_norm_clip / batch_size
      )
      prng_keys = jax.random.split(
          prng_key, len(jax.tree_leaves(clipped_grads_mean))
      )
      prng_tree = jax.tree_unflatten(
          jax.tree_structure(clipped_grads_mean), prng_keys
      )
      final_grads = jax.tree_map(
          lambda x, prng: x
          + noise_stddev * jax.random.normal(prng, shape=x.shape),
          clipped_grads_mean,
          prng_tree,
      )

    return final_grads, frac_clipped

  def process_aux_info(self, aux_info):
    aux_info = jax.tree_map(jax.tree_util.Partial(jnp.mean, axis=0), aux_info)
    return aux_info

  def grad_fn(
      self,
      loss_fn: Callable[..., Any],
      mdl_vars: Any,
      inputs: NestedMap,
      prng_key: PRNGKey,
  ) -> Tuple[Any, Any]:
    grad_fn = jax.vmap(
        jax.value_and_grad(loss_fn, has_aux=True, allow_int=True),
        in_axes=(None, 0, None),
        out_axes=0,
    )
    inputs = jax.tree_map(
        jax.tree_util.Partial(jnp.expand_dims, axis=1), inputs
    )
    (values, aux), grads = grad_fn(mdl_vars, inputs, prng_key)

    aux = self.process_aux_info(aux)
    grads, _ = self._clip_and_noise(grads, prng_key, aux.loss_weight)
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

  def grad_fn(self, loss_fn: Callable[..., Any], mdl_vars: Any,
              inputs: NestedMap, prng_key: PRNGKey) -> Tuple[Any, Any]:
    grad_fn = jax.vmap(
        jax.value_and_grad(loss_fn, has_aux=True, allow_int=True),
        in_axes=(None, 0, None),
        out_axes=0)
    inputs = jax.tree_map(self._prepare_for_microbatching, inputs)
    (values, aux), grads = grad_fn(mdl_vars, inputs, prng_key)

    aux = self.process_aux_info(aux)
    grads, _ = self._clip_and_noise(grads, prng_key, aux.loss_weight)
    return (values, aux), grads

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
