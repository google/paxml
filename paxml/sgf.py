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

"""Module with the stochastic gradient function classes.

TODO(b/259501483): This is is currently aliasing Praxis sgf.py's module
symbol(s), until sgf.py gets fully migrated into Paxml.
"""

from typing import Any, Callable, Tuple

import jax
from jax import numpy as jnp
from praxis import py_utils
from praxis import pytypes
from praxis import sgf

JTensor = pytypes.JTensor
NestedMap = py_utils.NestedMap
PRNGKey = pytypes.PRNGKey


GradAuxInfo = sgf.GradAuxInfo
BaseStochasticGradient = sgf.BaseStochasticGradient
StandardGradient = sgf.StandardGradient
DpSgdStochasticGradient = sgf.DpSgdStochasticGradient


class MicrobatchDpSgdStochasticGradient(DpSgdStochasticGradient):
  """DP-SGD stochastic gradient function with microbatch."""

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
