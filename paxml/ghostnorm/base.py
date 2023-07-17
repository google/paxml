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

"""Ghost norm support with jax.custom_vjp.

## Introduction

Ghost norm protocol compute per-example gradient norms using the information
available in standard back-propagation, without explicitly materializing each
of the per-example gradient tensors. This is used in DP-SGD to save memory cost
and to enable large batch size training without (sequential) gradient
accumulation.

This library supports ghost norm via jax.custom_vjp, which allows defining
custom gradient computation to carry out extra computations (of the per-example
gradient norms). However, because the custom_vjp interface does not allow
outputing arbitrary auxiliary computation results, we used a special protocol
for this.

Note: in order to use ghost norm clipping based DP-SGD, all the parameteric
layers (layers with trainable parameters) in the model need to implement this
protocol. This library provides support for basic layer types. For composite
layers that can be defined using more primitive layer types (e.g. layer norm can
be implemented using fully connected layers), there is no need for extra
modification as long as the primitive (parameteric) layers implement the
protocol.

## Ghost Norm Protocol

For a layer with trainable parameters, it should implement a `jax.custom_vjp`
rule. See `paxml.ghostnorm.linears.LinearGhostNorm` for a concrete example.

### Forward Pass

- If this layer is being called outside the context of ghost norm clipping
  DP-SGD (e.g. during evaluation, or training without DP), the parameters will
  be normal (a jax array or a PyTree).
- If this layer is being called for ghost norm clipping DP-SGD, the parameters
  will be wrapped in a `ParamWithAux` struct. The `param` attribute contains
  the original parameters (a jax array or a PyTree), and the `aux` attribute
  is a vector of length equals to the batch size. During forward pass, this
  `aux` vector should be interpreted as per-example reweighting coefficients.
  Normally, this vector is constant ones. But in the second pass of the 2-pass
  DP-SGD algorithm with ghost norm clipping, this reweighting is used to obtain
  the average clipped gradients. So the backward rule for trainable parameters
  should take this reweighting into consideration. However, the backward
  gradients with respect to the layer inputs (to be passed down to lower layers)
  should not be rescaled, because each layer takes care of its own scaling.

### Backward Pass

- If this layer is being called outside the context of ghost norm clipping
  DP-SGD, the backward procedure should proceed normally.
- If this layer is being called for ghost norm clipping DP-SGD, because the
  parameters in the forward pass is a `ParamWithAux` struct, the backward rule
  is also supposed to output such a struct. Here the gradients of the parameters
  should be put in the `param` attribute, and the per-example gradient norms
  should be put in the `aux` attribute.
"""

from flax import struct
import jax
from praxis import pytypes

JTensor = pytypes.JTensor
NestedJTensor = pytypes.NestedJTensor
Nested = pytypes.Nested


@struct.dataclass
class ParamWithAux:
  """Wrapper for parameters to pass auxiliary info to/from jax custom_vjp."""

  param: JTensor
  aux: JTensor | None

  @property
  def dtype(self):
    return self.param.dtype

  def astype(self, dtype):
    aux = None
    if self.aux is not None:
      aux = self.aux.astype(dtype)
    return self.replace(param=self.param.astype(dtype), aux=aux)


def _get_param(param: ParamWithAux | JTensor) -> JTensor:
  if isinstance(param, ParamWithAux):
    return param.param
  return param


def _get_aux(param: ParamWithAux | JTensor) -> JTensor | None:
  if isinstance(param, ParamWithAux):
    return param.aux
  return None


def get_param(params: Nested[ParamWithAux] | NestedJTensor) -> NestedJTensor:
  is_leaf = lambda x: isinstance(x, ParamWithAux)
  return jax.tree_util.tree_map(_get_param, params, is_leaf=is_leaf)


def get_aux(params: Nested[ParamWithAux] | NestedJTensor) -> JTensor | None:
  is_leaf = lambda x: isinstance(x, ParamWithAux)
  return _get_aux(jax.tree_util.tree_leaves(params, is_leaf=is_leaf)[0])
