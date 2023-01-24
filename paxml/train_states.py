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

"""TrainState class for encapsulating model weights and optimizer states."""

from __future__ import annotations

from typing import List

from flax import struct as flax_struct
import jax
import optax
from praxis import base_layer

NestedJTensor = base_layer.NestedJTensor
JTensorOrPartitionSpec = base_layer.JTensorOrPartitionSpec
NestedJTensorOrPartitionSpec = base_layer.NestedJTensorOrPartitionSpec


# A helper class for managing various train states. This struct may contain the
# actual Jax tensors, or simply PartitionSpecs for the corresponding tensor.
# If the latter, this struct is used for specifying the PartitionSpecs for
# input/output to/from a pjit-ed function.
class TrainState(flax_struct.PyTreeNode):
  """Simple train state."""

  step: JTensorOrPartitionSpec
  mdl_vars: NestedJTensorOrPartitionSpec
  opt_states: List[NestedJTensorOrPartitionSpec]

  def new_state(
      self, mdl_vars: NestedJTensor, opt_states: List[optax.OptState]
  ) -> TrainState:
    """Returns a new TrainState with updated mdl_vars and opt_states."""
    mdl_vars = jax.tree_util.tree_map(lambda x: x, mdl_vars)
    opt_states = jax.tree_util.tree_map(lambda x: x, opt_states)
    return TrainState(
        step=self.step + 1, mdl_vars=mdl_vars, opt_states=opt_states
    )

  def to_eval_state(self) -> TrainState:
    """Returns a new TrainState with opt_states removed, for eval purpose."""
    return TrainState(step=self.step, mdl_vars=self.mdl_vars, opt_states={})
