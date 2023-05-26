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

"""TrainState class for encapsulating model weights and optimizer states."""

from __future__ import annotations

import dataclasses
from typing import Any, List, Optional, Tuple

from flax import struct as flax_struct
import jax
from jax.sharding import PartitionSpec
import optax
from praxis import base_layer
from praxis import py_utils
from praxis import pytypes

JTensor = py_utils.JTensor
JTensorProvenance = Tuple[str, Optional[int]]
JTensorOrPartitionSpec = base_layer.JTensorOrPartitionSpec
Nested = pytypes.Nested
NestedJTensor = base_layer.NestedJTensor
NestedJTensorOrPartitionSpec = base_layer.NestedJTensorOrPartitionSpec
NestedMap = py_utils.NestedMap


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
    return TrainState(step=self.step, mdl_vars=self.mdl_vars, opt_states=[])


@dataclasses.dataclass
class TensorProvenance:
  checkpoint_path: str = 'random_init'
  checkpoint_step: Optional[int] = None

  def __repr__(self) -> str:
    if self.checkpoint_path == 'random_init':
      return f'"({self.checkpoint_path})"'

    checkpoint_step_repr = (
        self.checkpoint_step if self.checkpoint_step else 'latest')

    return f'"({self.checkpoint_path}:{checkpoint_step_repr})"'


@dataclasses.dataclass
class TrainStateProvenance:
  """Provenance for the TrainState pytree struct (not jax-transformable)."""
  step: TensorProvenance
  mdl_vars: Nested[TensorProvenance]
  opt_states: Nested[TensorProvenance]

  def replace(self, **changes: Any) -> TrainStateProvenance:
    return dataclasses.replace(self, **changes)


def build_train_state_provenance(
    train_state: TrainState,
    checkpoint_path: Optional[str] = None,
    step: Optional[int] = None) -> TrainStateProvenance:
  assert not isinstance(train_state.step, PartitionSpec), (
      'Tensor provenance is only for tensors')

  provenance = TensorProvenance()
  if checkpoint_path:
    provenance = TensorProvenance(checkpoint_path=checkpoint_path,
                                  checkpoint_step=step)
  return TrainStateProvenance(
      step=provenance,
      mdl_vars=jax.tree_map(lambda x: provenance, train_state.mdl_vars),
      opt_states=jax.tree_map(lambda x: provenance, train_state.opt_states))
