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

"""The basic program concept that encapsulates a per-step runnable."""
import abc
import dataclasses
from typing import Any, Callable, Dict, Optional, Protocol, Tuple

import jax

from paxml import tasks_lib
from paxml import train_states
from praxis import base_input
from praxis import pytypes


TrainState = train_states.TrainState


@dataclasses.dataclass
class ProgramOutput:
  # The train_state that's potentially modified by the program.
  # For example, a train program is expected to update the state to reflect
  # optimizer updates, while a eval program is expected to keep the state as is.
  state: TrainState
  # Auxilary dictionary that contains any information that program intends to
  # feedback to outer loop.
  aux: Dict[str, Any]


class Program(Protocol):
  """The basic interface for a program."""

  def run_step(
      self,
      train_state: TrainState,
      prng_key: Optional[jax.random.KeyArray],
      inputs: pytypes.NestedJTensor,
      unpadded_global_batch_size: int,
  ) -> ProgramOutput:
    """Runs a single step on the program."""
    ...


# TODO(laigd): We need to refactor partitioner out of trainer_lib or we will
# have circular dependencies.
class BasePartitionedProgram(Program, metaclass=abc.ABCMeta):
  """The base class for all programs that the step is under partitioning."""

  def __init__(self, partitioner):
    self._partitioner = partitioner
    self._partitioned_step_fn = None
    self._partitioned_input_spec = None

  @abc.abstractmethod
  def partition_step(self) -> Tuple[Any, Optional[pytypes.NestedPartitionSpec]]:
    """Partition the program step function.

    Default implementation will store partitioned function in
    partitioned_step_fn attribute, along with its partitioned input spec into
    partitioned_input_spec.
    """
    raise NotImplementedError('Subclass must implement partition_step().')

  def run_step(
      self,
      train_state: TrainState,
      prng_key: Optional[jax.random.KeyArray],
      inputs: pytypes.NestedJTensor,
      unpadded_global_batch_size: int,
  ) -> ProgramOutput:
    return self.partitioned_step_fn(
        train_state, prng_key, inputs, unpadded_global_batch_size
    )

  @property
  def partitioner(self):
    return self._partitioner

  @property
  def partitioned_step_fn(
      self,
  ) -> Callable[[TrainState, pytypes.PRNGKey, pytypes.NestedJTensor, int], Any]:
    if self._partitioned_step_fn is None:
      self._partitioned_step_fn, self._partitioned_input_spec = (
          self.partition_step()
      )
    return self._partitioned_step_fn

  @property
  def partitioned_input_spec(self) -> Optional[pytypes.NestedPartitionSpec]:
    # Note that partitioned_input_spec can be `None` for some partitioner, i.e.,
    # Pmap partitioner. So we always check if the partitioned_step_fn is cached.
    if self._partitioned_step_fn is None:
      self._partitioned_step_fn, self._partitioned_input_spec = (
          self.partition_step()
      )
    return self._partitioned_input_spec


class BaseTrainProgram(BasePartitionedProgram, metaclass=abc.ABCMeta):
  """A lean interface of a basic train program.

  Users should inherit from BaseTrainProgram and implement methods required to
  form a custom train program.

  TODO(hthu): Write a custom program example.
  """

  @property
  @abc.abstractmethod
  def train_inputs_shape_dtype(self) -> pytypes.NestedShapeDtypeLike:
    raise NotImplementedError(
        'Subclass must implement train_inputs_global_shape_dtype'
    )

  # TODO(hthu): Move TrainStateMetadata into common library to break up
  # dependency on trainer.
  @property
  @abc.abstractmethod
  def train_state_metadata(self):
    """Gets the TrainStateMetadata used for partitioning.

    Deliberately duplicate so that we can eventually move train_state_metadata
    away from partitioner.

    Returns:
      TrainStateMetadata that represents the metadata used during train.
    """
    raise NotImplementedError('Subclass must implement train_state_metadata')

  @property
  @abc.abstractmethod
  def task(self) -> tasks_lib.SingleTask:
    raise NotImplementedError('Subclass must implement task')

  @property
  @abc.abstractmethod
  def train_input(self) -> base_input.BaseInput:
    raise NotImplementedError('Subclass must implement train_input')

  @property
  @abc.abstractmethod
  def train_unpadded_global_batch_size(self) -> int:
    raise NotImplementedError(
        'Subclass must implement train_unpadded_global_batch_size'
    )
