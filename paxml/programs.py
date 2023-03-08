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

from absl import logging
from paxml import tasks_lib
from paxml import trainer_lib
from paxml import train_states
from praxis import base_hyperparams
from praxis import base_input
from praxis import pytypes
from praxis import py_utils


NestedJTensor = pytypes.NestedJTensor
NestedPartitionSpec = pytypes.NestedPartitionSpec
NestedShapeDtypeLike = pytypes.NestedShapeDtypeLike
PRNGKey = pytypes.PRNGKey

TrainState = train_states.TrainState

instantiate = base_hyperparams.instantiate


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
      inputs: NestedJTensor,
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
  def partition_step(self) -> Tuple[Any, Optional[NestedPartitionSpec]]:
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
      inputs: NestedJTensor,
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
  ) -> Callable[[TrainState, PRNGKey, NestedJTensor, int], Any]:
    if self._partitioned_step_fn is None:
      self._partitioned_step_fn, self._partitioned_input_spec = (
          self.partition_step()
      )
    return self._partitioned_step_fn

  @property
  def partitioned_input_spec(self) -> Optional[NestedPartitionSpec]:
    # Note that partitioned_input_spec can be `None` for some partitioner, i.e.,
    # Pmap partitioner. So we always check if the partitioned_step_fn is cached.
    if self._partitioned_step_fn is None:
      self._partitioned_step_fn, self._partitioned_input_spec = (
          self.partition_step()
      )
    return self._partitioned_input_spec


class BaseTrainProgram(BasePartitionedProgram):
  """A lean interface of a basic train program.

  Users should inherit from BaseTrainProgram and implement methods required to
  form a custom train program.

  TODO(hthu): Write a custom program example.
  """

  @property
  @abc.abstractmethod
  def train_inputs_shape_dtype(self) -> NestedShapeDtypeLike:
    raise NotImplementedError(
        'Subclass must implement train_inputs_global_shape_dtype'
    )

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


class SingleTaskTrainProgram(BaseTrainProgram):
  """Train program that assumes a single task on a single dataset."""

  def __init__(
      self,
      task: tasks_lib.SingleTask,
      train_input: base_input.BaseInput,
      partitioner: trainer_lib.Partitioner,
  ):
    super().__init__(partitioner)
    self._task = task
    self._train_input = train_input
    self._train_unpadded_global_batch_size = (
        train_input.hparams.cls.get_global_batch_size(train_input.hparams)
    )
    # Lazily initialized values.
    self._train_state_metadata = None
    self._step_fn = None
    self._input_pspecs = None
    self._train_inputs_shape_dtype = None

  def partition_step(self) -> Tuple[Any, Optional[NestedPartitionSpec]]:
    """PJIT and cache the pjit-ed step_fn.

    Returns:
      A tuple of (partitioned step function, partitioned input spec).
    """
    if self._partitioned_step_fn is None:
      self._partitioned_step_fn, self._partitioned_input_spec = (
          self._partitioner.partition(
              trainer_lib.train_step_single_learner,
              self._partitioner.train_inputs_shape_dtype,
              is_eval=False,
          )
      )
    return self._partitioned_step_fn, self._partitioned_input_spec

  def run_step(
      self,
      state: TrainState,
      prng_key: jax.random.KeyArray,
      inputs: Any,
      unpadded_global_batch_size: int,
  ) -> ProgramOutput:
    (
        partitioned_train_state,
        loss,
        weighted_scalars,
        per_example_out,
        summary_tensors,
    ) = self.partitioned_step_fn(
        state, prng_key, inputs, unpadded_global_batch_size
    )
    return ProgramOutput(
        partitioned_train_state,
        aux={
            'loss': loss,
            'weighted_scalars': weighted_scalars,
            'per_example_out': per_example_out,
            'summary_tensors': summary_tensors,
        },
    )

  @property
  def train_inputs_shape_dtype(self) -> NestedShapeDtypeLike:
    return self._partitioner.train_inputs_shape_dtype

  @property
  def train_state_metadata(self) -> trainer_lib.TrainStateMetadata:
    """Gets the TrainStateMetadata used for partitioning.

    Deliberately duplicate so that we can eventually move train_state_metadata
    away from partitioner.

    Returns:
      TrainStateMetadata that represents the
    """
    return self._partitioner.get_train_state_metadata()

  @property
  def task(self) -> tasks_lib.SingleTask:
    return self._task

  @property
  def train_input(self) -> base_input.BaseInput:
    return self._train_input

  @property
  def train_unpadded_global_batch_size(self) -> int:
    return self._train_unpadded_global_batch_size


class SingleTaskEvalProgram(BasePartitionedProgram):
  """Eval program that assumes a single task on a single dataset."""

  def __init__(
      self,
      task: tasks_lib.SingleTask,
      input_p: base_input.BaseInput.HParams,
      partitioner: trainer_lib.Partitioner,
  ):
    super().__init__(partitioner)
    self._task = task
    self._input_p = input_p
    # Lazily initialized per first use.
    self._eval_input_pipeline = None

  def _init_pipeline(self) -> base_input.BaseInput:
    """Initialize the pipeline for eval_input."""
    if self._eval_input_pipeline is None:
      return instantiate(
          self._partitioner.preprocess_input_params(self._input_p)
      )
    return self._eval_input_pipeline

  @property
  def task(self) -> tasks_lib.SingleTask:
    return self._task

  @property
  def eval_input(self) -> base_input.BaseInput:
    if self._eval_input_pipeline is None:
      logging.debug('Initializing eval_input pipeline : %s', self._input_p)
      self._eval_input_pipeline = self._init_pipeline()
    return self._eval_input_pipeline

  @property
  def eval_num_steps(self) -> int:
    return (
        -1
        if self._input_p.reset_for_eval
        else self._input_p.eval_loop_num_batches
    )

  def partition_step(self) -> Tuple[Any, Optional[NestedPartitionSpec]]:
    if self._partitioned_step_fn is None:
      # A bit of unfortunate conditioning but we have to branch out pmap/pjit
      # case here -- As Pmap can simply take the train_inputs_shape_dtype from
      # the partitioner whearas Pjit need to actually look at current eval input
      # and get shape from there.
      input_shape_dtype = self._partitioner.train_inputs_shape_dtype
      if isinstance(
          self._partitioner,
          (
              trainer_lib.PjitPartitioner,
              trainer_lib.AutoShardingPjitPartitioner,
          ),
      ):
        # Instantiate a stanalone pipeline for one-time use to get sample inputs
        # since the peek_padded() can return None if the pipeline is exhausted.
        # This can happen when the input_pipeline is used before the partitioned
        # step function is invoked as we do it lazily.
        cloned_input_p = self.eval_input.hparams.clone()
        # Note that the hparams from eval_input is already preprocessed by
        # partitioner, so we don't need to do another adjustment here.
        cloned_pipeline: base_input.BaseInput = instantiate(cloned_input_p)
        input_shape_dtype = jax.tree_map(
            py_utils.get_global_input_shape_dtype,
            cloned_pipeline.get_next_padded(),
        )
        # delete one-time usages.
        del cloned_pipeline, cloned_input_p
      self._partitioned_step_fn, self._partitioned_input_spec = (
          # TODO(laigd): Get rid of inputs_shape_dtype here.
          self._partitioner.partition(
              trainer_lib.eval_step_single_learner,
              inputs_shape_dtype=input_shape_dtype,
              is_eval=True,
          )
      )

    return self._partitioned_step_fn, self._partitioned_input_spec

  def run_step(
      self,
      state: TrainState,
      prng_key: jax.random.KeyArray,
      inputs: Any,
      unpadded_global_batch_size: int,
  ) -> ProgramOutput:
    (
        loss,
        weighted_scalars,
        per_example_out,
        summary_tensors,
    ) = self.partitioned_step_fn(
        state, prng_key, inputs, unpadded_global_batch_size
    )
    return ProgramOutput(
        state,
        aux={
            'loss': loss,
            'weighted_scalars': weighted_scalars,
            'per_example_out': per_example_out,
            'summary_tensors': summary_tensors,
        },
    )
