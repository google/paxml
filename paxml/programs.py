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
import time
from typing import Any, Callable, Dict, Optional, Protocol, Tuple

import jax
from jax import monitoring

from absl import logging
from paxml import metric_utils
from paxml import partitioning
from paxml import tasks_lib
from paxml import trainer_lib
from paxml import train_states
from praxis import base_hyperparams
from praxis import base_input
from praxis import pytypes
from praxis import py_utils

from paxml import profiling  # mapped to internal

JTensor = pytypes.JTensor
NestedJTensor = pytypes.NestedJTensor
NestedPartitionSpec = pytypes.NestedPartitionSpec
NestedShapeDtypeLike = pytypes.NestedShapeDtypeLike
PRNGKey = pytypes.PRNGKey
SummaryDict = pytypes.SummaryDict
WeightedScalars = pytypes.WeightedScalars

NestedMap = py_utils.NestedMap
TrainState = train_states.TrainState
instantiate = base_hyperparams.instantiate

_INIT_TIME = time.time()


def get_eval_train_state(task: tasks_lib.SingleTask, state: TrainState):
  task_p = task.hparams
  if task_p.train.eval_use_ema_states:
    if not tasks_lib.has_ema(task_p):
      raise ValueError(
          'eval_use_ema_states is requested but the '
          'learner does not seem to have ema enabled'
      )
    eval_state = tasks_lib.extract_ema(state).to_eval_state()
    logging.debug('  Converted train state to eval with ema state.')
  else:
    eval_state = state.to_eval_state()
  return eval_state


@dataclasses.dataclass
class ProgramOutput:
  # The train state that's potentially modified by the program.
  # For example, a train program is expected to update the state to reflect
  # optimizer updates, while a eval program is expected to keep the state as is.
  state: TrainState
  # Auxiliary dictionary that contains any information that program intends to
  # feedback to outer loop.
  aux: NestedMap


class Program(Protocol):
  """The basic interface for a program."""

  # TODO(laigd): add a unified setup() method here.

  def should_run(self, state: TrainState, step_i: int) -> bool:
    """Returns whether the .run() should be called at `state` and `step_i`."""
    ...

  def run(self, state: TrainState, step_i: int) -> ProgramOutput:
    """Returns the program on given state and step."""
    ...


class BaseTrainProgram(Program, metaclass=abc.ABCMeta):
  """A lean interface of a basic train program.

  Users should inherit from BaseTrainProgram and implement methods required to
  form a custom train program.

  TODO(hthu): Write a custom program example.
  """

  def __init__(
      self,
      task: tasks_lib.SingleTask,
      train_input: base_input.BaseInput,
      partitioner: partitioning.Partitioner,
  ):
    self._task = task
    self._train_input = train_input
    self._partitioner = partitioner

    self._train_unpadded_global_batch_size = (
        train_input.hparams.cls.get_global_batch_size(train_input.hparams)
    )
    train_p = self._task.hparams.train
    self._profiler = profiling.Profiler(
        num_steps=train_p.profiler_num_steps,
        min_duration_sec=train_p.profiler_min_duration_sec,
        max_num_hosts=train_p.profiler_max_num_hosts,
    )
    self._first_step_completion_time = None

    # States to initialize lazily by self.setup().
    self._train_prng_seed = None
    self._eval_prng_seed = None
    self._initial_step = -1
    self._init_duration_set = False
    self._train_summary_last_time = None
    self._train_summary_last_step = None
    self._train_summary_handler = None
    self._eval_train_summary_handler = None

  def setup(
      self,
      # TODO(laigd): it should take a root prng key and split it.
      train_prng_seed: pytypes.PRNGKey,
      eval_prng_seed: pytypes.PRNGKey,
      init_step: int,
      train_summary_handler: Any,
      eval_summary_handler: Any,
  ) -> None:
    self._train_prng_seed = train_prng_seed
    self._eval_prng_seed = eval_prng_seed
    self._initial_step = init_step
    self._init_duration_set = False
    self._train_summary_last_time = time.time()
    self._train_summary_last_step = init_step - 1
    self._train_summary_handler = train_summary_handler
    self._eval_train_summary_handler = eval_summary_handler

  def should_run(self, state: TrainState, step_i: int) -> bool:
    return step_i < self._task.hparams.train.num_train_steps

  # TODO(laigd): further split this into smaller modules and add program APIs
  # correspondingly.
  def run(self, state: TrainState, step_i: int) -> ProgramOutput:
    train_p = self._task.hparams.train
    logging.debug('  Retrieving inputs.')
    model_inputs = self._train_input.get_next_padded()
    model_inputs = self._partitioner.preprocess_inputs(
        self._train_input,
        model_inputs,
        self.train_input_partition_spec,
    )
    logging.debug('  Retrieved inputs.')

    profiler_capture_step = train_p.profiler_capture_step
    do_profile = profiler_capture_step is not None
    if do_profile and step_i - self._initial_step == profiler_capture_step:
      self._profiler.capture_async()

    logging.debug('  Performing train_step().')
    with jax.profiler.StepTraceAnnotation('train', step_num=step_i):
      with py_utils.timeit() as train_period:
        (
            new_state,
            loss,
            weighted_scalars,
            per_example_out,
            summary_tensors,
        ) = self.train_step(
            state,
            self._train_prng_seed,
            model_inputs,
            self._train_unpadded_global_batch_size,
        )
      del state  # Unused anymore.
    logging.debug(
        '  Completed train_step() in %f seconds.', train_period.elapsed
    )
    if step_i == self._initial_step:
      self._first_step_completion_time = time.time()

    if do_profile and step_i - self._initial_step < profiler_capture_step:
      self._profiler.update_step_moving_mean(train_period.elapsed)

    new_step_i, steps_per_sec = self._maybe_write_summaries(
        new_state,
        step_i,
        loss,
        weighted_scalars,
        summary_tensors,
        per_example_out,
    )
    logging.debug('  Writing summaries (attempt).')

    # Run eval at regular step interval.
    # While the eval ones below are post-model weight updates, hence we use the
    # new step counter new_step_i.
    eval_train_metrics = None
    if (
        train_p.eval_interval_steps
        and new_step_i % train_p.eval_interval_steps == 0
    ):
      eval_train_metrics = self._maybe_run_eval_train(new_state, new_step_i)

    return ProgramOutput(
        new_state,
        aux=NestedMap(
            loss=loss,
            weighted_scalars=weighted_scalars,
            new_step_i=new_step_i,
            steps_per_sec=steps_per_sec,
            eval_train_metrics=eval_train_metrics,
        ),
    )

  @abc.abstractmethod
  def train_step(
      self,
      state: train_states.TrainState,
      prng_key: PRNGKey,
      inputs: NestedJTensor,
      unpadded_global_batch_size: int,
  ) -> Tuple[TrainState, JTensor, WeightedScalars, NestedMap, SummaryDict]:
    """The train step function."""

  @abc.abstractmethod
  def eval_train_step(
      self,
      state: train_states.TrainState,
      prng_key: PRNGKey,
      inputs: NestedJTensor,
      unpadded_global_batch_size: int,
  ) -> Tuple[JTensor, WeightedScalars, NestedMap, SummaryDict]:
    """The eval step function on training inputs."""

  @property
  @abc.abstractmethod
  def train_input_partition_spec(self) -> Optional[NestedPartitionSpec]:
    """The partition spec for the model training inputs."""

  def _maybe_write_summaries(
      self,
      new_state: TrainState,
      step_i: int,
      loss,
      weighted_scalars,
      summary_tensors,
      per_example_out,
  ):
    new_step_i = step_i + 1
    steps_per_sec = None

    train_p = self._task.hparams.train
    if train_p.device_sync_interval_steps:
      should_sync_device = (
          new_step_i % train_p.device_sync_interval_steps
      ) == 0
    else:
      should_sync_device = self._train_summary_handler.should_write(new_step_i)
    if should_sync_device:
      new_step_i, steps_per_sec = self._sync_device(new_state, step_i)

    # Note: Train metrics are currently reported at step_i + 1, while these
    # training metrics/summaries are pre-model weight updates.
    # TODO(b/264635784): Update the logic to pass step_i instead.
    self._train_summary_handler.process(
        new_step_i,
        loss,
        weighted_scalars,
        summary_tensors,
        per_example_out=per_example_out,
        steps_per_sec=steps_per_sec,
    )
    logging.debug('  Wrote summaries (attempted).')

    return new_step_i, steps_per_sec

  def _sync_device(self, new_state: TrainState, step_i: int):
    # Synchronize step_i. This is performed at a fixed interval to avoid
    # a gap between steps.
    new_step_i = int(
        py_utils.maybe_unreplicate_for_fully_replicated(new_state.step)
    )
    steps_per_sec = self._compute_steps_per_sec(
        step_i, self._train_summary_last_time, self._train_summary_last_step
    )
    logging.info('steps/sec: %f', steps_per_sec)
    self._train_summary_last_time = time.time()
    self._train_summary_last_step = step_i

    if not self._init_duration_set:
      # Find estimated timestamp before the first execution call.
      # This enables us to include the first step's compile time but exclude
      # its execution time from the init duration.
      estimated_execute_duration = 1 / steps_per_sec
      first_step_execute_time = (
          self._first_step_completion_time - estimated_execute_duration
      )
      init_duration = first_step_execute_time - _INIT_TIME
      monitoring.record_event_duration_secs(
          '/jax/pax/init/time_before_first_step_secs', init_duration
      )
      self._init_duration_set = True
    return new_step_i, steps_per_sec

  def _compute_steps_per_sec(
      self, step_i, summary_last_time, summary_last_step
  ):
    """Computes the number of training steps per second."""
    # Note: This function doesn't account for the time spent on running
    # interleaved evaluation (if any) and/or evaluation on the training batch.
    # It's, hence, merely a raw underestimate.
    duration_sec = time.time() - summary_last_time
    num_steps = step_i - summary_last_step
    steps_per_sec = num_steps / duration_sec
    return steps_per_sec

  def _maybe_run_eval_train(self, new_state: TrainState, new_step_i: int):
    train_p = self._task.hparams.train
    eval_train_metrics = None

    if train_p.eval_skip_train:
      logging.debug('  train_p.eval_skip_train is True. Skipping eval_train.')
    else:
      logging.debug('  Retrieving eval model_inputs.')
      eval_inputs = self._train_input.peek_padded()
      if eval_inputs is None:
        logging.debug('  eval_inputs is None. Skipping eval_train.')
      else:
        logging.debug('  Retrieved eval model_inputs.')
        logging.debug('  Performing eval_step() runs on training split.')
        eval_inputs = self._partitioner.preprocess_inputs(
            self._train_input, eval_inputs, self.train_input_partition_spec
        )

        eval_state = get_eval_train_state(self._task, new_state)
        loss, weighted_scalars, _, summary_tensors = self.eval_train_step(
            eval_state,
            self._eval_prng_seed,
            eval_inputs,
            self._train_unpadded_global_batch_size,
        )
        logging.debug('  Completed eval_step() runs on training split.')
        if self._eval_train_summary_handler.process(
            new_step_i, loss, weighted_scalars, summary_tensors
        ):
          logging.debug('  Wrote eval summaries.')
        eval_train_metrics = metric_utils.as_float_dict(weighted_scalars)
    return eval_train_metrics

  # TODO(laigd): remove this.
  @property
  def train_unpadded_global_batch_size(self) -> int:
    return self._train_unpadded_global_batch_size


class SingleTaskTrainProgram(BaseTrainProgram):
  """Train program that assumes a single task on a single dataset."""

  def __init__(
      self,
      task: tasks_lib.SingleTask,
      train_input: base_input.BaseInput,
      partitioner: partitioning.Partitioner,
  ):
    super().__init__(task, train_input, partitioner)

    # Train step function information.
    self._train_step_created = False
    self._train_step_fn = None
    self._train_step_input_partition_spec = None

    # Eval train step function information. Note since this eval step runs on
    # training inputs, it'll have the same input shapes/dtypes and partition
    # spec as the train step function.
    self._eval_train_step_created = False
    self._eval_train_step_fn = None

  def _get_train_step(self) -> Tuple[Any, Optional[NestedPartitionSpec]]:
    """Creates the train step info (if not done before) and returns them."""
    if not self._train_step_created:
      self._train_step_fn, self._train_step_input_partition_spec = (
          self._partitioner.partition(
              trainer_lib.train_step_single_learner,
              self._partitioner.train_inputs_shape_dtype,
              is_eval=False,
          )
      )
      self._train_step_created = True
    return self._train_step_fn, self._train_step_input_partition_spec

  def _get_eval_train_step(self) -> Any:
    """Creates the train step info (if not done before) and returns them."""
    if not self._eval_train_step_created:
      # TODO(pax): Support auto-sharding for eval step. In this case, we would
      # have to fix the sharding of the input to be the same as what's derived
      # from the train_step.

      # Ignores the returned input partition spec. It should be the same as
      # self.train_input_partition_spec since the input shapes are the same.
      self._eval_train_step_fn, _ = self._partitioner.partition(
          trainer_lib.eval_step_single_learner,
          self._partitioner.train_inputs_shape_dtype,  # Train input shapes.
          is_eval=True,
      )
      self._eval_train_step_created = True
    return self._eval_train_step_fn

  def train_step(
      self,
      state: train_states.TrainState,
      prng_key: PRNGKey,
      inputs: NestedJTensor,
      unpadded_global_batch_size: int,
  ) -> Tuple[TrainState, JTensor, WeightedScalars, NestedMap, SummaryDict]:
    """The train step function."""
    train_step, _ = self._get_train_step()
    return train_step(state, prng_key, inputs, unpadded_global_batch_size)

  def eval_train_step(
      self,
      state: train_states.TrainState,
      prng_key: PRNGKey,
      inputs: NestedJTensor,
      unpadded_global_batch_size: int,
  ) -> Tuple[JTensor, WeightedScalars, NestedMap, SummaryDict]:
    """The eval step function on trianing inputs."""
    eval_train_step = self._get_eval_train_step()
    return eval_train_step(state, prng_key, inputs, unpadded_global_batch_size)

  @property
  def train_input_partition_spec(self) -> Optional[NestedPartitionSpec]:
    """The partition spec for the model training inputs."""
    _, input_partition_spec = self._get_train_step()
    return input_partition_spec


class SingleTaskEvalProgram(Program):
  """Eval program that assumes a single task on a single dataset."""

  def __init__(
      self,
      task: tasks_lib.SingleTask,
      input_p: base_input.BaseInput.HParams,
      partitioner: partitioning.Partitioner,
  ):
    self._task = task
    self._input_p = input_p
    self._partitioner = partitioner

    logging.debug('Initializing eval_input pipeline : %s', self._input_p)
    self._eval_input_pipeline = instantiate(
        self._partitioner.preprocess_input_params(self._input_p)
    )

    self._partitioned_step_fn, self._partitioned_input_spec = (
        self.partition_step()
    )

  @property
  def eval_input(self) -> base_input.BaseInput:
    return self._eval_input_pipeline

  @property
  def eval_num_steps(self) -> int:
    return (
        -1
        if self._input_p.reset_for_eval
        else self._input_p.eval_loop_num_batches
    )

  def partition_step(self) -> Tuple[Any, Optional[NestedPartitionSpec]]:
    # A bit of unfortunate conditioning but we have to branch out pmap/pjit
    # case here -- As Pmap can simply take the train_inputs_shape_dtype from
    # the partitioner whearas Pjit need to actually look at current eval input
    # and get shape from there.
    input_shape_dtype = self._partitioner.train_inputs_shape_dtype
    if isinstance(
        self._partitioner,
        (
            partitioning.PjitPartitioner,
            partitioning.AutoShardingPjitPartitioner,
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

    # TODO(laigd): Get rid of inputs_shape_dtype here.
    return self._partitioner.partition(
        trainer_lib.eval_step_single_learner,
        inputs_shape_dtype=input_shape_dtype,
        is_eval=True,
    )

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
    ) = self._partitioned_step_fn(
        state, prng_key, inputs, unpadded_global_batch_size
    )
    return ProgramOutput(
        state,
        aux=NestedMap(
            loss=loss,
            weighted_scalars=weighted_scalars,
            per_example_out=per_example_out,
            summary_tensors=summary_tensors,
        ),
    )

  @property
  def partitioner(self):
    return self._partitioner

  @property
  def partitioned_step_fn(
      self,
  ) -> Callable[[TrainState, PRNGKey, NestedJTensor, int], Any]:
    return self._partitioned_step_fn

  @property
  def partitioned_input_spec(self) -> Optional[NestedPartitionSpec]:
    return self._partitioned_input_spec

  # TODO(laigd): implement these.

  def should_run(self, state: TrainState, step_i: int) -> bool:
    raise NotImplementedError()

  def run(self, state: TrainState, step_i: int) -> ProgramOutput:
    raise NotImplementedError()
