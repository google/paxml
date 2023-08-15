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
import collections
import contextlib
import dataclasses
import queue
import time
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

from absl import flags
from absl import logging
from etils import epath
import jax
from jax import monitoring
from jax.experimental import multihost_utils
import numpy as np
from paxml import io_utils
from paxml import partitioning
from paxml import tasks_lib
from paxml import train_states
from paxml import trainer_lib
from paxml import xla_passthrough
from praxis import base_hyperparams
from praxis import base_input
from praxis import base_layer
from praxis import lazy_loader
from praxis import pax_fiddle
from praxis import py_utils
from praxis import pytypes
from praxis import trees

# Those modules are slow to import, so we do it lazily.
metric_utils = lazy_loader.LazyLoader(
    'metric_utils', globals(), 'paxml.metric_utils'
)
seqio_input = lazy_loader.LazyLoader(
    'seqio_input', globals(), 'paxml.seqio_input'
)
summary_utils = lazy_loader.LazyLoader(
    'summary_utils', globals(), 'paxml.summary_utils'
)
tf = lazy_loader.LazyLoader('tf', globals(), 'tensorflow.compat.v2')
profiling = lazy_loader.LazyLoader(
    'profiling', globals(), 'paxml.profiling'  # mapped to internal
)

JTensor = pytypes.JTensor
NestedJTensor = pytypes.NestedJTensor
NestedPartitionSpec = pytypes.NestedPartitionSpec
NestedShapeDtypeLike = pytypes.NestedShapeDtypeLike
PRNGKey = pytypes.PRNGKey
SummaryDict = pytypes.SummaryDict
WeightedScalars = pytypes.WeightedScalars
EvaluationMode = io_utils.EvaluationMode
SummaryWriter = 'tf.summary.SummaryWriter'  # pylint:disable=invalid-name
StepFnOutput = trainer_lib.StepFnOutput
BaseStepFnStaticArgs = trainer_lib.BaseStepFnStaticArgs

NestedMap = py_utils.NestedMap
TrainState = train_states.TrainState
instantiate = base_hyperparams.instantiate

_INIT_TIME = time.time()


def get_eval_train_state(
    task: tasks_lib.SingleTask, state: TrainState, use_ema: bool
):
  """Returns a TrainState for evaluation (eval/decode).

  Args:
    task: The jax task.
    state: The TrainState for training (i.e. with opt_states).
    use_ema: Whether to use ema variables for eval/decode.

  Returns:
    The TrainState for evaluation, with the opt_states stripped out.
  """
  if use_ema:
    if not tasks_lib.has_ema(task):
      raise ValueError(
          'use_ema is requested but the learner does not seem to have ema '
          'enabled.'
      )
    eval_state = tasks_lib.extract_ema(state).to_eval_state()
    logging.info('[PAX STATUS]: Converted train state to eval with EMA state.')
  else:
    eval_state = state.to_eval_state()
  return eval_state


def get_summary_base_dir(job_log_dir: epath.Path) -> epath.Path:
  return job_log_dir / 'summaries'


def _train_log_interval_steps(
    train_p: tasks_lib.SingleTask.TrainHParams,
) -> int:
  """Returns the interval to log train outputs."""
  if train_p.log_train_output_interval_steps is not None:
    return train_p.log_train_output_interval_steps
  else:
    return train_p.summary_interval_steps


@dataclasses.dataclass(frozen=True)
class ProgramOutput:
  # The train state that's potentially modified by the program.
  # For example, a train program is expected to update the state to reflect
  # optimizer updates, while a eval program is expected to keep the state as is.
  state: TrainState


@dataclasses.dataclass(frozen=True)
class TrainProgramOutput(ProgramOutput):
  loss: Optional[JTensor]
  weighted_scalars: Optional[WeightedScalars]
  new_train_step: int
  steps_per_sec: float
  eval_train_metrics: Optional[Mapping[str, float]]


@dataclasses.dataclass(frozen=True)
class EvalProgramOutput(ProgramOutput):
  eval_metrics: Optional[Mapping[str, float]] = None
  eval_scoring_metrics: Optional[Mapping[str, float]] = None
  num_eval_steps: int = 0


class Program(metaclass=abc.ABCMeta):
  """The basic interface for a program."""

  # TODO(laigd): add a unified setup() method here.

  @abc.abstractmethod
  def should_run(self, state: TrainState, step: int) -> bool:
    """Whether .run() should be called at `state` and train step `step`."""

  @abc.abstractmethod
  def run(self, state: TrainState, step: int) -> ProgramOutput:
    """Runs the program on given `state` and train step `step`."""

  @abc.abstractmethod
  def shutdown(self) -> None:
    """Runs any necessary cleanup."""


class _InflightQueue:
  """Tracks and limits the number of inflight computations."""

  def __init__(self, max_inflight: int):
    self._inflight_queue = None
    if max_inflight > 0:
      self._inflight_queue = queue.Queue(maxsize=max_inflight)

  def add_computation(self, computation: JTensor):
    """Adds a pending on-device computation."""
    if self._inflight_queue:
      self._inflight_queue.put(computation)

  def wait_for_next(self):
    """If the queue is full, wait for the next computation to finish."""
    if self._inflight_queue and self._inflight_queue.full():
      self._inflight_queue.get().block_until_ready()

  def wait_for_all(self):
    """Wait for all inflight computations to finish."""
    if self._inflight_queue:
      while not self._inflight_queue.empty():
        self._inflight_queue.get().block_until_ready()


class BaseTrainProgram(Program):
  """A lean interface of a basic train program.

  Users should inherit from BaseTrainProgram and implement methods required to
  form a custom train program.

  TODO(hthu): Write a custom program example.
  """

  def __init__(self):
    # States to set in self.setup().
    self._task: tasks_lib.SingleTask = None
    self._train_input: base_input.BaseInput = None
    self._partitioner: partitioning.Partitioner = None
    self._train_prng_seed: PRNGKey = None
    self._eval_prng_seed: PRNGKey = None
    self._initial_step = -1

    # States to initialize lazily in self.setup().
    self._train_unpadded_global_batch_size: int = None
    self._profiler: profiling.Profiler = None
    self._train_summary_writer: SummaryWriter = None
    self._train_summary_handler: summary_utils.SummaryHandler = None
    self._eval_train_summary_handler: summary_utils.SummaryHandler = None
    self._train_summary_last_time = None
    self._train_summary_last_step = None
    # Used to limit the number of inflight training steps.
    self._pending_train_losses: _InflightQueue = None

    # Other states used during training.
    self._first_step_completion_time: float = None
    self._init_duration_set = False

    # Used to enter context of various summary writer at .setup().
    self._exitstack = contextlib.ExitStack()

  @property
  def train_input(self) -> base_input.BaseInput:
    assert self._train_input
    return self._train_input

  @property
  def summary_writer(self) -> SummaryWriter:
    assert self._train_summary_writer
    return self._train_summary_writer

  def setup(
      self,
      task: tasks_lib.SingleTask,
      train_input: base_input.BaseInput,
      partitioner: partitioning.Partitioner,
      job_log_dir: epath.Path,
      # TODO(laigd): it should take a root prng key and split it.
      train_prng_seed: PRNGKey,
      eval_prng_seed: PRNGKey,
      init_step: int,
  ) -> None:
    logging.info('[PAX STATUS]: Setting up BaseTrainProgram.')
    self._task = task
    self._train_input = train_input
    self._partitioner = partitioner
    self._train_prng_seed = train_prng_seed
    self._eval_prng_seed = eval_prng_seed
    self._initial_step = init_step

    # Creates the train summary writer and handler.
    summary_base_dir = get_summary_base_dir(job_log_dir)
    summary_train_dir = summary_base_dir / 'train'
    self._train_summary_writer = self._exitstack.enter_context(
        summary_utils.get_summary_writer(summary_train_dir)
    )
    train_p = self._task.train
    self._train_summary_handler = summary_utils.SummaryHandler(
        self._train_summary_writer,
        train_p.summary_interval_steps,
        accumulate_interval_steps=train_p.summary_accumulate_interval_steps,
        log_interval_steps=_train_log_interval_steps(train_p),
        is_async=train_p.async_summary_writing,
        name='training',
    )

    # Creates the summary writer and handler for eval on train input.
    if not train_p.eval_skip_train:
      summary_eval_train_dir = summary_base_dir / 'eval_train'
      eval_train_summary_writer = self._exitstack.enter_context(
          summary_utils.get_summary_writer(summary_eval_train_dir)
      )
      self._eval_train_summary_handler = summary_utils.SummaryHandler(
          eval_train_summary_writer,
          train_p.summary_interval_steps,
          accumulate_interval_steps=train_p.summary_accumulate_interval_steps,
          name='eval',
      )

    # Initializes other states.
    self._train_unpadded_global_batch_size = (
        train_input.get_global_batch_size(train_input)
    )
    self._profiler = profiling.Profiler(
        num_steps=train_p.profiler_num_steps,
        min_duration_sec=train_p.profiler_min_duration_sec,
        max_num_hosts=train_p.profiler_max_num_hosts,
    )
    self._train_summary_last_time = time.time()
    self._train_summary_last_step = init_step - 1
    self._pending_train_losses = _InflightQueue(train_p.max_inflight_steps)

  def should_run(self, state: TrainState, step: int) -> bool:
    return step < self._task.train.num_train_steps

  # TODO(laigd): further split this into smaller modules and add program APIs
  # correspondingly.
  @py_utils.benchmark('[PAX STATUS]: ', first_n=20)
  def run(self, state: TrainState, step: int) -> TrainProgramOutput:
    train_p = self._task.train
    logging.log_first_n(logging.INFO, '[PAX STATUS]:  Retrieving inputs.', 5)
    model_inputs = self._train_input.get_next_padded()

    # Verify user-provided spec matches the first batch's structure.
    if step == self._initial_step and train_p.enforce_input_specs:
      self._partitioner.check_input_spec(model_inputs)

    model_inputs = self._partitioner.preprocess_inputs(
        self._train_input,
        model_inputs,  ## First two args can be consolidated
        self.train_input_partition_spec(model_inputs),
    )
    logging.log_first_n(logging.INFO, '[PAX STATUS]:  Retrieved inputs.', 5)

    # Waits if it reaches max inflight steps. We do this after retrieving the
    # inputs to maximize efficiency.
    self._pending_train_losses.wait_for_next()

    profiler_capture_step = train_p.profiler_capture_step
    do_profile = profiler_capture_step is not None
    if do_profile and step - self._initial_step == profiler_capture_step:
      self._profiler.capture_async()

    logging.log_first_n(
        logging.INFO, '[PAX STATUS]:  Performing train_step().', 5
    )
    with jax.profiler.StepTraceAnnotation('train', step_num=step):
      with py_utils.timeit() as train_period:
        new_step, new_state, train_outputs = self.train_step(
            step,
            state,
            self._train_prng_seed,
            model_inputs,
            BaseStepFnStaticArgs(
                unpadded_global_batch_size=self._train_unpadded_global_batch_size
            ),
        )
      del state  # Unused.
    jax.monitoring.record_event_duration_secs(
        '/jax/pax/train/duration_sec', train_period.elapsed
    )
    logging.log_first_n(
        logging.INFO,
        '[PAX STATUS]: train_step() took %f seconds.',
        20,
        train_period.elapsed,
    )
    self._pending_train_losses.add_computation(train_outputs.loss)
    if step == self._initial_step:
      self._first_step_completion_time = time.time()

    if do_profile and step - self._initial_step < profiler_capture_step:
      self._profiler.update_step_moving_mean(train_period.elapsed)
    logging.log_first_n(
        logging.INFO, '[PAX STATUS]:  Writing summaries (attempt).', 5
    )
    steps_per_sec = self._maybe_write_summaries(step, new_step, train_outputs)

    # Run eval at regular step interval.
    # While the eval ones below are post-model weight updates, hence we use the
    # new step counter new_step.
    eval_train_metrics = None
    if (
        train_p.eval_interval_steps
        and new_step % train_p.eval_interval_steps == 0
    ):
      eval_train_metrics = self._maybe_run_eval_train(new_state, new_step)

    return TrainProgramOutput(
        new_state,
        loss=train_outputs.loss,
        weighted_scalars=train_outputs.weighted_scalars,
        new_train_step=new_step,
        steps_per_sec=steps_per_sec,
        eval_train_metrics=eval_train_metrics,
    )

  @abc.abstractmethod
  def train_step(
      self,
      step: int,
      state: train_states.TrainState,
      prng_key: PRNGKey,
      inputs: NestedJTensor,
      static_args: BaseStepFnStaticArgs,
  ) -> Tuple[int, TrainState, StepFnOutput]:
    """The train step function.

    Args:
      step: The current train step counter.
      state: The current train state.
      prng_key: The PRNG key for this train step.
      inputs: The data input for this train step.
      static_args: Encapsulates any static arguments needed by the step
        function.

    Returns:
      A tuple (new_step, new_state, train_step_fn_out), where:

      - new_step: The updated train step counter. Usually this should be step+1.
      - new_state: The updated train state.
      - train_step_fn_out: A StepFnOutput instance encapsulating the output of
        the train step function.
    """

  @abc.abstractmethod
  def eval_train_step(
      self,
      state: train_states.TrainState,
      prng_key: PRNGKey,
      inputs: NestedJTensor,
      static_args: BaseStepFnStaticArgs,
  ) -> StepFnOutput:
    """The eval step function on training inputs.

    Args:
      state: The current train state.
      prng_key: The PRNG key for this eval step.
      inputs: The training data input for this eval step.
      static_args: Encapsulates any static arguments needed by the step
        function.

    Returns:
      A StepFnOutput instance encapsulating the output of the eval step
      function.
    """

  @abc.abstractmethod
  def train_input_partition_spec(
      self, inputs: NestedJTensor
  ) -> Optional[NestedPartitionSpec]:
    """Returns the partition spec for the model training inputs."""

  def _maybe_write_summaries(
      self, step: int, new_step: int, train_outputs: StepFnOutput
  ) -> Optional[float]:
    # Compute steps/sec every this many steps, revisit when necessary.
    compute_steps_per_sec_interval_steps = 10

    steps_per_sec = None
    should_compute_steps_per_sec = (
        new_step % compute_steps_per_sec_interval_steps == 0
    )
    if should_compute_steps_per_sec:
      steps_per_sec = self._compute_steps_per_sec(step)

    # Note: Train metrics are currently reported at step + 1, while these
    # training metrics/summaries are pre-model weight updates.
    # TODO(b/264635784): Update the logic to pass step instead.
    self._train_summary_handler.process(
        new_step,
        train_outputs.loss,
        train_outputs.weighted_scalars,
        train_outputs.summary_tensors,
        per_example_out=train_outputs.per_example_out,
        steps_per_sec=steps_per_sec,
    )
    logging.debug('[PAX STATUS]:  Wrote summaries (attempted).')
    return steps_per_sec

  def _compute_steps_per_sec(self, step: int):
    steps_per_sec = self._steps_per_sec(
        step, self._train_summary_last_time, self._train_summary_last_step
    )
    logging.info('steps/sec: %f', steps_per_sec)
    self._train_summary_last_time = time.time()
    self._train_summary_last_step = step

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
    return steps_per_sec

  def _steps_per_sec(self, step, summary_last_time, summary_last_step):
    """Computes the number of training steps per second."""
    # Note: This function doesn't account for the time spent on running
    # interleaved evaluation (if any) and/or evaluation on the training batch.
    # It's, hence, merely a raw underestimate.
    assert summary_last_time is not None
    duration_sec = time.time() - summary_last_time
    num_steps = step - summary_last_step
    steps_per_sec = num_steps / duration_sec
    return steps_per_sec

  def _maybe_run_eval_train(
      self, new_state: TrainState, new_step: int
  ) -> Optional[Mapping[str, float]]:
    train_p = self._task.train
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
            self._train_input,
            eval_inputs,
            self.train_input_partition_spec(eval_inputs),
        )

        eval_state = get_eval_train_state(
            self._task, new_state, self._task.train.eval_use_ema_states
        )
        eval_outputs = self.eval_train_step(
            eval_state,
            self._eval_prng_seed,
            eval_inputs,
            BaseStepFnStaticArgs(
                unpadded_global_batch_size=self._train_unpadded_global_batch_size
            ),
        )
        loss = eval_outputs.loss
        weighted_scalars = eval_outputs.weighted_scalars
        summary_tensors = eval_outputs.summary_tensors
        logging.debug('  Completed eval_step() runs on training split.')

        if self._eval_train_summary_handler.process(
            new_step, loss, weighted_scalars, summary_tensors
        ):
          logging.debug('[PAX STATUS]:  Wrote eval summaries.')
        eval_train_metrics = metric_utils.as_float_dict(weighted_scalars)
    return eval_train_metrics

  # TODO(laigd): remove this.
  @property
  def train_unpadded_global_batch_size(self) -> int:
    return self._train_unpadded_global_batch_size

  def shutdown(self) -> None:
    self._pending_train_losses.wait_for_all()
    self._train_summary_handler.close()
    if self._eval_train_summary_handler:
      self._eval_train_summary_handler.close()
    self._exitstack.close()


class SingleTaskTrainProgram(BaseTrainProgram):
  """Train program that assumes a single task on a single dataset."""

  def __init__(self):
    super().__init__()

    # Train step function information.
    self._train_step_created = False
    self._train_step_fn = None
    self._train_step_input_partition_spec = None

    # Eval train step function information. Note since this eval step runs on
    # training inputs, it'll have the same input shapes/dtypes and partition
    # spec as the train step function.
    self._eval_train_step_created = False
    self._eval_train_step_fn = None

  def _get_train_step(
      self, inputs: NestedJTensor
  ) -> Tuple[Any, Optional[NestedPartitionSpec]]:
    """Creates the train step info (if not done before) and returns them."""
    # Note: it doesn't matter what batch size `inputs` have, or whether it's the
    # original or preprocessed input, since:
    # - For pmap: `inputs` is not used;
    # - For pjit: batch size doesn't affect the partitioning result, while
    #   Partitioner.preprocess_inputs() keep other dims the same.
    if not self._train_step_created:
      self._train_step_fn, self._train_step_input_partition_spec = (
          self._partitioner.partition(
              trainer_lib.train_step_single_learner,
              trees.get_shape_dtype(inputs),
              is_eval=False,
          )
      )
      self._train_step_created = True
    return self._train_step_fn, self._train_step_input_partition_spec

  def _get_eval_train_step(self, inputs: NestedJTensor) -> Any:
    """Creates the train step info (if not done before) and returns them."""
    if not self._eval_train_step_created:
      # TODO(pax): Support auto-sharding for eval step. In this case, we would
      # have to fix the sharding of the input to be the same as what's derived
      # from the step.

      # Ignores the returned input partition spec. It should be the same as
      # self.train_input_partition_spec since the input shapes are the same.
      self._eval_train_step_fn, _ = self._partitioner.partition(
          trainer_lib.eval_step_single_learner,
          trees.get_shape_dtype(inputs),
          is_eval=True,
      )
      self._eval_train_step_created = True
    return self._eval_train_step_fn

  def train_step(
      self,
      step: int,
      state: train_states.TrainState,
      prng_key: PRNGKey,
      inputs: NestedJTensor,
      static_args: BaseStepFnStaticArgs,
  ) -> Tuple[int, TrainState, StepFnOutput]:
    """The train step function."""
    train_step, _ = self._get_train_step(inputs)
    return step + 1, *train_step(state, prng_key, inputs, static_args)

  def eval_train_step(
      self,
      state: train_states.TrainState,
      prng_key: PRNGKey,
      inputs: NestedJTensor,
      static_args: BaseStepFnStaticArgs,
  ) -> StepFnOutput:
    """The eval step function on trianing inputs."""
    eval_train_step = self._get_eval_train_step(inputs)
    unused_train_state, eval_step_fn_out = eval_train_step(
        state, prng_key, inputs, static_args
    )
    return eval_step_fn_out

  def train_input_partition_spec(
      self, inputs: NestedJTensor
  ) -> Optional[NestedPartitionSpec]:
    """The partition spec for the model training inputs."""
    _, input_partition_spec = self._get_train_step(inputs)
    return input_partition_spec


def can_load_written_outputs(
    basedir: epath.Path, pname: str, mode: EvaluationMode, step: int
) -> bool:
  """Returns whether we can load the eval/decoder outputs already."""
  success = np.array([0], dtype=np.int32)
  if jax.process_index() == 0:
    try:
      outputs = io_utils.load_outputs(basedir, pname, mode.value, step)
      success[0] = len(outputs)
    except Exception:  # pylint: disable=broad-except
      pass
  out = multihost_utils.broadcast_one_to_all(success)
  return out[0] > 0


def get_filename(
    step: Union[base_layer.JTensorOrPartitionSpec, int], prefix: str
) -> str:
  """Returns a filename for the given step."""
  step_num = py_utils.maybe_unreplicate_for_fully_replicated(step)
  return f'{prefix}_out_{step_num}_shard_{jax.process_index()}'


def safe_write_key_value_pairs(
    filename: epath.PathLike,
    key_value_pairs: Sequence[Tuple[Optional[str], Any]],
    cast_to_ndarray: bool = True,
    write_pickle: bool = True,
) -> None:
  try:
    io_utils.write_key_value_pairs(
        filename, key_value_pairs, cast_to_ndarray, write_pickle
    )
  except TypeError:
    logging.warning('Not serializable.')


def maybe_write_eval_outputs(
    mode: EvaluationMode,
    output_dir: epath.Path,
    step: int,
    eval_outputs: Sequence[Tuple[str, Any]],
    write_pickle: bool = True,
) -> None:
  """Writes model evaluation outputs to disk from leader process."""
  if jax.process_index() != 0 or flags.FLAGS.pax_only_aggregate_summaries:
    return

  fq_fname = output_dir / get_filename(step, mode.value)
  fq_fname.parent.mkdir(parents=True, exist_ok=True)
  logging.info(
      'Writing %s outputs to %s with %d entries',
      mode.value,
      fq_fname,
      len(eval_outputs),
  )
  safe_write_key_value_pairs(fq_fname, eval_outputs, write_pickle=write_pickle)


class BaseEvalProgram(Program):

  def __init__(self, input_p: pax_fiddle.Config[base_input.BaseInput]):
    self._input_p = input_p

    # States to set in self.setup()
    self._task: tasks_lib.SingleTask = None
    self._partitioner: partitioning.Partitioner = None
    self._job_log_dir: epath.Path = None
    self._eval_prng_seed: PRNGKey = None

    # States to initialize lazily in self.setup()
    self._eval_input_pipeline = None
    self._name: str = None
    self._eval_unpadded_global_batch_size: int = None
    self._eval_num_steps: int = None
    self._eval_summary_writer: SummaryWriter = None

    # Used to enter context of the summary writer at .setup().
    self._exitstack = contextlib.ExitStack()

  @property
  def eval_input(self) -> base_input.BaseInput:
    assert self._eval_input_pipeline
    return self._eval_input_pipeline

  def setup(
      self,
      task: tasks_lib.SingleTask,
      partitioner: partitioning.Partitioner,
      job_log_dir: epath.Path,
      eval_prng_seed: PRNGKey,
  ) -> None:
    self._task = task
    self._partitioner = partitioner
    self._job_log_dir = job_log_dir
    self._eval_prng_seed = eval_prng_seed

    # Creates the eval input pipeline.
    self._input_p = self._partitioner.preprocess_input_config(self._input_p)
    logging.info(
        '[PAX STATUS]: Initializing eval_input pipeline : %s', self._input_p
    )
    self._eval_input_pipeline = instantiate(self._input_p)
    self._name = self.eval_input.name
    self._eval_unpadded_global_batch_size = (
        self._eval_input_pipeline.get_global_batch_size(
            self._eval_input_pipeline
        )
    )
    self._eval_num_steps = (
        -1
        if self._input_p.reset_for_eval
        else self._input_p.eval_loop_num_batches
    )

    # Creates the eval summary writer.
    summary_base_dir = get_summary_base_dir(job_log_dir)
    summary_dir = summary_base_dir / f'eval_test_{self.eval_input.name}'
    self._eval_summary_writer = self._exitstack.enter_context(
        summary_utils.get_summary_writer(summary_dir)
    )

  def should_run(self, state: TrainState, step: int) -> bool:
    # TODO(laigd): implement and use this.
    raise NotImplementedError()

  def run(self, state: TrainState, step: int) -> EvalProgramOutput:
    if can_load_written_outputs(
        self._job_log_dir, self._name, EvaluationMode.EVAL, step
    ):
      logging.info(
          'Eval on %s (@ step %d) already done, skipping.', self._name, step
      )
      return EvalProgramOutput(state)

    logging.info(
        'Starting eval %s with num_steps=%d', self._name, self._eval_num_steps
    )
    num_steps, loss, summary_tensors, metrics, per_example_scores = (
        self._run_eval_loop(state)
    )
    logging.info('Finished eval on %s', self._name)

    # Flatten scoring outputs to simplify input for metrics eval computation.
    # Constructs a new flattened array of single example outputs from original
    # array containing batches of outputs.
    flat_scoring_outputs = []
    for batch in per_example_scores:
      for ex in py_utils.tree_unstack(batch, 0):
        flat_scoring_outputs.append((py_utils.get_enumeration_id(ex), ex))
    logging.info(
        'Finished unstacking %d batches into %d per example outputs.',
        len(per_example_scores),
        len(flat_scoring_outputs),
    )

    eval_scoring_metrics = None
    output_dir = (
        self._job_log_dir / f'{EvaluationMode.EVAL.value}_out' / self._name
    )

    # TODO(laigd): consider adding a method for this for subclass to overwrite.
    if seqio_input.should_process_outputs(self.eval_input):
      verbose_entries = 0 if flags.FLAGS.pax_only_aggregate_summaries else 1
      eval_scoring_metrics = seqio_input.process_outputs(
          self.eval_input,
          flat_scoring_outputs,
          self._eval_summary_writer,
          seqio_input.MetricType.SCORE,
          step,
          output_dir,
          verbose_entries=verbose_entries,
      )
      logging.info(
          'Finished processing %d outputs using seqio.',
          len(flat_scoring_outputs),
      )

    loss = np.array(loss)
    for k in summary_tensors:
      summary_tensors[k] = np.array([np.asarray(t) for t in summary_tensors[k]])
    loss = np.mean(loss, axis=0)
    logging.info('step: %d, eval test %s loss: %s', step, self._name, loss)

    for key, values in metrics.items():
      # `metric_utils.as_float` computes the average from a list of weighted
      # scalars.
      weighted_average = metric_utils.as_float(values)
      sum_metric_weights = np.sum(np.stack([v[1] for v in values])).item()
      logging.info(
          '  %s=%f (weight=%f)', key, weighted_average, sum_metric_weights
      )
    summary_utils.write_summary_entry(
        self._eval_summary_writer, step, loss, metrics, summary_tensors
    )
    maybe_write_eval_outputs(
        EvaluationMode.EVAL, output_dir, step, flat_scoring_outputs
    )

    return EvalProgramOutput(
        state,
        eval_metrics=metric_utils.as_float_dict(metrics),
        eval_scoring_metrics=eval_scoring_metrics,
        num_eval_steps=num_steps,
    )

  def _run_eval_loop(self, state: TrainState):
    losses = []
    summary_tensor_dict = {}
    metrics = collections.defaultdict(list)
    per_example_scores = []

    step_num = 0
    # self._eval_num_steps < 0 indicates running until input out of range.
    while self._eval_num_steps < 0 or step_num < self._eval_num_steps:
      try:
        eval_inputs = self.eval_input.get_next_padded()
      except (tf.errors.OutOfRangeError, StopIteration):
        if self._eval_num_steps > 0:
          raise
        logging.info('Data exhausted (%s) after %d steps', self._name, step_num)
        self.eval_input.reset()
        break

      step_num += 1
      eval_inputs, unsupported_inputs, supported_input_partition_spec = (
          xla_passthrough.split_out_xla_unsupported_batch(
              eval_inputs,
              partitioning_spec=self.eval_input_partition_spec(eval_inputs),
          )
      )
      eval_inputs = self._partitioner.preprocess_inputs(
          self.eval_input, eval_inputs, supported_input_partition_spec
      )
      eval_outputs = self.eval_step(
          state,
          self._eval_prng_seed,
          eval_inputs,
          BaseStepFnStaticArgs(
              unpadded_global_batch_size=self._eval_unpadded_global_batch_size
          ),
      )
      loss = eval_outputs.loss
      weighted_scalars = eval_outputs.weighted_scalars
      per_example_out = eval_outputs.per_example_out
      summary_tensors = eval_outputs.summary_tensors
      xla_passthrough.merge_back_xla_unsupported_batch(
          per_example_out, unsupported_inputs
      )
      logging.info('Finished eval step %d for %s', step_num, self._name)
      loss, weighted_scalars, per_example_out, summary_tensors = (
          py_utils.maybe_unreplicate_for_fully_replicated(out)
          for out in (loss, weighted_scalars, per_example_out, summary_tensors)
      )

      losses += [loss]
      for k, v in summary_utils.flatten_summary_dict(summary_tensors):
        if k in summary_tensor_dict:
          summary_tensor_dict[k] += [v]
        else:
          summary_tensor_dict[k] = [v]
      for k in weighted_scalars:
        metrics[k].append(weighted_scalars[k])
      # Use jax.device_get to overlap the device -> host memory transfer.
      # Make a copy on the transferred tensor since running
      # py_utils.tree_unstack() on XLA-backed numpy arrays is inefficient
      # (b/284371615).
      per_example_scores.append(
          jax.tree_map(lambda x: x.copy(), jax.device_get(per_example_out))
      )

    return step_num, losses, summary_tensor_dict, metrics, per_example_scores

  @abc.abstractmethod
  def eval_step(
      self,
      state: train_states.TrainState,
      prng_key: PRNGKey,
      inputs: NestedJTensor,
      static_args: BaseStepFnStaticArgs,
  ) -> StepFnOutput:
    """The eval step function.

    Args:
      state: The current train state.
      prng_key: The PRNG key for this eval step.
      inputs: The eval input data for this eval step.
      static_args: Encapsulates any static arguments needed by the step
        function.

    Returns:
      A StepFnOutput instance encapsulating the output of the eval step
      function.
    """

  @abc.abstractmethod
  def eval_input_partition_spec(
      self, inputs: NestedJTensor
  ) -> Optional[NestedPartitionSpec]:
    """Return the partition spec for the eval inputs."""

  def shutdown(self) -> None:
    self._exitstack.close()


class SingleTaskEvalProgram(BaseEvalProgram):
  """Eval program that assumes a single task on a single dataset."""

  def __init__(
      self,
      input_p: pax_fiddle.Config[base_input.BaseInput],
  ):
    super().__init__(input_p)

    # Eval step function information.
    self._eval_step_created = False
    self._eval_step_fn = None
    self._eval_step_input_spec = None

  def _get_eval_step(
      self, inputs: NestedJTensor
  ) -> Tuple[Any, Optional[NestedPartitionSpec]]:
    """Creates the eval step info if not done before."""
    if not self._eval_step_created:
      self._eval_step_fn, self._eval_step_input_spec = (
          self._partitioner.partition(
              trainer_lib.eval_step_single_learner,
              inputs_shape_dtype=trees.get_shape_dtype(inputs),
              is_eval=True,
          )
      )
      self._eval_step_created = True
    return self._eval_step_fn, self._eval_step_input_spec

  def eval_step(
      self,
      state: train_states.TrainState,
      prng_key: PRNGKey,
      inputs: NestedJTensor,
      static_args: BaseStepFnStaticArgs,
  ) -> StepFnOutput:
    """The eval step function."""
    eval_step, _ = self._get_eval_step(inputs)
    unused_train_state, eval_step_fn_out = eval_step(
        state, prng_key, inputs, static_args
    )
    return eval_step_fn_out

  def eval_input_partition_spec(
      self, inputs: NestedJTensor
  ) -> Optional[NestedPartitionSpec]:
    """The partition spec for the eval inputs."""
    _, input_partition_spec = self._get_eval_step(inputs)
    return input_partition_spec
