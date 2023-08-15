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

"""Programs for decoding."""

import collections
import contextlib
import dataclasses
import functools
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from absl import flags
from absl import logging
from clu import metrics as clu_metrics
from clu import platform
from etils import epath
import jax
import numpy as np
from paxml import base_metrics
from paxml import io_utils
from paxml import partitioning
from paxml import programs
from paxml import tasks_lib
from paxml import train_states
from paxml import trainer_lib
from paxml import xla_passthrough
from praxis import base_hyperparams
from praxis import base_input
from praxis import base_layer
from praxis import base_model
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

instantiate = base_hyperparams.instantiate
BaseMetrics = base_metrics.BaseMetrics
EvaluationMode = io_utils.EvaluationMode
JTensor = pytypes.JTensor
Metrics = pytypes.Metrics
NestedMap = py_utils.NestedMap
NestedJTensor = pytypes.NestedJTensor
NestedPartitionSpec = pytypes.NestedPartitionSpec
SummaryWriter = 'tf.summary.SummaryWriter'  # pylint:disable=invalid-name
StepFnOutput = trainer_lib.StepFnOutput
TrainState = train_states.TrainState
PRNGKey = pytypes.PRNGKey


def _merge_clu_metrics(metrics: Metrics, updated_metrics: Metrics) -> Metrics:
  """Merges existing eval metrics with updated metric data."""
  if metrics:
    if set(metrics.keys()) != set(updated_metrics.keys()):
      raise ValueError(
          'metrics and updated_metrics keys don`t match. '
          f'metrics keys: {metrics.keys()} '
          f'updated_metrics keys: {updated_metrics.keys()}'
      )

    for key in metrics:
      metrics[key] = metrics[key].merge(updated_metrics[key])
  else:
    metrics = updated_metrics
  return metrics


@dataclasses.dataclass(frozen=True)
class DecodeProgramOutput(programs.ProgramOutput):
  decode_metrics: Optional[Mapping[str, float]] = None
  processed_decode_metrics: Optional[Mapping[str, float]] = None
  seqio_metrics: Optional[Mapping[str, float]] = None
  num_decode_steps: int = 0


class SingleTaskDecodeProgram(programs.Program):
  """Decode program that assumes a single task on a single dataset."""

  def __init__(self, input_p: pax_fiddle.Config[base_input.BaseInput]):
    self._input_p = input_p

    # States to set in self.setup()
    self._task: tasks_lib.SingleTask = None
    self._partitioner: partitioning.Partitioner = None
    self._job_log_dir: epath.Path = None
    self._prng_key: PRNGKey = None
    self._output_pickle = True

    # States to initialize lazily in self.setup()
    self._input = None
    self._name: str = None
    self._unpadded_global_batch_size: int = None
    self._num_steps: int = None
    self._output_dir: epath.Path = None
    self._summary_writer: SummaryWriter = None
    self._metrics_p: pax_fiddle.Config[Any] = None

    # Used to enter context of the summary writer at .setup().
    self._exitstack = contextlib.ExitStack()

    # Decode step function information.
    self._decode_step_created = False
    self._decode_step_fn = None
    self._decode_step_input_spec = None

  def setup(
      self,
      task: tasks_lib.SingleTask,
      partitioner: partitioning.Partitioner,
      job_log_dir: epath.Path,
      decode_prng_seed: PRNGKey,
      output_pickle: bool = True,
  ) -> None:
    """Sets up the program.

    Args:
      task: The jax task.
      partitioner: The partitioner used to partition the decode function.
      job_log_dir: Directory for the job logs.
      decode_prng_seed: The prng key used for decoding.
      output_pickle: Whether to write decoding results to a pickle file.
    """
    self._task = task
    self._partitioner = partitioner
    self._job_log_dir = job_log_dir
    self._prng_key = decode_prng_seed
    self._output_pickle = output_pickle

    # Creates the decode input pipeline.
    self._input_p = self._partitioner.preprocess_input_config(self._input_p)
    logging.info(
        '[PAX STATUS]: Initializing decode input pipeline : %s', self._input_p
    )
    self._input = instantiate(self._input_p)

    # Creates other states.
    self._name = self._input.name
    self._unpadded_global_batch_size = self._input.get_global_batch_size(
        self._input
    )
    self._num_steps = (
        -1 if self._input.reset_for_eval else self._input.eval_loop_num_batches
    )
    self._output_dir = (
        job_log_dir / f'{EvaluationMode.DECODE.value}_out' / self._name
    )
    self._metrics_p = task.metrics or pax_fiddle.Config(
        base_metrics.MeanMetrics
    )

    # Creates the decode summary writer.
    summary_base_dir = programs.get_summary_base_dir(job_log_dir)
    summary_dir = summary_base_dir / f'decode_test_{self._input.name}'
    self._summary_writer = self._exitstack.enter_context(
        summary_utils.get_summary_writer(summary_dir)
    )

  def should_run(self, state: TrainState, train_step: int) -> bool:
    # TODO(wangpeng): Implement and use it.
    raise NotImplementedError()

  @property
  def decode_input(self):
    assert self._input is not None
    return self._input

  def run(self, state: TrainState, train_step: int) -> DecodeProgramOutput:
    # Skip decode if already completed.
    if programs.can_load_written_outputs(
        self._job_log_dir, self._name, EvaluationMode.DECODE, train_step
    ):
      logging.info(
          'Decoding on input %s at step %d already done, skipping.',
          self._name,
          train_step,
      )
      return DecodeProgramOutput(state)

    logging.info('Start decoding on input %s', self._name)
    # decode_metrics and process_decode_metrics work on WeightedScalars
    # which are string -> (value, weight) pairs where value and weight
    # scalars. These metrics are configured on the task.
    decode_metrics = instantiate(self._metrics_p)
    process_decode_metrics = instantiate(self._metrics_p)

    (
        step_num,
        metrics,
        processed_metrics,
        processed_decodes,
        all_summary_tensors,
    ) = self._run_decode_loop(state, decode_metrics, process_decode_metrics)

    # Now the decode loop of multiple batches on current dataset is done,
    # we start to aggregate copmuted metrics and put them in summary.
    seqio_metric_values = None
    if seqio_input.should_process_outputs(self.decode_input):
      logging.info(
          'Finished processing all %d examples.', len(processed_decodes)
      )
      filename = self._output_dir / programs.get_filename(
          train_step, EvaluationMode.DECODE.value
      )
      seqio_metric_values = seqio_input.process_outputs(
          self.decode_input,
          processed_decodes,
          self._summary_writer,
          seqio_input.MetricType.PREDICT,
          train_step,
          self._output_dir,
          plain_text_output_fname=f'{filename}.txt',
      )

    # Convert metrics to Dict[str, clu_values.Value] for summary writing.
    metric_values = metric_utils.compute_metric_values(metrics)
    process_metric_values = metric_utils.compute_metric_values(
        processed_metrics
    )

    with self._summary_writer.as_default():
      logging.info('Summarizing of decode_metrics.')
      decode_metric_dict = decode_metrics.summarize(
          train_step, 'decode_metrics'
      )
      logging.info('Summarizing of process_decode_metrics.')
      processed_metric_dict = process_decode_metrics.summarize(
          train_step, 'process_decode_metrics'
      )
      for key, tensor in all_summary_tensors.items():
        summary_type = base_layer.get_summary_type_from_key(key)
        summary_utils.write_summary_tensor(
            train_step,
            key,
            np.array(tensor),
            summary_type,
            sample_rate=(
                self._task.model.sample_rate
                if hasattr(self._task.model, 'sample_rate')
                else summary_utils.AUDIO_SUMMARY_SAMPLE_RATE
            ),
        )
      metric_utils.write_clu_metric_summaries(metric_values, train_step)
      metric_utils.write_clu_metric_summaries(process_metric_values, train_step)

    programs.maybe_write_eval_outputs(
        EvaluationMode.DECODE,
        self._output_dir,
        train_step,
        processed_decodes,
        write_pickle=self._output_pickle,
    )

    msg = f'Finished decoding input {self._name}'
    work_unit = platform.work_unit()
    work_unit.set_task_status(msg)
    logging.info(msg)

    return DecodeProgramOutput(
        state=state,
        decode_metrics=metric_utils.update_float_dict(
            metric_utils.as_float_dict(decode_metric_dict),
            metric_utils.as_float_dict(metric_values),
        ),
        processed_decode_metrics=metric_utils.update_float_dict(
            metric_utils.as_float_dict(processed_metric_dict),
            metric_utils.as_float_dict(process_metric_values),
        ),
        seqio_metrics=seqio_metric_values,
        num_decode_steps=step_num,
    )

  def _run_decode_loop(
      self,
      state: TrainState,
      decode_metrics: BaseMetrics,
      process_decode_metrics: BaseMetrics,
  ) -> Tuple[
      int,
      Dict[str, clu_metrics.Metric],
      Dict[str, clu_metrics.Metric],
      List[Tuple[str, Any]],
      Dict[str, List[JTensor]],
  ]:
    # metrics and processed_metrics are dictionaries of
    # strings -> clu_metrics.Metric objects. metrics is returned from decode()
    # and processed_metrics is returned from process_decode_out.
    metrics = {}
    processed_metrics = {}
    processed_decodes = []
    all_summary_tensors = collections.defaultdict(list)
    # profile xprof for decoding.
    profiler = None
    if self._task.decode.profiler_num_steps > 0:
      profiler = profiling.Profiler(
          num_steps=self._task.decode.profiler_num_steps,
          min_duration_sec=self._task.decode.profiler_min_duration_sec,
          max_num_hosts=self._task.decode.profiler_max_num_hosts,
      )

    step_num = 0
    # self._num_steps < 0 indicates running until input out of range.
    while self._num_steps < 0 or step_num < self._num_steps:
      step_num += 1
      if (
          profiler is not None
          and step_num == self._task.decode.profiler_capture_step
      ):
        profiler.capture_async()
      # Instrument decode step.
      with jax.profiler.StepTraceAnnotation('decode', step_num=step_num):
        try:
          batch = self.decode_input.get_next_padded()
        except (tf.errors.OutOfRangeError, StopIteration):
          self.decode_input.reset()
          if step_num == 1:
            logging.error('Input %s yields zero batch.', self._name)
          else:
            logging.info('Input %s exhausted at step %d.', self._name, step_num)
          break
        batch, tpu_unsupported_batch, inputs_partition_spec = (
            xla_passthrough.split_out_xla_unsupported_batch(
                batch,
                partitioning_spec=self.decode_input_partition_spec(batch),
            )
        )
        batch = self._partitioner.preprocess_inputs(
            self.decode_input, batch, inputs_partition_spec
        )

        if self._task and self._task.decode.prng_key_fold_with_batch_index:
          # In this case, the key is a scalar and we need to preprocess it
          # (broadcast/split) after folding in step_num.
          decode_key = jax.random.fold_in(self._prng_key, step_num)
          decode_key = self._partitioner.preprocess_prng_key(decode_key)
        else:
          decode_key = self._prng_key

        decode_out = self.decode_step(
            state,
            decode_key,
            batch,
            trainer_lib.BaseStepFnStaticArgs(
                unpadded_global_batch_size=self._unpadded_global_batch_size
            ),
        )
      # Synchronize all the hosts to ensure their executions don't diverge.
      py_utils.sync_global_devices(f'spmd_decode-{self._name}-{step_num}')

      # Output is fully replicated now, so it's ok to unreplicate it by
      # retrieving from device 0 only.
      unreplicated_decode_out = py_utils.maybe_unreplicate_for_fully_replicated(
          decode_out
      )
      del decode_out  # release Jax Arrays memory allocations
      per_example_out = unreplicated_decode_out.per_example_out
      weighted_scalars = unreplicated_decode_out.weighted_scalars
      updated_metrics = unreplicated_decode_out.clu_metrics
      summary_tensors = unreplicated_decode_out.summary_tensors

      # Merge clu.metrics to update for each minibatch.
      metrics = _merge_clu_metrics(metrics, updated_metrics)

      for key, tensor in summary_utils.flatten_summary_dict(summary_tensors):
        all_summary_tensors[key].append(tensor)

      logging.info(
          'Finished decoding input batch %d for %s', step_num, self._name
      )

      if jax.process_index() == 0:
        # Copy the tensor from device memory to ram, since accumulating such
        # tensor on devices may cause HBM OOM, when
        # task_p.train.summary_accumulate_interval_steps is set.
        weighted_scalars = jax.tree_map(np.array, weighted_scalars)
        decode_metrics.store(weighted_scalars)

        xla_passthrough.merge_back_xla_unsupported_batch(
            per_example_out, tpu_unsupported_batch
        )

        # Run `process_decode_out` on CPU device as its implementation
        # is not expected to be JIT friendly. Since we keep track of
        # its outputs, we also don't want on-device allocation as
        # would eventually lead to HBM OOM.
        with jax.default_device(jax.devices('cpu')[0]):
          per_example_out = jax.tree_map(np.asarray, per_example_out)
          process_weighted_scalars, processed_out, processed_metric_updates = (
              self._task.model.process_decode_out(
                  self.decode_input, per_example_out
              )
          )

        processed_out = seqio_input.maybe_update_decode_output_keys(
            processed_out, per_example_out
        )

        process_decode_metrics.store(process_weighted_scalars)
        processed_decodes.extend(processed_out)
        if processed_metric_updates:
          processed_metrics = _merge_clu_metrics(
              processed_metrics, processed_metric_updates
          )

    return (
        step_num,
        metrics,
        processed_metrics,
        processed_decodes,
        all_summary_tensors,
    )

  def _get_decode_step(
      self, inputs: NestedJTensor
  ) -> Tuple[Any, Optional[NestedPartitionSpec]]:
    """Creates the decode step info if not done before."""
    if not self._decode_step_created:
      self._decode_step_fn, self._decode_step_input_spec = (
          self._partitioner.partition(
              trainer_lib._decode_step_for_partitioner,
              inputs_shape_dtype=trees.get_shape_dtype(inputs),
              is_eval=True,
          )
      )
      self._decode_step_created = True
    return self._decode_step_fn, self._decode_step_input_spec

  def decode_step(
      self,
      state: train_states.TrainState,
      prng_key: PRNGKey,
      inputs: NestedJTensor,
      static_args: trainer_lib.BaseStepFnStaticArgs,
  ) -> StepFnOutput:
    """The decode step function."""
    decode_step, _ = self._get_decode_step(inputs)
    unused_train_state, decode_step_fn_out = decode_step(
        state, prng_key, inputs, static_args
    )
    return decode_step_fn_out

  def decode_input_partition_spec(
      self, inputs: NestedJTensor
  ) -> Optional[NestedPartitionSpec]:
    """The partition spec for the decode inputs."""
    _, input_partition_spec = self._get_decode_step(inputs)
    return input_partition_spec

  def shutdown(self) -> None:
    self._exitstack.close()
