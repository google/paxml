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
import functools
import sys
from typing import Callable, Dict, List, Mapping, Optional, Tuple

from absl import flags
from absl import logging
from clu import metrics as clu_metrics
from clu import platform
from etils import epath
import jax
import numpy as np
from paxml import base_metrics
from paxml import io_utils
from paxml import metric_tracker_utils as trk_utils
from paxml import metric_utils
from paxml import partitioning
from paxml import programs
from paxml import seqio_input
from paxml import summary_utils
from paxml import tasks_lib
from paxml import train_states
from paxml import xla_passthrough
from praxis import base_hyperparams
from praxis import base_input
from praxis import base_layer
from praxis import base_model
from praxis import pax_fiddle
from praxis import py_utils
from praxis import pytypes
import tensorflow.compat.v2 as tf

from paxml import checkpoints  # mapped to internal


instantiate = base_hyperparams.instantiate
EvaluationMode = io_utils.EvaluationMode
JTensor = pytypes.JTensor
Metrics = pytypes.Metrics
NestedMap = py_utils.NestedMap
NestedJTensor = pytypes.NestedJTensor
NestedPartitionSpec = pytypes.NestedPartitionSpec
SummaryWriter = tf.summary.SummaryWriter
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


class DecodeOutput(NestedMap):
  """The output of a decode program."""

  def __init__(
      self,
      *,
      decode_metrics: Optional[Dict[str, float]],
      processed_decode_metrics: Optional[Dict[str, float]],
      seqio_metrics: Optional[Dict[str, float]],
      num_decode_steps: int,
      raw_decode_metrics: Optional[Mapping[str, clu_metrics.Metric]],
  ):
    """The constructor.

    Args:
      decode_metrics: Decode metrics.
      processed_decode_metrics: Processed decode metrics.
      seqio_metrics: Decode (seqio) metrics.
      num_decode_steps: The number of performed decode steps.
      raw_decode_metrics: Raw decode metrics.
    """
    super().__init__()
    self.decode_metrics = decode_metrics
    self.processed_decode_metrics = processed_decode_metrics
    self.seqio_metrics = seqio_metrics
    self.num_decode_steps = num_decode_steps
    self.raw_decode_metrics = raw_decode_metrics


class SingleTaskDecodeProgram(programs.Program):
  """Decode program that assumes a single task on a single dataset."""

  def __init__(
      self,
      *,
      model: base_model.BaseModel,
      partitioner: partitioning.Partitioner,
      decode_input: base_input.BaseInput,
      # This argument is used for `sync_global_devices`.
      # TODO(wangpeng): Try removing this argument.
      input_index: Optional[int] = None,
  ):
    """The constructor.

    Args:
      model: The model to be used for decoding.
      partitioner: The partitioner used to partition the decode function.
      decode_input: The instantiated input to be decoded.
      input_index: The index of this input among a list of inputs.
    """
    self._model = model
    self._partitioner = partitioner
    self._input = decode_input
    self._input_index = input_index
    self._num_steps = (
        -1 if self._input.reset_for_eval else self._input.eval_loop_num_batches
    )
    self._dirname = epath.Path(self._input.name)

    self._prng_key: PRNGKey = None
    self._job_log_dir: epath.Path = None
    self._basedir: epath.Path = None
    self._summary_writer: SummaryWriter = None
    self._use_pmap = None

    self._pmap_decode_step = None
    self._task_p = None
    self._output_pickle = None
    self._enable_checkpoint_saving = None

    self._spmd_decode_step = None
    self._inputs_partition_spec = None
    self._metrics_p: pax_fiddle.Config[base_metrics.BaseMetrics] = None

  def setup(
      self,
      *,
      prng_key: JTensor,
      job_log_dir: epath.Path,
      summary_writer: SummaryWriter,
      use_pmap: bool,
      pmap_decode_step: Optional[
          Callable[
              [TrainState, PRNGKey, NestedJTensor],
              Tuple[NestedMap, NestedMap, NestedMap],
          ]
      ],
      task_p: Optional[pax_fiddle.Config[tasks_lib.SingleTask]],
      output_pickle: bool,
      enable_checkpoint_saving: bool,
      spmd_decode_step: Optional[
          Callable[
              [TrainState, PRNGKey, NestedJTensor, Optional[int]],
              Tuple[Tuple[NestedMap, NestedMap], NestedMap],
          ]
      ],
      inputs_partition_spec: Optional[NestedPartitionSpec],
      metrics_p: pax_fiddle.Config[base_metrics.BaseMetrics],
  ) -> None:
    """Sets up the program.

    Args:
      prng_key: The prng key used for decoding.
      job_log_dir: Directory for the job logs.
      summary_writer: The summary writer to log summaries.
      use_pmap: Whether to use PMAP (instead of SPMD/pjit). If this is True,
        `task_p`, `var_weight_params`, `output_pickle` and
        `enable_checkpoint_saving` should be set; otherwise, `spmd_decode_step`,
        `inputs_partition_spec` and `metrics_p` should be set.
      pmap_decode_step: pmap'ed decode function.
      task_p: Params for the task encapsulating a data parallel model.
      output_pickle: Whether to write decoding results to a pickle file.
      enable_checkpoint_saving: Whether to perform checkpoint saving or not.
      spmd_decode_step: pjit'ed decode function.
      inputs_partition_spec: Partition specs for inputs.
      metrics_p: Parameters to configure how to aggregate the metrics.
    """
    self._prng_key = prng_key
    self._job_log_dir = job_log_dir
    self._basedir = self._job_log_dir / f'{EvaluationMode.DECODE.value}_out'
    self._summary_writer = summary_writer
    self._use_pmap = use_pmap

    self._pmap_decode_step = pmap_decode_step
    self._task_p = task_p
    self._output_pickle = output_pickle
    self._enable_checkpoint_saving = enable_checkpoint_saving

    self._spmd_decode_step = spmd_decode_step
    self._inputs_partition_spec = inputs_partition_spec
    self._metrics_p = metrics_p

  def should_run(self, state: TrainState, train_step: int) -> bool:
    # TODO(wangpeng): Implement and use it.
    raise NotImplementedError()

  def shutdown(self) -> None:
    pass

  @property
  def model(self):
    assert self._model is not None
    return self._model

  @property
  def partitioner(self):
    assert self._partitioner is not None
    return self._partitioner

  @property
  def decode_input(self):
    assert self._input is not None
    return self._input

  def _find_and_maybe_update_tracked_metric(
      self,
      step_i: int,
      replicated_model_states: TrainState,
      task_p: pax_fiddle.Config[tasks_lib.SingleTask],
      decode_metrics_list: List[Dict[str, float]],
  ) -> None:
    basedir = self._basedir
    dirname = self._dirname
    input_name = self._input.name
    enable_checkpoint_saving = self._enable_checkpoint_saving

    tracked_metric = task_p.track_decoder_metric
    track_min_or_max = task_p.track_decoder_metric_min_or_max
    if not track_min_or_max:
      raise ValueError(
          'Must also set track_decoder_metric_min_or_max when '
          f'enabling metric tracking: {task_p}'
      )
    m_value = None
    for d in decode_metrics_list:
      if tracked_metric in d:
        m_value = d[tracked_metric]
        break

    if m_value:
      # Filesystem friendly name for the tracked metric.
      tracked_metric_name = tracked_metric.replace('/', '-')
      tracker_dir_path = (
          basedir
          / dirname
          / f'{tracked_metric_name}_{track_min_or_max}_tracker'
      )
      _maybe_update_tracked_metric(
          m_value,
          step_i,
          tracker_dir_path,
          tracked_metric_name,
          track_min_or_max,
          input_name,
          replicated_model_states,
          enable_checkpoint_saving=enable_checkpoint_saving,
      )
    else:
      logging.info(
          'Cannot track metric %s on input %s.',
          tracked_metric,
          input_name,
      )

  def run(self, state: TrainState, train_step: int) -> programs.ProgramOutput:
    work_unit = platform.work_unit()
    use_pmap = self._use_pmap
    train_state = state
    step_i = train_step
    partitioner = self._partitioner
    model = self._model
    job_log_dir = self._job_log_dir
    decode_input = self._input
    input_name = self._input.name
    num_steps = self._num_steps
    summary_writer = self._summary_writer
    prng_key = self._prng_key
    basedir = self._basedir
    dirname = self._dirname
    raw_filename = programs.get_filename(
        train_state.step if use_pmap else step_i, EvaluationMode.DECODE.value
    )
    filename = basedir / dirname / raw_filename
    metrics_p = self._metrics_p

    if use_pmap:
      pmap_decode_step = self._pmap_decode_step
      output_pickle = self._output_pickle
      decode_step_func = functools.partial(
          pmap_decode_step, train_state.to_eval_state()
      )

    else:
      spmd_decode_step = self._spmd_decode_step
      inputs_partition_spec = self._inputs_partition_spec
      split = self._input_index

      # We do not fold in jax.process_index in contrast to the pmap version and
      # use a single global key instead to rely on pjit to split for different
      # replicas.
      assert spmd_decode_step is not None
      spmd_decode_step_fn = functools.partial(
          spmd_decode_step,
          train_state.to_eval_state(),
      )

    if programs.can_load_written_outputs(
        job_log_dir, input_name, EvaluationMode.DECODE, step_i
    ):
      logging.info(
          'Decoding on input %s at step %d already done, skipping.',
          input_name,
          step_i,
      )
      return programs.ProgramOutput(
          state=state,
          aux=DecodeOutput(
              decode_metrics=None,
              processed_decode_metrics=None,
              seqio_metrics=None,
              num_decode_steps=0,
              raw_decode_metrics=None,
          ),
      )
    logging.info('Start decoding on input %s', input_name)
    step_num = 0
    # decode_metrics and process_decode_metrics work on WeightedScalars
    # which are string -> (value, weight) pairs where value and weight
    # scalars. These metrics are configured on the task.
    decode_metrics = instantiate(metrics_p)
    process_decode_metrics = instantiate(metrics_p)

    # metrics and processed_metrics are dictionaries of
    # strings -> clu_metrics.Metric objects. metrics is returned from decode()
    # and processed_metrics is returned from process_decode_out.
    metrics = {}
    processed_metrics = {}
    processed_decodes = []
    all_summary_tensors = collections.defaultdict(list)
    while num_steps < 0 or step_num < num_steps:
      step_num += 1
      try:
        batch = decode_input.get_next_padded()
      except (tf.errors.OutOfRangeError, StopIteration):
        decode_input.reset()
        logging.log_if(
            logging.ERROR,
            'Input %s yields zero batch.',
            step_num == 1,
            input_name,
        )
        break
      batch, tpu_unsupported_batch, inputs_partition_spec = (
          xla_passthrough.split_out_xla_unsupported_batch(
              batch,
              partitioning_spec=None if use_pmap else inputs_partition_spec,
          )
      )
      batch = partitioner.preprocess_inputs(
          decode_input, batch, None if use_pmap else inputs_partition_spec
      )

      task_p = self._task_p
      if task_p and task_p.decode.prng_key_fold_with_batch_index:
        # In this case, the key is a scalar we need to preprocess it
        # (broadcast/split) after folding in step_num.
        decode_key = jax.random.fold_in(prng_key, step_num)
        decode_key = partitioner.preprocess_prng_key(decode_key)
      else:
        decode_key = prng_key

      if use_pmap:
        (weighted_scalars, out, summary_tensors, updated_metrics) = (
            decode_step_func(decode_key, batch)
        )
      else:
        unused_train_state, decode_out = spmd_decode_step_fn(
            decode_key,
            batch,
            decode_input.get_global_batch_size(decode_input.hparams),
        )
        # Cross host synchronization happens at this point.
        py_utils.sync_global_devices(f'spmd_decode_step_fn{split}_{step_num}')
        # Output is fully replicated now, so it's ok to unreplicate it by
        # retrieving from device 0 only.
        out = py_utils.maybe_unreplicate_for_fully_replicated(
            decode_out.per_example_out
        )
        weighted_scalars = py_utils.maybe_unreplicate_for_fully_replicated(
            decode_out.weighted_scalars
        )

        # Because outputs of the decode step in pjit are annotated to
        # be Jax Arrays, they are already fully replicated across
        # shards and we can just unreplicate.
        # This also means we don't need to call an all_gather and a reduce()
        # on each clu.metric like we do in pmap mode.
        updated_metrics = py_utils.maybe_unreplicate_for_fully_replicated(
            decode_out.summary_tensors
        )

        summary_tensors = decode_out.updated_vars.get(base_layer.SUMMARIES, {})
        summary_tensors = summary_utils.flatten_flax_summaries(summary_tensors)
        del decode_out  # release Jax Arrays memory allocations

        summary_tensors = py_utils.maybe_unreplicate_for_fully_replicated(
            summary_tensors
        )

      # Merge clu.metrics to update for each minibatch.
      metrics = _merge_clu_metrics(metrics, updated_metrics)

      for key, tensor in summary_utils.flatten_summary_dict(summary_tensors):
        all_summary_tensors[key].append(tensor)

      logging.info(
          'Finished decoding input batch %d for %s', step_num, input_name
      )

      if use_pmap:
        # we store the metric directly as it has already been aggregated in
        # side decode_step_fun
        decode_metrics.store(weighted_scalars)
      elif jax.process_index() == 0:
        weighted_scalars = jax.tree_map(np.array, weighted_scalars)
        decode_metrics.store(weighted_scalars)

      xla_passthrough.merge_back_xla_unsupported_batch(
          out, tpu_unsupported_batch
      )

      if jax.process_index() == 0:
        # Run `process_decode_out` on CPU device as its implementation
        # is not expected to be JIT friendly. Since we keep track of
        # its outputs, we also don't want on-device allocation as
        # would eventually lead to HBM OOM.
        with jax.default_device(jax.devices('cpu')[0]):
          out = jax.tree_map(np.asarray, out)
          process_decode_output = model.process_decode_out(decode_input, out)

        (process_weighted_scalars, processed_out, processed_metric_updates) = (
            process_decode_output
        )
        processed_out = seqio_input.maybe_update_decode_output_keys(
            processed_out, out
        )

        process_decode_metrics.store(process_weighted_scalars)
        processed_decodes.extend(processed_out)
        if processed_metric_updates:
          processed_metrics = _merge_clu_metrics(
              processed_metrics, processed_metric_updates
          )

    # Now the decode loop of multiple batches on current dataset is done,
    # we start to aggregate copmuted metrics and put them in summary.
    seqio_metric_values = None
    if seqio_input.should_process_outputs(decode_input):
      logging.info(
          'Finished processing all %d examples.', len(processed_decodes)
      )
      seqio_metric_values = seqio_input.process_outputs(
          decode_input,
          processed_decodes,
          summary_writer,
          seqio_input.MetricType.PREDICT,
          step_i,
          basedir / dirname,
          plain_text_output_fname=f'{filename}.txt',
      )

    # Convert metrics to Dict[str, clu_values.Value] for summary writing.
    metric_values = metric_utils.compute_metric_values(metrics)
    process_metric_values = metric_utils.compute_metric_values(
        processed_metrics
    )

    with summary_writer.as_default():
      logging.info('Summarizing of decode_metrics.')
      decode_metric_dict = decode_metrics.summarize(step_i, 'decode_metrics')
      logging.info('Summarizing of process_decode_metrics.')
      processed_metric_dict = process_decode_metrics.summarize(
          step_i, 'process_decode_metrics'
      )
      for key, tensor in all_summary_tensors.items():
        summary_type = base_layer.get_summary_type_from_key(key)
        summary_utils.write_summary_tensor(
            step_i, key, np.array(tensor), summary_type
        )
      metric_utils.write_clu_metric_summaries(metric_values, step_i)
      metric_utils.write_clu_metric_summaries(process_metric_values, step_i)

    if jax.process_index() == 0 and not (
        use_pmap and flags.FLAGS.pax_only_aggregate_summaries
    ):
      dir_path = basedir / dirname
      dir_path.mkdir(parents=True, exist_ok=True)
      output_file = filename
      logging.info(
          'Writing decoder output to %s with %d entries',
          output_file,
          len(processed_decodes),
      )
      programs.safe_write_key_value_pairs(
          output_file,
          processed_decodes,
          write_pickle=output_pickle if use_pmap else True,
      )

    msg = f'Finished decoding input batch at step {step_num} for {input_name}'
    work_unit.set_task_status(msg)
    logging.info(msg)

    merged_decode_metrics = metric_utils.update_float_dict(
        metric_utils.as_float_dict(decode_metric_dict),
        metric_utils.as_float_dict(metric_values),
    )
    merged_processed_decode_metrics = metric_utils.update_float_dict(
        metric_utils.as_float_dict(processed_metric_dict),
        metric_utils.as_float_dict(process_metric_values),
    )

    if use_pmap:
      task_p = self._task_p
      assert task_p is not None
      if task_p.track_decoder_metric:
        # Track metric specified by task_p.track_decoder_metric.
        self._find_and_maybe_update_tracked_metric(
            step_i,
            train_state,
            task_p,
            [merged_decode_metrics, merged_processed_decode_metrics],
        )

    decode_output = DecodeOutput(
        decode_metrics=merged_decode_metrics,
        processed_decode_metrics=merged_processed_decode_metrics,
        seqio_metrics=seqio_metric_values,
        num_decode_steps=step_num,
        raw_decode_metrics=metrics,
    )
    return programs.ProgramOutput(state=state, aux=decode_output)


def _maybe_update_tracked_metric(
    m_value: float,
    step: int,
    tracker_dir_path: epath.Path,
    tracked_metric: str,
    min_or_max: tasks_lib.SingleTask.TrackDecoderMetricMode,
    data_partition_name: str,
    replicated_model_states: TrainState,
    enable_checkpoint_saving: bool = True,
) -> None:
  """Updates tracked metric if new value (m_value) is lower that the stored one.

  Also updates the status file maintained by the tracker and writes
  new checkpoint assets in the same tracker directory.

  Args:
    m_value: new metric value.
    step: current training step.
    tracker_dir_path: directory where the tracker should store the status file
      and also write and garbage collect checkpoint assets.
    tracked_metric: name of metric being tracked, e.g. 'wer'.
    min_or_max: min or max tracker.
    data_partition_name: data partition on which the value of the metric is
      being tracked.
    replicated_model_states: replicated model states used to save the best
      checkpoint.
    enable_checkpoint_saving: Whether to perform checkpoint saving or not.
  """
  if jax.process_index() == 0:
    tracker_dir_path.mkdir(parents=True, exist_ok=True)
    initial_value = sys.float_info.max
    if min_or_max == tasks_lib.SingleTask.TrackDecoderMetricMode.MAX:
      initial_value = -sys.float_info.max
    tracker = trk_utils.MetricTracker(
        dir_name=tracker_dir_path,
        metric_name=tracked_metric,
        metric_partition=data_partition_name,
        initial_metric_value=initial_value,
    )
    if (
        min_or_max == tasks_lib.SingleTask.TrackDecoderMetricMode.MIN
        and m_value < tracker.metric_value
    ) or (
        min_or_max == tasks_lib.SingleTask.TrackDecoderMetricMode.MAX
        and m_value > tracker.metric_value
    ):
      logging.info('Updating tracked %s value and checkpoint.', tracked_metric)
      tracker.update(value=m_value, global_step=step)
      # Also save checkpoint; we just need to save the first model replica.
      # WARNING: the checkpoint saved here will not contain optimizer state
      # if it is written by a separate decoding job; if decoding is done
      # interleaved with training as part of the trainer then it will
      # contain them.
      # Decoding with this checkpoint may thus produce different results
      # than those obtained during training if the model state cannot be
      # fully recovered due to the missing optimizer state, e.g. when using
      # EMA during training and separate decoding jobs.
      # TODO(ciprianchelba): specify the checkpoint format and/or async
      # checkpointing.
      if enable_checkpoint_saving:
        unreplicated_model_states = jax.tree_map(
            lambda x: x[0], replicated_model_states
        )
        checkpoints.save_checkpoint(unreplicated_model_states, tracker_dir_path)
