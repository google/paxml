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
import copy
import functools
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from absl import flags
from absl import logging
from clu import metrics as clu_metrics
from clu import platform
from etils import epath
import jax
import numpy as np
from paxml import base_metrics
from paxml import io_utils
from paxml import metric_utils
from paxml import partitioning
from paxml import programs
from paxml import seqio_input
from paxml import summary_utils
from paxml import tasks_lib
from paxml import train_states
from paxml import trainer_lib
from paxml import xla_passthrough
from praxis import base_hyperparams
from praxis import base_input
from praxis import base_layer
from praxis import base_model
from praxis import pax_fiddle
from praxis import py_utils
from praxis import pytypes
from praxis import trees
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
  ):
    """The constructor.

    Args:
      model: The model to be used for decoding.
      partitioner: The partitioner used to partition the decode function.
      decode_input: The instantiated input to be decoded.
    """
    self._model = model
    self._partitioner = partitioner
    self._input = decode_input
    self._num_steps = (
        -1 if self._input.reset_for_eval else self._input.eval_loop_num_batches
    )
    self._dirname = epath.Path(self._input.name)

    self._prng_key: PRNGKey = None
    self._job_log_dir: epath.Path = None
    self._basedir: epath.Path = None
    self._summary_writer: SummaryWriter = None
    self._use_pmap = None

    self._task = None
    self._output_pickle = None
    self._enable_checkpoint_saving = None

    self._metrics_p: pax_fiddle.Config[base_metrics.BaseMetrics] = None

    # Decode step function information.
    self._decode_step_created = False
    self._decode_step_fn = None
    self._decode_step_input_spec = None

  def setup(
      self,
      *,
      prng_key: JTensor,
      job_log_dir: epath.Path,
      summary_writer: SummaryWriter,
      use_pmap: bool,
      task: Optional[tasks_lib.SingleTask],
      output_pickle: bool,
      enable_checkpoint_saving: bool,
      metrics_p: pax_fiddle.Config[base_metrics.BaseMetrics],
  ) -> None:
    """Sets up the program.

    Args:
      prng_key: The prng key used for decoding.
      job_log_dir: Directory for the job logs.
      summary_writer: The summary writer to log summaries.
      use_pmap: Whether to use PMAP (instead of SPMD/pjit). If this is True,
        `task`, `var_weight_params`, `output_pickle` and
        `enable_checkpoint_saving` should be set; otherwise, `metrics` should
        be set.
      task: Params for the task encapsulating a data parallel model.
      output_pickle: Whether to write decoding results to a pickle file.
      enable_checkpoint_saving: Whether to perform checkpoint saving or not.
      metrics_p: Parameters to configure how to aggregate the metrics.
    """
    self._prng_key = prng_key
    self._job_log_dir = job_log_dir
    self._basedir = self._job_log_dir / f'{EvaluationMode.DECODE.value}_out'
    self._summary_writer = summary_writer
    self._use_pmap = use_pmap

    self._task = task
    self._output_pickle = output_pickle
    self._enable_checkpoint_saving = enable_checkpoint_saving

    self._metrics_p = metrics_p

  def should_run(self, state: TrainState, train_step: int) -> bool:
    # TODO(wangpeng): Implement and use it.
    raise NotImplementedError()

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

  def run(self, state: TrainState, train_step: int) -> programs.ProgramOutput:
    work_unit = platform.work_unit()
    use_pmap = self._use_pmap
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
        state.step if use_pmap else step_i, EvaluationMode.DECODE.value
    )
    filename = basedir / dirname / raw_filename
    metrics_p = self._metrics_p

    if use_pmap:
      output_pickle = self._output_pickle

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
              partitioning_spec=None
              if use_pmap
              else self.decode_input_partition_spec(batch),
          )
      )
      batch = partitioner.preprocess_inputs(
          decode_input, batch, inputs_partition_spec
      )

      if self._task and self._task.decode.prng_key_fold_with_batch_index:
        # In this case, the key is a scalar we need to preprocess it
        # (broadcast/split) after folding in step_num.
        decode_key = jax.random.fold_in(prng_key, step_num)
        decode_key = partitioner.preprocess_prng_key(decode_key)
      else:
        decode_key = prng_key

      decode_out = self.decode_step(
          state,
          decode_key,
          batch,
          trainer_lib.BaseStepFnStaticArgs(
              unpadded_global_batch_size=decode_input.get_global_batch_size(
                  decode_input.hparams
              )
          ),
      )
      if not use_pmap:  # TODO(laigd): investigate if we can remove this sync.
        # Cross host synchronization happens at this point.
        py_utils.sync_global_devices(f'spmd_decode-{input_name}-{step_num}')

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
          'Finished decoding input batch %d for %s', step_num, input_name
      )

      if use_pmap:
        # we store the metric directly as it has already been aggregated in
        # side decode_step_fun
        decode_metrics.store(weighted_scalars)
      elif jax.process_index() == 0:
        # Copy the tensor from device memory to ram, since accumulating such
        # tensor on devices may cause HBM OOM, when
        # task_p.train.summary_accumulate_interval_steps is set.
        # TODO(laigd): investigate whether we should apply this to pmap as well.
        weighted_scalars = jax.tree_map(np.array, weighted_scalars)
        decode_metrics.store(weighted_scalars)

      xla_passthrough.merge_back_xla_unsupported_batch(
          per_example_out, tpu_unsupported_batch
      )

      if jax.process_index() == 0:
        # Run `process_decode_out` on CPU device as its implementation
        # is not expected to be JIT friendly. Since we keep track of
        # its outputs, we also don't want on-device allocation as
        # would eventually lead to HBM OOM.
        with jax.default_device(jax.devices('cpu')[0]):
          per_example_out = jax.tree_map(np.asarray, per_example_out)
          process_decode_output = model.process_decode_out(
              decode_input, per_example_out
          )

        (process_weighted_scalars, processed_out, processed_metric_updates) = (
            process_decode_output
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
    decode_output = DecodeOutput(
        decode_metrics=merged_decode_metrics,
        processed_decode_metrics=merged_processed_decode_metrics,
        seqio_metrics=seqio_metric_values,
        num_decode_steps=step_num,
        raw_decode_metrics=metrics,
    )
    return programs.ProgramOutput(state=state, aux=decode_output)

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
    pass
