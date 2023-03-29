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

"""Evaluation loop for Pax model."""

import abc
import collections
import contextlib
import functools
import gc
import sys
import time
import typing
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from absl import flags
from absl import logging
from clu import platform
from etils import epath
import jax
from jax.experimental import multihost_utils
import jax.numpy as jnp
import numpy as np
from paxml import base_experiment
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
from paxml import trainer_lib
from paxml import tuning_lib
from praxis import base_hyperparams
from praxis import base_input
from praxis import base_layer
from praxis import optimizer_prefix_vectorization
from praxis import pax_fiddle
from praxis import py_utils
from praxis import pytypes
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

from paxml import checkpoints  # mapped to internal

instantiate = base_hyperparams.instantiate
CheckpointType = checkpoints.CheckpointType
EvaluationMode = io_utils.EvaluationMode
JTensor = pytypes.JTensor
Metrics = pytypes.Metrics
NestedMap = py_utils.NestedMap
JTensor = pytypes.JTensor
NestedJTensor = pytypes.NestedJTensor
NestedPartitionSpec = pytypes.NestedPartitionSpec
NestedShapeDtypeLike = pytypes.NestedShapeDtypeLike
NestedWeightHParams = base_layer.NestedWeightHParams
RunningMode = trainer_lib.RunningMode
SummaryWriter = tf.summary.SummaryWriter
TrainState = train_states.TrainState
WeightedScalars = pytypes.WeightedScalars
WeightedScalarsList = pytypes.WeightedScalarsList
PMAP_PARALLEL_AXIS_NAME = base_layer.PMAP_PARALLEL_AXIS_NAME
PRNGKey = pytypes.PRNGKey
NestedShapeDtypeLike = pytypes.NestedShapeDtypeLike
NO_PREFIX_KEY = optimizer_prefix_vectorization.NO_PREFIX_KEY


def _get_dir_names(
    inputs: Sequence[base_input.BaseInput],
) -> Sequence[epath.Path]:
  """Returns a list of same length for parent dir names for each dataset."""
  return [epath.Path(p.name) for p in inputs]


def _wait_until_step(checkpointer, start_step):
  """Waits until start_step is reached."""
  if not start_step:
    return

  while True:
    cur_step = checkpointer.retrieve_latest_checkpoint_step()
    if cur_step is not None and start_step <= cur_step:
      break
    time.sleep(300)


def _get_train_input_specs(
    task_p: tasks_lib.SingleTask.HParams,
    experiment_config: base_experiment.BaseExperiment,
):
  """Gets the shape/dtype of the inputs to the model."""
  if not task_p.train.always_use_train_for_model_init:
    return None

  input_specs_provider = instantiate(
      experiment_config.get_input_specs_provider_params()
  )
  train_input_specs = input_specs_provider.get_input_specs()
  if task_p.model.mesh_shape is not None:
    train_input_specs = jax.tree_map(
        py_utils.get_global_input_shape_dtype, train_input_specs
    )
  if train_input_specs is None:
    raise ValueError(
        'No training input specs available, while enabling '
        '`task_p.train.always_use_train_for_model_init` requires it.'
    )
  return train_input_specs


class _EvalCheckpointer(metaclass=abc.ABCMeta):
  """Adapts particular implementations of checkpointing into a common API."""

  restore_checkpoint_dir: epath.Path

  def __init__(
      self,
      jax_task: tasks_lib.SingleTask,
      job_log_dir: epath.Path,
      checkpoint_type: checkpoints.CheckpointType,
      restore_checkpoint_dir: epath.Path,
      restore_checkpoint_step: int,
      partitioner: partitioning.Partitioner,
      enforce_restore_shape_check: bool = False,
  ):
    self._jax_task = jax_task
    self._partitioner = partitioner
    self.checkpoint_type = checkpoint_type
    self.job_log_dir = job_log_dir
    self.restore_checkpoint_dir: epath.Path = restore_checkpoint_dir
    self.restore_checkpoint_step: int = restore_checkpoint_step
    self.use_ema: bool = tasks_lib.has_ema(jax_task.hparams)
    self._enforce_restore_shape_check = enforce_restore_shape_check

  def retrieve_latest_checkpoint_step(self) -> Optional[int]:
    return checkpoints.retrieve_latest_checkpoint_step(
        self.restore_checkpoint_dir
    )

  def wait_for_new_step(self, last_checkpoint_step: int) -> int:
    new_checkpoint_step = self.retrieve_latest_checkpoint_step()
    while new_checkpoint_step == last_checkpoint_step:
      logging.info('Sleep before checking for new latest checkpoint.')
      time.sleep(60)
      new_checkpoint_step = self.retrieve_latest_checkpoint_step()
    # There must be a new checkpoint here.
    assert new_checkpoint_step is not None
    logging.info('Found new checkpoint at step: %d', new_checkpoint_step)
    return new_checkpoint_step

  @abc.abstractmethod
  def load_checkpoint_for_step(
      self, step: int, train_state_metadata: trainer_lib.TrainStateMetadata
  ) -> TrainState:
    raise NotImplementedError

  @abc.abstractmethod
  def get_model_states(
      self,
      root_prng_key: PRNGKey,
  ) -> Tuple[TrainState, trainer_lib.TrainStateMetadata, PRNGKey]:
    """Restore the train state from checkpoint or initialize it.

    Args:
      root_prng_key: The root prng key.

    Returns:
      (train_state, train_state_metadata, initialized_root_prng_key).
    """


class _SpmdEvalCheckpointer(_EvalCheckpointer):

  def _restore(
      self, step: int, train_state_metadata: trainer_lib.TrainStateMetadata
  ) -> Optional[TrainState]:
    partitioned_train_state = checkpoints.restore_checkpoint(
        train_state_metadata.padded_global_shapes,
        self.restore_checkpoint_dir,
        global_mesh=self._partitioner.global_mesh,
        checkpoint_type=self.checkpoint_type,
        state_specs=train_state_metadata.partition_specs,
        step=step,
        enforce_restore_shape_check=self._enforce_restore_shape_check,
    )
    py_utils.sync_global_devices(
        f'checkpointer:restored:{self.restore_checkpoint_dir}'
    )
    if partitioned_train_state and self.use_ema:
      partitioned_train_state = tasks_lib.extract_ema(partitioned_train_state)
    return partitioned_train_state

  def load_checkpoint_for_step(
      self, step: int, train_state_metadata: trainer_lib.TrainStateMetadata
  ) -> TrainState:
    partitioned_train_state = self._restore(step, train_state_metadata)
    assert partitioned_train_state
    return partitioned_train_state

  def get_model_states(
      self,
      root_prng_key: PRNGKey,
  ) -> Tuple[TrainState, trainer_lib.TrainStateMetadata, PRNGKey]:
    """Gets a partitioned model states and the step function."""
    train_state_metadata = self._partitioner.get_train_state_metadata(
        discard_opt_states=not self.use_ema
    )
    partition_specs = train_state_metadata.partition_specs
    assert partition_specs is not None, 'must be in pjit mode'
    if self.use_ema and not partition_specs.opt_states:
      # Make sure the opt_states exists before restoring.
      # This is combined with the decoding test.
      raise ValueError(
          "The partition spec doesn't include opt states but ema is enabled."
      )

    partitioned_train_state = self._restore(
        self.restore_checkpoint_step, train_state_metadata
    )
    root_prng_key, partitioned_train_state = (
        self._partitioner.initialize_prng_key_and_train_state(
            root_prng_key,
            partitioned_train_state,
            self.checkpoint_type,
            discard_opt_states=True,
        )
    )
    return partitioned_train_state, train_state_metadata, root_prng_key


class _PmapEvalCheckpointer(_EvalCheckpointer):

  def __init__(
      self,
      jax_task: tasks_lib.SingleTask,
      job_log_dir: epath.Path,
      checkpoint_type: checkpoints.CheckpointType,
      restore_checkpoint_dir: epath.Path,
      restore_checkpoint_step: int,
      partitioner: partitioning.Partitioner,
      mode: EvaluationMode,
      enforce_restore_shape_check: bool = False,
  ):
    super().__init__(
        jax_task,
        job_log_dir,
        checkpoint_type,
        restore_checkpoint_dir,
        restore_checkpoint_step,
        partitioner,
        enforce_restore_shape_check=enforce_restore_shape_check,
    )
    self.track_metric: bool = (mode != EvaluationMode.EVAL) and bool(
        jax_task.hparams.track_decoder_metric
    )
    if py_utils.pmap_use_tensorstore() and self.track_metric:
      raise ValueError(
          '`track_decoder_metric` is currently unsupported with TensorStore '
          'checkpoints.')

  def _restore(
      self, step: int, train_state_global_shapes: TrainState
  ) -> Optional[TrainState]:
    if py_utils.pmap_use_tensorstore():
      model_states = tasks_lib.restore_pmap_from_tensorstore(
          train_state_global_shapes,
          self.restore_checkpoint_dir,
          step=step,
          checkpoint_type=self.checkpoint_type,
          enforce_restore_shape_check=self._enforce_restore_shape_check,
      )
    else:
      model_states = checkpoints.restore_checkpoint(
          train_state_global_shapes,
          self.restore_checkpoint_dir,
          checkpoint_type=self.checkpoint_type,
          step=step,
          enforce_restore_shape_check=self._enforce_restore_shape_check,
      )
    if model_states:
      if self.use_ema:
        model_states = tasks_lib.extract_ema(model_states)
      elif not self.track_metric:
        model_states = model_states.to_eval_state()
    return model_states

  def load_checkpoint_for_step(
      self, step: int, train_state_metadata: trainer_lib.TrainStateMetadata
  ) -> TrainState:
    model_states = self._restore(
        step, train_state_metadata.unpadded_global_shapes
    )
    replicated_model_states = trainer_lib.replicate_model_state(model_states)
    del model_states  # Unused at that point.
    return replicated_model_states

  def get_model_states(
      self,
      root_prng_key: PRNGKey,
  ) -> Tuple[TrainState, trainer_lib.TrainStateMetadata, PRNGKey]:
    # Note: `discard_opt_states` is not supported when restoring pmap flax ckpt.
    # We must restore the entire checkpoint and then trim the opt states.
    train_state_metadata = self._partitioner.get_train_state_metadata(
        discard_opt_states=py_utils.pmap_use_tensorstore() and not self.use_ema,
    )

    model_states = self._restore(
        self.restore_checkpoint_step,
        train_state_metadata.unpadded_global_shapes,
    )
    root_prng_key, replicated_model_states = (
        self._partitioner.initialize_prng_key_and_train_state(
            root_prng_key,
            model_states,
            self.checkpoint_type,
            discard_opt_states=not self.use_ema,
        )
    )
    return replicated_model_states, train_state_metadata, root_prng_key


def _create_checkpointer(
    jax_task: tasks_lib.SingleTask,
    job_log_dir: epath.Path,
    checkpoint_type: checkpoints.CheckpointType,
    mode: Optional[EvaluationMode],
    restore_checkpoint_dir: Optional[epath.PathLike],
    restore_checkpoint_step: Optional[int],
    partitioner: partitioning.Partitioner,
    enforce_restore_shape_check: bool = False,
) -> _EvalCheckpointer:
  if not restore_checkpoint_dir:
    # bool(Path(''))==True, so guarding against this odd Optional explicitly ^
    restore_checkpoint_dir = job_log_dir / 'checkpoints'

  if restore_checkpoint_step is None and mode is not None:
    restore_checkpoint_step = io_utils.get_checkpoint_step(
        job_log_dir, restore_checkpoint_dir, mode
    )
    # TODO(pax-team): Enforce that a checkpoint exists / a checkpoint step was
    # retrieved.

  checkpoints.reregister_type_handlers(
      jax_task.hparams.train.tensorstore_metadata_key
  )
  if jax_task.hparams.model.mesh_shape is not None:
    checkpointer_cls = _SpmdEvalCheckpointer
    extra_kwargs = {}
  else:
    checkpointer_cls = _PmapEvalCheckpointer
    extra_kwargs = dict(mode=mode)
  return checkpointer_cls(
      jax_task,
      job_log_dir,
      checkpoint_type,
      restore_checkpoint_dir,
      restore_checkpoint_step,
      partitioner,
      enforce_restore_shape_check=enforce_restore_shape_check,
      **extra_kwargs,
  )


def run_eval_loop_over_test_splits(
    test_eval_programs: Sequence[programs.SingleTaskEvalProgram],
    eval_partitioned_train_state: TrainState,
    eval_prng_seed: jax.random.KeyArray,
    summary_writers: List[SummaryWriter],
    step: int,
    job_log_dir: epath.Path,
) -> Tuple[
    List[Optional[Dict[str, float]]],  # eval metrics.
    List[Optional[Dict[str, float]]],  # eval scoring metrics.
    List[int],  # performed eval steps.
]:
  """Run evaluation in a loop over a list of test sets.

  Args:
    test_eval_programs: A list of EvalPrograms to conduct eval.
    eval_partitioned_train_state: Train State to use for eval.
    eval_prng_seed: RNG seed for eval programs.
    summary_writers: The summary writer objects to log summaries.
    step: The step at which we are evaling the model.
    job_log_dir: Job's log directory in which scoring outputs will be written.

  Returns:
    A tuple of (a list of eval metrics,
                a list of optional scoring metrics (seqio)
                a list of integer as performed evaluation steps).
      Items from each list are aligned with the `model_inputs`.
  """
  eval_metrics_list = []
  eval_scoring_metrics_list = []
  num_eval_steps = []
  assert len(summary_writers) == len(test_eval_programs)
  for writer, eval_program in zip(summary_writers, test_eval_programs):
    # TODO(laigd): call setup in eval runner.
    eval_program.setup(job_log_dir, eval_prng_seed, writer)
    program_out = eval_program.run(eval_partitioned_train_state, step)
    eval_metrics_list.append(program_out.aux.eval_metrics)
    eval_scoring_metrics_list.append(program_out.aux.eval_scoring_metrics)
    num_eval_steps.append(program_out.aux.num_eval_steps)

  return (eval_metrics_list, eval_scoring_metrics_list, num_eval_steps)


def evaluate(
    experiment_config: base_experiment.BaseExperiment,
    job_log_dir: epath.Path,
    maybe_use_persistence_checkpointing: bool,
    restore_checkpoint_dir: Optional[epath.Path] = None,
    restore_checkpoint_step: Optional[int] = None,
    early_stopping_fn: Optional[trainer_lib.EarlyStoppingFn] = None,
    enable_auto_sharding: bool = False,
    enforce_restore_shape_check: bool = False,
) -> None:
  """Runs the evaluation loop on the entire eval data set.

  Args:
    experiment_config: an instance of BaseExperiment for the experiment to
      evaluate.
    job_log_dir: The directory for the job logs.
    maybe_use_persistence_checkpointing: If set, it will try to use
      persistence-based checkpointing if suitable.
    restore_checkpoint_dir: Optional directory from which to restore a
      checkpoint.
    restore_checkpoint_step: If set, the checkpoint step to restore.
    early_stopping_fn: An optional callable object for reporting eval metrics
      and determining whether to early stop current training. The callable
      object has signature: (metrics, running_mode, ckpt_step, is_final_ckpt) ->
      should_stop_early.
    enable_auto_sharding: Enables the XLA AutoSharding pass to generate SPMD
      shardings.
    enforce_restore_shape_check: Raises an error if restore shapes do not match
      checkpoint shapes.
  """
  jax.monitoring.record_event('/jax/pax/evaluate/beacon')
  eval_input_p = [v for v in experiment_config.datasets() if not v.is_training]
  if not eval_input_p:
    logging.info('No eval datasets defined. Returning early.')
    return
  for inp in eval_input_p:
    if inp.num_infeed_hosts == 0:
      inp.num_infeed_hosts = jax.process_count()
    inp.infeed_host_index = jax.process_index()

  task_p = experiment_config.task()
  task_p = typing.cast(tasks_lib.SingleTask.HParams, task_p)
  jax_task = instantiate(task_p)
  train_input_specs = _get_train_input_specs(task_p, experiment_config)
  prng_key = jax.random.PRNGKey(task_p.evaluate.random_seed)

  checkpoint_type = checkpoints.retrieve_checkpoint_type(
      maybe_use_persistence_checkpointing, jax_task.hparams
  )
  reshard_inputs = checkpoint_type != CheckpointType.PERSISTENCE
  partitioner = partitioning.create_partitioner(
      jax_task,
      init_is_eval=True,
      reshard_inputs=reshard_inputs,
      auto_sharding_mode=RunningMode.EVAL if enable_auto_sharding else None,
  )
  input_for_shape = None
  if not task_p.train.always_use_train_for_model_init:
    assert train_input_specs is None
    # TODO(pax-dev): Investigate if we can use model input specs
    # instead of instantiating this input pipeline.
    input_p = partitioner.preprocess_input_params(eval_input_p[0])
    input_for_shape = instantiate(input_p)
  partitioner.setup(
      jax_task, prng_key, train_input_specs, input_for_shape, job_log_dir
  )

  checkpointer = _create_checkpointer(
      jax_task,
      job_log_dir,
      checkpoint_type,
      EvaluationMode.EVAL,
      restore_checkpoint_dir=restore_checkpoint_dir,
      restore_checkpoint_step=restore_checkpoint_step,
      partitioner=partitioner,
      enforce_restore_shape_check=enforce_restore_shape_check,
  )

  partitioned_train_state, train_state_metadata, prng_key = (
      checkpointer.get_model_states(prng_key)
  )
  eval_runner = _EvalRunner(jax_task, partitioner, eval_input_p, job_log_dir)
  prng_key, eval_key = jax.random.split(prng_key)
  eval_one_step_fn = eval_runner.get_partition_run_one_step_fn(eval_key)
  eval_programs = eval_runner.eval_programs

  decode_once_fn = None
  decode_inputs = None
  continuous_decode = True
  _common_eval_or_decode_loop(
      EvaluationMode.EVAL,
      checkpointer,
      jax_task.hparams,
      job_log_dir,
      eval_one_step_fn,
      decode_once_fn,
      partitioned_train_state,
      train_state_metadata,
      early_stopping_fn,
      continuous_decode,
      eval_programs,
      decode_inputs,
  )


class _EvalRunner:
  """A runner class that runs evaluate with spmd."""

  def __init__(
      self,
      jax_task: tasks_lib.SingleTask,
      partitioner: partitioning.Partitioner,
      eval_input_ps: Sequence[base_input.BaseInput.HParams],
      job_log_dir: epath.Path,
  ):
    self._jax_task = jax_task
    self._partitioner = partitioner
    self._job_log_dir = job_log_dir
    self._eval_programs = [
        programs.SingleTaskEvalProgram(jax_task, input_p, partitioner)
        for input_p in eval_input_ps
    ]
    trainer_lib.check_unique_names(
        [program.eval_input for program in self._eval_programs]
    )

  @property
  def eval_programs(self):
    return self._eval_programs

  def get_partition_run_one_step_fn(self, eval_key):
    logging.info('eval prng_key: %s', eval_key)
    eval_key = self._partitioner.preprocess_prng_key(eval_key)

    def eval_one_step_fn(train_state, eval_summary_writers):
      if not self._eval_programs:
        return tuning_lib.EvalMetrics(
            metrics_list=[],
            scoring_metrics_list=[],
            steps_per_sec=0,
            input_names=[],
        )

      with py_utils.timeit() as eval_period:
        step_i = int(
            py_utils.maybe_unreplicate_for_fully_replicated(train_state.step)
        )
        eval_metrics_list, eval_scoring_metrics_list, num_eval_steps = (
            run_eval_loop_over_test_splits(
                self._eval_programs,
                train_state.to_eval_state(),
                eval_key,
                eval_summary_writers,
                step_i,
                self._job_log_dir,
            )
        )
      return tuning_lib.EvalMetrics(
          metrics_list=eval_metrics_list,
          scoring_metrics_list=eval_scoring_metrics_list,
          steps_per_sec=sum(num_eval_steps) / eval_period.elapsed,
          input_names=[
              program.eval_input.name for program in self._eval_programs
          ],
      )

    return eval_one_step_fn


def decode(
    experiment_config: base_experiment.BaseExperiment,
    job_log_dir: epath.PathLike,
    maybe_use_persistence_checkpointing: bool,
    restore_checkpoint_dir: Optional[epath.PathLike],
    restore_checkpoint_step: Optional[int],
    continuous_decode: bool,
    run_eval: Optional[bool] = False,
    early_stopping_fn: Optional[trainer_lib.EarlyStoppingFn] = None,
    enable_auto_sharding: bool = False,
    enable_checkpoint_saving: bool = True,
    output_pickle: bool = True,
    enforce_restore_shape_check: bool = False,
) -> None:
  """Runs decoding on the decoder datasets.

  Args:
    experiment_config: an instance of BaseExperiment for the experiment to
      decode.
    job_log_dir: The directory for the job logs.
    maybe_use_persistence_checkpointing: If set, it will try to use
      persistence-based checkpointing if suitable.
    restore_checkpoint_dir: The directory from which to restore checkpoint.
    restore_checkpoint_step: If set, the checkpoint step to restore. If unset,
      try to restore from the latest checkpoint if any.
    continuous_decode: whether to continuously decode on the latest ckpt.
    run_eval: whether to run evaluate() (i.e. to obtain scoring based metrics)
      as well.
    early_stopping_fn: An optional callable object for reporting metrics and
      determining whether to early stop current training. The callable object
      has signature: (metrics, running_mode, ckpt_step, is_final_ckpt) ->
      should_stop_early.
    enable_auto_sharding: Enables the XLA AutoSharding pass to generate SPMD
      shardings.
    enable_checkpoint_saving: Whether to perform checkpoint saving or not.
    output_pickle: Output .pickle file alongside the .jsonl file when decoding.
    enforce_restore_shape_check: Raises an error if restore shapes do not match
      checkpoint shapes.
  """
  jax.monitoring.record_event('/jax/pax/decode/beacon')
  job_log_dir = epath.Path(job_log_dir)
  if restore_checkpoint_dir:
    restore_checkpoint_dir = epath.Path(restore_checkpoint_dir)

  decoder_inputs = experiment_config.decoder_datasets()
  eval_inputs = [v for v in experiment_config.datasets() if not v.is_training]

  if not run_eval:
    eval_inputs = []
  if not decoder_inputs and not eval_inputs:
    logging.info('No input datasets defined.')
    return
  for inp in decoder_inputs + eval_inputs:
    if inp.num_infeed_hosts == 0:
      inp.num_infeed_hosts = jax.process_count()
    inp.infeed_host_index = jax.process_index()

  # TODO(laigd): the logic below is very similar to the logic in evaluate(),
  # merge them.
  task_p = experiment_config.task()
  task_p = typing.cast(tasks_lib.SingleTask.HParams, task_p)
  jax_task = instantiate(task_p)
  train_input_specs = _get_train_input_specs(task_p, experiment_config)
  prng_key = jax.random.PRNGKey(task_p.decode.random_seed)

  checkpoint_type = checkpoints.retrieve_checkpoint_type(
      maybe_use_persistence_checkpointing, jax_task.hparams
  )
  reshard_inputs = checkpoint_type != CheckpointType.PERSISTENCE
  partitioner = partitioning.create_partitioner(
      jax_task,
      init_is_eval=True,
      reshard_inputs=reshard_inputs,
      auto_sharding_mode=RunningMode.DECODE if enable_auto_sharding else None,
  )
  input_for_shape = None
  if not task_p.train.always_use_train_for_model_init:
    assert train_input_specs is None
    # We assume that either eval_input or decoder_input can be used to retrieve
    # all the model variable shapes, which is needed for restoring checkpoints.
    #
    # TODO(zhangqiaorjc): If we can no longer assume variable shapes will be the
    # same regardless of which eval_input or decoder_input we use to draw the
    # sample inputs, we need to revisit the design here.

    # TODO(pax-dev): Investigate if we can use model input specs
    # instead of instantiating this input pipeline.
    input_p = partitioner.preprocess_input_params(
        (decoder_inputs + eval_inputs)[0]
    )
    input_for_shape = instantiate(input_p)
  partitioner.setup(
      jax_task, prng_key, train_input_specs, input_for_shape, job_log_dir
  )

  checkpointer = _create_checkpointer(
      jax_task,
      job_log_dir,
      checkpoint_type,
      EvaluationMode.DECODE,
      restore_checkpoint_dir,
      restore_checkpoint_step,
      partitioner=partitioner,
      enforce_restore_shape_check=enforce_restore_shape_check,
  )

  if continuous_decode:
    logging.info(
        'running continuous_decode from %s', checkpointer.restore_checkpoint_dir
    )
  else:
    logging.info(
        'running decode_once restored from %s',
        checkpointer.restore_checkpoint_dir,
    )

  if task_p.model.mesh_shape is not None:
    decode_method = decode_spmd_model
    extra_kwargs = {}
  else:
    decode_method = decode_pmap_model
    extra_kwargs = dict(
        output_pickle=output_pickle,
        enable_checkpoint_saving=enable_checkpoint_saving,
    )
  decode_method(
      jax_task,
      prng_key,
      partitioner,
      checkpointer,
      decoder_inputs,
      eval_inputs,
      job_log_dir,
      continuous_decode,
      early_stopping_fn,
      **extra_kwargs,
  )


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


def decode_pmap_model(
    jax_task: tasks_lib.SingleTask,
    prng_key: PRNGKey,
    partitioner: partitioning.Partitioner,
    checkpointer: _EvalCheckpointer,
    input_p: Sequence[base_input.BaseInput.HParams],
    eval_input_p: Sequence[base_input.BaseInput.HParams],
    job_log_dir: epath.Path,
    continuous_decode: bool,
    early_stopping_fn: Optional[trainer_lib.EarlyStoppingFn] = None,
    output_pickle: bool = True,
    enable_checkpoint_saving: bool = True,
) -> None:
  """Runs the decoding on the entire decoder datasets for a PMAP model.

  Args:
    jax_task: The task encapsulating a the data parallel model.
    prng_key: Root PRNGKey for the decode pipeline.
    partitioner: The partitioner, will be used to partition the step function.
    checkpointer: The model checkpointing method to use.
    input_p: List of input params to be decoded.
    eval_input_p: List of input params to be evaluated.
    job_log_dir: Directory for the job logs.
    continuous_decode: whether to continuously decode on the latest ckpt.
    early_stopping_fn: An optional callable object for reporting metrics and
      determining whether to early stop current training. The callable object
      has signature: (metrics, running_mode, ckpt_step, is_final_ckpt) ->
      should_stop_early.
    output_pickle: Output .pickle file alongside the .jsonl file when decoding.
    enable_checkpoint_saving: Whether to perform checkpoint saving or not.
  """
  task_p = jax_task.hparams
  # Either decoder or eval inputs is not empty.
  assert list(input_p) + list(eval_input_p)

  if continuous_decode:
    # Waits until train.decode_start_after_n_steps is reached.
    _wait_until_step(
        checkpointer, jax_task.hparams.train.decode_start_after_n_steps
    )

  partitioned_train_state, train_state_metadata, prng_key = (
      checkpointer.get_model_states(prng_key)
  )

  eval_runner = _EvalRunner(jax_task, partitioner, eval_input_p, job_log_dir)
  prng_key, eval_key = jax.random.split(prng_key)
  eval_one_step_fn = eval_runner.get_partition_run_one_step_fn(eval_key)
  eval_programs = eval_runner.eval_programs

  # JaxContext needed for parameter sharing.
  context_p = base_layer.JaxContext.HParams(do_eval=True)
  with base_layer.JaxContext.new_context(hparams=context_p):
    trainer_lib.write_post_init_model_hparams_file(
        jax_task.model,
        train_state_metadata.var_weight_hparams,
        job_log_dir / 'decoder_out',
        do_eval=True,
    )

  prng_key, decode_key = jax.random.split(prng_key)
  prng_seed = jax.random.split(decode_key, num=jax.local_device_count())
  logging.info('decoder prng_seed: %s', prng_seed)

  inputs = [instantiate(p) for p in input_p]
  trainer_lib.check_unique_names(inputs)
  decode_once_fn = partition_decode_once_pmap_model(
      jax_task,
      partitioner,
      task_p,
      train_state_metadata.var_weight_hparams,
      inputs,
      prng_seed,
      job_log_dir,
      output_pickle,
      enable_checkpoint_saving=enable_checkpoint_saving,
  )

  decode_inputs = inputs
  _common_eval_or_decode_loop(
      EvaluationMode.DECODE,
      checkpointer,
      task_p,
      job_log_dir,
      eval_one_step_fn,
      decode_once_fn,
      partitioned_train_state,
      train_state_metadata,
      early_stopping_fn,
      continuous_decode,
      eval_programs,
      decode_inputs,
  )


def partition_decode_once_pmap_model(
    jax_task: tasks_lib.SingleTask,
    partitioner: partitioning.Partitioner,
    task_p: tasks_lib.SingleTask.HParams,
    var_weight_hparams: NestedWeightHParams,
    inputs: List[base_input.BaseInput],
    prng_seed: JTensor,
    job_log_dir: epath.Path,
    output_pickle: bool = True,
    enable_checkpoint_saving: bool = True,
) -> Callable[[TrainState, List[SummaryWriter]], tuning_lib.DecodeMetrics]:
  def decode_once_fn(partitioned_train_state, summary_writers):
    with py_utils.timeit() as decode_period:
      (
          decode_metrics_list,
          processed_decode_metrics_list,
          decode_seqio_metrics_list,
          num_decode_steps,
      ) = decode_once_pmap_model(
          jax_task,
          partitioner,
          task_p,
          var_weight_hparams,
          inputs,
          prng_seed,
          job_log_dir,
          partitioned_train_state,
          summary_writers,
          output_pickle,
          enable_checkpoint_saving=enable_checkpoint_saving,
      )
    decode_steps_per_sec = sum(num_decode_steps) / decode_period.elapsed
    return tuning_lib.DecodeMetrics(
        metrics_list=decode_metrics_list,
        processed_metrics_list=processed_decode_metrics_list,
        seqio_metrics_list=decode_seqio_metrics_list,
        steps_per_sec=decode_steps_per_sec,
        input_names=[inp.name for inp in inputs],
    )

  return decode_once_fn


def decode_once_pmap_model(
    jax_task: tasks_lib.SingleTask,
    partitioner: partitioning.Partitioner,
    task_p: tasks_lib.SingleTask.HParams,
    var_weight_hparams: NestedWeightHParams,
    inputs: List[base_input.BaseInput],
    prng_seed: JTensor,
    job_log_dir: epath.Path,
    replicated_model_states: TrainState,
    summary_writers: List[SummaryWriter],
    output_pickle: bool = True,
    enable_checkpoint_saving: bool = True,
) -> Tuple[
    List[Optional[Dict[str, float]]],  # decode metrics.
    List[Optional[Dict[str, float]]],  # processed decode metrics.
    List[Optional[Dict[str, float]]],  # decode (seqio) metrics.
    List[int],  # performed decode steps.
]:
  """Runs the decoding on the entire decoder datasets for a PMAP model.

  Args:
    jax_task: instantiated model from task_p.
    partitioner: The Partitioner used to partition the computations.
    task_p: Params for the task encapsulating a data parallel model.
    var_weight_hparams: Nested structure of HParams for the model weights.
    inputs: instantiated inputs.
    prng_seed: The prng seed used for decoding.
    job_log_dir: Directory for the job logs.
    replicated_model_states: A TrainState object.
    summary_writers: The summary writer objects to log summaries.
    enable_checkpoint_saving: Whether to perform checkpoint saving or not.

  Returns:
    A tuple of (a list of decode metrics,
                a list of processed decode metrics,
                a list of optional decoder (seqio) metrics.
                 list of integers as performed decode steps for each input).
      Items from each list are aligned with each input from inputs.
  """
  if not inputs:
    return [], [], [], []
  work_unit = platform.work_unit()
  model = jax_task.model
  model_p = task_p.model
  metrics_p = task_p.metrics
  if not metrics_p:
    metrics_p = base_metrics.MeanMetrics.HParams()

  step_i = int(
      py_utils.maybe_unreplicate_for_fully_replicated(
          replicated_model_states.step
      )
  )

  logging.info('step=%d', step_i)

  def decode_step(mdl_states, prng_key, inputs, batch_idx):
    if task_p.decode.prng_key_fold_with_batch_index:
      prng_seed_decode = jax.random.fold_in(prng_key, batch_idx)
    else:
      prng_seed_decode = prng_key
    mdl_states = mdl_states.to_eval_state()
    (weighted_scalars, per_example_out, updated_metrics), updated_vars = (
        trainer_lib.decode_step(
            model,
            mdl_states,
            prng_seed_decode,
            var_weight_hparams,
            inputs,
            model_p.fprop_dtype,
            task_p.decode.prng_key_fold_with_global_step,
        )
    )

    weighted_scalars = decode_metrics.aggregate(weighted_scalars)
    aggregated_per_example_out = jax.lax.all_gather(
        per_example_out, axis_name=PMAP_PARALLEL_AXIS_NAME, tiled=True
    )

    summary_tensors = updated_vars.get(base_layer.SUMMARIES, {})
    summary_tensors = summary_utils.flatten_flax_summaries(summary_tensors)
    aggregated_summaries = summary_utils.aggregate_per_replica_summaries(
        summary_tensors
    )

    # We want to aggregate metrics across workers.
    # In pmap we do an all gather of the metric state across workers, and then
    # call reduce() on the metric which by default calls merge across workers.
    aggregated_metrics = {}
    for metric_name, metric in updated_metrics.items():
      aggregated_metrics[metric_name] = jax.lax.all_gather(
          metric, axis_name=PMAP_PARALLEL_AXIS_NAME
      ).reduce()

    return (
        weighted_scalars,
        aggregated_per_example_out,
        aggregated_summaries,
        aggregated_metrics,
    )

  # As an example, suppose the output leaf from trainer_lib.decoder_step()
  # for each core has shape: [per_core_batch_size, decoding_length].
  # In the all_gather we set tiled=True, so the output chunks are all
  # concatenated into the existing batch axis, so we get shape
  # [num_cores x per_core_batch_size, decoding_length].
  # In the pmap call we set out_axes=None to not have to manually unreplicate,
  # so the output of pmap_decode_step() will have the same shape.
  #
  # Example code snippet showing this:
  #   # shape (8, 3, 2)
  #   x = jnp.tile(jnp.arange(8)[:, None, None],[1, 3, 2])
  #   # shape (24, 2)
  #   z = jax.pmap(
  #       lambda y: jax.lax.all_gather(y+1, axis_name='i', tiled=True),
  #       axis_name='i', out_axes=None)(x)
  #
  # We aggregate all outputs from decode_step.
  pmap_decode_step = jax.pmap(
      decode_step,
      axis_name=PMAP_PARALLEL_AXIS_NAME,
      out_axes=(None, None, None, None),
  )

  def decode_step_func(inputs, batch_idx):
    # TODO(pax): shall we eval all sub-models during eval?
    return pmap_decode_step(
        replicated_model_states,
        prng_seed,
        inputs,
        batch_idx * jnp.ones((jax.local_device_count(),)),
    )

  num_steps_per_input = [
      -1 if input.reset_for_eval else input.eval_loop_num_batches
      for input in inputs
  ]
  basedir = job_log_dir / f'{EvaluationMode.DECODE.value}_out'
  dirnames = _get_dir_names(inputs)
  filename = programs.get_filename(
      replicated_model_states.step, EvaluationMode.DECODE.value
  )
  filenames = [basedir / s / filename for s in dirnames]

  decode_metrics_list = []
  processed_decode_metrics_list = []
  seqio_metrics_list = []
  num_decode_steps = []

  for split, num_split_steps in enumerate(num_steps_per_input):
    input_name = inputs[split].name
    if programs.can_load_written_outputs(
        job_log_dir, input_name, EvaluationMode.DECODE, step_i
    ):
      logging.info(
          'Decoding on input %s at step %d already done, skipping.',
          input_name,
          step_i,
      )
      decode_metrics_list.append(None)
      processed_decode_metrics_list.append(None)
      seqio_metrics_list.append(None)
      num_decode_steps.append(0)
      continue
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
    while num_split_steps < 0 or step_num < num_split_steps:
      step_num += 1
      try:
        batch = inputs[split].get_next()
      except (tf.errors.OutOfRangeError, StopIteration):
        inputs[split].reset()
        break
      batch = partitioner.preprocess_inputs(inputs[split], batch, None)
      (batch_metrics, out, summary_tensors, updated_metrics) = decode_step_func(
          batch, batch_idx=step_num
      )
      for key, tensor in summary_utils.flatten_summary_dict(summary_tensors):
        all_summary_tensors[key].append(tensor)
      # we store the metric directly as it has already been aggregated in
      # side decode_step_fun
      decode_metrics.store(batch_metrics)
      logging.info(
          'Finished decoding input batch %d for %s', step_num, input_name
      )

      # Merge clu.metrics to update for each minibatch.
      metrics = _merge_clu_metrics(metrics, updated_metrics)

      # Run `process_decode_out` on CPU device as its implementation is not
      # expected to be JIT friendly. Since we keep track of its outputs, we also
      # don't want on-device allocation as would eventually lead to HBM OOM.
      if jax.process_index() == 0:
        with jax.default_device(jax.devices('cpu')[0]):
          out = jax.tree_map(np.asarray, out)
          process_decode_output = model.process_decode_out(inputs[split], out)

        (processed_scalars, processed_out, processed_metric_updates) = (
            process_decode_output
        )
        processed_out = seqio_input.maybe_update_decode_output_keys(
            processed_out, out
        )

        process_decode_metrics.store(processed_scalars)
        processed_decodes.extend(processed_out)
        if processed_metric_updates:
          processed_metrics = _merge_clu_metrics(
              processed_metrics, processed_metric_updates
          )

        logging.info('Finished processing decoded input batch %d', step_num)

      work_unit.set_task_status(
          f'Finished decoding on {input_name} (batches={step_num})'
      )
      logging.info('Finished decoding on %s (batches=%s)', input_name, step_num)

    # Now the decode loop of multiple batches on current dataset is done,
    # we start to aggregate copmuted metrics and put them in summary.
    seqio_metric_values = None
    if seqio_input.should_process_outputs(inputs[split]):
      logging.info(
          'Finished processing all %d examples.', len(processed_decodes)
      )
      seqio_metric_values = seqio_input.process_outputs(
          inputs[split],
          processed_decodes,
          summary_writers[split],
          seqio_input.MetricType.PREDICT,
          step_i,
          basedir / dirnames[split],
          plain_text_output_fname=f'{filenames[split]}.txt',
      )

    # Convert metrics to Dict[str, clu_values.Value] for summary writing.
    metric_values = metric_utils.compute_metric_values(metrics)
    process_metric_values = metric_utils.compute_metric_values(
        processed_metrics
    )

    with summary_writers[split].as_default():
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

    if (
        jax.process_index() == 0
        and not flags.FLAGS.pax_only_aggregate_summaries
    ):
      dir_path = basedir / dirnames[split]
      dir_path.mkdir(parents=True, exist_ok=True)
      output_file = filenames[split]
      logging.info(
          'Writing decoder output to %s with %d entries',
          output_file,
          len(processed_decodes),
      )
      programs.safe_write_key_value_pairs(
          output_file, processed_decodes, write_pickle=output_pickle
      )

    merged_decode_metrics = metric_utils.update_float_dict(
        metric_utils.as_float_dict(decode_metric_dict),
        metric_utils.as_float_dict(metric_values),
    )
    decode_metrics_list.append(merged_decode_metrics)

    merged_processed_decode_metrics = metric_utils.update_float_dict(
        metric_utils.as_float_dict(processed_metric_dict),
        metric_utils.as_float_dict(process_metric_values),
    )
    processed_decode_metrics_list.append(merged_processed_decode_metrics)
    seqio_metrics_list.append(seqio_metric_values)
    num_decode_steps.append(step_num)

    # Track metric specified by task_p.track_decoder_metric.
    if task_p.track_decoder_metric:
      input_names = [inp.name for inp in inputs]
      _find_and_maybe_update_tracked_metric(
          basedir,
          split,
          dirnames,
          step_i,
          input_names,
          replicated_model_states,
          task_p,
          [merged_decode_metrics, merged_processed_decode_metrics],
          enable_checkpoint_saving=enable_checkpoint_saving,
      )

  return (
      decode_metrics_list,
      processed_decode_metrics_list,
      seqio_metrics_list,
      num_decode_steps,
  )


def decode_spmd_model(
    jax_task: tasks_lib.SingleTask,
    prng_key: PRNGKey,
    partitioner: partitioning.Partitioner,
    checkpointer: _EvalCheckpointer,
    input_p: Sequence[base_input.BaseInput.HParams],
    eval_input_p: Sequence[base_input.BaseInput.HParams],
    job_log_dir: epath.Path,
    continuous_decode: bool,
    early_stopping_fn: Optional[trainer_lib.EarlyStoppingFn],
) -> None:
  """Runs the decoding on the entire decoder datasets for SPMD model.

  Args:
    jax_task: The task that encapsulates an SPMD model.
    prng_key: Root PRNGKey for the decode pipeline.
    partitioner: The partitioner used to partition the step function.
    checkpointer: The model checkpointing method to use.
    input_p: List of input params to be decoded.
    eval_input_p: List of input params to be evaluated.
    job_log_dir: Directory for the job logs.
    continuous_decode: whether to continuously decode on the latest ckpt.
    early_stopping_fn: An optional callable object for reporting metrics and
      determining whether to early stop current training. The callable object
      has signature: (metrics, running_mode, ckpt_step, is_final_ckpt) ->
      should_stop_early.
  """
  task_p = jax_task.hparams
  padded_input_p = [
      trainer_lib.adjust_input_params_for_small_batch(
          inp, partitioner.global_mesh
      )
      for inp in input_p
  ]
  inputs = [instantiate(p) for p in padded_input_p]
  trainer_lib.check_unique_names(inputs)

  # TODO(hthu): Remove eval_input_p as it basically isn't effective.
  # Either decoder or eval inputs is not empty.
  assert list(input_p) + list(eval_input_p)

  if continuous_decode:
    # Waits until train.decode_start_after_n_steps is reached.
    _wait_until_step(
        checkpointer, jax_task.hparams.train.decode_start_after_n_steps
    )

  partitioned_train_state, train_state_metadata, prng_key = (
      checkpointer.get_model_states(prng_key)
  )

  eval_runner = _EvalRunner(jax_task, partitioner, eval_input_p, job_log_dir)
  prng_key, eval_key = jax.random.split(prng_key, 2)
  eval_one_step_fn = eval_runner.get_partition_run_one_step_fn(eval_key)
  eval_programs = eval_runner.eval_programs

  if inputs:
    # Peek to avoid exhausting the input pipeline.
    sample_inputs = inputs[0].peek_padded()
    inputs_shape_dtype = jax.tree_map(
        py_utils.get_global_input_shape_dtype, sample_inputs
    )
  else:
    # This means there is no input to run, and currently this happens only
    # when user runs decode() with eval input, i.e. `mode` is set to DECODE
    # above. In that case, the partitioned decode_step_fn won't get used since
    # there is no decode input, and we simply use the training input shapes
    # to partition.
    # TODO(laigd): avoid cases like this.
    inputs_shape_dtype = partitioner.train_inputs_shape_dtype

  decode_step_fn, is_eval = partitioning.get_step_fn(RunningMode.DECODE)
  assert is_eval
  decode_step_fn, inputs_partition_spec = partitioner.partition(
      decode_step_fn, inputs_shape_dtype, is_eval
  )
  decode_once_fn = partition_decode_once_spmd_model(
      jax_task,
      partitioner,
      task_p,
      inputs,
      job_log_dir,
      prng_key,
      decode_step_fn,
      inputs_partition_spec,
  )
  trainer_lib.write_post_init_model_hparams_file(
      jax_task.model,
      train_state_metadata.var_weight_hparams,
      job_log_dir / 'decoder_out',
      do_eval=True,
  )

  decode_inputs = inputs
  _common_eval_or_decode_loop(
      EvaluationMode.DECODE,
      checkpointer,
      task_p,
      job_log_dir,
      eval_one_step_fn,
      decode_once_fn,
      partitioned_train_state,
      train_state_metadata,
      early_stopping_fn,
      continuous_decode,
      eval_programs,
      decode_inputs,
  )


def partition_decode_once_spmd_model(
    jax_task: tasks_lib.SingleTask,
    partitioner: partitioning.Partitioner,
    task_p: tasks_lib.SingleTask.HParams,
    inputs: List[base_input.BaseInput],
    job_log_dir: epath.Path,
    prng_key: JTensor,
    decode_step_fn: Callable[
        [TrainState, PRNGKey, NestedJTensor, Optional[int]],
        Tuple[Tuple[NestedMap, NestedMap], NestedMap],
    ],
    inputs_partition_spec: NestedPartitionSpec,
) -> Callable[[TrainState, List[SummaryWriter]], tuning_lib.DecodeMetrics]:
  """Returns a function that runs decode over all decoder datasets."""

  def decode_once_fn(partitioned_train_state, summary_writers):
    with py_utils.timeit() as decode_period:
      (
          decode_metrics_list,
          processed_decode_metrics_list,
          decode_seqio_metrics_list,
          num_decode_steps,
      ) = decode_once_spmd_model(
          jax_task,
          partitioner,
          task_p,
          inputs,
          job_log_dir,
          partitioned_train_state,
          summary_writers,
          prng_key,
          decode_step_fn,
          inputs_partition_spec,
      )
    decode_steps_per_sec = sum(num_decode_steps) / decode_period.elapsed
    return tuning_lib.DecodeMetrics(
        metrics_list=decode_metrics_list,
        processed_metrics_list=processed_decode_metrics_list,
        seqio_metrics_list=decode_seqio_metrics_list,
        steps_per_sec=decode_steps_per_sec,
        input_names=[inp.name for inp in inputs],
    )

  return decode_once_fn


def _is_shape_dtype_struct(x):
  """Indicates whether the input is of type ShapeDtypeStruct or not."""
  return isinstance(x, jax.ShapeDtypeStruct)


def decode_once_spmd_model(
    jax_task: tasks_lib.SingleTask,
    partitioner: partitioning.Partitioner,
    task_p: tasks_lib.SingleTask.HParams,
    inputs: List[base_input.BaseInput],
    job_log_dir: epath.Path,
    train_state: TrainState,
    summary_writers: List[SummaryWriter],
    prng_key: JTensor,
    decode_step_fn: Callable[
        [TrainState, PRNGKey, NestedJTensor, Optional[int]],
        Tuple[Tuple[NestedMap, NestedMap], NestedMap],
    ],
    inputs_partition_spec: NestedPartitionSpec,
) -> Tuple[
    List[Optional[Dict[str, float]]],  # decode metrics.
    List[Optional[Dict[str, float]]],  # processed decode metrics.
    List[Optional[Dict[str, float]]],  # decode (seqio) metrics.
    List[int],
]:  # performed decode steps.
  """Runs the decoding once on the entire decoder datasets for an SPMD model.

  Args:
    jax_task: instantiated model from task_p.
    task_p: Params for the task that encapsulates an SPMD model.
    inputs: instantiated inputs.
    job_log_dir: Directory for the job logs.
    train_state: A TrainState object.
    summary_writers: The summary writer objects to log summaries.
    prng_key: The prng key used for decoding.
    decode_step_fn: pjit'ed decode functions.
    inputs_partition_spec: Partition specs for inputs.

  Returns:
    A tuple of (a list of decode metrics,
                a list of processed decode metrics,
                a list of optional decoder (seqio) metrics.
                 list of integers as performed decode steps for each input).
      Items from each list are aligned with each input from inputs.
  """
  work_unit = platform.work_unit()
  metrics_p = task_p.metrics
  if not metrics_p:
    metrics_p = base_metrics.MeanMetrics.HParams()

  step_i = int(
      py_utils.maybe_unreplicate_for_fully_replicated(train_state.step)
  )
  basedir = job_log_dir / f'{EvaluationMode.DECODE.value}_out'
  dirnames = _get_dir_names(inputs)
  filenames = [
      basedir / s / programs.get_filename(step_i, EvaluationMode.DECODE.value)
      for s in dirnames
  ]

  logging.info(
      'partitioned_train_state: %s',
      jax.tree_map(lambda x: x.shape, train_state),
  )
  # We do not fold in jax.process_index in contrast to the pmap version and
  # use a single global key instead to rely on pjit to split for different
  # replicas.
  logging.info('decode prng_key: %s', prng_key)
  spmd_decode_step_fn = functools.partial(
      decode_step_fn, train_state.to_eval_state(), prng_key
  )

  num_steps_per_input = [
      -1 if input.reset_for_eval else input.eval_loop_num_batches
      for input in inputs
  ]
  decode_metrics_list = []
  processed_decode_metrics_list = []
  seqio_metrics_list = []
  num_decode_steps = []

  for split, num_split_steps in enumerate(num_steps_per_input):
    input_name = inputs[split].name
    if programs.can_load_written_outputs(
        job_log_dir, input_name, EvaluationMode.DECODE, step_i
    ):
      logging.info(
          'Decoding on input %s at step %d already done, skipping.',
          input_name,
          step_i,
      )
      decode_metrics_list.append(None)
      processed_decode_metrics_list.append(None)
      seqio_metrics_list.append(None)
      num_decode_steps.append(0)
      continue
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
    while num_split_steps < 0 or step_num < num_split_steps:
      step_num += 1
      try:
        batch = inputs[split].get_next_padded()
      except (tf.errors.OutOfRangeError, StopIteration):
        inputs[split].reset()
        break
      batch = partitioner.preprocess_inputs(
          inputs[split], batch, inputs_partition_spec
      )
      (weighted_scalars, out, updated_metrics), updated_vars = (
          spmd_decode_step_fn(
              batch, inputs[split].get_global_batch_size(inputs[split].hparams)
          )
      )

      # Cross host synchronization happens at this point.
      py_utils.sync_global_devices(f'spmd_decode_step_fn{split}_{step_num}')
      # Output is fully replicated now, so it's ok to unreplicate it by
      # retrieving from device 0 only.
      out = py_utils.maybe_unreplicate_for_fully_replicated(out)
      weighted_scalars = py_utils.maybe_unreplicate_for_fully_replicated(
          weighted_scalars
      )

      # Because outputs of the decode step in pjit are annotated to be Jax
      # Arrays, they are already fully replicated across shards and we can just
      # unreplicate.
      # This also means we don't need to call an all_gather and a reduce()
      # on each clu.metric like we do in pmap mode.
      updated_metrics = py_utils.maybe_unreplicate_for_fully_replicated(
          updated_metrics
      )

      # Merge clu.metrics to update for each minibatch.
      metrics = _merge_clu_metrics(metrics, updated_metrics)

      summary_tensors = updated_vars.get(base_layer.SUMMARIES, {})
      summary_tensors = summary_utils.flatten_flax_summaries(summary_tensors)
      del updated_vars  # release Jax Arrays memory allocations

      summary_tensors = py_utils.maybe_unreplicate_for_fully_replicated(
          summary_tensors
      )
      for key, tensor in summary_utils.flatten_summary_dict(summary_tensors):
        all_summary_tensors[key].append(tensor)

      logging.info(
          'Finished decoding input batch %d for %s', step_num, input_name
      )
      if jax.process_index() != 0:
        continue
      weighted_scalars = jax.tree_map(np.array, weighted_scalars)
      decode_metrics.store(weighted_scalars)

      # Run `process_decode_out` on CPU device as its implementation is not
      # expected to be JIT friendly. Since we keep track of its outputs, we also
      # don't want on-device allocation as would eventually lead to HBM OOM.
      with jax.default_device(jax.devices('cpu')[0]):
        out = jax.tree_map(np.asarray, out)
        process_decode_output = jax_task.model.process_decode_out(
            inputs[split], out
        )

      (process_weighted_scalars, processed, processed_metric_updates) = (
          process_decode_output
      )
      processed = seqio_input.maybe_update_decode_output_keys(processed, out)

      process_decode_metrics.store(process_weighted_scalars)
      processed_decodes.extend(processed)
      if processed_metric_updates:
        processed_metrics = _merge_clu_metrics(
            processed_metrics, processed_metric_updates
        )

      logging.info('Finished processing decoded input batch %d', step_num)

    logging.info('Finished decoding on %s (batches=%s)', input_name, step_num)

    # Now the decode loop of multiple batches on current dataset is done,
    # we start to aggregate copmuted metrics and put them in summary.
    seqio_metric_values = None
    if seqio_input.should_process_outputs(inputs[split]):
      logging.info(
          'Finished processing all %d examples.', len(processed_decodes)
      )
      seqio_metric_values = seqio_input.process_outputs(
          inputs[split],
          processed_decodes,
          summary_writers[split],
          seqio_input.MetricType.PREDICT,
          step_i,
          basedir / dirnames[split],
          plain_text_output_fname=f'{filenames[split]}.txt',
      )

    # Convert metrics to Dict[str, clu_values.Value] for summary writing.
    metric_values = metric_utils.compute_metric_values(metrics)
    process_metric_values = metric_utils.compute_metric_values(
        processed_metrics
    )

    with summary_writers[split].as_default():
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

    if jax.process_index() == 0:
      dir_path = basedir / dirnames[split]
      dir_path.mkdir(parents=True, exist_ok=True)
      output_file = filenames[split]
      logging.info(
          'Writing decoder output to %s with %d entries',
          output_file,
          len(processed_decodes),
      )
      programs.safe_write_key_value_pairs(output_file, processed_decodes)

    work_unit.set_task_status(
        f'Finished processing decoded input batch for {input_name}'
    )

    decode_metrics_list.append(
        metric_utils.update_float_dict(
            metric_utils.as_float_dict(decode_metric_dict),
            metric_utils.as_float_dict(metric_values),
        )
    )
    processed_decode_metrics_list.append(
        metric_utils.update_float_dict(
            metric_utils.as_float_dict(processed_metric_dict),
            metric_utils.as_float_dict(process_metric_values),
        )
    )
    seqio_metrics_list.append(seqio_metric_values)
    num_decode_steps.append(step_num)

    # Track metric specified by task_p.track_decoder_metric.
    if task_p.track_decoder_metric:
      logging.warn(
          'Decoder metric tracking is not implemented yet for pjit '
          'models. Ignoring metric tracking.'
      )

  return (
      decode_metrics_list,
      processed_decode_metrics_list,
      seqio_metrics_list,
      num_decode_steps,
  )


def _common_eval_or_decode_loop(
    mode: io_utils.EvaluationMode,
    checkpointer: _EvalCheckpointer,
    task_p: tasks_lib.SingleTask.HParams,
    job_log_dir: epath.Path,
    eval_one_step_fn: Callable[..., tuning_lib.EvalMetrics],
    decode_once_fn: Optional[Callable[..., tuning_lib.DecodeMetrics]],
    partitioned_train_state: TrainState,
    train_state_metadata: trainer_lib.TrainStateMetadata,
    early_stopping_fn: Optional[trainer_lib.EarlyStoppingFn],
    continuous_decode: bool,
    eval_programs: Sequence[programs.SingleTaskEvalProgram],
    decode_inputs: Optional[Sequence[base_input.BaseInput]],
):
  last_checkpoint_step = checkpointer.retrieve_latest_checkpoint_step()
  logging.info('Evaluation loop starting...')
  summary_base_dir = job_log_dir / 'summaries'
  if decode_inputs:
    summary_decode_dirs = [
        summary_base_dir / f'decode_test_{inp.name}' for inp in decode_inputs
    ]
  summary_eval_dirs = [
      summary_base_dir / f'eval_test_{prg.eval_input.name}'
      for prg in eval_programs
  ]
  with contextlib.ExitStack() as exit_stack:
    if decode_inputs:
      summary_writers = [
          exit_stack.enter_context(summary_utils.get_summary_writer(d))
          for d in summary_decode_dirs
      ]
    eval_summary_writers = [
        exit_stack.enter_context(summary_utils.get_summary_writer(d))
        for d in summary_eval_dirs
    ]

    # Collect then freeze GC, so that GC in the eval loop will not touch the
    # python objects used to initialize the model. Unfreeze at the end of the
    # loop.
    gc.collect()
    gc.freeze()
    while True:
      with io_utils.checkpoint_progress(
          job_log_dir, last_checkpoint_step, mode
      ):
        decode_metrics = None
        if decode_inputs:
          logging.info('Decoding step %s ckpt ...', last_checkpoint_step)
          decode_metrics = decode_once_fn(
              partitioned_train_state, summary_writers
          )

        logging.info('Evaling step %s ckpt ...', last_checkpoint_step)
        eval_metrics = eval_one_step_fn(
            partitioned_train_state, eval_summary_writers
        )

      if not continuous_decode:
        last_checkpoint_step = last_checkpoint_step or 1

      if last_checkpoint_step is not None:
        exceeded_ckpt = last_checkpoint_step + task_p.train.save_interval_steps
        is_last_ckpt = (
            exceeded_ckpt > task_p.train.num_train_steps
            or not continuous_decode
        )
        if tuning_lib.should_early_stop(
            early_stopping_fn,
            last_checkpoint_step,
            is_last_ckpt,
            eval_metrics=eval_metrics,
            decode_metrics=decode_metrics,
        ):
          logging.info(
              (
                  'Early stopped at checkpoint step %d by the'
                  'tuner, while the num_train_steps is %d'
              ),
              last_checkpoint_step,
              task_p.train.num_train_steps,
          )
          break
        if is_last_ckpt:
          break
      # Release partitioned_train_state.
      jax.tree_util.tree_map(lambda x: x.delete(), partitioned_train_state)
      del partitioned_train_state
      new_checkpoint_step = checkpointer.wait_for_new_step(last_checkpoint_step)
      partitioned_train_state = checkpointer.load_checkpoint_for_step(
          new_checkpoint_step, train_state_metadata
      )
      last_checkpoint_step = new_checkpoint_step
    gc.unfreeze()


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


def _find_and_maybe_update_tracked_metric(
    basedir: epath.Path,
    split: int,
    dirnames: Sequence[epath.Path],
    step_i: int,
    input_names: Sequence[str],
    replicated_model_states: TrainState,
    task_p: tasks_lib.SingleTask.HParams,
    decode_metrics_list: List[Dict[str, float]],
    enable_checkpoint_saving: bool = True,
) -> None:
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
        / dirnames[split]
        / f'{tracked_metric_name}_{track_min_or_max}_tracker'
    )
    _maybe_update_tracked_metric(
        m_value,
        step_i,
        tracker_dir_path,
        tracked_metric_name,
        track_min_or_max,
        input_names[split],
        replicated_model_states,
        enable_checkpoint_saving=enable_checkpoint_saving,
    )
  else:
    logging.info(
        'Cannot track metric %s on input %s.',
        tracked_metric,
        input_names[split],
    )


def infer_and_write(
    experiment_config: base_experiment.BaseExperiment,
    job_log_dir: epath.Path,
    enforce_restore_shape_check: bool = False,
) -> None:
  """Generates output from a model and writes it out.

  Args:
    experiment_config: an instance of BaseExperiment for the experiment with
      output generators configured.
    job_log_dir: The base directory for writing the outputs.
    enforce_restore_shape_check: Raises an error if restore shapes do not match
      checkpoint shapes.
  """
  jax.monitoring.record_event('/jax/pax/infer_and_write/beacon')
  task_p = experiment_config.task()
  task_p = typing.cast(tasks_lib.SingleTask.HParams, task_p)
  task = instantiate(task_p)
  model_p = task_p.model
  inputs_p = experiment_config.decoder_datasets()
  prng_key = jax.random.PRNGKey(task_p.infer.random_seed)
  train_input_specs = _get_train_input_specs(task_p, experiment_config)

  maybe_use_persistence_checkpointing = False
  checkpoint_type = checkpoints.retrieve_checkpoint_type(
      maybe_use_persistence_checkpointing, task.hparams
  )
  reshard_inputs = checkpoint_type != CheckpointType.PERSISTENCE
  partitioner = partitioning.create_partitioner(
      task, reshard_inputs=reshard_inputs
  )
  input_for_shape = None
  if not task_p.train.always_use_train_for_model_init:
    assert train_input_specs is None
    # TODO(pax-dev): Investigate if we can use model input specs
    # instead of instantiating this input pipeline.
    input_p = partitioner.preprocess_input_params(inputs_p[0])
    input_for_shape = instantiate(input_p)
  partitioner.setup(
      task, prng_key, train_input_specs, input_for_shape, job_log_dir
  )

  checkpointer = _create_checkpointer(
      task,
      job_log_dir,
      checkpoint_type,
      mode=None,
      restore_checkpoint_dir=task_p.infer_writer.restore_checkpoint_dir,
      restore_checkpoint_step=task_p.infer_writer.restore_checkpoint_step,
      partitioner=partitioner,
      enforce_restore_shape_check=enforce_restore_shape_check,
  )
  for inp in inputs_p:
    if inp.num_infeed_hosts == 0:
      inp.num_infeed_hosts = jax.process_count()
    inp.infeed_host_index = jax.process_index()

  if model_p.mesh_shape is not None:
    # TODO(b/238416854): add support for SPMD models
    raise NotImplementedError('SPMD infer_and_write not implemented yet')
  else:
    infer_and_write_pmap(
        task, prng_key, partitioner, checkpointer, inputs_p, job_log_dir
    )


def infer_and_write_pmap(
    task: tasks_lib.SingleTask,
    prng_key: PRNGKey,
    partitioner: partitioning.Partitioner,
    checkpointer: _EvalCheckpointer,
    inputs_p: Sequence[base_input.BaseInput.HParams],
    job_log_dir: epath.Path,
) -> None:
  """Runs the infer_and_write for each of the inputs given task in pmap."""
  task_p = task.hparams
  infer_writer_p = task_p.infer_writer

  if not inputs_p:
    return

  assert isinstance(checkpointer, _PmapEvalCheckpointer)
  replicated_model_states, train_state_metadata, prng_key = (
      checkpointer.get_model_states(prng_key)
  )

  @functools.partial(jax.pmap, axis_name=PMAP_PARALLEL_AXIS_NAME, out_axes=None)
  def infer_pmap_step(mdl_states, prng_seeds, input_batch):
    outputs = task.inference_runner.infer(
        mdl_states,
        prng_seeds,
        train_state_metadata.var_weight_hparams,
        input_batch,
    )
    # tiled=True folds in first axis into second axis [2,8,5] -> [2*8,5]
    replicated_outputs = jax.lax.all_gather(
        outputs, axis_name=PMAP_PARALLEL_AXIS_NAME, tiled=True
    )

    return replicated_outputs

  # Instantiate inputs to infer on
  inputs = [instantiate(p) for p in inputs_p]
  trainer_lib.check_unique_names(inputs)
  num_steps = [
      -1 if p.reset_for_eval else p.eval_loop_num_batches for p in inputs_p
  ]

  for input_gen, num_steps in zip(inputs, num_steps):
    name = input_gen.hparams.name
    logging.info('Starting output generation on input "%s"', name)

    # Feed each (device, input) pair a unique seed
    prng_key, output_seed = jax.random.split(prng_key)
    output_seeds = jax.random.split(output_seed, jax.local_device_count())

    if num_steps > 0:
      logging.info('total number of steps: %d', num_steps)

    # Only write from one process
    dirname = job_log_dir / 'output' / name
    fq_filename = dirname / 'output'
    if jax.process_index() == 0:
      # Create output dirs if DNE
      if not dirname.exists():
        dirname.mkdir(parents=True, exist_ok=True)

      # Write example schema, metadata, and serialized example protos
      logging.info('writing output to %s', fq_filename)
      features_dict = tfds.features.FeaturesDict(
          task.inference_runner.output_schema
      )
      features_dict.save_config(dirname.as_posix())
      tfds.core.MetadataDict(
          restore_checkpoint_dir=infer_writer_p.restore_checkpoint_dir,
          restore_checkpoint_step=infer_writer_p.restore_checkpoint_step,
          input_name=name,
          model_name=task_p.model.name,
      ).save_metadata(dirname)

      writer = io_utils.ShardedParallelWriter(
          fq_filename,
          infer_writer_p.output_num_shards,
          output_format=infer_writer_p.output_format,
      )

    step = 0
    while num_steps < 0 or step < num_steps:
      step += 1
      logging.info('processing input batch %d', step)
      try:
        batch = input_gen.get_next()
      except (tf.errors.OutOfRangeError, StopIteration):
        input_gen.reset()
        break

      pmap_batch = partitioner.preprocess_inputs(input_gen, batch, None)
      outputs = infer_pmap_step(
          replicated_model_states, output_seeds, pmap_batch
      )
      # Get first device's output since it's been replicated by all-gather
      outputs = py_utils.maybe_unreplicate_for_fully_replicated(outputs)
      outputs_cpu = jax.tree_map(np.asarray, outputs)

      if jax.process_index() == 0:
        serialized_outputs = task.inference_runner.serialize_outputs(
            outputs_cpu
        )
        # fire-and-forget writing
        writer.write(serialized_outputs)

    if jax.process_index() == 0:
      writer.close()
