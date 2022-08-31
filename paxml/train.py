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

"""Training loop for Pax model."""

import abc
import contextlib
import datetime
import functools
import os
import re
import time
import typing
from typing import Any, Dict, Optional, Sequence, Tuple, Union

from absl import logging
import jax
from jax.experimental import maps
from jax.experimental.gda_serialization import serialization as gda_serialization
import jax.numpy as jnp
from paxml import base_experiment
from paxml import checkpoint_managers
from paxml import checkpoint_pb2
from paxml import eval_lib
from paxml import metric_utils
from paxml import summary_utils
from paxml import tasks_lib
from paxml import trainer_lib
from paxml import tuning_lib
from praxis import base_hyperparams
from praxis import base_input
from praxis import base_layer
from praxis import py_utils
from praxis import pytypes
from praxis import train_states
import tensorflow.compat.v2 as tf

from paxml import checkpoints  # mapped to internal

CheckpointType = checkpoint_pb2.CheckpointType
instantiate = base_hyperparams.instantiate
PMAP_PARALLEL_AXIS_NAME = base_layer.PMAP_PARALLEL_AXIS_NAME
NON_PAX_RNG_KEY = base_layer.NON_PAX_RNG_KEY
PARAMS = base_layer.PARAMS
NestedShapeDtypeLike = pytypes.NestedShapeDtypeLike
SummaryWriter = tf.summary.SummaryWriter
CheckpointManager = Union[checkpoint_managers.CheckpointManager,
                          checkpoint_managers.OrbaxCheckpointManager]
Checkpointer = checkpoints.Checkpointer
PaxCheckpointHandler = checkpoints.PaxCheckpointHandler
TRAIN_STATE_KEY = checkpoint_managers.TRAIN_STATE_KEY

_N_STEPS_WARMUP_LOGGING = 5


def _checkpoint_dir(job_log_dir: str) -> str:
  """Returns the checkpoint directory from the root `job_log_dir`."""
  return os.path.join(job_log_dir, 'checkpoints')


def _make_checkpoint_dir(job_log_dir: str) -> str:
  checkpoint_dir = _checkpoint_dir(job_log_dir)
  if jax.process_index() == 0 and not tf.io.gfile.exists(checkpoint_dir):
    tf.io.gfile.makedirs(checkpoint_dir)
  # Block all hosts until directory is ready.
  py_utils.sync_global_devices(f'checkpointer:makedirs:{checkpoint_dir}')
  return checkpoint_dir


def _parse_duration(
    duration_str: Optional[str]) -> Optional[datetime.timedelta]:
  """Parses a duration string and returns the datetime.timedelta instance.

  Args:
    duration_str: A string representing a duration or None. Either (a) an
      integer, the implicit unit being the second, (b) an integer followed by
      's', e.g. '30s', the unit being the second, (c) an integer followed by
      'm', e.g. '15m', the unit being the minute, (d) an integer followed by
      'h', e.g. '2h', the unit being the hour or (e) an integer followed by 'd',
      e.g. '1d' the unit being the hour.

  Returns:
    The corresponding duration as a datetime.timedelta instance or None if the
    input was None.
  """
  if not duration_str:
    return None
  pattern = re.compile(r'(\d+)(\w)*')
  match = pattern.match(duration_str)
  if (not match or len(match.groups()) != 2 or
      match.group(2) not in {None, 's', 'm', 'h', 'd'}):
    raise ValueError(f'Unable to parse string duration `{duration_str}`.')
  int_value = int(match.group(1))
  if match.group(2) is None or match.group(2) == 's':
    pass
  elif match.group(2) == 'm':
    int_value *= 60
  elif match.group(2) == 'h':
    int_value *= 3600
  elif match.group(2) == 'd':
    int_value *= 86400
  else:
    raise ValueError(f'Unable to parse string duration `{duration_str}`.')
  return datetime.timedelta(seconds=int_value)


class _TrainingCheckpointer(metaclass=abc.ABCMeta):
  """Adapts particular implementations of checkpointing into a common API."""

  @abc.abstractmethod
  def save_if_needed(self, step_i, partitioned_train_state, train_state_pspecs):
    raise NotImplementedError

  @abc.abstractmethod
  def save_final(self, step_i, partitioned_train_state, train_state_pspecs):
    raise NotImplementedError

  @abc.abstractmethod
  def restore(self, train_state_global_shapes, global_mesh, train_state_pspecs):
    raise NotImplementedError

  def maybe_sync_multihostcheckpointing(self):
    pass


class _PjitTrainingCheckpointer(_TrainingCheckpointer):

  def __init__(self, checkpoint_manager: checkpoint_managers.CheckpointManager,
               checkpoint_type, async_ckpt_manager, async_checkpointer,
               job_log_dir):
    self.checkpoint_dir = _make_checkpoint_dir(job_log_dir)
    checkpoints.delete_temp_directories(self.checkpoint_dir)
    self.async_ckpt_manager = async_ckpt_manager
    self.async_checkpointer = async_checkpointer

    self.checkpoint_type = checkpoint_type
    self.multi_host_checkpointing = checkpoint_type == CheckpointType.CHECKPOINT_GDA
    self.checkpoint_manager = checkpoint_manager

  def save(self,
           step_i,
           partitioned_train_state,
           train_state_pspecs,
           is_final=False):
    logging.info('Saving a ckpt at %sstep: %d', 'final ' if is_final else '',
                 step_i)
    py_utils.sync_global_devices(
        f'checkpointer:saving:{self.checkpoint_dir}:step-{step_i}')
    checkpoints.save_checkpoint(
        partitioned_train_state,
        self.checkpoint_dir,
        checkpoint_type=self.checkpoint_type,
        state_specs=train_state_pspecs,
        async_ckpt_manager=self.async_ckpt_manager,
        use_orbax=False,
        async_checkpointer=self.async_checkpointer)
    self.checkpoint_manager.save_metadata(global_step_id=step_i, force=is_final)
    py_utils.sync_global_devices(
        f'checkpointer:saved:{self.checkpoint_dir}:step-{step_i}')

  def save_if_needed(self, step_i, partitioned_train_state, train_state_pspecs):
    if self.checkpoint_manager.should_save(step_i):
      self.save(
          step_i, partitioned_train_state, train_state_pspecs, is_final=False)

    if self.async_ckpt_manager is not None:
      # Since the checkpoint is happening asynchronously, the errors may
      # be caught after some time (when the training is continuing). So
      # check on every step if there were any errors raised by the
      # manager and raise them in the main thread.
      self.async_ckpt_manager.check_for_errors()

  def save_final(self, step_i, partitioned_train_state, train_state_pspecs):
    if self.checkpoint_manager.last_checkpoint_step < step_i:
      self.save(
          step_i, partitioned_train_state, train_state_pspecs, is_final=True)

  def maybe_sync_multihostcheckpointing(self):
    if self.multi_host_checkpointing:
      py_utils.sync_global_devices(
          f'checkpointer:restored:{self.checkpoint_dir}')

  def restore(self, train_state_global_shapes, global_mesh, train_state_pspecs):
    return checkpoints.restore_checkpoint(
        train_state_global_shapes,
        self.checkpoint_dir,
        global_mesh=global_mesh,
        checkpoint_type=self.checkpoint_type,
        state_specs=train_state_pspecs,
        use_orbax=False)


class _OrbaxPjitTrainingCheckpointer(_TrainingCheckpointer):

  def __init__(self,
               checkpoint_manager: checkpoint_managers.OrbaxCheckpointManager):
    self.checkpoint_manager = checkpoint_manager

  def save_final(self, step_i, partitioned_train_state, train_state_pspecs):
    logging.info('Saving a ckpt at final step: %d', step_i)
    self.checkpoint_manager.save(step_i,
                                 {TRAIN_STATE_KEY: partitioned_train_state})

  def save_if_needed(self, step_i, partitioned_train_state, train_state_pspecs):
    self.checkpoint_manager.save(step_i,
                                 {TRAIN_STATE_KEY: partitioned_train_state})
    self.checkpoint_manager.check_for_errors()

  def restore(self, train_state_global_shapes, global_mesh, train_state_pspecs):
    items = {TRAIN_STATE_KEY: train_state_global_shapes}
    restore_kwargs = {
        TRAIN_STATE_KEY: {
            'specs': train_state_pspecs,
            'mesh': global_mesh,
        }
    }
    step = self.checkpoint_manager.latest_step()
    if step is None:
      partitioned_train_state = None
    else:
      partitioned_train_state = self.checkpoint_manager.restore(
          step, items=items, restore_kwargs=restore_kwargs)
    return partitioned_train_state


class _PmapTrainingCheckpointer(_TrainingCheckpointer):

  def __init__(self, job_log_dir,
               checkpoint_manager: checkpoint_managers.CheckpointManager,
               async_ckpt_manager, async_checkpointer):
    self.job_log_dir = job_log_dir
    self.checkpoint_dir = _checkpoint_dir(job_log_dir)
    self.checkpoint_manager = checkpoint_manager
    self.async_ckpt_manager = async_ckpt_manager
    self.async_checkpointer = async_checkpointer

  def restore_from_tensorstore(self, train_state_global_shapes):
    _make_checkpoint_dir(self.job_log_dir)
    checkpoints.delete_temp_directories(self.checkpoint_dir)
    logging.info('Pmap restore from TensorStore checkpoint...')
    # Restored from GDA checkpoint dir.
    return tasks_lib.restore_pmap_from_tensorstore(train_state_global_shapes,
                                                   self.checkpoint_dir)

  def restore(self, train_state_global_shapes, global_mesh, train_state_pspecs):
    if py_utils.pmap_use_tensorstore():
      return self.restore_from_tensorstore(train_state_global_shapes)
    else:
      return checkpoints.restore_checkpoint(
          train_state_global_shapes, self.checkpoint_dir)

  def save(self, step_i, partitioned_train_state, is_final=False):
    logging.info('Saving a ckpt at %sstep: %d', 'final ' if is_final else '',
                 step_i)
    if py_utils.pmap_use_tensorstore():
      logging.info('Pmap saving a TensorStore ckpt at step: %d', step_i)
      py_utils.sync_global_devices(
          f'checkpointer:saving:{self.checkpoint_dir}:step-{step_i}')
      fully_replicated_gda_model_states = jax.tree_map(
          py_utils.convert_fully_replicated_sda_to_gda, partitioned_train_state)
      checkpoints.save_checkpoint(
          fully_replicated_gda_model_states,
          self.checkpoint_dir,
          checkpoint_type=CheckpointType.CHECKPOINT_GDA,
          async_ckpt_manager=self.async_ckpt_manager,
          use_orbax=False,
          async_checkpointer=self.async_checkpointer)
      py_utils.sync_global_devices(
          f'checkpointer:saved:{self.checkpoint_dir}:step-{step_i}')
    else:
      if jax.process_index() == 0:
        # We just need to save the first model replica.
        unreplicated_model_states = jax.tree_map(lambda x: x[0],
                                                 partitioned_train_state)
        checkpoints.save_checkpoint(
            unreplicated_model_states,
            self.checkpoint_dir,
            use_orbax=False,
            async_checkpointer=self.async_checkpointer)
    self.checkpoint_manager.save_metadata(global_step_id=step_i, force=is_final)

  def save_if_needed(self, step_i, partitioned_train_state, train_state_pspecs):
    if self.checkpoint_manager.should_save(step_i):
      self.save(step_i, partitioned_train_state, is_final=False)

    if self.async_ckpt_manager is not None:
      # Since the checkpoint is happening asynchronously, the errors may
      # be caught after some time (when the training is continuing). So
      # check on every step if there were any errors raised by the
      # manager and raise them in the main thread.
      self.async_ckpt_manager.check_for_errors()

  def save_final(self, step_i, partitioned_train_state, train_state_pspecs):
    if self.checkpoint_manager.last_checkpoint_step < step_i:
      self.save(step_i, partitioned_train_state, is_final=True)


class _OrbaxPmapTrainingCheckpointer(_TrainingCheckpointer):

  def __init__(self, job_log_dir,
               checkpoint_manager: checkpoint_managers.OrbaxCheckpointManager):
    self.job_log_dir = job_log_dir
    self.checkpoint_dir = _checkpoint_dir(job_log_dir)
    self.checkpoint_manager = checkpoint_manager

  def restore_from_tensorstore(self, train_state_global_shapes):
    # TODO(cpgaffney): Use Orbax APIs.
    _make_checkpoint_dir(self.job_log_dir)
    checkpoints.delete_temp_directories(self.checkpoint_dir)
    logging.info('Pmap restore from TensorStore checkpoint...')
    # Restored from GDA checkpoint dir.
    return tasks_lib.restore_pmap_from_tensorstore(train_state_global_shapes,
                                                   self.checkpoint_dir)

  def restore(self, train_state_global_shapes, global_mesh, train_state_pspecs):
    if py_utils.pmap_use_tensorstore():
      return self.restore_from_tensorstore(train_state_global_shapes)
    else:
      step = self.checkpoint_manager.latest_step()
      items = {TRAIN_STATE_KEY: train_state_global_shapes}
      if step is None:
        train_state = None
      else:
        train_state = self.checkpoint_manager.restore(step, items=items)
    return train_state

  def save(self, step_i, partitioned_train_state, is_final=False):
    if py_utils.pmap_use_tensorstore():
      logging.info('Saving a ckpt at %sstep: %d', 'final ' if is_final else '',
                   step_i)
      py_utils.sync_global_devices(
          f'checkpointer:saving:{self.checkpoint_dir}:step-{step_i}')
      fully_replicated_gda_train_state = jax.tree_map(
          py_utils.convert_fully_replicated_sda_to_gda, partitioned_train_state)
      self.checkpoint_manager.save(
          step_i, {TRAIN_STATE_KEY: fully_replicated_gda_train_state})
      py_utils.sync_global_devices(
          f'checkpointer:saved:{self.checkpoint_dir}:step-{step_i}')
    else:
      if jax.process_index() == 0:
        # We just need to save the first model replica.
        unreplicated_train_state = jax.tree_map(lambda x: x[0],
                                                partitioned_train_state)
        self.checkpoint_manager.save(
            step_i, {TRAIN_STATE_KEY: unreplicated_train_state})

  def save_if_needed(self, step_i, partitioned_train_state, train_state_pspecs):
    self.save(step_i, partitioned_train_state)

    self.checkpoint_manager.check_for_errors()

  def save_final(self, step_i, partitioned_train_state, train_state_pspecs):
    if self.checkpoint_manager.latest_step() < step_i:
      self.save(step_i, partitioned_train_state, is_final=True)

def _create_checkpoint_manager(
    task_p: tasks_lib.SingleTask.HParams,
    job_log_dir: str,
    checkpoint_type: CheckpointType,
    todelete_subdir: Optional[str],
    use_orbax: bool = False,
    async_checkpointer: Optional[checkpoints.AsyncCheckpointer] = None
) -> CheckpointManager:
  """Creates a checkpoint manager."""
  checkpoint_dir = _make_checkpoint_dir(job_log_dir)
  train_p = task_p.train
  max_to_keep = train_p.save_max_to_keep
  save_interval_steps = train_p.save_interval_steps
  keep_interval_timedelta = _parse_duration(train_p.save_keep_interval_duration)
  if use_orbax:
    options = checkpoint_managers.CheckpointManagerOptions(
        max_to_keep=max_to_keep,
        save_interval_steps=save_interval_steps,
        keep_time_interval=keep_interval_timedelta,
        todelete_subdir=todelete_subdir)
    checkpointer = async_checkpointer
    if checkpointer is None:
      checkpointer = Checkpointer(PaxCheckpointHandler(enable_flax=False))
    return checkpoint_managers.OrbaxCheckpointManager(
        checkpoint_dir, {TRAIN_STATE_KEY: checkpointer}, options)
  else:
    return checkpoint_managers.CheckpointManager(
        config_name='',
        root_dir=checkpoint_dir,
        checkpoint_type=checkpoint_type,
        max_to_keep=max_to_keep,
        save_interval_steps=save_interval_steps,
        keep_interval_timedelta=keep_interval_timedelta,
        todelete_subdir=todelete_subdir)



def _update_latest_model_step(train_input_p: base_input.BaseInput.HParams,
                              initial_global_step: int,
                              eval_interval_steps: int) -> None:
  """Updates `train_input_p` in place its latest model step."""
  del eval_interval_steps
  if not hasattr(train_input_p, 'deterministic_input_start_index'):
    return
  dp = train_input_p.deterministic_input_start_index
  dp._latest_model_step = initial_global_step  # pylint: disable=protected-access


def _should_early_stop_training(early_stopping_fn: trainer_lib.EarlyStoppingFn,
                                running_mode: trainer_lib.RunningMode,
                                task_p: tasks_lib.SingleTask.HParams,
                                step_i: int, metrics: Dict[str, float]) -> bool:
  """Returns True if current training should be stopped."""
  assert early_stopping_fn is not None
  train_p = task_p.train

  remaining = train_p.num_train_steps - step_i
  is_last_ckpt = remaining == 0
  if not is_last_ckpt:
    last_eval = False
    if running_mode & trainer_lib.RunningMode.EVAL:
      last_eval = remaining < max(train_p.eval_interval_steps,
                                  train_p.save_interval_steps)
    last_decode = False
    if running_mode & trainer_lib.RunningMode.DECODE:
      last_decode = remaining < max(train_p.decode_interval_steps,
                                    train_p.save_interval_steps)
    is_last_ckpt = last_eval or last_decode
  return early_stopping_fn(metrics, running_mode, step_i, is_last_ckpt)


def _compute_steps_per_sec(step_i, summary_last_time, summary_last_step):
  """Computes the number of training steps per second."""
  # Note: This function doesn't account for the time spent on running
  # interleaved evaluation (if any) and/or evaluation on the training batch.
  # It's, hence, merely a raw underestimate.
  duration_sec = time.time() - summary_last_time
  num_steps = step_i - summary_last_step
  steps_per_sec = num_steps / duration_sec
  return steps_per_sec


def _should_log_train(step: int,
                      train_p: tasks_lib.SingleTask.TrainHParams) -> bool:
  """Indicates whether train output must be logged to the INFO stream."""
  if train_p.log_train_output_interval_steps is not None:
    return step % train_p.log_train_output_interval_steps == 0
  else:
    return step % train_p.summary_interval_steps == 0


def write_hparams_file(model_config: base_experiment.BaseExperiment,
                       job_log_dir: str, filename_prefix: str = '') -> None:
  """Writes a params file into the root `job_log_dir`."""
  if jax.process_index() == 0:
    params_fpath = os.path.join(job_log_dir,
                                filename_prefix + 'model_params.txt')
    if not tf.io.gfile.exists(job_log_dir):
      tf.io.gfile.makedirs(job_log_dir)
    with tf.io.gfile.GFile(params_fpath, 'w') as hparams_file:
      for dataset in model_config.datasets():
        hparams_file.write(dataset.to_text())
        hparams_file.write('\n\n')
      for decoder_dataset in model_config.decoder_datasets():
        hparams_file.write('decoder dataset hparams\n')
        hparams_file.write(decoder_dataset.to_text())
        hparams_file.write('\n\n')
      hparams_file.write(model_config.task().to_text())


def train_and_evaluate(
    experiment_config: base_experiment.BaseExperiment,
    job_log_dir: Optional[str],
    maybe_use_persistence_checkpointing: bool,
    eval_on_test: Optional[bool],
    checkpoint_todelete_subdir: Optional[str] = None,
    early_stopping_fn: Optional[trainer_lib.EarlyStoppingFn] = None,
    async_ckpt_manager: Optional[
        gda_serialization.GlobalAsyncCheckpointManagerBase] = None,
    run_decode: bool = False,
    enable_auto_sharding: bool = False,
    use_orbax: bool = False,
    async_checkpointer: Optional[checkpoints.AsyncCheckpointer] = None) -> None:
  """The shared path to run the training and evaluation loop.

  Args:
    experiment_config: an instance of BaseExperiment for the experiment to train
      and evaluate.
    job_log_dir: The directory for the job logs.
    maybe_use_persistence_checkpointing: If set, it will try to use
      persistence-based checkpointing if suitable.
    eval_on_test: Whether to eval on test as a part of the training loop.
    checkpoint_todelete_subdir: If set, checkpoints to be deleted will be only
      renamed into the provided subdirectory. Otherwise, they will be directly
      deleted from the file system. This is useful, when checkpoint deletion is
      time consuming.
    early_stopping_fn: An optional callable object for reporting eval metrics
      and determining whether to early stop current training. The callable
      object has signature: (metrics_by_dataset, ckpt_step, is_final_ckpt) ->
      should_stop_early.
    async_ckpt_manager: Asynchronous checkpoint manager which manages
      serialization and deserialization of GDA arrays. This manager allows
      training to continue when checkpointing is going on as checkpointing
      happens in a different thread.
    run_decode: whether to periodically run decode as part of the training loop.
      If and only if this is True, every `task_p.train.decode_interval_steps` of
      training, model runs decode.
    enable_auto_sharding: Enables the XLA Auto SPMD partitioner.
    use_orbax: Enables checkpointing backed by Orbax.
    async_checkpointer: When async checkpointing and Orbax are enabled, allows
      training to continue when checkpointing is going on as checkpointing
      happens in a different thread.
  """
  task_p = experiment_config.task()
  task_p = typing.cast(tasks_lib.SingleTask.HParams, task_p)

  input_p = experiment_config.datasets()
  # Note that we modify input params below with runtime information, therefore
  # experiment_config.datasets() should not be called again as it won't have the
  # correct runtime information populated.
  for inp in input_p:
    if not isinstance(inp, base_input.BaseInput.HParams):
      raise ValueError('Expecting BaseInput.HParams from datasets(), got: '
                       f'{inp.ToText()}')
    if inp.num_infeed_hosts == 0:
      inp.num_infeed_hosts = jax.process_count()
    inp.infeed_host_index = jax.process_index()
  train_input_p = [v for v in input_p if v.is_training]
  if len(train_input_p) != 1:
    raise ValueError(
        f'Expecting exactly one training split. Got `{len(train_input_p)}`.')
  train_input_p = train_input_p[0]
  logging.info('train_input_p=%s', train_input_p.to_text())
  logging.info('task_p=%s', task_p.to_text())
  eval_input_p = None
  if eval_on_test:
    eval_input_p = [v for v in input_p if not v.is_training]

  if (run_decode and task_p.train.decode_interval_steps is not None and
      task_p.train.decode_interval_steps > 0):
    decode_input_p = experiment_config.decoder_datasets()
  else:
    decode_input_p = []
  for inp in decode_input_p:
    if inp.num_infeed_hosts == 0:
      inp.num_infeed_hosts = jax.process_count()
    inp.infeed_host_index = jax.process_index()

  checkpoint_type = checkpoints.retrieve_checkpoint_type(
      maybe_use_persistence_checkpointing, task_p)

  checkpoint_manager = _create_checkpoint_manager(
      task_p,
      job_log_dir,
      checkpoint_type,
      checkpoint_todelete_subdir,
      use_orbax=use_orbax)

  if task_p.model.ici_mesh_shape is not None:
    if use_orbax:
      checkpointer = _OrbaxPjitTrainingCheckpointer(checkpoint_manager)
    else:
      checkpointer = _PjitTrainingCheckpointer(checkpoint_manager,
                                               checkpoint_type,
                                               async_ckpt_manager,
                                               async_checkpointer, job_log_dir)
    train_and_evaluate_spmd_model(
        task_p,
        train_input_p,
        job_log_dir,
        checkpointer,
        checkpoint_type,
        eval_input_p,
        decode_input_p,
        early_stopping_fn,
        enable_auto_sharding)
  else:
    if use_orbax:
      checkpointer = _OrbaxPmapTrainingCheckpointer(job_log_dir,
                                                    checkpoint_manager)
    else:
      checkpointer = _PmapTrainingCheckpointer(job_log_dir, checkpoint_manager,
                                               async_ckpt_manager,
                                               async_checkpointer)
    train_and_evaluate_pmap(
        task_p,
        train_input_p,
        job_log_dir,
        checkpointer,
        eval_input_p,
        decode_input_p,
        early_stopping_fn)


class _PeekableInput:
  """Wraps a BaseInput to provide a peek() method. Single thread access only."""

  def __init__(self, inp: base_input.BaseInput) -> None:
    self._inp = inp
    self._peek = None

  def get_next_padded(self):
    if self._peek is None:
      return self._inp.get_next_padded()
    peek = self._peek
    self._peek = None
    return peek

  def peek_padded(self):
    if self._peek is None:
      try:
        self._peek = self._inp.get_next_padded()
      except (tf.errors.OutOfRangeError, StopIteration):
        self._peek = None
    return self._peek


class _SummaryContextManager(contextlib.ExitStack):
  """Manage summary writers."""

  _exit_callbacks = []

  def __init__(self,
               job_log_dir: str,
               eval_input_p: Sequence[base_input.BaseInput.HParams],
               decode_input_p: Sequence[base_input.BaseInput.HParams],
               eval_skip_train: bool = False):
    """Initialize context manager.

    Args:
      job_log_dir: Directory for the job logs.
      eval_input_p: Optional list of params for the eval input pipelines.
      decode_input_p: Optional list of hparams for the decode input pipelines.
      eval_skip_train: By default, we also run eval on the training data input
        (`eval_train`), specifically on a batch not yet used for training. When
        set to True, this is skipped.
    """
    super().__init__()
    self.summary_base_dir = os.path.join(job_log_dir, 'summaries')
    self.summary_train_dir = os.path.join(self.summary_base_dir, 'train')
    self.summary_eval_dir = os.path.join(self.summary_base_dir, 'eval_train')
    self.summary_writer = summary_utils.get_summary_writer
    if eval_input_p:
      self.summary_eval_test_dirs = [
          os.path.join(self.summary_base_dir, f'eval_test_{p.name}')
          for p in eval_input_p
      ]
    else:
      self.summary_eval_test_dirs = []
    if decode_input_p:
      self.summary_decode_dirs = [
          os.path.join(self.summary_base_dir, f'decode_test_{p.name}')
          for p in decode_input_p
      ]
    else:
      self.summary_decode_dirs = []
    self.eval_skip_train = eval_skip_train

  def __enter__(
      self
  ) -> Tuple[SummaryWriter, SummaryWriter, SummaryWriter, SummaryWriter]:
    self.train_summary_writer = self.enter_context(
        self.summary_writer(self.summary_train_dir))
    self.eval_summary_writer = None
    if not self.eval_skip_train:
      self.eval_summary_writer = self.enter_context(
          self.summary_writer(self.summary_eval_dir))
    self.eval_test_summary_writers = [
        self.enter_context(self.summary_writer(d))
        for d in self.summary_eval_test_dirs
    ]
    self.decode_summary_writers = [
        self.enter_context(self.summary_writer(d))
        for d in self.summary_decode_dirs
    ]
    return (self.train_summary_writer, self.eval_summary_writer,
            self.eval_test_summary_writers, self.decode_summary_writers)


def train_and_evaluate_pmap(
    task_p: tasks_lib.SingleTask.HParams,
    train_input_p: base_input.BaseInput.HParams,
    job_log_dir: Optional[str],
    checkpointer: _TrainingCheckpointer,
    eval_input_p: Optional[Sequence[base_input.BaseInput.HParams]],
    decode_input_p: Sequence[base_input.BaseInput.HParams],
    early_stopping_fn: Optional[trainer_lib.EarlyStoppingFn] = None) -> None:
  """Runs the training and evaluation loop with PMAP.

  Args:
    task_p: HParams for the task encapsulating the data parallel model.
    train_input_p: HParams for the train data input pipeline.
    job_log_dir: Directory for the job logs.
    checkpointer: Callbacks for checkpointing.
    eval_input_p: Optional list of hparams for the eval input pipelines.
    decode_input_p: Optional list of hparams for the decode input pipelines.
    early_stopping_fn: An optional callable object for reporting eval metrics
      and determining whether to early stop current training. The callable
      object has signature: (metrics_by_dataset, ckpt_step, is_final_ckpt) ->
      should_stop_early.
  """
  logging.info('Using pmap for data parallelism.')
  if jax.config.jax_parallel_functions_output_gda:
    logging.warning('--jax_use_gda is set to True but ignored for pmap.')
  jax_task = instantiate(task_p)

  if eval_input_p:
    eval_input_pipelines = [instantiate(input_p) for input_p in eval_input_p]
    trainer_lib.check_unique_names(eval_input_pipelines)
  if decode_input_p:
    decode_input_pipelines = [
        instantiate(input_p) for input_p in decode_input_p
    ]
    trainer_lib.check_unique_names(decode_input_pipelines)

  # TODO(shafey): Retrieve the seeds from the model definition instead.
  prng_key = jax.random.PRNGKey(1234)
  prng_key, init_key = jax.random.split(prng_key)


  train_input_pipeline = _PeekableInput(instantiate(train_input_p))

  # Get shape and dtype of model_inputs.
  sample_inputs = train_input_pipeline.peek_padded()
  vars_weight_params = jax_task.model.abstract_init_with_metadata(
      init_key, sample_inputs)
  # JaxContext needed for shared layer lookup from global scope.
  with base_layer.JaxContext.new_context():
    # Dump out model meta info for debugging.
    trainer_lib.write_post_init_model_hparams_file(jax_task.model,
                                                   vars_weight_params,
                                                   job_log_dir)

  train_state_global_shapes = jax_task.create_train_state_unpadded_shapes(
      vars_weight_params)
  train_state_pspecs = None  # For consistency with the spmd path.

  # TODO(zhangqiaorjc): Memory optimization by restoring to local_devices()[0]
  # and rely on replicate_model_state to replicate to all local_devices on each
  # process.
  model_states = checkpointer.restore(train_state_global_shapes, None,
                                      train_state_pspecs)
  # Randomly initialized variables if no files in checkpoint dir.
  if model_states is None:
    model_states = trainer_lib.initialize_model_state(jax_task, init_key,
                                                      sample_inputs)

  logging.info('model_states=%s', jax.tree_map(lambda x: x.shape, model_states))

  total_num_params = py_utils.total_num_vars(model_states.mdl_vars)
  replicated_model_states = trainer_lib.replicate_model_state(model_states)

  train_p = task_p.train
  initial_global_step = int(jax.device_get(replicated_model_states.step)[0])
  logging.info('Model initial global_step=%d', initial_global_step)
  _update_latest_model_step(train_input_p, initial_global_step,
                            train_p.eval_interval_steps)

  # Unreplicated model states are not needed anymore at that point.
  del model_states

  logging.info('replicated_model_states shapes: %s',
               jax.tree_map(lambda x: x.shape, replicated_model_states))
  # From now on, different replicas should use different random seeds.
  # Here, each process will have its unique prng_key.
  # prng_key will be further split so that each core on a host will get
  # different prng_key.
  prng_key = jax.random.fold_in(prng_key, jax.process_index())
  logging.info('root prng_key: %s', prng_key)

  fprop_dtype = task_p.model.fprop_dtype

  def train_step(states, prng_key, inputs):
    """Train model for a single step."""
    return trainer_lib.train_step_single_learner(
        jax_task,
        states,
        prng_key,
        inputs,
        fprop_dtype=fprop_dtype)

  def eval_step(states, prng_key, inputs):
    eval_states = trainer_lib.train_state_for_eval_step(states)
    return trainer_lib.eval_step_single_learner(
        jax_task,
        eval_states,
        prng_key,
        inputs,
        fprop_dtype=fprop_dtype)

  num_devices = jax.local_device_count()
  prng_key, train_key, eval_key = jax.random.split(prng_key, 3)
  train_prng_seed = jax.random.split(train_key, num=num_devices)
  eval_prng_seed = jax.random.split(eval_key, num=num_devices)
  logging.info('train prng_seed: %s', train_prng_seed)
  logging.info('eval prng_seed: %s', eval_prng_seed)

  if decode_input_p:
    prng_key, decode_key = jax.random.split(prng_key, 2)
    decode_prng_seed = jax.random.split(decode_key, num=num_devices)
    logging.info('decode prng_seed: %s', decode_prng_seed)

  p_train_step = jax.pmap(
      train_step,
      donate_argnums=(0,),
      axis_name=PMAP_PARALLEL_AXIS_NAME)
  p_eval_step = jax.pmap(
      eval_step,
      axis_name=PMAP_PARALLEL_AXIS_NAME)

  logging.info('Training loop starting...')
  if eval_input_p:
    # We either run p.eval_loop_num_batches steps or one epoch (when supported
    # by a resettable input) per eval loop during training. When
    # p.reset_for_eval is set to True, we run the eval loop until
    # tf.errors.OutOfRangeError (or StopIteration) is raised, which can be
    # triggered either because input pipeline has reached the end of the input
    # sequence, or a pre-determined num_batches has reached.
    eval_num_steps = [
        -1 if p.reset_for_eval else p.eval_loop_num_batches
        for p in eval_input_p
    ]

  with _SummaryContextManager(
      job_log_dir, eval_input_p, decode_input_p,
      train_p.eval_skip_train) as (train_summary_writer, eval_summary_writer,
                                   eval_test_summary_writers,
                                   decode_summary_writers):
    summary_utils.write_model_structure(
        train_summary_writer, replicated_model_states, is_vars_replicated=True)
    summary_utils.write_total_num_params(train_summary_writer, total_num_params)
    train_global_batch_size = (
        train_input_p.cls.get_batch_size(train_input_p) *
        train_input_p.num_infeed_hosts)
    summary_utils.write_global_batch_size(train_summary_writer,
                                          train_global_batch_size)

    train_summary_handler = summary_utils.SummaryHandler(
        train_summary_writer,
        train_p.summary_interval_steps,
        accumulate_interval_steps=train_p.summary_accumulate_interval_steps)
    eval_summary_handler = summary_utils.SummaryHandler(
        eval_summary_writer,
        train_p.summary_interval_steps,
        accumulate_interval_steps=train_p.summary_accumulate_interval_steps)

    summary_last_time = time.time()
    summary_last_step = None

    step_i = int(
        py_utils.maybe_unreplicate_for_fully_replicated(
            replicated_model_states.step))
    while True:
      logging.debug('step=`%d`: Beginning', step_i)

      if summary_last_step is None:
        summary_last_step = step_i - 1

      checkpointer.save_if_needed(step_i, replicated_model_states,
                                  train_state_pspecs)

      if step_i >= train_p.num_train_steps:
        logging.info(
            'Training loop completed (step (`%d`) greater than '
            'num_train_step (`%d`).', step_i, train_p.num_train_steps)
        break

      if step_i <= _N_STEPS_WARMUP_LOGGING:
        logging.info('step=`%d`: Retrieving model inputs.', step_i)
      logging.debug('  Retrieving inputs.')
      model_inputs = tf.nest.map_structure(
          py_utils.reshard, train_input_pipeline.get_next_padded())
      logging.debug('  Retrieved inputs.')
      logging.debug('  Performing train_step().')
      with jax.profiler.StepTraceAnnotation('train', step_num=step_i):
        with py_utils.timeit() as train_period:
          (replicated_model_states, loss, weighted_scalars, per_example_out,
           summary_tensors) = p_train_step(replicated_model_states,
                                           train_prng_seed, model_inputs)
      logging.debug('  Completed train_step() in %f seconds.',
                    train_period.elapsed)

      logging.debug('  Writing summaries (attempt).')
      step_i += 1
      should_accumulate = train_summary_handler.should_accumulate(step_i)
      should_write_summary = train_summary_handler.should_write(step_i)
      should_log = _should_log_train(step_i, train_p)
      if should_log or should_accumulate or should_write_summary:
        loss = py_utils.maybe_unreplicate_for_fully_replicated(loss)
        weighted_scalars = py_utils.maybe_unreplicate_for_fully_replicated(
            weighted_scalars)
        summary_tensors = py_utils.maybe_unreplicate_for_fully_replicated(
            summary_tensors)
      if should_log:
        logging.info('step_i: %d, training loss: %s', step_i, loss)
        logging.info('weighted_scalars: %s', weighted_scalars)
        per_example_out = py_utils.maybe_unreplicate_for_first_shard(
            per_example_out)
        logging.info('per_example_out: %s', per_example_out)
        logging.info('summary_tensors: %s', summary_tensors)

      # Train metrics.
      train_weighted_scalars = weighted_scalars
      steps_per_sec = _compute_steps_per_sec(step_i, summary_last_time,
                                             summary_last_step)
      if should_write_summary:
        logging.info('steps/sec: %f', steps_per_sec)
        summary_last_time = time.time()
        summary_last_step = step_i
      if train_summary_handler.process(
          step_i,
          loss,
          weighted_scalars,
          summary_tensors,
          steps_per_sec=steps_per_sec):
        # Synchronize step_i
        step_i = int(
            py_utils.maybe_unreplicate_for_fully_replicated(
                replicated_model_states.step))
      logging.debug('  Wrote summaries (attempted).')

      eval_metrics_list = None
      eval_scoring_metrics_list = None
      eval_steps_per_sec = None
      eval_train_metrics = None
      # Run eval at regular step interval.
      if (train_p.eval_interval_steps and
          step_i % train_p.eval_interval_steps == 0):

        def eval_step_fn(inputs):
          logging.info('step=%d', step_i)
          # TODO(pax): shall we eval all sub-models during eval?
          return p_eval_step(replicated_model_states, eval_prng_seed, inputs)

        logging.debug('  Starting eval_step().')
        if eval_input_p:
          # Eval on the test sets.
          logging.debug('  Performing eval_step() runs on test splits.')
          with py_utils.timeit() as eval_period:
            eval_metrics_list, eval_scoring_metrics_list, num_eval_steps = (
                eval_lib.run_eval_loop_over_test_splits(
                    eval_num_steps,
                    eval_step_fn,
                    eval_test_summary_writers,
                    step_i,
                    eval_input_pipelines,
                    reshard_inputs=True))
          eval_steps_per_sec = eval_period.elapsed / sum(num_eval_steps)
          logging.debug(
              '  Completed eval_step() runs on test splits in %f seconds.',
              eval_period.elapsed)
        if train_p.eval_skip_train:
          logging.debug('  train_p.eval_skip_train is True. '
                        'Skipping eval_train.')
        else:
          logging.debug('  Retrieving eval model_inputs.')
          eval_inputs = train_input_pipeline.peek_padded()
          if eval_inputs is None:
            logging.debug('  eval model_inputs is None. Skipping eval_train.')
          else:
            logging.debug('  Retrieved eval model_inputs.')
            logging.debug('  Performing eval_step() runs on training split.')

            (loss, weighted_scalars, _,
             summary_tensors) = eval_lib.run_eval_one_step(
                 eval_inputs, eval_step_fn, reshard_inputs=True)
            logging.debug('  Completed eval_step() runs on training split.')
            loss = py_utils.maybe_unreplicate_for_fully_replicated(loss)
            weighted_scalars = py_utils.maybe_unreplicate_for_fully_replicated(
                weighted_scalars)
            summary_tensors = py_utils.maybe_unreplicate_for_fully_replicated(
                summary_tensors)
            logging.info('step=`%d`', step_i)
            logging.info('  eval loss: %s', loss)
            logging.info('  weighted_scalars: %s', weighted_scalars)
            logging.info('  summary_tensors: %s', summary_tensors)
            if eval_summary_handler.process(step_i, loss, weighted_scalars,
                                            summary_tensors):
              logging.debug('  Wrote eval summaries.')
            eval_train_metrics = metric_utils.as_float_dict(weighted_scalars)

      decode_metrics_list = None
      processed_decode_metrics_list = None
      decode_seqio_metrics_list = None
      decode_steps_per_sec = None
      if (decode_input_p and train_p.decode_interval_steps and
          step_i % train_p.decode_interval_steps == 0):
        with py_utils.timeit() as decode_period:
          (decode_metrics_list, processed_decode_metrics_list,
           decode_seqio_metrics_list, num_decode_steps) = (
               eval_lib.decode_once_pmap_model(jax_task, task_p,
                                               decode_input_pipelines,
                                               decode_input_p, decode_prng_seed,
                                               job_log_dir,
                                               replicated_model_states,
                                               decode_summary_writers))
        decode_steps_per_sec = sum(num_decode_steps) / decode_period.elapsed

      logging.debug('step=`%d`: End', step_i - 1)

      if early_stopping_fn is not None:
        if tuning_lib.should_early_stop(
            early_stopping_fn, step_i,
            is_last_ckpt=tuning_lib.is_last_checkpoint(
                trainer_lib.RunningMode.detect(
                    has_train_metrics=True,
                    has_eval_metrics=bool(eval_metrics_list),
                    has_decode_metrics=bool(decode_metrics_list)),
                step_i,
                task_p.train.num_train_steps,
                task_p.train.eval_interval_steps,
                task_p.train.decode_interval_steps,
                task_p.train.save_interval_steps),
            train_weighted_scalars=train_weighted_scalars,
            eval_train_metrics=eval_train_metrics,
            eval_input_p=eval_input_p,
            eval_metrics_list=eval_metrics_list,
            eval_scoring_metrics_list=eval_scoring_metrics_list,
            decode_input_p=decode_input_p,
            decode_metrics_list=decode_metrics_list,
            processed_decode_metrics_list=processed_decode_metrics_list,
            decode_seqio_metrics_list=decode_seqio_metrics_list,
            train_steps_per_sec=steps_per_sec,
            eval_steps_per_sec=eval_steps_per_sec,
            decode_steps_per_sec=decode_steps_per_sec,
            num_params=total_num_params):
          logging.info(
              'Training loop is early stopped at step `%d` by the '
              'tuner, while num_train_step is `%d`.', step_i,
              train_p.num_train_steps)
          break

    # Save checkpoint for the last step.
    checkpointer.save_final(step_i, replicated_model_states, train_state_pspecs)


def compile_for_auto_sharding(train_step: Any,
                              train_state: train_states.TrainState,
                              train_key: jnp.ndarray,
                              inputs_shape_dtype: NestedShapeDtypeLike):
  """Compiles train_step ahead of time to extract the shardings.

  The sharding is returned by the auto spmd partitioner and is attached on the
  compiled object.

  Args:
    train_step: The train_step function which will be compiled ahead of time.
    train_state: Train state which contains abstract values for ahead of time
      compilation.
    train_key: Prng key used for training.
    inputs_shape_dtype: Inputs with shape/dtype attributes to be used for shape
      inference.

  Returns:
    * A compiled train_step function
    * The input shardings returned by the auto spmd partitioner.
  """

  def _create_aval(x):
    return jax.ShapedArray(x.shape, x.dtype)

  train_key = jax.tree_map(_create_aval, train_key)
  inputs_shape_dtype = jax.tree_map(_create_aval, inputs_shape_dtype)
  compiled = train_step.lower(
      train_state, train_key, inputs_shape_dtype, _global_avals=True).compile()
  return compiled, compiled.input_shardings


def train_and_evaluate_spmd_model(
    task_p: tasks_lib.SingleTask.HParams,
    train_input_p: base_input.BaseInput.HParams,
    job_log_dir: Optional[str],
    checkpointer: _TrainingCheckpointer,
    checkpoint_type: CheckpointType,
    eval_input_p: Optional[Sequence[base_input.BaseInput.HParams]],
    decode_input_p: Sequence[base_input.BaseInput.HParams],
    early_stopping_fn: Optional[trainer_lib.EarlyStoppingFn] = None,
    enable_auto_sharding: bool = False) -> None:
  """Runs the training and evaluation loop with PJIT.

  Args:
    task_p: Params for task encapsulating the SPMD model.
    train_input_p: Params for the train data pipeline.
    job_log_dir: Directory for the job logs.
    checkpointer: Callbacks for checkpointing.
    checkpoint_type: The type of checkpoint to use.
    eval_input_p: Optional list of params for the eval input pipelines.
    decode_input_p: Optional list of hparams for the decode input pipelines.
    early_stopping_fn: An optional callable object for reporting eval metrics
      and determining whether to early stop current training. The callable
      object has signature: (metrics_by_dataset, ckpt_step, is_final_ckpt) ->
      should_stop_early.
    enable_auto_sharding: Enables the XLA Auto SPMD partitioner.
  """
  logging.info('Using SPMD sharding for model parallelism.')
  model_p = task_p.model
  local_device_count = jax.local_device_count()

  # TODO(bf-jax): Retrieve the seeds from the model definition instead.
  prng_key = jax.random.PRNGKey(1234)
  prng_key, init_key = jax.random.split(prng_key)

  assert jax.config.jax_parallel_functions_output_gda, 'GDA must be enabled'

  device_mesh = py_utils.create_device_mesh(model_p.ici_mesh_shape,
                                            model_p.dcn_mesh_shape)
  logging.info('device_mesh: %s', device_mesh)

  global_mesh = maps.Mesh(device_mesh, model_p.mesh_axis_names)

  logging.info('Retrieving model inputs for shape info.')
  create_gda_for_inputs = (
      jax.config.jax_parallel_functions_output_gda and
      checkpoint_type != CheckpointType.CHECKPOINT_PERSISTENCE)
  train_unpadded_global_batch_size = (
      train_input_p.cls.get_batch_size(train_input_p) *
      train_input_p.num_infeed_hosts)
  train_input_p = trainer_lib.adjust_input_params_for_small_batch(
      train_input_p, global_mesh)
  train_input_for_shape = instantiate(train_input_p)
  sample_inputs = train_input_for_shape.get_next_padded()
  inputs_shape_dtype = tf.nest.map_structure(
      py_utils.get_global_input_shape_dtype, sample_inputs)

  if eval_input_p:
    eval_input_p = [
        trainer_lib.adjust_input_params_for_small_batch(input_p, global_mesh)
        for input_p in eval_input_p
    ]
    eval_input_pipelines = [instantiate(input_p) for input_p in eval_input_p]
    trainer_lib.check_unique_names(eval_input_pipelines)
    # Do not mutate eval_input_pipelines itself. Instantiate a new one
    # to get sample input.
    sample_eval_model_inputs = instantiate(eval_input_p[0]).get_next_padded()
    eval_test_inputs_shape_dtype = tf.nest.map_structure(
        py_utils.get_global_input_shape_dtype, sample_eval_model_inputs)
    eval_test_inputs_pspecs = trainer_lib.get_input_partition_specs(
        model_p.mesh_axis_names, eval_test_inputs_shape_dtype)
  if decode_input_p:
    decode_input_p = [
        trainer_lib.adjust_input_params_for_small_batch(input_p, global_mesh)
        for input_p in decode_input_p
    ]
    decode_input_pipelines = [
        instantiate(input_p) for input_p in decode_input_p
    ]
    trainer_lib.check_unique_names(decode_input_pipelines)
    decode_sample_inputs = instantiate(decode_input_p[0]).get_next_padded()
    decode_inputs_shape_dtype = tf.nest.map_structure(
        py_utils.get_global_input_shape_dtype, decode_sample_inputs)

  with global_mesh:
    jax_task = instantiate(task_p)
    vars_weight_params = jax_task.model.abstract_init_with_metadata(
        init_key, inputs_shape_dtype)
    # Dump out model meta info for debugging.
    trainer_lib.write_post_init_model_hparams_file(jax_task.model,
                                                   vars_weight_params,
                                                   job_log_dir)
    train_state_global_shapes = jax_task.create_train_state_padded_shapes(
        vars_weight_params)
    train_state_pspecs = jax_task.create_train_state_partition_specs(
        vars_weight_params)

    # We do not fold in jax.process_index in contrast to the pmap version and
    # use a single global key instead to rely on pjit to split for different
    # replicas.
    logging.info('root prng_key: %s', prng_key)
    prng_key, train_key, eval_key = jax.random.split(prng_key, 3)
    logging.info('train prng_key: %s', train_key)
    logging.info('eval prng_key: %s', eval_key)
    if decode_input_p:
      prng_key, decode_key = jax.random.split(prng_key, 2)
      logging.info('decode prng_key: %s', decode_key)

    if enable_auto_sharding:
      if train_input_p.num_infeed_hosts < jax.process_count() or (
          train_input_p.cls.get_batch_size(train_input_p) < local_device_count):
        raise NotImplementedError(
            'Per-device batch size < 1 not supported for auto sharding.')
      logging.info('Auto sharding is enabled in PAX.')
      # If auto sharding is enabled, create abstract train state and ahead of
      # time compile the `train_step`. Then we can extract the input shardings
      # returned by XLA's auto spmd partitioner from the compiled object.
      train_step, _ = trainer_lib.get_partitioned_spmd_model_step_fn_auto_shard(
          jax_task,
          init_key=None,
          model_state_partition_specs=None,
          inputs_shape_dtype=inputs_shape_dtype,
          is_eval=False)
      # NOTE(pax-dev): The following is currently incompatible with variable
      # uneven-sharding padding. When enable_auto_sharding is False,
      # train_state_pspecs correspond to padded train_states.
      abstract_train_state = jax_task.create_train_state_unpadded_shapes(
          vars_weight_params,
          # TODO(pax-dev): set discard_opt_states according to is_eval.
          discard_opt_states=False)
      train_step, input_shardings = compile_for_auto_sharding(
          train_step, abstract_train_state, train_key, inputs_shape_dtype)
      train_state_pspecs = jax.tree_map(lambda x: x.spec, input_shardings[0])
      inputs_pspecs = jax.tree_map(lambda x: x.spec, input_shardings[2])
    else:
      train_step, inputs_pspecs = (
          trainer_lib.get_partitioned_spmd_model_step_fn(
              jax_task,
              init_key,
              train_state_pspecs,
              inputs_shape_dtype,
              is_eval=False,
              unpadded_global_batch_size=train_unpadded_global_batch_size))

    # Try to restore from checkpoint.
    partitioned_train_state = checkpointer.restore(train_state_global_shapes,
                                                   global_mesh,
                                                   train_state_pspecs)

    # Randomly initialized variables if no files in checkpoint dir.
    if partitioned_train_state is None:
      if create_gda_for_inputs:
        # pjit(model.init) requires a GDA input.
        sample_inputs = py_utils.create_gda(sample_inputs, inputs_shape_dtype,
                                            global_mesh, inputs_pspecs)
      _, partitioned_train_state = (
          trainer_lib.initialize_partitioned_model_states(
              jax_task,
              init_key,
              sample_inputs,
              global_mesh=global_mesh,
              # Note: We currently enforce that the checkpoint to reload via
              # init_checkpoint_rules are in the same format as the checkpoint
              # solution used by the experiment.
              checkpoint_type=checkpoint_type,
              state_specs=train_state_pspecs))

    total_num_params = py_utils.total_num_vars(partitioned_train_state.mdl_vars)
    # TODO(pax): Support auto-sharding for eval step. In this case, we would
    # have to fix the sharding of the input to be the same as what's derived
    # from the train_step.
    eval_step, _ = trainer_lib.get_partitioned_spmd_model_step_fn(
        jax_task,
        init_key,
        trainer_lib.train_state_for_eval_step(train_state_pspecs),
        inputs_shape_dtype,
        is_eval=True,
        unpadded_global_batch_size=train_unpadded_global_batch_size)
    if decode_input_p:
      # TODO(pax-dev): Support auto-sharding for decoder step.
      decode_step_fn, decode_inputs_partition_spec = (
          trainer_lib.get_partitioned_spmd_model_decode_fn(
              jax_task, init_key,
              trainer_lib.train_state_for_eval_step(train_state_pspecs),
              decode_inputs_shape_dtype))

    logging.info(
        'partitioned_train_state shapes '
        '(global shape for GDA, host-local shape for non-GDA: %s',
        jax.tree_map(lambda x: x.shape, partitioned_train_state))
    checkpointer.maybe_sync_multihostcheckpointing()

    train_p = task_p.train
    initial_global_step = int(
        py_utils.maybe_unreplicate_for_fully_replicated(
            partitioned_train_state.step))
    logging.info('Model initial global_step=%d', initial_global_step)
    _update_latest_model_step(train_input_p, initial_global_step,
                              train_p.eval_interval_steps)
    train_input_pipeline = _PeekableInput(instantiate(train_input_p))

    logging.info('Training loop starting...')
    if eval_input_p:
      # We either run p.eval_loop_num_batches steps or one epoch (when supported
      # by a resettable input) per eval loop during training. When
      # p.reset_for_eval is set to True, we run the eval loop until
      # tf.errors.OutOfRangeError (or StopIteration) is raised, which can be
      # triggered either because input pipeline has reached the end of the input
      # sequence, or a pre-determined num_batches has reached.
      eval_num_steps = [
          -1 if p.reset_for_eval else p.eval_loop_num_batches
          for p in eval_input_p
      ]

    with _SummaryContextManager(
        job_log_dir, eval_input_p, decode_input_p,
        train_p.eval_skip_train) as (train_summary_writer, eval_summary_writer,
                                     eval_test_summary_writers,
                                     decode_summary_writers):
      # This only prints the view from the first host machine.
      summary_utils.write_model_structure(
          train_summary_writer,
          partitioned_train_state,
          is_vars_replicated=False)
      summary_utils.write_total_num_params(train_summary_writer,
                                           total_num_params)
      summary_utils.write_global_batch_size(train_summary_writer,
                                            train_unpadded_global_batch_size)

      train_summary_handler = summary_utils.SummaryHandler(
          train_summary_writer, train_p.summary_interval_steps,
          accumulate_interval_steps=train_p.summary_accumulate_interval_steps)
      eval_summary_handler = summary_utils.SummaryHandler(
          eval_summary_writer, train_p.summary_interval_steps,
          accumulate_interval_steps=train_p.summary_accumulate_interval_steps)

      summary_last_time = time.time()
      summary_last_step = None

      step_i = int(
          py_utils.maybe_unreplicate_for_fully_replicated(
              partitioned_train_state.step))
      step_counter = 0

      # Start the train loop. Make sure all at the same step.
      py_utils.sync_global_devices(f'Start training loop from step: {step_i}')
      while True:
        logging.debug('step=`%d`: Beginning', step_i)

        if summary_last_step is None:
          summary_last_step = step_i - 1
        checkpointer.save_if_needed(step_i, partitioned_train_state,
                                    train_state_pspecs)

        if step_i >= train_p.num_train_steps:
          logging.info(
              'Training loop completed (step (`%d`) greater than '
              'num_train_step (`%d`).', step_i, train_p.num_train_steps)
          break

        # Get new model inputs
        if step_counter <= _N_STEPS_WARMUP_LOGGING:
          logging.info('step=`%d`: Retrieving model inputs.', step_i)
        logging.debug('  Retrieving inputs.')
        model_inputs = train_input_pipeline.get_next_padded()

        if create_gda_for_inputs:
          if step_counter <= _N_STEPS_WARMUP_LOGGING:
            start = time.time()
          py_utils.assert_same_shape_and_dtype(
              inputs_shape_dtype,
              tf.nest.map_structure(py_utils.get_global_input_shape_dtype,
                                    model_inputs))
          model_inputs = py_utils.create_gda(model_inputs, inputs_shape_dtype,
                                             global_mesh, inputs_pspecs)
          if step_counter <= _N_STEPS_WARMUP_LOGGING:
            logging.info('GDA train batch input creation time %s',
                         time.time() - start)

        logging.debug('  Retrieved inputs.')

        logging.debug('  Performing train_step().')
        with jax.profiler.StepTraceAnnotation('train', step_num=step_i):
          with py_utils.timeit() as train_period:
            (partitioned_train_state, loss, weighted_scalars, per_example_out,
             summary_tensors) = train_step(partitioned_train_state, train_key,
                                           model_inputs)
        logging.debug('  Completed train_step() in %f seconds.',
                      train_period.elapsed)
        logging.debug('  Writing summaries (attempt).')
        should_accumulate = train_summary_handler.should_accumulate(step_i)
        should_write_summary = train_summary_handler.should_write(step_i)
        should_log = _should_log_train(step_i, train_p)
        if should_log or should_accumulate or should_write_summary:
          loss = py_utils.maybe_unreplicate_for_fully_replicated(loss)
          weighted_scalars = py_utils.maybe_unreplicate_for_fully_replicated(
              weighted_scalars)
          summary_tensors = py_utils.maybe_unreplicate_for_fully_replicated(
              summary_tensors)
        if should_log:
          logging.info('step_i: %d, training loss: %s', step_i, loss)
          logging.info('weighted_scalars: %s', weighted_scalars)
          per_example_out = py_utils.maybe_unreplicate_for_first_shard(
              per_example_out)
          logging.info('per_example_out: %s', per_example_out)
          logging.info('summary_tensors: %s', summary_tensors)

        # Train metrics.
        train_weighted_scalars = weighted_scalars
        steps_per_sec = _compute_steps_per_sec(step_i, summary_last_time,
                                               summary_last_step)
        if should_write_summary:
          logging.info('steps/sec: %f', steps_per_sec)
          summary_last_time = time.time()
          summary_last_step = step_i
        if train_summary_handler.process(
            step_i,
            loss,
            weighted_scalars,
            summary_tensors,
            steps_per_sec=steps_per_sec):
          step_i = int(
              py_utils.maybe_unreplicate_for_fully_replicated(
                  partitioned_train_state.step))
        else:
          # Increment train step locally to avoid an explicit device sync.
          step_i += 1
        logging.debug('  Wrote summaries (attempted).')

        eval_metrics_list = None
        eval_scoring_metrics_list = None
        eval_steps_per_sec = None
        eval_train_metrics = None
        # Run eval at regular step interval.
        if (train_p.eval_interval_steps and
            step_i % train_p.eval_interval_steps == 0):
          eval_step_fn = functools.partial(
              eval_step,
              trainer_lib.train_state_for_eval_step(partitioned_train_state),
              eval_key)

          logging.debug('  Starting eval_step().')
          # If we have eval test then also evaluate on test.
          if eval_input_p:
            logging.debug('  Performing eval_step() runs on test splits.')
            with py_utils.timeit() as eval_period:
              eval_metrics_list, eval_scoring_metrics_list, num_eval_steps = (
                  eval_lib.run_eval_loop_over_test_splits(
                      eval_num_steps,
                      eval_step_fn,
                      eval_test_summary_writers,
                      step_i,
                      eval_input_pipelines,
                      eval_test_inputs_pspecs,
                      eval_test_inputs_shape_dtype,
                      global_mesh,
                      reshard_inputs=False,
                      create_gda_for_inputs=create_gda_for_inputs))
            eval_steps_per_sec = eval_period.elapsed / sum(num_eval_steps)
            logging.debug(
                '  Completed eval_step() runs on test splits in %f seconds.',
                eval_period.elapsed)
          if train_p.eval_skip_train:
            logging.debug('  train_p.eval_skip_train is True. Skipping '
                          'eval_train.')
          else:
            logging.debug('  Retrieving eval model_inputs.')
            eval_inputs = train_input_pipeline.peek_padded()
            if eval_inputs is None:
              logging.debug('  eval_inputs is None. Skipping eval_train.')
            else:
              if create_gda_for_inputs:
                eval_inputs = py_utils.create_gda(eval_inputs,
                                                  inputs_shape_dtype,
                                                  global_mesh, inputs_pspecs)
              logging.debug('  Retrieved eval model_inputs.')
              logging.debug('  Performing eval_step() runs on training split.')
              loss, weighted_scalars, _, summary_tensors = (
                  eval_lib.run_eval_one_step(
                      eval_inputs, eval_step_fn, reshard_inputs=False))
              logging.debug('  Completed eval_step() runs on training split.')
              loss = py_utils.maybe_unreplicate_for_fully_replicated(loss)
              weighted_scalars = py_utils.maybe_unreplicate_for_fully_replicated(
                  weighted_scalars)
              summary_tensors = py_utils.maybe_unreplicate_for_fully_replicated(
                  summary_tensors)

              logging.info('step=`%d`', step_i)
              logging.info('  eval loss: %s', loss)
              logging.info('  weighted_scalars: %s', weighted_scalars)
              logging.info('  summary_tensors: %s', summary_tensors)
              if eval_summary_handler.process(step_i, loss, weighted_scalars,
                                              summary_tensors):
                logging.debug('  Wrote eval summaries.')
              eval_train_metrics = metric_utils.as_float_dict(weighted_scalars)

        decode_metrics_list = None
        processed_decode_metrics_list = None
        decode_seqio_metrics_list = None
        decode_steps_per_sec = None
        if (decode_input_p and train_p.decode_interval_steps and
            step_i % train_p.decode_interval_steps == 0):
          with py_utils.timeit() as decode_period:
            (decode_metrics_list, processed_decode_metrics_list,
             decode_seqio_metrics_list, num_decode_steps) = (
                 eval_lib.decode_once_spmd_model(
                     jax_task, task_p, decode_input_pipelines, decode_input_p,
                     job_log_dir, partitioned_train_state,
                     decode_summary_writers, decode_key, global_mesh,
                     decode_step_fn, create_gda_for_inputs,
                     decode_inputs_shape_dtype, decode_inputs_partition_spec))
          decode_steps_per_sec = sum(num_decode_steps) / decode_period.elapsed
        logging.debug('step=`%d`: End', step_i - 1)
        if early_stopping_fn is not None:
          if tuning_lib.should_early_stop(
              early_stopping_fn, step_i,
              is_last_ckpt=tuning_lib.is_last_checkpoint(
                  trainer_lib.RunningMode.detect(
                      has_train_metrics=True,
                      has_eval_metrics=bool(eval_metrics_list),
                      has_decode_metrics=bool(decode_metrics_list)),
                  step_i,
                  task_p.train.num_train_steps,
                  task_p.train.eval_interval_steps,
                  task_p.train.decode_interval_steps,
                  task_p.train.save_interval_steps),
              train_weighted_scalars=train_weighted_scalars,
              eval_train_metrics=eval_train_metrics,
              eval_input_p=eval_input_p,
              eval_metrics_list=eval_metrics_list,
              eval_scoring_metrics_list=eval_scoring_metrics_list,
              decode_input_p=decode_input_p,
              decode_metrics_list=decode_metrics_list,
              processed_decode_metrics_list=processed_decode_metrics_list,
              decode_seqio_metrics_list=decode_seqio_metrics_list,
              train_steps_per_sec=steps_per_sec,
              eval_steps_per_sec=eval_steps_per_sec,
              decode_steps_per_sec=decode_steps_per_sec,
              num_params=total_num_params):
            logging.info(
                'Training loop is early stopped at step `%d` by the '
                'tuner, while num_train_step is `%d`.', step_i,
                train_p.num_train_steps)
            break
        step_counter += 1

    # Save checkpoint for the last step.
    checkpointer.save_final(step_i, partitioned_train_state, train_state_pspecs)
