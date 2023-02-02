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
import dataclasses
import datetime
import functools
import gc
import re
import time
import typing
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Type

from absl import logging
from etils import epath
import jax
from jax import monitoring
from jax.experimental import pjit
from paxml import base_experiment
from paxml import checkpoint_managers
from paxml import checkpoint_pb2
from paxml import eval_lib
from paxml import experiment_utils
from paxml import metric_utils
from paxml import summary_utils
from paxml import tasks_lib
from paxml import train_states
from paxml import trainer_lib
from paxml import tuning_lib
from praxis import base_hyperparams
from praxis import base_input
from praxis import base_layer
from praxis import py_utils
from praxis import pytypes
import tensorflow.compat.v2 as tf

from paxml import checkpoints  # mapped to internal
from paxml import profiling  # mapped to internal

Checkpointer = checkpoints.Checkpointer
CheckpointType = checkpoint_pb2.CheckpointType
instantiate = base_hyperparams.instantiate
NestedShapeDtypeLike = pytypes.NestedShapeDtypeLike
FlaxCheckpointer = checkpoints.FlaxCheckpointer
FlaxCheckpointHandler = checkpoints.FlaxCheckpointHandler
PaxCheckpointHandler = checkpoints.PaxCheckpointHandler
PRNGKey = pytypes.PRNGKey
RunningMode = trainer_lib.RunningMode
SummaryWriter = tf.summary.SummaryWriter
TrainState = train_states.TrainState

PARAMS = base_layer.PARAMS
NON_PAX_RNG_KEY = base_layer.NON_PAX_RNG_KEY
PMAP_PARALLEL_AXIS_NAME = base_layer.PMAP_PARALLEL_AXIS_NAME

_INIT_TIME = time.time()
_READ_CHECKPOINT_EVENT: str = '/jax/checkpoint/read/durations_sec'
_WRITE_CHECKPOINT_EVENT: str = '/jax/checkpoint/write/durations_sec'


def _checkpoint_dir(job_log_dir: epath.Path) -> epath.Path:
  """Returns the checkpoint directory from the root `job_log_dir`."""
  return job_log_dir / 'checkpoints'


def _make_checkpoint_dir(job_log_dir: epath.Path) -> epath.Path:
  checkpoint_dir = _checkpoint_dir(job_log_dir)
  if jax.process_index() == 0 and not checkpoint_dir.exists():
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
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
  def get_model_states(
      self,
      jax_task: tasks_lib.SingleTask,
      global_mesh: Optional[jax.sharding.Mesh],
      metadata: trainer_lib.TrainStateMetadata,
      init_key: PRNGKey,
  ) -> Tuple[TrainState, int]:
    """Restores TrainState from checkpoint or initializes it.

    Args:
      jax_task: A SingleTask instance.
      global_mesh: Use this mesh to restore the checkpoint.
      metadata: A TrainStateMetadata instance.
      init_key: PRNGKey for initializing the model variables.

    Returns:
      (train_state, total_num_params).
    """
    raise NotImplementedError

  @property
  @abc.abstractmethod
  def checkpoint_type(self) -> CheckpointType:
    raise NotImplementedError


class _OrbaxPjitTrainingCheckpointer(_TrainingCheckpointer):

  def __init__(self,
               checkpoint_manager: checkpoint_managers.OrbaxCheckpointManager,
               checkpoint_type: CheckpointType,
               enable_checkpoint_saving: bool = True):
    self.checkpoint_manager = checkpoint_manager
    self._checkpoint_type = checkpoint_type
    if checkpoint_type == CheckpointType.CHECKPOINT_FLAX:
      raise ValueError('FLAX checkpointing not supported for pjit models.')
    self._enable_checkpoint_saving = enable_checkpoint_saving

  def _save_with_args(self, step_i, partitioned_train_state):
    if not self._enable_checkpoint_saving:
      return
    with py_utils.timeit() as save_period:
      self.checkpoint_manager.save(step_i, partitioned_train_state)
    monitoring.record_event_duration_secs(_WRITE_CHECKPOINT_EVENT,
                                          save_period.elapsed)

  def _restore_with_args(self, step_i, train_state_global_shapes, global_mesh,
                         train_state_pspecs):
    restore_args = {}
    if self._checkpoint_type == CheckpointType.CHECKPOINT_GDA:
      restore_args = {'specs': train_state_pspecs, 'mesh': global_mesh}
    elif self._checkpoint_type == CheckpointType.CHECKPOINT_PERSISTENCE:
      restore_args = {
          'state_specs': train_state_pspecs,
          'global_mesh': global_mesh
      }
    return self.checkpoint_manager.restore(
        step_i, train_state_global_shapes, restore_kwargs=restore_args
    )

  def save_final(self, step_i, partitioned_train_state, train_state_pspecs):
    logging.info('Saving a ckpt at final step: %d', step_i)
    self._save_with_args(step_i, partitioned_train_state)

  def save_if_needed(self, step_i, partitioned_train_state, train_state_pspecs):
    if not self.checkpoint_manager.should_save(step_i):
      return
    self._save_with_args(step_i, partitioned_train_state)
    self.checkpoint_manager.check_for_errors()

  # TODO(laigd): merge this with _SpmdEvalCheckpointer.get_model_states().
  def get_model_states(
      self,
      jax_task: tasks_lib.SingleTask,
      global_mesh: Optional[jax.sharding.Mesh],
      metadata: trainer_lib.TrainStateMetadata,
      init_key: PRNGKey,
  ) -> Tuple[TrainState, int]:
    step = self.checkpoint_manager.latest_step()
    if step is None:
      partitioned_train_state = None
    else:
      with py_utils.timeit() as restore_period:
        partitioned_train_state = self._restore_with_args(
            step,
            metadata.padded_global_shapes,
            global_mesh,
            metadata.partition_specs,
        )
      monitoring.record_event_duration_secs(_READ_CHECKPOINT_EVENT,
                                            restore_period.elapsed)

    # Randomly initialized variables if no files in checkpoint dir.
    if partitioned_train_state is None:
      _, partitioned_train_state = (
          trainer_lib.initialize_partitioned_model_states(
              jax_task,
              init_key,
              metadata.input_shape_dtype,
              global_mesh=global_mesh,
              # Note: We currently enforce that the checkpoint to reload via
              # init_checkpoint_rules are in the same format as the checkpoint
              # solution used by the experiment.
              checkpoint_type=self.checkpoint_type,
              state_specs=metadata.partition_specs,
          )
      )

    logging.info(
        'partitioned_train_state shapes '
        '(global shape for GDA, host-local shape for non-GDA: %s',
        jax.tree_map(lambda x: x.shape, partitioned_train_state))

    total_num_params = py_utils.total_num_vars(partitioned_train_state.mdl_vars)
    return partitioned_train_state, total_num_params

  @property
  def checkpoint_type(self) -> CheckpointType:
    return self._checkpoint_type


class _OrbaxPmapTrainingCheckpointer(_TrainingCheckpointer):

  def __init__(self,
               job_log_dir: epath.Path,
               checkpoint_manager: checkpoint_managers.OrbaxCheckpointManager,
               checkpoint_type: CheckpointType,
               enable_checkpoint_saving: bool = True):
    self.job_log_dir = job_log_dir
    self.checkpoint_dir = _checkpoint_dir(job_log_dir)
    self.checkpoint_manager = checkpoint_manager
    self._checkpoint_type = checkpoint_type
    self._enable_checkpoint_saving = enable_checkpoint_saving

  def _restore_from_tensorstore(self, train_state_global_shapes):
    _make_checkpoint_dir(self.job_log_dir)
    logging.info('Pmap restore from TensorStore checkpoint...')
    # Restored from GDA checkpoint dir.
    return tasks_lib.restore_pmap_from_tensorstore(
        train_state_global_shapes,
        self.checkpoint_dir,
        checkpoint_type=self._checkpoint_type)

  # TODO(laigd): merge this with _PmapEvalCheckpointer.get_model_states().
  def get_model_states(
      self,
      jax_task: tasks_lib.SingleTask,
      global_mesh: Optional[jax.sharding.Mesh],
      metadata: trainer_lib.TrainStateMetadata,
      init_key: PRNGKey,
  ) -> Tuple[TrainState, int]:
    train_state_global_shapes = metadata.unpadded_global_shapes
    with py_utils.timeit() as restore_period:
      if py_utils.pmap_use_tensorstore():
        train_state = self._restore_from_tensorstore(train_state_global_shapes)
      else:
        step = self.checkpoint_manager.latest_step()
        if step is None:
          train_state = None
        else:
          train_state = self.checkpoint_manager.restore(
              step, train_state_global_shapes
          )
    monitoring.record_event_duration_secs(_READ_CHECKPOINT_EVENT,
                                          restore_period.elapsed)
    # Randomly initialized variables if no files in checkpoint dir.
    if train_state is None:
      train_state = trainer_lib.initialize_model_state(
          jax_task,
          init_key,
          metadata.input_shape_dtype,
          checkpoint_type=self.checkpoint_type,
      )
    logging.info('train_state=%s', jax.tree_map(lambda x: x.shape, train_state))

    partitioned_train_state = trainer_lib.replicate_model_state(train_state)
    logging.info(
        'partitioned_train_state shapes: %s',
        jax.tree_map(lambda x: x.shape, partitioned_train_state),
    )
    # Unreplicated model states are not needed anymore at that point.
    del train_state

    total_num_params = py_utils.total_num_vars(partitioned_train_state.mdl_vars)
    assert total_num_params % jax.local_device_count() == 0
    total_num_params = total_num_params // jax.local_device_count()
    return partitioned_train_state, total_num_params

  def _save_with_args(self, step_i, train_state):
    self.checkpoint_manager.save(step_i, train_state)

  def _save(self, step_i, partitioned_train_state, is_final=False):
    if not self._enable_checkpoint_saving:
      return

    with py_utils.timeit() as save_period:
      if py_utils.pmap_use_tensorstore():
        logging.info('Saving a ckpt at %sstep: %d',
                     'final ' if is_final else '', step_i)
        if jax.config.jax_array:
          fully_replicated_gda_train_state = jax.tree_map(
              py_utils.convert_host_local_array_to_global_array,
              partitioned_train_state)
        else:
          fully_replicated_gda_train_state = jax.tree_map(
              py_utils.convert_fully_replicated_sda_to_gda,
              partitioned_train_state)
        self._save_with_args(step_i, fully_replicated_gda_train_state)
      else:
        unreplicated_train_state = jax.tree_map(lambda x: x[0],
                                                partitioned_train_state)
        self._save_with_args(step_i, unreplicated_train_state)
    monitoring.record_event_duration_secs(_WRITE_CHECKPOINT_EVENT,
                                          save_period.elapsed)

  def save_if_needed(self, step_i, partitioned_train_state, train_state_pspecs):
    if not self.checkpoint_manager.should_save(step_i):
      return
    self._save(step_i, partitioned_train_state)
    self.checkpoint_manager.check_for_errors()

  def save_final(self, step_i, partitioned_train_state, train_state_pspecs):
    latest_step = self.checkpoint_manager.latest_step()
    if latest_step is None or latest_step < step_i:
      self._save(step_i, partitioned_train_state, is_final=True)

  @property
  def checkpoint_type(self) -> CheckpointType:
    return self._checkpoint_type


def _create_checkpointer(
    task_p: tasks_lib.SingleTask.HParams,
    job_log_dir: epath.Path,
    checkpoint_type: CheckpointType,
    todelete_subdir: Optional[str],
    async_checkpointer: Optional[checkpoints.AsyncCheckpointer] = None,
    enable_checkpoint_saving: bool = True,
) -> _TrainingCheckpointer:
  """Creates a checkpoint manager."""
  checkpoint_dir = _make_checkpoint_dir(job_log_dir)
  train_p = task_p.train
  max_to_keep = train_p.save_max_to_keep
  save_interval_steps = train_p.save_interval_steps
  keep_interval_timedelta = _parse_duration(train_p.save_keep_interval_duration)

  options = checkpoint_managers.CheckpointManagerOptions(
      max_to_keep=max_to_keep,
      save_interval_steps=save_interval_steps,
      keep_time_interval=keep_interval_timedelta,
      todelete_subdir=todelete_subdir)
  checkpointer = async_checkpointer
  if checkpoint_type == CheckpointType.CHECKPOINT_FLAX:
    checkpointer = FlaxCheckpointer(FlaxCheckpointHandler())
  if checkpointer is None:
    if checkpoint_type == CheckpointType.CHECKPOINT_GDA:
      checkpointer = Checkpointer(
          PaxCheckpointHandler())
    elif checkpoint_type == CheckpointType.CHECKPOINT_PERSISTENCE:
      raise ValueError('Checkpointer must already be initialized.')
    else:
      raise ValueError(f'Unsupported Orbax checkpoint type: {checkpoint_type}')
  checkpoint_manager = checkpoint_managers.OrbaxCheckpointManager(
      checkpoint_dir,
      checkpointer,
      options=options,
      checkpoint_type=checkpoint_type,
  )

  if task_p.model.ici_mesh_shape is not None:
    checkpointer = _OrbaxPjitTrainingCheckpointer(
        checkpoint_manager,
        checkpoint_type,
        enable_checkpoint_saving=enable_checkpoint_saving)
  else:
    checkpointer = _OrbaxPmapTrainingCheckpointer(
        job_log_dir,
        checkpoint_manager,
        checkpoint_type,
        enable_checkpoint_saving=enable_checkpoint_saving)

  return checkpointer


def _compute_steps_per_sec(step_i, summary_last_time, summary_last_step):
  """Computes the number of training steps per second."""
  # Note: This function doesn't account for the time spent on running
  # interleaved evaluation (if any) and/or evaluation on the training batch.
  # It's, hence, merely a raw underestimate.
  duration_sec = time.time() - summary_last_time
  num_steps = step_i - summary_last_step
  steps_per_sec = num_steps / duration_sec
  return steps_per_sec


def _train_log_interval_steps(
    train_p: tasks_lib.SingleTask.TrainHParams) -> int:
  """Returns the interval to log train outputs."""
  if train_p.log_train_output_interval_steps is not None:
    return train_p.log_train_output_interval_steps
  else:
    return train_p.summary_interval_steps


def write_hparams_file(model_config: base_experiment.BaseExperiment,
                       job_log_dir: epath.Path,
                       filename_prefix: str = '') -> None:
  """Writes a params file into the root `job_log_dir`."""
  if jax.process_index() == 0:
    job_log_dir.mkdir(parents=True, exist_ok=True)
    params_fpath = job_log_dir / f'{filename_prefix}model_params.txt'
    with params_fpath.open('w') as hparams_file:
      for dataset in model_config.datasets():
        hparams_file.write(dataset.to_text())
        hparams_file.write('\n\n')
      for decoder_dataset in model_config.decoder_datasets():
        hparams_file.write('decoder dataset hparams\n')
        hparams_file.write(decoder_dataset.to_text())
        hparams_file.write('\n\n')
      hparams_file.write(model_config.task().to_text())


def write_experiment_class_vars_file(exp_cls: Type[
    base_experiment.BaseExperiment],
                                     job_log_dir: epath.Path,
                                     filename_prefix: str = '') -> None:
  """Writes a params file into the root `job_log_dir`."""
  if jax.process_index() == 0:
    exp_summary_fpath = job_log_dir / f'{filename_prefix}experiment_cls_vars.txt'
    job_log_dir.mkdir(parents=True, exist_ok=True)

    cls_vars_summary = experiment_utils.get_cls_vars_summary(exp_cls)
    exp_summary_fpath.write_text(cls_vars_summary)


def train_and_evaluate(
    experiment_config: base_experiment.BaseExperiment,
    job_log_dir: epath.PathLike,
    maybe_use_persistence_checkpointing: bool,
    eval_on_test: Optional[bool],
    checkpoint_todelete_subdir: Optional[str] = None,
    early_stopping_fn: Optional[trainer_lib.EarlyStoppingFn] = None,
    run_decode: bool = False,
    enable_auto_sharding: bool = False,
    async_checkpointer: Optional[checkpoints.AsyncCheckpointer] = None,
    enable_checkpoint_saving: bool = True) -> None:
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
    run_decode: whether to periodically run decode as part of the training loop.
      If and only if this is True, every `task_p.train.decode_interval_steps` of
      training, model runs decode.
    enable_auto_sharding: Enables the XLA Auto SPMD partitioner.
    async_checkpointer: When async checkpointing and Orbax are enabled, allows
      training to continue when checkpointing is going on as checkpointing
      happens in a different thread.
    enable_checkpoint_saving: Whether to perform checkpoint saving or not.
  """
  jax.monitoring.record_event('/jax/pax/train_and_evaluate/beacon')
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

  logging.info('train_input_p:')
  for line in train_input_p.to_text().splitlines():
    logging.info('  %s', line)
  logging.info('task_p:')
  for line in task_p.to_text().splitlines():
    logging.info('  %s', line)

  eval_input_p = []
  if (eval_on_test and task_p.train.eval_interval_steps is not None and
      task_p.train.eval_interval_steps > 0):
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

  job_log_dir = epath.Path(job_log_dir)
  checkpointer = _create_checkpointer(
      task_p,
      job_log_dir,
      checkpoint_type,
      checkpoint_todelete_subdir,
      async_checkpointer=async_checkpointer,
      enable_checkpoint_saving=enable_checkpoint_saving)
  if not enable_checkpoint_saving:
    logging.info(
        'Checkpointing is disabled and no checkpoint will be saved to disk.')

  if task_p.model.ici_mesh_shape is not None:
    train_and_evaluate_spmd_model(task_p, train_input_p, job_log_dir,
                                  checkpointer, checkpoint_type, eval_input_p,
                                  decode_input_p, early_stopping_fn,
                                  enable_auto_sharding)
  else:
    train_and_evaluate_pmap(task_p, train_input_p, job_log_dir, checkpointer,
                            eval_input_p, decode_input_p, early_stopping_fn)


class _PeekableInput:
  """Wraps a BaseInput to provide a peek() method. Single thread access only."""

  def __init__(self, inp: base_input.BaseInput) -> None:
    self._inp = inp
    self._peek = None

  @property
  def hparams(self):
    return self._inp.hparams

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

  def reshard_for_pmap(self, arrays):
    return self._inp.reshard_for_pmap(arrays)

  def reshard_for_spmd(self, arrays, global_mesh, pspecs):
    return self._inp.reshard_for_spmd(arrays, global_mesh, pspecs)


def _maybe_update_latest_model_step(
    train_input: _PeekableInput,
    train_input_p: base_input.BaseInput.HParams,
    initial_global_step: int,
) -> _PeekableInput:
  """Updates `train_input_p` in place its latest model step."""
  if not hasattr(train_input_p, 'deterministic_input_start_index'):
    return train_input
  dp = train_input_p.deterministic_input_start_index
  dp._latest_model_step = initial_global_step  # pylint: disable=protected-access
  logging.info('Reinstanting input because _latest_model_step is updated.')
  return _PeekableInput(instantiate(train_input_p))


class _SummaryContextManager(contextlib.ExitStack):
  """Manage summary writers."""

  _exit_callbacks = []

  def __init__(self,
               job_log_dir: epath.Path,
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
    self.summary_base_dir = job_log_dir / 'summaries'
    self.summary_train_dir = self.summary_base_dir / 'train'
    self.summary_eval_dir = self.summary_base_dir / 'eval_train'
    self.summary_writer = summary_utils.get_summary_writer
    if eval_input_p:
      self.summary_eval_test_dirs = [
          self.summary_base_dir / f'eval_test_{p.name}' for p in eval_input_p
      ]
    else:
      self.summary_eval_test_dirs = []
    if decode_input_p:
      self.summary_decode_dirs = [
          self.summary_base_dir / f'decode_test_{p.name}'
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
    job_log_dir: epath.Path,
    checkpointer: _TrainingCheckpointer,
    eval_input_p: Sequence[base_input.BaseInput.HParams],
    decode_input_p: Sequence[base_input.BaseInput.HParams],
    early_stopping_fn: Optional[trainer_lib.EarlyStoppingFn] = None) -> None:
  """Runs the training and evaluation loop with PMAP.

  Args:
    task_p: HParams for the task encapsulating the data parallel model.
    train_input_p: HParams for the train data input pipeline.
    job_log_dir: Directory for the job logs.
    checkpointer: Callbacks for checkpointing.
    eval_input_p: list of hparams for the eval input pipelines.
    decode_input_p: list of hparams for the decode input pipelines.
    early_stopping_fn: An optional callable object for reporting eval metrics
      and determining whether to early stop current training. The callable
      object has signature: (metrics_by_dataset, ckpt_step, is_final_ckpt) ->
      should_stop_early.
  """
  logging.info('Using pmap for data parallelism.')
  if jax.config.jax_parallel_functions_output_gda:
    logging.warning('--jax_use_gda is set to True but ignored for pmap.')
  if jax.config.jax_array:
    logging.warning('--jax_array is set to True')

  (
      jax_task,
      partitioner,
      train_state_metadata,
      partitioned_train_state,
      total_num_params,
      prng_key,
      init_key,
      train_input_pipeline,
      train_unpadded_global_batch_size,
  ) = _create_task_and_states(
      task_p, train_input_p, job_log_dir, checkpointer,
  )
  assert not partitioner.global_mesh
  inputs_shape_dtype = partitioner.train_inputs_shape_dtype

  # From now on, different replicas should use different random seeds.
  # Here, each process will have its unique prng_key.
  # prng_key will be further split so that each core on a host will get
  # different prng_key.
  prng_key = jax.random.fold_in(prng_key, jax.process_index())
  logging.info('root prng_key: %s', prng_key)
  num_devices = jax.local_device_count()
  prng_key, train_key, eval_key = jax.random.split(prng_key, 3)
  train_prng_seed = jax.random.split(train_key, num=num_devices)
  eval_prng_seed = jax.random.split(eval_key, num=num_devices)
  logging.info('train prng_seed: %s', train_prng_seed)
  logging.info('eval prng_seed: %s', eval_prng_seed)

  train_step_fn, is_eval = trainer_lib.get_step_fn(RunningMode.TRAIN)
  assert not is_eval
  p_train_step, _ = partitioner.partition(
      train_step_fn,
      inputs_shape_dtype,  # TODO(laigd): this is not needed by pmap.
      is_eval,
  )

  fprop_dtype = task_p.model.fprop_dtype

  def eval_step(eval_states, prng_key, inputs):
    return trainer_lib.eval_step_single_learner(
        jax_task, eval_states, prng_key, inputs, fprop_dtype=fprop_dtype)

  p_eval_step = jax.pmap(eval_step, axis_name=PMAP_PARALLEL_AXIS_NAME)
  p_eval_steps = [p_eval_step] * len(eval_input_p)

  global_mesh = None
  reshard_inputs = True
  create_gda_for_inputs = False
  is_vars_replicated = True

  def prepare_model_inputs(train_input_pipeline, model_inputs):
    return train_input_pipeline.reshard_for_pmap(model_inputs)

  def prepare_eval_inputs(train_input_pipeline, eval_inputs):
    return train_input_pipeline.reshard_for_pmap(eval_inputs)

  def partition_eval_input_fns(eval_input_p):
    eval_input_pipelines = [instantiate(input_p) for input_p in eval_input_p]
    trainer_lib.check_unique_names(eval_input_pipelines)
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
    eval_test_inputs_pspecs = None
    return (
        eval_input_pipelines,
        eval_num_steps,
        eval_test_inputs_pspecs,
        eval_input_p,
    )

  def partition_decode_once_fns(prng_key, decode_input_p):
    decode_input_pipelines = [
        instantiate(input_p) for input_p in decode_input_p
    ]
    trainer_lib.check_unique_names(decode_input_pipelines)
    prng_key, decode_key = jax.random.split(prng_key, 2)
    decode_prng_seed = jax.random.split(decode_key, num=num_devices)
    logging.info('decode prng_seed: %s', decode_prng_seed)
    decode_once_fn = eval_lib.partition_decode_once_pmap_model(
        jax_task, task_p, train_state_metadata.var_weight_hparams,
        decode_input_pipelines, decode_input_p, decode_prng_seed, job_log_dir)
    return decode_once_fn, prng_key, decode_input_p

  _train_and_evaluate_common(
      partitioned_train_state, prng_key, eval_input_p, decode_input_p, task_p,
      total_num_params, early_stopping_fn, checkpointer,
      partition_eval_input_fns, partition_decode_once_fns, job_log_dir,
      eval_prng_seed, reshard_inputs, task_p.train, prepare_eval_inputs,
      prepare_model_inputs, is_vars_replicated,
      train_unpadded_global_batch_size, train_state_metadata,
      train_input_pipeline, p_train_step, p_eval_steps, p_eval_step,
      global_mesh, create_gda_for_inputs, train_prng_seed)


def train_and_evaluate_spmd_model(
    task_p: tasks_lib.SingleTask.HParams,
    train_input_p: base_input.BaseInput.HParams,
    job_log_dir: epath.Path,
    checkpointer: _TrainingCheckpointer,
    checkpoint_type: CheckpointType,
    eval_input_p: Sequence[base_input.BaseInput.HParams],
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
    eval_input_p: list of params for the eval input pipelines.
    decode_input_p: list of hparams for the decode input pipelines.
    early_stopping_fn: An optional callable object for reporting eval metrics
      and determining whether to early stop current training. The callable
      object has signature: (metrics_by_dataset, ckpt_step, is_final_ckpt) ->
      should_stop_early.
    enable_auto_sharding: Enables the XLA Auto SPMD partitioner.
  """
  logging.info('Using SPMD sharding for model parallelism.')
  assert py_utils.gda_or_jax_array(), 'GDA or jax.Array must be enabled'
  (
      jax_task,
      partitioner,
      train_state_metadata,
      partitioned_train_state,
      total_num_params,
      prng_key,
      init_key,
      train_input_pipeline,
      train_unpadded_global_batch_size,
  ) = _create_task_and_states(
      task_p, train_input_p, job_log_dir, checkpointer, enable_auto_sharding
  )
  inputs_shape_dtype = partitioner.train_inputs_shape_dtype

  # We do not fold in jax.process_index in contrast to the pmap version and
  # use a single global key instead to rely on pjit to split for different
  # replicas.
  logging.info('root prng_key: %s', prng_key)
  prng_key, train_prng_seed, eval_prng_seed = jax.random.split(prng_key, 3)
  logging.info('train prng_key: %s', train_prng_seed)
  logging.info('eval prng_key: %s', eval_prng_seed)
  train_prng_seed = partitioner.preprocess_prng_key(train_prng_seed)
  eval_prng_seed = partitioner.preprocess_prng_key(eval_prng_seed)

  train_step_fn, is_eval = trainer_lib.get_step_fn(RunningMode.TRAIN)
  assert not is_eval
  p_train_step, inputs_pspecs = partitioner.partition(
      train_step_fn,
      inputs_shape_dtype,
      is_eval,
      train_unpadded_global_batch_size,
  )

  # TODO(pax): Support auto-sharding for eval step. In this case, we would
  # have to fix the sharding of the input to be the same as what's derived
  # from the train_step.

  eval_step_fn, is_eval = trainer_lib.get_step_fn(RunningMode.EVAL)
  assert is_eval
  p_eval_on_train_step, eval_inp_pspec = partitioner.partition(
      eval_step_fn,
      inputs_shape_dtype,
      is_eval,
      # Use train_unpadded_global_batch_size since this is eval on train input.
      unpadded_global_batch_size=train_unpadded_global_batch_size,
  )

  global_mesh = partitioner.global_mesh
  create_gda_for_inputs = (
      py_utils.gda_or_jax_array() and
      checkpoint_type != CheckpointType.CHECKPOINT_PERSISTENCE)
  reshard_inputs = False
  is_vars_replicated = False

  p_eval_steps = []
  padded_eval_input_p = eval_input_p
  if eval_input_p:
    padded_eval_input_p = [
        trainer_lib.adjust_input_params_for_small_batch(input_p, global_mesh)
        for input_p in eval_input_p
    ]
    p_eval_steps, _ = trainer_lib.get_spmd_model_step_fns_from_inputs(
        padded_eval_input_p,
        eval_input_p,
        partitioner,
        RunningMode.EVAL,
    )

  def prepare_model_inputs(input_pipeline, model_inputs):
    if (create_gda_for_inputs or
        input_pipeline.hparams.experimental_remote_input):
      model_inputs = input_pipeline.reshard_for_spmd(model_inputs,
                                                     global_mesh, inputs_pspecs)
    return model_inputs

  def prepare_eval_inputs(train_input_pipeline, eval_inputs):
    if (create_gda_for_inputs or
        train_input_pipeline.hparams.experimental_remote_input):
      eval_inputs = train_input_pipeline.reshard_for_spmd(
          eval_inputs, global_mesh, eval_inp_pspec
      )
    return eval_inputs

  def partition_eval_input_fns(padded_eval_input_ps):
    assert padded_eval_input_ps, 'padded_eval_input_ps must not be empty'
    eval_input_pipelines = [
        instantiate(input_p) for input_p in padded_eval_input_ps]
    trainer_lib.check_unique_names(eval_input_pipelines)

    eval_num_steps = []
    eval_test_inputs_pspecs = []
    for p in padded_eval_input_ps:
      # Do not mutate eval_input_pipelines itself. Instantiate a new one
      # to get sample input.
      sample_eval_model_inputs = instantiate(p).get_next_padded()
      inputs_shape_dtype = jax.tree_util.tree_map(
          py_utils.get_global_input_shape_dtype, sample_eval_model_inputs)
      eval_test_inputs_pspecs.append(
          trainer_lib.get_input_partition_specs(
              task_p.model.mesh_axis_names, inputs_shape_dtype
          )
      )

      # We either run p.eval_loop_num_batches steps or one epoch (when supported
      # by a resettable input) per eval loop during training. When
      # p.reset_for_eval is set to True, we run the eval loop until
      # tf.errors.OutOfRangeError (or StopIteration) is raised, which can be
      # triggered either because input pipeline has reached the end of the input
      # sequence, or a pre-determined num_batches has reached.
      eval_num_steps.append(-1 if p.reset_for_eval else p.eval_loop_num_batches)

    return (
        eval_input_pipelines,
        eval_num_steps,
        eval_test_inputs_pspecs,
        padded_eval_input_ps,
    )

  def partition_decode_once_fns(
      prng_key: jax.random.KeyArray,
      decode_input_ps: Sequence[base_input.BaseInput.HParams]) -> Tuple[
          Callable[..., tuning_lib.DecodeMetrics],
          jax.random.KeyArray, Sequence[base_input.BaseInput.HParams],
      ]:
    assert decode_input_ps, 'decode_input_p must not be empty'
    prng_key, decode_key = jax.random.split(prng_key, 2)
    logging.info('decode prng_key: %s', decode_key)
    decode_key = partitioner.preprocess_prng_key(decode_key)

    padded_decode_input_ps = [
        trainer_lib.adjust_input_params_for_small_batch(input_p, global_mesh)
        for input_p in decode_input_ps
    ]
    padded_decode_input_pipelines = [
        instantiate(input_p) for input_p in decode_input_ps
    ]
    trainer_lib.check_unique_names(padded_decode_input_pipelines)

    # TODO(pax-dev): Support auto-sharding for decoder step.
    (
        decode_step_fns,
        decode_input_partition_specs,
    ) = trainer_lib.get_spmd_model_step_fns_from_inputs(
        padded_decode_input_ps,
        decode_input_ps,
        partitioner,
        RunningMode.DECODE,
    )

    decode_once_fn = eval_lib.partition_decode_once_spmd_model(
        jax_task,
        task_p,
        padded_decode_input_pipelines,
        padded_decode_input_ps,
        job_log_dir,
        decode_key,
        global_mesh,
        decode_step_fns,
        create_gda_for_inputs,
        decode_input_partition_specs,
    )

    return decode_once_fn, prng_key, decode_input_ps

  _train_and_evaluate_common(
      partitioned_train_state,
      prng_key,
      padded_eval_input_p,
      decode_input_p,
      task_p,
      total_num_params,
      early_stopping_fn,
      checkpointer,
      partition_eval_input_fns,
      partition_decode_once_fns,
      job_log_dir,
      eval_prng_seed,
      reshard_inputs,
      task_p.train,
      prepare_eval_inputs,
      prepare_model_inputs,
      is_vars_replicated,
      train_unpadded_global_batch_size,
      train_state_metadata,
      train_input_pipeline,
      p_train_step,
      p_eval_steps,
      p_eval_on_train_step,
      global_mesh,
      create_gda_for_inputs,
      train_prng_seed,
  )


def _create_task_and_states(
    task_p: tasks_lib.SingleTask.HParams,
    train_input_p: base_input.BaseInput.HParams,
    job_log_dir: epath.Path,
    checkpointer: _TrainingCheckpointer,
    enable_auto_sharding: bool = False,
):
  jax_task = instantiate(task_p)
  prng_key = jax.random.PRNGKey(task_p.train.random_seed)
  prng_key, init_key = jax.random.split(prng_key)

  # The partitioner only needs shape/dtype information of the prng key.
  # TODO(laigd): let the partitioner take ShapeDtypeStruct of prng key instead.
  partitioner = trainer_lib.create_partitioner(
      jax_task,
      init_key,
      auto_sharding_mode=RunningMode.TRAIN if enable_auto_sharding else None,
      job_log_dir=job_log_dir,
  )
  train_unpadded_global_batch_size = train_input_p.cls.get_global_batch_size(
      train_input_p
  )
  train_input_p = partitioner.preprocess_input_params(train_input_p)
  train_input_pipeline = _PeekableInput(instantiate(train_input_p))
  partitioner.set_train_inputs_shape_dtype(train_input_pipeline)
  train_state_metadata = partitioner.get_train_state_metadata()

  # JaxContext needed for shared layer lookup from global scope.
  with base_layer.JaxContext.new_context():
    # Dump out model meta info for debugging.
    trainer_lib.write_post_init_model_hparams_file(
        jax_task.model, train_state_metadata.var_weight_hparams, job_log_dir
    )

  # Restore TrainState from checkpoint or initialize it.
  partitioned_train_state, total_num_params = checkpointer.get_model_states(
      jax_task, partitioner.global_mesh, train_state_metadata, init_key
  )

  initial_global_step = int(
      py_utils.maybe_unreplicate_for_fully_replicated(
          partitioned_train_state.step
      )
  )
  logging.info('Model initial global_step=%d', initial_global_step)
  train_input_pipeline = _maybe_update_latest_model_step(
      train_input_pipeline, train_input_p, initial_global_step
  )
  return (
      jax_task,
      partitioner,
      train_state_metadata,
      partitioned_train_state,
      total_num_params,
      prng_key,
      init_key,
      train_input_pipeline,
      train_unpadded_global_batch_size,
  )


def _train_and_evaluate_common(
    partitioned_train_state, prng_key, eval_input_p, decode_input_p, task_p,
    total_num_params, early_stopping_fn, checkpointer, partition_eval_input_fns,
    partition_decode_once_fns, job_log_dir, eval_prng_seed, reshard_inputs,
    train_p, prepare_eval_inputs, prepare_model_inputs, is_vars_replicated,
    train_unpadded_global_batch_size, train_state_metadata,
    train_input_pipeline, p_train_step, p_eval_steps, p_eval_on_train_step,
    global_mesh, create_gda_for_inputs, train_prng_seed):
  """Training loop code common to both pmap and spmd."""

  if eval_input_p:
    (
        eval_input_pipelines,
        eval_num_steps,
        eval_test_inputs_pspecs,
        eval_input_p,
    ) = partition_eval_input_fns(eval_input_p)

  if decode_input_p:
    decode_once_fn, prng_key, decode_input_p = partition_decode_once_fns(
        prng_key, decode_input_p)

  logging.info('Training loop starting...')

  with _SummaryContextManager(
      job_log_dir, eval_input_p, decode_input_p,
      train_p.eval_skip_train) as (train_summary_writer, eval_summary_writer,
                                   eval_test_summary_writers,
                                   decode_summary_writers):
    # This only prints the view from the first host machine.
    summary_utils.write_model_structure(
        train_summary_writer,
        partitioned_train_state,
        is_vars_replicated=is_vars_replicated)
    summary_utils.write_total_num_params(train_summary_writer, total_num_params)
    summary_utils.write_global_batch_size(train_summary_writer,
                                          train_unpadded_global_batch_size)

    train_summary_handler = summary_utils.SummaryHandler(
        train_summary_writer,
        train_p.summary_interval_steps,
        accumulate_interval_steps=train_p.summary_accumulate_interval_steps,
        log_interval_steps=_train_log_interval_steps(train_p),
        is_async=bool(train_p.device_sync_interval_steps),
        name='training')
    eval_summary_handler = summary_utils.SummaryHandler(
        eval_summary_writer,
        train_p.summary_interval_steps,
        accumulate_interval_steps=train_p.summary_accumulate_interval_steps,
        name='eval')

    summary_last_time = time.time()
    summary_last_step = None

    profiler = profiling.Profiler(
        num_steps=train_p.profiler_num_steps,
        min_duration_sec=train_p.profiler_min_duration_sec,
        max_num_hosts=train_p.profiler_max_num_hosts)

    step_i = int(
        py_utils.maybe_unreplicate_for_fully_replicated(
            partitioned_train_state.step))
    initial_step = step_i
    init_duration_set = False

    # Start the train loop. Make sure all at the same step.
    py_utils.sync_global_devices(f'Start training loop from step: {step_i}')
    # Collect then freeze GC, so that GC in the training loop will not touch the
    # python objects used to initialize the model. Unfreeze at the end of the
    # loop.
    gc.collect()
    gc.freeze()
    while True:
      logging.debug('step=`%d`: Beginning', step_i)

      if summary_last_step is None:
        summary_last_step = step_i - 1

      checkpointer.save_if_needed(
          step_i, partitioned_train_state, train_state_metadata.partition_specs
      )

      if step_i >= train_p.num_train_steps:
        logging.info(
            'Training loop completed (step (`%d`) greater than '
            'num_train_step (`%d`).', step_i, train_p.num_train_steps)
        break

      # Get new model inputs
      logging.debug('  Retrieving inputs.')
      model_inputs = train_input_pipeline.get_next_padded()
      model_inputs = prepare_model_inputs(train_input_pipeline, model_inputs)
      logging.debug('  Retrieved inputs.')

      do_profile = train_p.profiler_capture_step is not None
      if (do_profile and
          step_i - initial_step == train_p.profiler_capture_step):
        profiler.capture_async()

      logging.debug('  Performing train_step().')
      with jax.profiler.StepTraceAnnotation('train', step_num=step_i):
        with py_utils.timeit() as train_period:
          (partitioned_train_state, loss, weighted_scalars, per_example_out,
           summary_tensors) = p_train_step(partitioned_train_state,
                                           train_prng_seed, model_inputs)
      logging.debug('  Completed train_step() in %f seconds.',
                    train_period.elapsed)
      if step_i == initial_step:
        first_step_completion_time = time.time()

      if do_profile and step_i - initial_step < train_p.profiler_capture_step:
        profiler.update_step_moving_mean(train_period.elapsed)

      logging.debug('  Writing summaries (attempt).')
      new_step_i = step_i + 1

      # Train metrics.
      train_weighted_scalars = weighted_scalars
      if train_p.device_sync_interval_steps:
        should_sync_device = (
            new_step_i % train_p.device_sync_interval_steps
        ) == 0
      else:
        should_sync_device = train_summary_handler.should_write(new_step_i)
      steps_per_sec = None
      if should_sync_device:
        # Synchronize step_i. This is performed at a fixed interval to avoid
        # a gap between steps.
        new_step_i = int(
            py_utils.maybe_unreplicate_for_fully_replicated(
                partitioned_train_state.step))
        steps_per_sec = _compute_steps_per_sec(step_i, summary_last_time,
                                               summary_last_step)
        logging.info('steps/sec: %f', steps_per_sec)
        summary_last_time = time.time()
        summary_last_step = step_i
        if not init_duration_set:
          # Find estimated timestamp before the first execution call.
          # This enables us to include the first step's compile time but exclude
          # its execution time from the init duration.
          estimated_execute_duration = 1 / steps_per_sec
          first_step_execute_time = (
              first_step_completion_time - estimated_execute_duration)
          init_duration = first_step_execute_time - _INIT_TIME
          monitoring.record_event_duration_secs(
              '/jax/pax/init/time_before_first_step_secs', init_duration)
          init_duration_set = True

      # Note: Train metrics are currently reported at step_i + 1, while these
      # training metrics/summaries are pre-model weight updates.
      # TODO(b/264635784): Update the logic to pass step_i instead.
      train_summary_handler.process(
          new_step_i,
          loss,
          weighted_scalars,
          summary_tensors,
          per_example_out=per_example_out,
          steps_per_sec=steps_per_sec)
      logging.debug('  Wrote summaries (attempted).')

      # While the eval ones below are post-model weight updates, hence the step
      # counter is incremented in between.
      step_i = new_step_i

      eval_train_metrics = None
      eval_metrics: Optional[tuning_lib.EvalMetrics] = None
      # Run eval at regular step interval.
      if (train_p.eval_interval_steps and
          step_i % train_p.eval_interval_steps == 0):

        logging.debug('  Starting eval_step().')

        if train_p.eval_use_ema_states:
          if not eval_lib.has_ema(task_p):
            raise ValueError(
                'eval_use_ema_states is requested but the '
                'learner does not seem to have ema enabled'
            )
          eval_partitioned_train_state = eval_lib.extract_ema(
              partitioned_train_state).to_eval_state()
          logging.debug('  Performing eval_step() with ema states.')
        else:
          eval_partitioned_train_state = partitioned_train_state.to_eval_state()
        # If we have eval test then also evaluate on test.
        if eval_input_p:
          logging.debug('  Performing eval_step() runs on test splits.')
          eval_step_fns = [functools.partial(
              step_fn, eval_partitioned_train_state,
              eval_prng_seed) for step_fn in p_eval_steps]
          with py_utils.timeit() as eval_period:
            eval_metrics_list, eval_scoring_metrics_list, num_eval_steps = (
                eval_lib.run_eval_loop_over_test_splits(
                    eval_num_steps,
                    eval_step_fns,
                    eval_test_summary_writers,
                    step_i,
                    eval_input_pipelines,
                    job_log_dir,
                    eval_test_inputs_pspecs,
                    global_mesh,
                    reshard_inputs=reshard_inputs,
                    create_gda_for_inputs=create_gda_for_inputs))
          eval_steps_per_sec = sum(num_eval_steps) / eval_period.elapsed
          eval_metrics = tuning_lib.EvalMetrics(
              input_p=eval_input_p,
              metrics_list=eval_metrics_list,
              scoring_metrics_list=eval_scoring_metrics_list,
              steps_per_sec=eval_steps_per_sec)
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
            logging.debug('  eval_inputs is None. Skipping eval_train.')
          else:
            logging.debug('  Retrieved eval model_inputs.')
            logging.debug('  Performing eval_step() runs on training split.')
            eval_inputs = prepare_eval_inputs(train_input_pipeline, eval_inputs)

            eval_on_train_step_fn = functools.partial(
                p_eval_on_train_step,
                eval_partitioned_train_state,
                eval_prng_seed)
            loss, weighted_scalars, _, summary_tensors = eval_on_train_step_fn(
                eval_inputs)
            logging.debug('  Completed eval_step() runs on training split.')
            if eval_summary_handler.process(step_i, loss, weighted_scalars,
                                            summary_tensors):
              logging.debug('  Wrote eval summaries.')
            eval_train_metrics = metric_utils.as_float_dict(weighted_scalars)

      decode_metrics: Optional[tuning_lib.DecodeMetrics] = None
      if (decode_input_p and train_p.decode_interval_steps and
          step_i % train_p.decode_interval_steps == 0):
        if train_p.decode_use_ema_states:
          if not eval_lib.has_ema(task_p):
            raise ValueError(
                'decode_use_ema_states is requested but the '
                'learner does not seem to have ema enabled'
            )
          decode_partitioned_train_state = eval_lib.extract_ema(
              partitioned_train_state)
          logging.debug('  Performing decode_once_fn() with ema states.')
        else:
          decode_partitioned_train_state = partitioned_train_state
        decode_metrics = decode_once_fn(decode_partitioned_train_state,
                                        decode_summary_writers)

      logging.debug('step=`%d`: End', step_i - 1)

      if early_stopping_fn is not None:
        if tuning_lib.should_early_stop(
            early_stopping_fn,
            step_i,
            is_last_ckpt=tuning_lib.is_last_checkpoint(
                RunningMode.detect(
                    has_train_metrics=True,
                    has_eval_metrics=bool(eval_metrics),
                    has_decode_metrics=bool(decode_metrics)), step_i,
                task_p.train.num_train_steps, task_p.train.eval_interval_steps,
                task_p.train.decode_interval_steps,
                task_p.train.save_interval_steps),
            train_weighted_scalars=train_weighted_scalars,
            eval_train_metrics=eval_train_metrics,
            eval_metrics=eval_metrics,
            decode_metrics=decode_metrics,
            train_steps_per_sec=steps_per_sec,
            num_params=total_num_params):
          logging.info(
              'Training loop is early stopped at step `%d` by the '
              'tuner, while num_train_step is `%d`.', step_i,
              train_p.num_train_steps)
          break
    gc.unfreeze()
    # Save checkpoint for the last step.
    checkpointer.save_final(
        step_i, partitioned_train_state, train_state_metadata.partition_specs
    )

    train_summary_handler.close()
    eval_summary_handler.close()
