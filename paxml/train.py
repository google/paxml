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

"""Training loop for Pax model."""

import abc
import contextlib
import datetime
import gc
import re
import time
import typing
from typing import Any, Callable, Optional, Sequence, Tuple, Type

from absl import logging
from etils import epath
import jax
from jax import monitoring
import jax.numpy as jnp
import numpy as np
from paxml import base_experiment
from paxml import checkpoint_managers
from paxml import eval_lib
from paxml import experiment_utils
from paxml import metric_utils
from paxml import partitioning
from paxml import programs
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
CheckpointType = checkpoints.CheckpointType
instantiate = base_hyperparams.instantiate
NestedShapeDtypeLike = pytypes.NestedShapeDtypeLike
FlaxCheckpointer = checkpoints.FlaxCheckpointer
FlaxCheckpointHandler = checkpoints.FlaxCheckpointHandler
PaxCheckpointHandler = checkpoints.PaxCheckpointHandler
# alias to internal checkpointer
BaseInputCheckpointHandler = checkpoints.BaseInputCheckpointHandler
PRNGKey = pytypes.PRNGKey
RunningMode = trainer_lib.RunningMode
SummaryWriter = tf.summary.SummaryWriter
TrainState = train_states.TrainState

PARAMS = base_layer.PARAMS
NON_PAX_RNG_KEY = base_layer.NON_PAX_RNG_KEY

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
    duration_str: Optional[str],
) -> Optional[datetime.timedelta]:
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
  if (
      not match
      or len(match.groups()) != 2
      or match.group(2) not in {None, 's', 'm', 'h', 'd'}
  ):
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
  def save_if_needed(
      self,
      step_i,
      partitioned_train_state,
      train_state_pspecs,
      train_input_pipeline,
  ):
    raise NotImplementedError

  @abc.abstractmethod
  def save_final(
      self,
      step_i,
      *,
      partitioned_train_state,
      train_state_pspecs,
      train_input_pipeline,
  ):
    raise NotImplementedError

  @abc.abstractmethod
  def get_model_states(
      self,
      partitioner: partitioning.Partitioner,
      metadata: trainer_lib.TrainStateMetadata,
      root_prng_key: PRNGKey,
      train_input_pipeline: Optional[base_input.BaseInput] = None,
  ) -> Tuple[TrainState, int, PRNGKey]:
    """Restores TrainState from checkpoint or initializes it.

    Args:
      partitioner: The partitioner used to initialized the model states and root
        prng key.
      metadata: A TrainStateMetadata instance.
      root_prng_key: PRNGKey for initializing the model variables.
      train_input_pipeline: Training input pipeline instance

    Returns:
      (train_state, total_num_params, initialized_root_prng_key).
    """
    raise NotImplementedError

  @property
  @abc.abstractmethod
  def checkpoint_type(self) -> CheckpointType:
    raise NotImplementedError

  def wait_until_finished(self):
    """Waits for any incomplete save operations to complete."""
    raise NotImplementedError


class _OrbaxPjitTrainingCheckpointer(_TrainingCheckpointer):

  def __init__(
      self,
      checkpoint_manager: checkpoint_managers.OrbaxCheckpointManager,
      checkpoint_type: CheckpointType,
      enable_checkpoint_saving: bool = True,
  ):
    self.checkpoint_manager = checkpoint_manager
    self._checkpoint_type = checkpoint_type
    if checkpoint_type == CheckpointType.FLAX:
      raise ValueError('FLAX checkpointing not supported for pjit models.')
    self._enable_checkpoint_saving = enable_checkpoint_saving

  def wait_until_finished(self):
    self.checkpoint_manager.wait_until_finished()

  def _save_with_args(
      self,
      step_i: int,
      *,
      partitioned_train_state: Any,
      train_input_pipeline: Optional[base_input.BaseInput] = None,
      force: Optional[bool] = False,
  ):
    if not self._enable_checkpoint_saving:
      return
    with py_utils.timeit() as save_period:
      self.checkpoint_manager.save(
          step_i,
          partitioned_train_state,
          train_input_pipeline,
          force=force,
      )
    monitoring.record_event_duration_secs(
        _WRITE_CHECKPOINT_EVENT, save_period.elapsed
    )

  def _restore_with_args(
      self,
      step_i,
      train_state_global_shapes,
      global_mesh,
      train_state_pspecs,
      train_input_pipeline,
  ):
    restore_args = {}
    if self._checkpoint_type == CheckpointType.GDA:
      restore_args = {'specs': train_state_pspecs, 'mesh': global_mesh}
    elif self._checkpoint_type == CheckpointType.PERSISTENCE:
      restore_args = {
          'state_specs': train_state_pspecs,
          'global_mesh': global_mesh,
      }
    return self.checkpoint_manager.restore(
        step_i,
        train_state_global_shapes,
        train_input_pipeline,
        restore_kwargs=restore_args,
    )

  def save_final(
      self,
      step_i,
      *,
      partitioned_train_state,
      train_state_pspecs=None,
      train_input_pipeline: Optional[base_input.BaseInput] = None,
  ):
    del train_state_pspecs
    latest_step = self.checkpoint_manager.latest_step()
    if latest_step is None or latest_step < step_i:
      logging.info('Saving a ckpt at final step: %d', step_i)
      self._save_with_args(
          step_i,
          partitioned_train_state=partitioned_train_state,
          train_input_pipeline=train_input_pipeline,
          force=True,
      )

  def save_if_needed(
      self,
      step_i,
      partitioned_train_state,
      train_state_pspecs=None,
      train_input_pipeline=None,
  ):
    del train_state_pspecs
    if not self.checkpoint_manager.should_save(step_i):
      return
    self._save_with_args(
        step_i,
        partitioned_train_state=partitioned_train_state,
        train_input_pipeline=train_input_pipeline,
    )
    self.checkpoint_manager.check_for_errors()

  # TODO(laigd): merge this with _SpmdEvalCheckpointer.get_model_states().
  def get_model_states(
      self,
      partitioner: partitioning.Partitioner,
      metadata: trainer_lib.TrainStateMetadata,
      root_prng_key: PRNGKey,
      train_input_pipeline: Optional[base_input.BaseInput] = None,
  ) -> Tuple[TrainState, int, PRNGKey]:
    step = self.checkpoint_manager.latest_step()
    if step is None:
      partitioned_train_state = None
    else:
      with py_utils.timeit() as restore_period:
        partitioned_train_state = self._restore_with_args(
            step,
            metadata.padded_global_shapes,
            partitioner.global_mesh,
            metadata.partition_specs,
            train_input_pipeline,
        )
      monitoring.record_event_duration_secs(
          _READ_CHECKPOINT_EVENT, restore_period.elapsed
      )

    root_prng_key, partitioned_train_state = (
        partitioner.initialize_prng_key_and_train_state(
            root_prng_key,
            partitioned_train_state,
            self.checkpoint_type,
        )
    )

    total_num_params = py_utils.total_num_vars(partitioned_train_state.mdl_vars)
    return partitioned_train_state, total_num_params, root_prng_key

  @property
  def checkpoint_type(self) -> CheckpointType:
    return self._checkpoint_type


class _OrbaxPmapTrainingCheckpointer(_TrainingCheckpointer):

  def __init__(
      self,
      job_log_dir: epath.Path,
      checkpoint_manager: checkpoint_managers.OrbaxCheckpointManager,
      checkpoint_type: CheckpointType,
      enable_checkpoint_saving: bool = True,
  ):
    self.job_log_dir = job_log_dir
    self.checkpoint_dir = _checkpoint_dir(job_log_dir)
    self.checkpoint_manager = checkpoint_manager
    self._checkpoint_type = checkpoint_type
    self._enable_checkpoint_saving = enable_checkpoint_saving

  def wait_until_finished(self):
    self.checkpoint_manager.wait_until_finished()

  def _restore_with_args(
      self,
      step_i: int,
      train_state_global_shapes: Optional[TrainState],
      train_input_pipeline: Optional[base_input.BaseInput],
  ):
    """Restore using CheckpointManager, setting up additional args."""
    restore_args = None
    if py_utils.pmap_use_tensorstore():
      # TODO(gaoi) add support for input checkpointing using tensorstore
      if train_input_pipeline:
        raise ValueError(
            'Input checkpointing is not supported when using'
            ' pmap_use_tensorstore=True. Please set'
            ' Task.Train.enable_input_checkpoint=False'
        )

      def _get_spec(shape):
        if shape.shape:
          return jax.sharding.PartitionSpec(None)
        else:
          return jax.sharding.PartitionSpec()

      global_mesh = jax.sharding.Mesh(
          np.array(jax.devices()), axis_names=('x',)
      )
      fully_replicated_state_specs = jax.tree_map(
          _get_spec, train_state_global_shapes
      )
      restore_args = {
          'specs': fully_replicated_state_specs,
          'mesh': global_mesh,
      }
    restored_state = self.checkpoint_manager.restore(
        step_i,
        train_state_global_shapes,
        train_input_pipeline=train_input_pipeline,
        restore_kwargs=restore_args,
    )
    if not py_utils.pmap_use_tensorstore():
      return restored_state
    if self._checkpoint_type == CheckpointType.PERSISTENCE:
      return jax.tree_map(
          py_utils.convert_fully_replicated_array_to_pmap_array,
          restored_state,
      )
    # model_states is jax.Array; we convert back to DA or jax.Array with
    # single device sharding for pmap.
    return jax.tree_map(lambda x: x.addressable_data(0), restored_state)

  # TODO(laigd): merge this with _PmapEvalCheckpointer.get_model_states().
  def get_model_states(
      self,
      partitioner: partitioning.Partitioner,
      metadata: trainer_lib.TrainStateMetadata,
      root_prng_key: PRNGKey,
      train_input_pipeline: Optional[base_input.BaseInput] = None,
  ) -> Tuple[TrainState, int, PRNGKey]:
    train_state_global_shapes = metadata.unpadded_global_shapes
    with py_utils.timeit() as restore_period:
      step = self.checkpoint_manager.latest_step()
      if step is None:
        train_state = None
      else:
        train_state = self._restore_with_args(
            step, train_state_global_shapes, train_input_pipeline
        )
    monitoring.record_event_duration_secs(
        _READ_CHECKPOINT_EVENT, restore_period.elapsed
    )

    # TODO(laigd): move the logic below outside of get_model_states.
    root_prng_key, replicated_train_state = (
        partitioner.initialize_prng_key_and_train_state(
            root_prng_key, train_state, self.checkpoint_type
        )
    )

    total_num_params = py_utils.total_num_vars(replicated_train_state.mdl_vars)
    assert total_num_params % jax.local_device_count() == 0
    total_num_params = total_num_params // jax.local_device_count()
    return replicated_train_state, total_num_params, root_prng_key

  def _save_with_args(
      self,
      step_i: int,
      *,
      train_state: Any,
      train_input_pipeline: Optional[base_input.BaseInput] = None,
      force: Optional[bool] = False,
  ):
    self.checkpoint_manager.save(
        step_i,
        train_state,
        train_input_pipeline,
        force=force,
    )

  def _save(
      self,
      step_i,
      partitioned_train_state,
      train_input_pipeline,
      is_final=False,
  ):
    if not self._enable_checkpoint_saving:
      return

    with py_utils.timeit() as save_period:
      if py_utils.pmap_use_tensorstore():
        logging.info(
            'Saving a ckpt at %sstep: %d', 'final ' if is_final else '', step_i
        )
        fully_replicated_gda_train_state = jax.tree_map(
            py_utils.convert_host_local_array_to_global_array,
            partitioned_train_state,
        )
        self._save_with_args(
            step_i,
            train_state=fully_replicated_gda_train_state,
            train_input_pipeline=train_input_pipeline,
            force=is_final,
        )
      else:
        unreplicated_train_state = jax.tree_map(
            lambda x: x[0], partitioned_train_state
        )
        self._save_with_args(
            step_i,
            train_state=unreplicated_train_state,
            train_input_pipeline=train_input_pipeline,
            force=is_final,
        )
    monitoring.record_event_duration_secs(
        _WRITE_CHECKPOINT_EVENT, save_period.elapsed
    )

  def save_if_needed(
      self,
      step_i,
      partitioned_train_state,
      train_state_pspecs,
      train_input_pipeline=None,
  ):
    if not self.checkpoint_manager.should_save(step_i):
      return
    self._save(
        step_i,
        partitioned_train_state,
        train_input_pipeline,
    )
    self.checkpoint_manager.check_for_errors()

  def save_final(
      self,
      step_i,
      *,
      partitioned_train_state,
      train_state_pspecs,
      train_input_pipeline: Optional[base_input.BaseInput] = None,
  ):
    latest_step = self.checkpoint_manager.latest_step()
    if latest_step is None or latest_step < step_i:
      self._save(
          step_i, partitioned_train_state, train_input_pipeline, is_final=True
      )

  @property
  def checkpoint_type(self) -> CheckpointType:
    return self._checkpoint_type


def _create_checkpointer(
    task_p: tasks_lib.SingleTask.HParams,
    job_log_dir: epath.Path,
    checkpoint_type: CheckpointType,
    todelete_subdir: Optional[str],
    train_input_p: Optional[base_input.BaseInput.HParams] = None,
    enable_async_checkpointing: bool = True,
    enable_checkpoint_saving: bool = True,
    enforce_restore_shape_check: bool = False,
    maybe_use_persistence_checkpointing: bool = False,
) -> _TrainingCheckpointer:
  """Creates a checkpoint manager."""
  checkpoint_dir = _make_checkpoint_dir(job_log_dir)
  train_p = task_p.train
  max_to_keep = train_p.save_max_to_keep
  save_interval_steps = train_p.save_interval_steps
  keep_interval_timedelta = _parse_duration(train_p.save_keep_interval_duration)

  checkpoints.reregister_type_handlers(train_p.tensorstore_metadata_key)
  options = checkpoint_managers.CheckpointManagerOptions(
      max_to_keep=max_to_keep,
      save_interval_steps=save_interval_steps,
      keep_time_interval=keep_interval_timedelta,
      todelete_subdir=todelete_subdir,
  )

  if checkpoint_type == CheckpointType.FLAX:
    checkpointer = FlaxCheckpointer(FlaxCheckpointHandler())
  elif enable_async_checkpointing:
    if maybe_use_persistence_checkpointing:
      raise NotImplementedError('Persistence checkpointer not supported.')
    else:
      checkpointer = checkpoints.AsyncCheckpointer(
          checkpoints.PaxCheckpointHandler(
              enforce_restore_shape_check=enforce_restore_shape_check
          ),
          timeout_secs=600,
      )
  else:
    if checkpoint_type == CheckpointType.GDA:
      checkpointer = Checkpointer(
          PaxCheckpointHandler(
              enforce_restore_shape_check=enforce_restore_shape_check
          )
      )
    elif checkpoint_type == CheckpointType.PERSISTENCE:
      raise ValueError('Checkpointer must already be initialized.')
    else:
      raise ValueError(f'Unsupported Orbax checkpoint type: {checkpoint_type}')

  train_input_checkpointer = None
  if train_p.enable_input_checkpointing:
    if (
        hasattr(train_input_p, 'deterministic_input')
        and train_input_p.deterministic_input
    ):
      raise ValueError(
          'Checkpointing deterministic Seqio inputs is not supported via Orbax'
          ' (will be checkpointed independently). Please set'
          ' enable_input_checkpointing=False.'
      )
    train_input_checkpointer = checkpoints.BaseInputCheckpointHandler()
  checkpoint_manager = checkpoint_managers.OrbaxCheckpointManager(
      checkpoint_dir,
      checkpointer,
      train_input_checkpointer=train_input_checkpointer,
      options=options,
      checkpoint_type=checkpoint_type,
  )

  if task_p.model.ici_mesh_shape is not None:
    checkpointer = _OrbaxPjitTrainingCheckpointer(
        checkpoint_manager,
        checkpoint_type,
        enable_checkpoint_saving=enable_checkpoint_saving,
    )
  else:
    checkpointer = _OrbaxPmapTrainingCheckpointer(
        job_log_dir,
        checkpoint_manager,
        checkpoint_type,
        enable_checkpoint_saving=enable_checkpoint_saving,
    )

  return checkpointer


def _train_log_interval_steps(
    train_p: tasks_lib.SingleTask.TrainHParams,
) -> int:
  """Returns the interval to log train outputs."""
  if train_p.log_train_output_interval_steps is not None:
    return train_p.log_train_output_interval_steps
  else:
    return train_p.summary_interval_steps


def write_hparams_file(
    model_config: base_experiment.BaseExperiment,
    job_log_dir: epath.Path,
    filename_prefix: str = '',
) -> None:
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


def write_experiment_class_vars_file(
    exp_cls: Type[base_experiment.BaseExperiment],
    job_log_dir: epath.Path,
    filename_prefix: str = '',
) -> None:
  """Writes a params file into the root `job_log_dir`."""
  if jax.process_index() == 0:
    exp_summary_fpath = (
        job_log_dir / f'{filename_prefix}experiment_cls_vars.txt'
    )
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
    enable_async_checkpointing: bool = False,
    enable_checkpoint_saving: bool = True,
    enforce_restore_shape_check: bool = False,
) -> None:
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
    enable_async_checkpointing: Allows training to continue when checkpointing
      is going on as checkpointing happens in a different thread.
    enable_checkpoint_saving: Whether to perform checkpoint saving or not.
    enforce_restore_shape_check: Raises an error if restore shapes do not match
      checkpoint shapes.
  """
  jax.monitoring.record_event('/jax/pax/train_and_evaluate/beacon')
  task_p = experiment_config.task()
  task_p = typing.cast(tasks_lib.SingleTask.HParams, task_p)

  # in case the user passed in a string dtype, convert it to an actual dtype
  task_p.model.fprop_dtype = jnp.dtype(task_p.model.fprop_dtype)

  input_p = experiment_config.datasets()
  # Note that we modify input params below with runtime information, therefore
  # experiment_config.datasets() should not be called again as it won't have the
  # correct runtime information populated.
  for inp in input_p:
    if not isinstance(
        inp, (base_input.BaseInput.HParams, base_input.DistributedInputHParams)
    ):
      raise ValueError(
          f'Expecting BaseInput.HParams from datasets(), got: {inp.ToText()}'
      )
    if inp.num_infeed_hosts == 0:
      inp.num_infeed_hosts = jax.process_count()
    inp.infeed_host_index = jax.process_index()
  train_input_p = [v for v in input_p if v.is_training]
  if len(train_input_p) != 1:
    raise ValueError(
        f'Expecting exactly one training split. Got `{len(train_input_p)}`.'
    )
  train_input_p = train_input_p[0]

  logging.info('train_input_p:')
  for line in train_input_p.to_text().splitlines():
    logging.info('  %s', line)
  logging.info('task_p:')
  for line in task_p.to_text().splitlines():
    logging.info('  %s', line)

  eval_input_p = []
  if (
      eval_on_test
      and task_p.train.eval_interval_steps is not None
      and task_p.train.eval_interval_steps > 0
  ):
    eval_input_p = [v for v in input_p if not v.is_training]

  if (
      run_decode
      and task_p.train.decode_interval_steps is not None
      and task_p.train.decode_interval_steps > 0
  ):
    decode_input_p = experiment_config.decoder_datasets()
  else:
    decode_input_p = []
  for inp in decode_input_p:
    if inp.num_infeed_hosts == 0:
      inp.num_infeed_hosts = jax.process_count()
    inp.infeed_host_index = jax.process_index()

  checkpoint_type = checkpoints.retrieve_checkpoint_type(
      maybe_use_persistence_checkpointing, task_p
  )

  job_log_dir = epath.Path(job_log_dir)
  checkpointer = _create_checkpointer(
      task_p,
      job_log_dir,
      checkpoint_type,
      checkpoint_todelete_subdir,
      train_input_p=train_input_p,
      enable_async_checkpointing=enable_async_checkpointing,
      enable_checkpoint_saving=enable_checkpoint_saving,
      enforce_restore_shape_check=enforce_restore_shape_check,
      maybe_use_persistence_checkpointing=maybe_use_persistence_checkpointing,
  )
  if not enable_checkpoint_saving:
    logging.info(
        'Checkpointing is disabled and no checkpoint will be saved to disk.'
    )

  if task_p.model.ici_mesh_shape is not None:
    train_and_evaluate_spmd_model(
        task_p,
        train_input_p,
        job_log_dir,
        checkpointer,
        eval_input_p,
        decode_input_p,
        early_stopping_fn,
        enable_auto_sharding,
        experiment_train_program=experiment_config.train_program(),
    )
  else:
    train_and_evaluate_pmap(
        task_p,
        train_input_p,
        job_log_dir,
        checkpointer,
        eval_input_p,
        decode_input_p,
        early_stopping_fn,
        experiment_train_program=experiment_config.train_program(),
    )
  checkpointer.wait_until_finished()  # No-op if not async.


def _maybe_update_latest_model_step(
    train_input: base_input.BaseInput,
    train_input_p: base_input.BaseInput.HParams,
    initial_global_step: int,
) -> base_input.BaseInput:
  """Updates `train_input_p` in place its latest model step."""
  if not hasattr(train_input_p, 'deterministic_input_start_index'):
    return train_input
  dp = train_input_p.deterministic_input_start_index
  dp._latest_model_step = (
      initial_global_step  # pylint: disable=protected-access
  )
  logging.info('Reinstanting input because _latest_model_step is updated.')
  return instantiate(train_input_p)


class _SummaryContextManager(contextlib.ExitStack):
  """Manage summary writers."""

  _exit_callbacks = []

  def __init__(
      self,
      job_log_dir: epath.Path,
      eval_input_names: Sequence[str],
      decode_input_names: Sequence[str],
      eval_skip_train: bool = False,
  ):
    """Initialize context manager.

    Args:
      job_log_dir: Directory for the job logs.
      eval_input_names: list of names for the eval input pipelines.
      decode_input_names: list of names for the decode input pipelines.
      eval_skip_train: By default, we also run eval on the training data input
        (`eval_train`), specifically on a batch not yet used for training. When
        set to True, this is skipped.
    """
    super().__init__()
    self.summary_base_dir = job_log_dir / 'summaries'
    self.summary_train_dir = self.summary_base_dir / 'train'
    self.summary_eval_dir = self.summary_base_dir / 'eval_train'
    self.summary_writer = summary_utils.get_summary_writer
    if eval_input_names:
      self.summary_eval_test_dirs = [
          self.summary_base_dir / f'eval_test_{name}'
          for name in eval_input_names
      ]
    else:
      self.summary_eval_test_dirs = []
    if decode_input_names:
      self.summary_decode_dirs = [
          self.summary_base_dir / f'decode_test_{name}'
          for name in decode_input_names
      ]
    else:
      self.summary_decode_dirs = []
    self.eval_skip_train = eval_skip_train

  def __enter__(
      self,
  ) -> Tuple[SummaryWriter, SummaryWriter, SummaryWriter, SummaryWriter]:
    self.train_summary_writer = self.enter_context(
        self.summary_writer(self.summary_train_dir)
    )
    self.eval_summary_writer = None
    if not self.eval_skip_train:
      self.eval_summary_writer = self.enter_context(
          self.summary_writer(self.summary_eval_dir)
      )
    self.eval_test_summary_writers = [
        self.enter_context(self.summary_writer(d))
        for d in self.summary_eval_test_dirs
    ]
    self.decode_summary_writers = [
        self.enter_context(self.summary_writer(d))
        for d in self.summary_decode_dirs
    ]
    return (
        self.train_summary_writer,
        self.eval_summary_writer,
        self.eval_test_summary_writers,
        self.decode_summary_writers,
    )


def train_and_evaluate_pmap(
    task_p: tasks_lib.SingleTask.HParams,
    train_input_p: base_input.BaseInput.HParams,
    job_log_dir: epath.Path,
    checkpointer: _TrainingCheckpointer,
    eval_input_p: Sequence[base_input.BaseInput.HParams],
    decode_input_p: Sequence[base_input.BaseInput.HParams],
    early_stopping_fn: Optional[trainer_lib.EarlyStoppingFn] = None,
    experiment_train_program: Optional[programs.BaseTrainProgram] = None,
) -> None:
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
    experiment_train_program: An embedded train_program that's constructed in
      experiment. If specified, the program will be used for train steps.
  """

  logging.info('Using pmap for data parallelism.')
  (
      task,
      partitioner,
      train_program,
      test_eval_programs,
      partitioned_train_state,
      total_num_params,
      prng_key,
      train_prng_seed,
      eval_prng_seed,
      early_stopping_fn,
  ) = _create_program_and_states(
      task_p,
      train_input_p,
      eval_input_p,
      job_log_dir,
      checkpointer,
      early_stopping_fn=early_stopping_fn,
      experiment_train_program=experiment_train_program,
  )

  def partition_decode_once_fns(prng_key, decode_input_p):
    decode_input_pipelines = [
        instantiate(input_p) for input_p in decode_input_p
    ]
    trainer_lib.check_unique_names(decode_input_pipelines)
    prng_key, decode_key = jax.random.split(prng_key, 2)
    logging.info('decode prng_seed: %s', decode_key)
    decode_key = partitioner.preprocess_prng_key(decode_key)
    decode_once_fn = eval_lib.partition_decode_once_pmap_model(
        task,
        partitioner,
        task_p,
        partitioner.get_train_state_metadata().var_weight_hparams,
        decode_input_pipelines,
        decode_key,
        job_log_dir,
    )
    decode_input_names = [inp.name for inp in decode_input_pipelines]
    return decode_once_fn, prng_key, decode_input_names

  _train_and_evaluate_common(
      task,
      partitioner,
      train_program,
      partitioned_train_state,
      prng_key,
      test_eval_programs,
      decode_input_p,
      total_num_params,
      early_stopping_fn,
      checkpointer,
      partition_decode_once_fns,
      job_log_dir,
      eval_prng_seed,
      is_vars_replicated=True,
      train_prng_seed=train_prng_seed,
  )


def train_and_evaluate_spmd_model(
    task_p: tasks_lib.SingleTask.HParams,
    train_input_p: base_input.BaseInput.HParams,
    job_log_dir: epath.Path,
    checkpointer: _TrainingCheckpointer,
    eval_input_p: Sequence[base_input.BaseInput.HParams],
    decode_input_p: Sequence[base_input.BaseInput.HParams],
    early_stopping_fn: Optional[trainer_lib.EarlyStoppingFn] = None,
    enable_auto_sharding: bool = False,
    experiment_train_program: Optional[programs.BaseTrainProgram] = None,
) -> None:
  """Runs the training and evaluation loop with PJIT.

  Args:
    task_p: Params for task encapsulating the SPMD model.
    train_input_p: Params for the train data pipeline.
    job_log_dir: Directory for the job logs.
    checkpointer: Callbacks for checkpointing.
    eval_input_p: list of params for the eval input pipelines.
    decode_input_p: list of hparams for the decode input pipelines.
    early_stopping_fn: An optional callable object for reporting eval metrics
      and determining whether to early stop current training. The callable
      object has signature: (metrics_by_dataset, ckpt_step, is_final_ckpt) ->
      should_stop_early.
    enable_auto_sharding: Enables the XLA Auto SPMD partitioner.
    experiment_train_program: An embedded train_program that's constructed in
      experiment. If specified, the program will be used for train steps.
  """
  logging.info('Using SPMD sharding for model parallelism.')
  (
      task,
      partitioner,
      train_program,
      test_eval_programs,
      partitioned_train_state,
      total_num_params,
      prng_key,
      train_prng_seed,
      eval_prng_seed,
      early_stopping_fn,
  ) = _create_program_and_states(
      task_p,
      train_input_p,
      eval_input_p,
      job_log_dir,
      checkpointer,
      enable_auto_sharding,
      early_stopping_fn,
      experiment_train_program,
  )

  def partition_decode_once_fns(
      prng_key: jax.random.KeyArray,
      decode_input_ps: Sequence[base_input.BaseInput.HParams],
  ) -> Tuple[
      Callable[..., tuning_lib.DecodeMetrics],
      jax.random.KeyArray,
      Sequence[str],
  ]:
    assert decode_input_ps, 'decode_input_p must not be empty'
    prng_key, decode_key = jax.random.split(prng_key, 2)
    logging.info('decode prng_key: %s', decode_key)
    decode_key = partitioner.preprocess_prng_key(decode_key)

    padded_decode_input_ps = [
        trainer_lib.adjust_input_params_for_small_batch(
            input_p, partitioner.global_mesh
        )
        for input_p in decode_input_ps
    ]
    padded_decode_input_pipelines = [
        instantiate(input_p) for input_p in padded_decode_input_ps
    ]
    trainer_lib.check_unique_names(padded_decode_input_pipelines)
    _, decode_inputs_shape_dtype = trainer_lib.get_inputs_shape_dtype(
        padded_decode_input_ps[0]
    )

    # TODO(pax-dev): Support auto-sharding for decoder step.
    step_fn, is_eval = partitioning.get_step_fn(RunningMode.DECODE)
    assert is_eval
    decode_step_fn, decode_input_partition_spec = partitioner.partition(
        step_fn, decode_inputs_shape_dtype, is_eval
    )

    decode_once_fn = eval_lib.partition_decode_once_spmd_model(
        task,
        partitioner,
        task_p,
        padded_decode_input_pipelines,
        job_log_dir,
        decode_key,
        decode_step_fn,
        decode_input_partition_spec,
    )

    decode_input_names = [inp.name for inp in padded_decode_input_pipelines]
    return decode_once_fn, prng_key, decode_input_names

  _train_and_evaluate_common(
      task,
      partitioner,
      train_program,
      partitioned_train_state,
      prng_key,
      test_eval_programs,
      decode_input_p,
      total_num_params,
      early_stopping_fn,
      checkpointer,
      partition_decode_once_fns,
      job_log_dir,
      eval_prng_seed,
      is_vars_replicated=False,
      train_prng_seed=train_prng_seed,
  )


def _create_program_and_states(
    task_p: tasks_lib.SingleTask.HParams,
    train_input_p: base_input.BaseInput.HParams,
    eval_input_p: Sequence[base_input.BaseInput.HParams],
    job_log_dir: epath.Path,
    checkpointer: _TrainingCheckpointer,
    enable_auto_sharding: bool = False,
    early_stopping_fn: Optional[trainer_lib.EarlyStoppingFn] = None,
    experiment_train_program: Optional[programs.BaseTrainProgram] = None,
):
  reshard_inputs = checkpointer.checkpoint_type != CheckpointType.PERSISTENCE
  jax_task = instantiate(task_p)
  root_prng_key = jax.random.PRNGKey(task_p.train.random_seed)

  # Avoid creating the paritioner/input twice if the train program is defined.
  # TODO(laigd): move this to the BaseExecutor class.
  if experiment_train_program is not None:
    partitioner = experiment_train_program.partitioner
    train_input_pipeline = experiment_train_program.train_input
  else:
    # The partitioner only needs shape/dtype information of the prng key.
    # TODO(laigd): let the partitioner take ShapeDtypeStruct of prng key
    # instead.
    partitioner = partitioning.create_partitioner(
        jax_task,
        reshard_inputs=reshard_inputs,
        auto_sharding_mode=RunningMode.TRAIN if enable_auto_sharding else None,
    )
    train_input_p = partitioner.preprocess_input_params(train_input_p)
    train_input_pipeline = instantiate(train_input_p)
    partitioner.setup(
        jax_task,
        root_prng_key,
        train_inputs_shape_dtype=None,
        train_input_pipeline=train_input_pipeline,
        job_log_dir=job_log_dir,
    )
  train_state_metadata = partitioner.get_train_state_metadata()

  # JaxContext needed for shared layer lookup from global scope.
  with base_layer.JaxContext.new_context():
    # Dump out model meta info for debugging.
    trainer_lib.write_post_init_model_hparams_file(
        jax_task.model, train_state_metadata.var_weight_hparams, job_log_dir
    )

  # Restore TrainState from checkpoint or initialize it.
  train_input = (
      train_input_pipeline if task_p.train.enable_input_checkpointing else None
  )
  partitioned_train_state, total_num_params, root_prng_key = (
      checkpointer.get_model_states(
          partitioner, train_state_metadata, root_prng_key, train_input
      )
  )

  initial_global_step = int(
      py_utils.maybe_unreplicate_for_fully_replicated(
          partitioned_train_state.step
      )
  )
  logging.info('Model initial global_step=%d', initial_global_step)
  if not task_p.train.enable_input_checkpointing:
    train_input_pipeline = _maybe_update_latest_model_step(
        train_input_pipeline, train_input_p, initial_global_step
    )
  if experiment_train_program:
    logging.info('Using customized train program.')
    train_program = experiment_train_program
  else:
    train_program = programs.SingleTaskTrainProgram(
        jax_task, train_input_pipeline, partitioner
    )

  prng_key, train_prng_seed, eval_prng_seed = jax.random.split(root_prng_key, 3)
  logging.info('train prng seed: %s', train_prng_seed)
  logging.info('eval prng seed: %s', eval_prng_seed)
  train_prng_seed = partitioner.preprocess_prng_key(train_prng_seed)
  eval_prng_seed = partitioner.preprocess_prng_key(eval_prng_seed)

  if jax_task.early_stopping_fn is not None:
    if early_stopping_fn is None:
      early_stopping_fn = jax_task.early_stopping_fn
    else:
      raise ValueError(
          'early_stopping_fn is set in both task and '
          'train_and_evel function parameter.'
      )

  # Construct a list of Eval programs on test data.
  test_eval_programs = [
      programs.SingleTaskEvalProgram(jax_task, e_input_p, partitioner)
      for e_input_p in eval_input_p
  ]
  trainer_lib.check_unique_names(
      [eval_program.eval_input for eval_program in test_eval_programs]
  )

  return (
      jax_task,
      partitioner,
      train_program,
      test_eval_programs,
      partitioned_train_state,
      total_num_params,
      prng_key,
      train_prng_seed,
      eval_prng_seed,
      early_stopping_fn,
  )


def _train_and_evaluate_common(
    task: tasks_lib.SingleTask,
    partitioner: partitioning.Partitioner,
    train_program: programs.BaseTrainProgram,
    partitioned_train_state,
    prng_key,
    # TODO(hthu): Take a more generalized form of EvalProgram interface.
    test_eval_programs: Sequence[programs.SingleTaskEvalProgram],
    decode_input_p,
    total_num_params,
    early_stopping_fn,
    checkpointer,
    partition_decode_once_fns,
    job_log_dir,
    eval_prng_seed,
    is_vars_replicated,
    train_prng_seed,
):
  """Training loop code common to both pmap and spmd."""
  task_p = task.hparams
  train_p = task_p.train
  train_state_metadata = partitioner.get_train_state_metadata()
  train_input = (
      train_program.train_input if train_p.enable_input_checkpointing else None
  )

  if decode_input_p:
    decode_once_fn, prng_key, decode_input_names = partition_decode_once_fns(
        prng_key, decode_input_p
    )
  else:
    decode_input_names = []

  eval_input_names = [prg.eval_input.name for prg in test_eval_programs]

  logging.info('Training loop starting...')

  with _SummaryContextManager(
      job_log_dir,
      eval_input_names,
      decode_input_names,
      train_p.eval_skip_train,
  ) as (
      train_summary_writer,
      eval_summary_writer,
      eval_test_summary_writers,
      decode_summary_writers,
  ):
    # This only prints the view from the first host machine.
    summary_utils.write_model_structure(
        train_summary_writer,
        partitioned_train_state,
        is_vars_replicated=is_vars_replicated,
    )
    summary_utils.write_total_num_params(train_summary_writer, total_num_params)
    summary_utils.write_global_batch_size(
        train_summary_writer, train_program.train_unpadded_global_batch_size
    )

    # TODO(laigd): consider moving this into train program.
    train_summary_handler = summary_utils.SummaryHandler(
        train_summary_writer,
        train_p.summary_interval_steps,
        accumulate_interval_steps=train_p.summary_accumulate_interval_steps,
        log_interval_steps=_train_log_interval_steps(train_p),
        is_async=bool(train_p.device_sync_interval_steps),
        name='training',
    )
    eval_summary_handler = summary_utils.SummaryHandler(
        eval_summary_writer,
        train_p.summary_interval_steps,
        accumulate_interval_steps=train_p.summary_accumulate_interval_steps,
        name='eval',
    )

    step_i = int(
        py_utils.maybe_unreplicate_for_fully_replicated(
            partitioned_train_state.step
        )
    )
    train_program.setup(
        train_prng_seed,
        eval_prng_seed,
        step_i,
        train_summary_handler,
        eval_summary_handler,
    )

    # Start the train loop. Make sure all at the same step.
    py_utils.sync_global_devices(f'Start training loop from step: {step_i}')
    # Collect then freeze GC, so that GC in the training loop will not touch the
    # python objects used to initialize the model. Unfreeze at the end of the
    # loop.
    gc.collect()
    gc.freeze()
    while True:
      logging.debug('step=`%d`: Beginning', step_i)
      checkpointer.save_if_needed(
          step_i,
          partitioned_train_state,
          train_state_metadata.partition_specs,
          train_input,
      )

      if not train_program.should_run(partitioned_train_state, step_i):
        logging.info(
            (
                'Training loop completed (step (`%d`) greater than '
                'num_train_step (`%d`).'
            ),
            step_i,
            train_p.num_train_steps,
        )
        break

      program_output = train_program.run(partitioned_train_state, step_i)
      partitioned_train_state = program_output.state
      train_weighted_scalars = program_output.aux.weighted_scalars
      steps_per_sec = program_output.aux.steps_per_sec
      eval_train_metrics = program_output.aux.eval_train_metrics

      # While the eval ones below are post-model weight updates, hence the step
      # counter is incremented in between.
      step_i = program_output.aux.new_train_step

      eval_metrics: Optional[tuning_lib.EvalMetrics] = None
      # Run eval at regular step interval.
      if (
          train_p.eval_interval_steps
          and step_i % train_p.eval_interval_steps == 0
      ):
        logging.debug('  Starting eval_step().')
        eval_partitioned_train_state = programs.get_eval_train_state(
            task, partitioned_train_state
        )
        # If we have eval test then also evaluate on test.
        if test_eval_programs:
          logging.debug('  Performing eval_step() runs on test splits.')
          with py_utils.timeit() as eval_period:
            eval_metrics_list, eval_scoring_metrics_list, num_eval_steps = (
                eval_lib.run_eval_loop_over_test_splits(
                    test_eval_programs,
                    eval_partitioned_train_state,
                    eval_prng_seed,
                    eval_test_summary_writers,
                    step_i,
                    job_log_dir,
                )
            )
          eval_steps_per_sec = sum(num_eval_steps) / eval_period.elapsed
          eval_metrics = tuning_lib.EvalMetrics(
              metrics_list=eval_metrics_list,
              scoring_metrics_list=eval_scoring_metrics_list,
              steps_per_sec=eval_steps_per_sec,
              input_names=[
                  eval_program.eval_input.name
                  for eval_program in test_eval_programs
              ],
          )
          logging.debug(
              '  Completed eval_step() runs on test splits in %f seconds.',
              eval_period.elapsed,
          )

      decode_metrics: Optional[tuning_lib.DecodeMetrics] = None
      if (
          decode_input_p
          and train_p.decode_interval_steps
          and step_i % train_p.decode_interval_steps == 0
      ):
        if train_p.decode_use_ema_states:
          if not tasks_lib.has_ema(task_p):
            raise ValueError(
                'decode_use_ema_states is requested but the '
                'learner does not seem to have ema enabled'
            )
          decode_partitioned_train_state = tasks_lib.extract_ema(
              partitioned_train_state
          )
          logging.debug('  Performing decode_once_fn() with ema states.')
        else:
          decode_partitioned_train_state = partitioned_train_state
        decode_metrics = decode_once_fn(
            decode_partitioned_train_state, decode_summary_writers
        )

      logging.debug('step=`%d`: End', step_i - 1)

      if early_stopping_fn is not None:
        if tuning_lib.should_early_stop(
            early_stopping_fn,
            step_i,
            is_last_ckpt=tuning_lib.is_last_checkpoint(
                RunningMode.detect(
                    has_train_metrics=True,
                    has_eval_metrics=bool(eval_metrics),
                    has_decode_metrics=bool(decode_metrics),
                ),
                step_i,
                task_p.train.num_train_steps,
                task_p.train.eval_interval_steps,
                task_p.train.decode_interval_steps,
                task_p.train.save_interval_steps,
            ),
            train_weighted_scalars=train_weighted_scalars,
            eval_train_metrics=eval_train_metrics,
            eval_metrics=eval_metrics,
            decode_metrics=decode_metrics,
            train_steps_per_sec=steps_per_sec,
            num_params=total_num_params,
        ):
          logging.info(
              (
                  'Training loop is early stopped at step `%d` by the '
                  'tuner, while num_train_step is `%d`.'
              ),
              step_i,
              train_p.num_train_steps,
          )
          break
    gc.unfreeze()
    # Save checkpoint for the last step.
    checkpointer.save_final(
        step_i,
        partitioned_train_state=partitioned_train_state,
        train_state_pspecs=train_state_metadata.partition_specs,
        train_input_pipeline=train_input,
    )

    checkpointer.wait_until_finished()
    train_summary_handler.close()
    eval_summary_handler.close()
