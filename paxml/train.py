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
import re
import typing
from typing import Any, Dict, Optional, Tuple, Type

from absl import logging
from etils import epath
import jax
from jax import monitoring
import jax.numpy as jnp
import numpy as np
from paxml import base_experiment
from paxml import checkpoint_managers
from paxml import checkpoint_types
from paxml import executors
from paxml import experiment_utils
from paxml import partitioning
from paxml import programs
from paxml import tasks_lib
from paxml import train_states
from paxml import trainer_lib
from praxis import base_hyperparams
from praxis import base_input
from praxis import py_utils
from praxis import pytypes
import tensorflow.compat.v2 as tf

from paxml import checkpoints  # mapped to internal

Checkpointer = checkpoints.Checkpointer
CheckpointType = checkpoints.CheckpointType
instantiate = base_hyperparams.instantiate
FlaxCheckpointer = checkpoints.FlaxCheckpointer
FlaxCheckpointHandler = checkpoints.FlaxCheckpointHandler
PaxCheckpointHandler = checkpoints.PaxCheckpointHandler
# alias to internal checkpointer
BaseInputCheckpointHandler = checkpoints.BaseInputCheckpointHandler
PRNGKey = pytypes.PRNGKey
RunningMode = trainer_lib.RunningMode
SummaryWriter = tf.summary.SummaryWriter
TrainState = train_states.TrainState
TrainStateProvenance = train_states.TrainStateProvenance

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
      train_state_unpadded_shape_dtype_struct,
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
      train_state_unpadded_shape_dtype_struct,
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
  ) -> Tuple[TrainState, Optional[TrainStateProvenance], int, PRNGKey]:
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
      ocdbt_coordinator_server: Optional[Any] = None,
      restore_transformations: Optional[Dict[str, Any]] = None,
  ):
    self.checkpoint_manager = checkpoint_manager
    self._checkpoint_type = checkpoint_type
    if checkpoint_type == CheckpointType.FLAX:
      raise ValueError('FLAX checkpointing not supported for pjit models.')
    self._enable_checkpoint_saving = enable_checkpoint_saving
    self._ocdbt_coordinator_server = ocdbt_coordinator_server
    self._restore_transformations = restore_transformations

  def wait_until_finished(self):
    self.checkpoint_manager.wait_until_finished()

  def _save_with_args(
      self,
      step_i: int,
      *,
      partitioned_train_state: Any,
      train_state_unpadded_shape_dtype_struct: Any = None,
      train_input_pipeline: Optional[base_input.BaseInput] = None,
      force: Optional[bool] = False,
  ):
    if not self._enable_checkpoint_saving:
      return
    with py_utils.timeit() as save_period:
      self.checkpoint_manager.save(
          step_i,
          partitioned_train_state,
          train_state_unpadded_shape_dtype_struct,
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
      train_state_unpadded_shape_dtype_struct,
      global_mesh,
      train_state_pspecs,
      train_input_pipeline,
  ):
    restore_args = {}
    if self._checkpoint_type == CheckpointType.GDA:
      restore_args = {
          'specs': train_state_pspecs,
          'mesh': global_mesh,
          'transforms': self._restore_transformations,
      }
    elif self._checkpoint_type == CheckpointType.PERSISTENCE:
      restore_args = {
          'state_specs': train_state_pspecs,
          'global_mesh': global_mesh,
      }
    return self.checkpoint_manager.restore(
        step_i,
        train_state_global_shapes,
        train_state_unpadded_shape_dtype_struct,
        train_input_pipeline,
        restore_kwargs=restore_args,
    )

  def save_final(
      self,
      step_i,
      *,
      partitioned_train_state,
      train_state_unpadded_shape_dtype_struct=None,
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
          train_state_unpadded_shape_dtype_struct=train_state_unpadded_shape_dtype_struct,
          train_input_pipeline=train_input_pipeline,
          force=True,
      )

  def save_if_needed(
      self,
      step_i,
      partitioned_train_state,
      train_state_unpadded_shape_dtype_struct=None,
      train_state_pspecs=None,
      train_input_pipeline=None,
  ):
    del train_state_pspecs
    if not self.checkpoint_manager.should_save(step_i):
      return
    self._save_with_args(
        step_i,
        partitioned_train_state=partitioned_train_state,
        train_state_unpadded_shape_dtype_struct=train_state_unpadded_shape_dtype_struct,
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
  ) -> Tuple[TrainState, Optional[TrainStateProvenance], int, PRNGKey]:
    step = self.checkpoint_manager.latest_step()
    if step is None:
      partitioned_train_state = None
    else:
      with py_utils.timeit() as restore_period:
        partitioned_train_state = self._restore_with_args(
            step,
            metadata.padded_global_shapes,
            metadata.unpadded_global_shapes,
            partitioner.global_mesh,
            metadata.partition_specs,
            train_input_pipeline,
        )
      monitoring.record_event_duration_secs(
          _READ_CHECKPOINT_EVENT, restore_period.elapsed
      )

    root_prng_key, partitioned_train_state, train_state_provenance = (
        partitioner.initialize_prng_key_and_train_state(
            root_prng_key,
            partitioned_train_state,
            self.checkpoint_type,
        )
    )

    total_num_params = py_utils.total_num_vars(partitioned_train_state.mdl_vars)
    return (
        partitioned_train_state,
        train_state_provenance,
        total_num_params,
        root_prng_key,
    )

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
      ocdbt_coordinator_server: Optional[Any] = None,
      restore_transformations: Optional[Dict[str, Any]] = None,
  ):
    self.job_log_dir = job_log_dir
    self.checkpoint_dir = _checkpoint_dir(job_log_dir)
    self.checkpoint_manager = checkpoint_manager
    self._checkpoint_type = checkpoint_type
    self._enable_checkpoint_saving = enable_checkpoint_saving
    self._ocdbt_coordinator_server = ocdbt_coordinator_server
    self._restore_transformations = restore_transformations

  def wait_until_finished(self):
    self.checkpoint_manager.wait_until_finished()

  def _restore_with_args(
      self,
      step_i: int,
      train_state_global_shapes: Optional[TrainState],
      train_state_unpadded_shape_dtype_struct: Optional[TrainState],
      train_input_pipeline: Optional[base_input.BaseInput],
  ):
    """Restore using CheckpointManager, setting up additional args."""
    restore_args = None
    if py_utils.pmap_use_tensorstore():
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
        train_state_unpadded_shape_dtype_struct=train_state_unpadded_shape_dtype_struct,
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
  ) -> Tuple[TrainState, Optional[TrainStateProvenance], int, PRNGKey]:
    train_state_global_shapes = metadata.unpadded_global_shapes
    with py_utils.timeit() as restore_period:
      step = self.checkpoint_manager.latest_step()
      if step is None:
        train_state = None
      else:
        train_state = self._restore_with_args(
            step,
            train_state_global_shapes,
            metadata.unpadded_global_shapes,
            train_input_pipeline,
        )
    monitoring.record_event_duration_secs(
        _READ_CHECKPOINT_EVENT, restore_period.elapsed
    )

    # TODO(laigd): move the logic below outside of get_model_states.
    root_prng_key, replicated_train_state, train_state_provenance = (
        partitioner.initialize_prng_key_and_train_state(
            root_prng_key, train_state, self.checkpoint_type
        )
    )

    total_num_params = py_utils.total_num_vars(replicated_train_state.mdl_vars)
    assert total_num_params % jax.local_device_count() == 0
    total_num_params = total_num_params // jax.local_device_count()
    return (
        replicated_train_state,
        train_state_provenance,
        total_num_params,
        root_prng_key,
    )

  def _save_with_args(
      self,
      step_i: int,
      *,
      train_state: Any,
      train_state_unpadded_shape_dtype_struct: Any,
      train_input_pipeline: Optional[base_input.BaseInput] = None,
      force: Optional[bool] = False,
  ):
    self.checkpoint_manager.save(
        step_i,
        train_state,
        train_state_unpadded_shape_dtype_struct,
        train_input_pipeline,
        force=force,
    )

  def _save(
      self,
      step_i,
      partitioned_train_state,
      train_state_unpadded_shape_dtype_struct,
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
            train_state_unpadded_shape_dtype_struct=(
                train_state_unpadded_shape_dtype_struct
            ),
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
            train_state_unpadded_shape_dtype_struct=(
                train_state_unpadded_shape_dtype_struct
            ),
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
      train_state_unpadded_shape_dtype_struct,
      train_state_pspecs,
      train_input_pipeline=None,
  ):
    if not self.checkpoint_manager.should_save(step_i):
      return
    self._save(
        step_i,
        partitioned_train_state,
        train_state_unpadded_shape_dtype_struct,
        train_input_pipeline,
    )
    self.checkpoint_manager.check_for_errors()

  def save_final(
      self,
      step_i,
      *,
      partitioned_train_state,
      train_state_unpadded_shape_dtype_struct,
      train_state_pspecs,
      train_input_pipeline: Optional[base_input.BaseInput] = None,
  ):
    latest_step = self.checkpoint_manager.latest_step()
    if latest_step is None or latest_step < step_i:
      self._save(
          step_i,
          partitioned_train_state,
          train_state_unpadded_shape_dtype_struct,
          train_input_pipeline,
          is_final=True,
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
    tensorstore_use_ocdbt: bool = False,
) -> _TrainingCheckpointer:
  """Creates a checkpoint manager."""
  checkpoint_dir = _make_checkpoint_dir(job_log_dir)
  train_p = task_p.train
  max_to_keep = train_p.save_max_to_keep
  save_interval_steps = train_p.save_interval_steps
  keep_interval_timedelta = _parse_duration(train_p.save_keep_interval_duration)
  restore_transformations = train_p.restore_transformations

  ocdbt_coordinator_server = checkpoints.reregister_type_handlers(
      tensorstore_metadata_key=train_p.tensorstore_metadata_key,
      tensorstore_use_ocdbt=tensorstore_use_ocdbt,
  )
  options = checkpoint_managers.CheckpointManagerOptions(
      max_to_keep=max_to_keep,
      save_interval_steps=save_interval_steps,
      keep_time_interval=keep_interval_timedelta,
      todelete_subdir=todelete_subdir,
  )

  if checkpoint_type == CheckpointType.FLAX:
    if tensorstore_use_ocdbt:
      checkpointer = Checkpointer(
          PaxCheckpointHandler(
              enforce_restore_shape_check=enforce_restore_shape_check,
              use_ocdbt=tensorstore_use_ocdbt,
          )
      )
    else:
      checkpointer = FlaxCheckpointer(FlaxCheckpointHandler())
  elif enable_async_checkpointing:
    if maybe_use_persistence_checkpointing:
      raise NotImplementedError('Persistence checkpointer not supported.')
    else:
      checkpointer = checkpoints.AsyncCheckpointer(
          checkpoints.PaxCheckpointHandler(
              enforce_restore_shape_check=enforce_restore_shape_check,
              use_ocdbt=tensorstore_use_ocdbt,
          ),
          timeout_secs=600,
      )
  else:
    if checkpoint_type == CheckpointType.GDA:
      checkpointer = Checkpointer(
          PaxCheckpointHandler(
              enforce_restore_shape_check=enforce_restore_shape_check,
              use_ocdbt=tensorstore_use_ocdbt,
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

  if task_p.train.enable_input_checkpointing:
    train_input_p.input_checkpointing_enabled = True
  checkpoint_manager = checkpoint_managers.OrbaxCheckpointManager(
      checkpoint_dir,
      checkpointer,
      train_input_checkpointer=train_input_checkpointer,
      options=options,
      checkpoint_type=checkpoint_type,
      tensorstore_use_ocdbt=tensorstore_use_ocdbt,
  )

  if task_p.model.ici_mesh_shape is not None:
    checkpointer = _OrbaxPjitTrainingCheckpointer(
        checkpoint_manager,
        checkpoint_type,
        enable_checkpoint_saving=enable_checkpoint_saving,
        ocdbt_coordinator_server=ocdbt_coordinator_server,
        restore_transformations=restore_transformations,
    )
  else:
    checkpointer = _OrbaxPmapTrainingCheckpointer(
        job_log_dir,
        checkpoint_manager,
        checkpoint_type,
        enable_checkpoint_saving=enable_checkpoint_saving,
        ocdbt_coordinator_server=ocdbt_coordinator_server,
        restore_transformations=restore_transformations,
    )

  return checkpointer


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
        hparams_file.write(base_hyperparams.nested_struct_to_text(dataset))
        hparams_file.write('\n\n')
      for decoder_dataset in model_config.decoder_datasets():
        hparams_file.write('decoder dataset hparams\n')
        hparams_file.write(
            base_hyperparams.nested_struct_to_text(decoder_dataset)
        )
        hparams_file.write('\n\n')
      hparams_file.write(
          base_hyperparams.nested_struct_to_text(model_config.task()))


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
    tensorstore_use_ocdbt: bool = False,
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
    tensorstore_use_ocdbt: Uses OCDBT format for saving new checkpoints.
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
  for line in base_hyperparams.nested_struct_to_text(
      train_input_p
  ).splitlines():  # pytype: disable=attribute-error
    logging.info('  %s', line)
  logging.info('task_p:')
  for line in base_hyperparams.nested_struct_to_text(task_p).splitlines():  # pytype: disable=attribute-error
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

  checkpoint_type = checkpoint_types.retrieve_checkpoint_type(
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
      tensorstore_use_ocdbt=tensorstore_use_ocdbt,
  )
  if not enable_checkpoint_saving:
    logging.info(
        'Checkpointing is disabled and no checkpoint will be saved to disk.'
    )

  # Creates the task.
  jax_task = instantiate(task_p)
  if jax_task.early_stopping_fn is not None:
    if early_stopping_fn is None:
      early_stopping_fn = jax_task.early_stopping_fn
    else:
      raise ValueError(
          'early_stopping_fn is set in both task and '
          'train_and_evel function parameter.'
      )

  # Creates the partitioner, which will be set up later.
  partitioner = experiment_config.partitioner()
  if not partitioner:
    reshard_inputs = checkpointer.checkpoint_type != CheckpointType.PERSISTENCE
    partitioner = partitioning.create_partitioner(
        jax_task,
        reshard_inputs=reshard_inputs,
        auto_sharding_mode=RunningMode.TRAIN if enable_auto_sharding else None,
    )

  # Creates the train and eval programs.
  train_program = experiment_config.train_program()
  # TODO(laigd): make eval programs configurable.
  eval_programs = [
      programs.SingleTaskEvalProgram(jax_task, input_p, partitioner)
      for input_p in eval_input_p
  ]
  trainer_lib.check_unique_names([prog.eval_input for prog in eval_programs])

  # Creates the executor and run the training pipeline.
  executor = experiment_config.executor()
  if not executor:
    executor = executors.DefaultExecutor()
  with partitioner.global_mesh or contextlib.nullcontext():
    executor.setup(
        jax_task,
        job_log_dir,
        checkpointer,
        partitioner,
        train_input_p,
        decode_input_p,
        train_program,
        eval_programs,
        early_stopping_fn,
    )
    executor.start()
