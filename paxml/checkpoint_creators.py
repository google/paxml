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

"""Utilities to create training checkpointers."""

import datetime
import re
from typing import Any, Dict, Optional, Tuple

from absl import logging
from etils import epath
import jax
from jax import monitoring
import numpy as np
import orbax.checkpoint as ocp
from paxml import checkpoint_managers
from paxml import partitioning
from paxml import tasks_lib
from paxml import train_states
from paxml import trainer_lib
from praxis import base_input
from praxis import pax_fiddle
from praxis import py_utils
from praxis import pytypes

from paxml import checkpoints  # mapped to internal

CheckpointType = checkpoints.CheckpointType
Checkpointer = checkpoints.Checkpointer
FlaxCheckpointer = checkpoints.FlaxCheckpointer
FlaxCheckpointHandler = checkpoints.FlaxCheckpointHandler
PaxCheckpointHandler = checkpoints.PaxCheckpointHandler
# alias to internal checkpointer
BaseInputCheckpointHandler = checkpoints.BaseInputCheckpointHandler
PRNGKey = pytypes.PRNGKey
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


def _restore_from_external_checkpoint(
    path: epath.Path,
    checkpoint_handler: Optional[ocp.CheckpointHandler],
    train_state_metadata: trainer_lib.TrainStateMetadata,
    partitioner: partitioning.Partitioner,
    train_input_pipeline: Optional[base_input.BaseInput] = None,
    transformations: Optional[Dict[str, Any]] = None,
):
  """Restores a checkpoint from an external, possibly non-Pax, location."""
  if checkpoint_handler is None:
    raise ValueError(
        'Must provide CheckpointHandler to load external checkpoint.'
    )
  # TODO(b/278628399): Also support customized loading of train_input.
  del train_input_pipeline
  checkpointer = Checkpointer(checkpoint_handler)
  return checkpointer.restore(
      path,
      item=train_state_metadata,
      partitioner=partitioner,
      transformations=transformations,
  )


class _OrbaxPjitTrainingCheckpointer(checkpoints.TrainingCheckpointer):

  def __init__(
      self,
      checkpoint_manager: checkpoint_managers.OrbaxCheckpointManager,
      checkpoint_type: CheckpointType,
      enable_checkpoint_saving: bool = True,
      restore_transformations: Optional[Dict[str, Any]] = None,
      external_checkpoint_path: Optional[epath.Path] = None,
      external_checkpoint_handler: Optional[ocp.CheckpointHandler] = None,
  ):
    self.checkpoint_manager = checkpoint_manager
    self._checkpoint_type = checkpoint_type
    if checkpoint_type == CheckpointType.FLAX:
      raise ValueError('FLAX checkpointing not supported for pjit models.')
    self._enable_checkpoint_saving = enable_checkpoint_saving
    self._restore_transformations = restore_transformations

    self._external_checkpoint_path = external_checkpoint_path
    # TODO(b/278628399) Consider providing default implementation.
    self._external_checkpoint_handler = external_checkpoint_handler
    self._step_to_restore = self.checkpoint_manager.latest_step()

  @property
  def step_to_restore(self) -> Optional[int]:
    return self._step_to_restore

  def wait_until_finished(self):
    self.checkpoint_manager.wait_until_finished()

  def reached_preemption(self, step: int) -> bool:
    return self.checkpoint_manager.reached_preemption(step)

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
    if (
        self._checkpoint_type == CheckpointType.GDA
        or self._checkpoint_type == CheckpointType.PERSISTENCE
    ):
      restore_args = {
          'specs': train_state_pspecs,
          'mesh': global_mesh,
          'transforms': self._restore_transformations,
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
    with py_utils.timeit() as restore_period:
      if self._step_to_restore is None:
        if self._external_checkpoint_path is None:
          partitioned_train_state = None
        else:
          partitioned_train_state = _restore_from_external_checkpoint(
              self._external_checkpoint_path,
              self._external_checkpoint_handler,
              metadata,
              partitioner,
              train_input_pipeline=train_input_pipeline,
              transformations=self._restore_transformations,
          )
      else:
        partitioned_train_state = self._restore_with_args(
            self._step_to_restore,
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
            root_prng_key, partitioned_train_state, self.checkpoint_type
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


class _OrbaxPmapTrainingCheckpointer(checkpoints.TrainingCheckpointer):

  def __init__(
      self,
      job_log_dir: epath.Path,
      checkpoint_manager: checkpoint_managers.OrbaxCheckpointManager,
      checkpoint_type: CheckpointType,
      enable_checkpoint_saving: bool = True,
      restore_transformations: Optional[Dict[str, Any]] = None,
      external_checkpoint_path: Optional[epath.Path] = None,
      external_checkpoint_handler: Optional[ocp.CheckpointHandler] = None,
  ):
    self.job_log_dir = job_log_dir
    self.checkpoint_dir = _checkpoint_dir(job_log_dir)
    self.checkpoint_manager = checkpoint_manager
    self._checkpoint_type = checkpoint_type
    self._enable_checkpoint_saving = enable_checkpoint_saving
    self._restore_transformations = restore_transformations
    self._external_checkpoint_path = external_checkpoint_path
    self._external_checkpoint_handler = external_checkpoint_handler
    self._step_to_restore = self.checkpoint_manager.latest_step()

  @property
  def step_to_restore(self) -> Optional[int]:
    return self._step_to_restore

  def wait_until_finished(self):
    self.checkpoint_manager.wait_until_finished()

  def reached_preemption(self, step: int) -> bool:
    return self.checkpoint_manager.reached_preemption(step)

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
      if self._step_to_restore is None:
        if self._external_checkpoint_path is None:
          train_state = None
        else:
          train_state = _restore_from_external_checkpoint(
              self._external_checkpoint_path,
              self._external_checkpoint_handler,
              metadata,
              partitioner,
              train_input_pipeline=train_input_pipeline,
              transformations=self._restore_transformations,
          )
      else:
        train_state = self._restore_with_args(
            self._step_to_restore,
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


@py_utils.benchmark('[PAX STATUS]: ', first_n=2)
def _create_checkpointer(
    task_p: pax_fiddle.Config[tasks_lib.SingleTask],
    job_log_dir: epath.Path,
    checkpoint_type: CheckpointType,
    todelete_subdir: Optional[str],
    train_input_p: Optional[pax_fiddle.Config[base_input.BaseInput]] = None,
    enable_async_checkpointing: bool = True,
    enable_checkpoint_saving: bool = True,
    enforce_restore_shape_check: bool = False,
    maybe_use_persistence_checkpointing: bool = False,
    tensorstore_use_ocdbt: bool = False,
) -> checkpoints.TrainingCheckpointer:
  """Creates a checkpoint manager."""
  logging.info('[PAX STATUS]: Creating checkpointer.')
  checkpoint_dir = _make_checkpoint_dir(job_log_dir)
  train_p = task_p.train
  max_to_keep = train_p.save_max_to_keep
  save_interval_steps = train_p.save_interval_steps
  keep_interval_timedelta = _parse_duration(train_p.save_keep_interval_duration)
  restore_transformations = train_p.restore_transformations

  checkpoints.reregister_type_handlers(
      tensorstore_metadata_key=train_p.tensorstore_metadata_key,
  )
  options = checkpoint_managers.CheckpointManagerOptions(
      max_to_keep=max_to_keep,
      save_interval_steps=save_interval_steps,
      keep_time_interval=keep_interval_timedelta,
      todelete_subdir=todelete_subdir,
      cleanup_tmp_directories=True,
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
        restore_transformations=restore_transformations,
        external_checkpoint_path=task_p.train.external_checkpoint_path,
        external_checkpoint_handler=task_p.train.external_checkpoint_handler,
    )
  else:
    checkpointer = _OrbaxPmapTrainingCheckpointer(
        job_log_dir,
        checkpoint_manager,
        checkpoint_type,
        enable_checkpoint_saving=enable_checkpoint_saving,
        restore_transformations=restore_transformations,
        external_checkpoint_path=task_p.train.external_checkpoint_path,
        external_checkpoint_handler=task_p.train.external_checkpoint_handler,
    )

  return checkpointer
