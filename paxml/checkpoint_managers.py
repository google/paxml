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

"""Module to manage checkpoint metadata and automatic checkpoint deletion."""

import dataclasses
import typing
from typing import Any, Mapping, Optional, Sequence, Union

from etils import epath
import jax
import orbax.checkpoint
from paxml import checkpoint_pb2
from paxml import checkpoints
from praxis import py_utils
import tensorflow.compat.v2 as tf

from paxml import preemption  # mapped to internal


CheckpointType = checkpoint_pb2.CheckpointType

CHECKPOINT_PREFIX = 'checkpoint_'
STATE_ITEM_NAME = checkpoints.STATE_ITEM_NAME
METADATA_ITEM_NAME = checkpoints.METADATA_ITEM_NAME

_SUPPORTED_ITEMS = frozenset({STATE_ITEM_NAME, METADATA_ITEM_NAME})


def _get_checkpoint_version(
    step: int, checkpoint_type: CheckpointType, directory: epath.Path
) -> float:
  """Gets checkpoint version from saved metadata."""
  checkpoint_step_dir = checkpoints.make_checkpoint_step_dir(
      directory, step, checkpoint_type=checkpoint_type
  )
  version = 0.
  # Necessary because some checkpoints do not conform to Orbax directory
  # structure. Could rely exclusively on actual version if all checkpoints
  # conformed.
  if checkpoints.metadata_exists(checkpoint_step_dir):
    version = checkpoints.restore_metadata(checkpoint_step_dir)[
        checkpoints.get_version_key()
    ]
  return version


def _update_args_with_version(item_kwargs, version):
  kwargs = {STATE_ITEM_NAME: {checkpoints.get_version_key(): version}}
  if item_kwargs is not None:
    kwargs[STATE_ITEM_NAME].update(item_kwargs)
  return kwargs


def _create_items_dict_with_metadata(item, version):
  items = {
      STATE_ITEM_NAME: item,
  }
  if version > 0:
    items.update(
        {METADATA_ITEM_NAME: checkpoints.make_metadata(version=version)}
    )
  return items


@dataclasses.dataclass
class CheckpointManagerOptions(orbax.checkpoint.CheckpointManagerOptions):
  """Options for constructing OrbaxCheckpointManager.

  See superclass.

  Attributes:
    todelete_subdir: If set, checkpoints to be deleted will be only renamed into
      a subdirectory with the provided string. Otherwise, they will be directly
      deleted from the file system. Useful if checkpoint deletion is time
      consuming. By default, delete the checkpoint assets.
  """
  todelete_subdir: Optional[str] = None


class _CheckpointManagerImpl(orbax.checkpoint.CheckpointManager):
  """Provides Pax-specific logic for orbax.checkpoint.CheckpointManager.

  Pax only supports a single checkpointable item (TrainState) and checkpoints
  are saved under a different folder name in a flat manner (no per-item
  sub-directories).

  Additionally, Pax supports extra options provided via CheckpointManagerOptions
  (see above).

  An instance of this class can be created on several JAX processes.
  All public APIs may be called by all processes.
  """

  def __init__(
      self,
      directory: epath.PathLike,
      *args,
      checkpoint_type: CheckpointType = CheckpointType.CHECKPOINT_UNSPECIFIED,
      **kwargs,
  ):
    if checkpoint_type == CheckpointType.CHECKPOINT_UNSPECIFIED:
      raise ValueError('Must specify checkpoint type.')
    self._checkpoint_type = checkpoint_type

    self._version = checkpoints.get_version()
    # Check for existing checkpoints and retrieve version information. The
    # specific version may impact the checkpoint format, so it must be known in
    # advance of any operations.
    self._directory = epath.Path(directory)
    if self._directory.exists():
      steps = self.all_steps(read=True)
      if steps:
        versions = [
            _get_checkpoint_version(s, self._checkpoint_type, self._directory)
            for s in steps
        ]
        if not all(v == versions[0] for v in versions):
          raise ValueError('Expected all checkpoints to have the same version.')
        self._version = versions[0]

    super().__init__(directory, *args, **kwargs)
    # Set to 1 if not provided or set to 0.
    self._options.save_interval_steps = self._options.save_interval_steps or 1

  @property
  def version(self) -> float:
    return self._version

  def _checkpoint_name(self, step: int) -> str:
    return checkpoints.checkpoint_name(
        step, checkpoint_type=self._checkpoint_type
    )

  def should_save(self, step: int) -> bool:
    """Indicates whether there is a need to save a checkpoint."""
    # Whether to save an on-demand checkpoint due to preemption
    if preemption.reached_preemption_sync_point(step):
      return True
    last_checkpoint_step = (
        self._last_checkpoint.step if self._last_checkpoint else None)
    # Ensure current step is between the last step and next step (accounting for
    # save interval). The `last_checkpoint_step` may not be initialized, in
    # which case we should save. Otherwise, step must fall on the specified
    # save interval. This condition accounts for the possibility of saving
    # on preemption, in which case we want to maintain the same save period as
    # if preemption had not happened.
    return last_checkpoint_step is None or (
        last_checkpoint_step < step and
        step % self._options.save_interval_steps == 0)

  def _get_save_directory(self,
                          step: int,
                          directory: epath.Path,
                          key_name: Optional[str] = None) -> epath.Path:
    """Returns the standardized path to a save directory for a single item."""
    step_dir = checkpoints.make_checkpoint_step_dir(
        directory, step, checkpoint_type=self._checkpoint_type
    )
    if self._version < 1 or key_name is None:
      return step_dir
    return step_dir / key_name

  def _cleanup_tmp_directories(self):
    if py_utils.is_mock_tpu_backend():
      return
    super()._cleanup_tmp_directories()

  def _delete_directory(self, step: int):
    if jax.process_index() != 0:
      return
    options = typing.cast(CheckpointManagerOptions, self._options)
    todelete_subdir = options.todelete_subdir
    checkpoint_name = self._checkpoint_name(step)

    if todelete_subdir:
      rename_dir = self.directory / todelete_subdir
      if not rename_dir.exists():
        rename_dir.mkdir(parents=True)
      src = self.directory / checkpoint_name
      dst = rename_dir / checkpoint_name
      # TODO(pax-team): Check if dst already exists?
      tf.io.gfile.rename(src, dst)
    else:
      super()._delete_directory(step)

  def structure(self) -> Union[Any, Mapping[str, Any]]:
    if self._checkpoint_type == CheckpointType.CHECKPOINT_FLAX:
      raise ValueError('`structure` not supported for Flax format checkpoints.')
    return super().structure()


class OrbaxCheckpointManager:
  """Wrapper class for overridden _CheckpointManagerImpl."""

  def __init__(
      self,
      directory: epath.Path,
      checkpointer: orbax.checkpoint.AbstractCheckpointer,
      options: Optional[CheckpointManagerOptions] = None,
      checkpoint_type: CheckpointType = CheckpointType.CHECKPOINT_UNSPECIFIED,
  ):
    checkpointers = {
        checkpoints.STATE_ITEM_NAME: checkpointer,
        METADATA_ITEM_NAME: orbax.checkpoint.Checkpointer(
            orbax.checkpoint.JsonCheckpointHandler()
        ),
    }
    self._manager = _CheckpointManagerImpl(
        directory,
        checkpointers,
        options=options,
        checkpoint_type=checkpoint_type,
    )

  @property
  def version(self) -> float:
    return self._manager.version

  @property
  def directory(self) -> epath.Path:
    return self._manager.directory

  def all_steps(self) -> Sequence[int]:
    return self._manager.all_steps()

  def latest_step(self) -> Optional[int]:
    return self._manager.latest_step()

  def check_for_errors(self):
    self._manager.check_for_errors()

  def wait_until_finished(self):
    self._manager.wait_until_finished()

  def should_save(self, step: int) -> bool:
    return self._manager.should_save(step)

  def save(
      self,
      step: int,
      train_state: Any,
      force: Optional[bool] = False,
  ) -> bool:
    save_kwargs = _update_args_with_version(None, self.version)
    items = _create_items_dict_with_metadata(train_state, self.version)
    return self._manager.save(step, items, save_kwargs=save_kwargs, force=force)

  def restore(
      self,
      step: int,
      train_state: Any,
      restore_kwargs: Optional[Any] = None,
  ) -> Union[Any, Mapping[str, Any]]:
    """See superclass documentation."""
    # Propagate version to CheckpointHandler.
    restore_kwargs = _update_args_with_version(restore_kwargs, self.version)
    items = _create_items_dict_with_metadata(train_state, self.version)
    return self._manager.restore(
        step, items=items, restore_kwargs=restore_kwargs
    )[STATE_ITEM_NAME]
