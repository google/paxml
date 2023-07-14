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

"""Shared checkpointing utility functions."""

import re
from typing import Any, Optional

from absl import logging
from etils import epath
from jax.experimental import multihost_utils
import numpy as np
import orbax.checkpoint as ocp
from paxml import checkpoint_types
from paxml import checkpoint_version
from praxis import pytypes


_CHECKPOINT_PREFIX = 'checkpoint'
CHECKPOINT_PREFIX = f'{_CHECKPOINT_PREFIX}_'
_STEP_FORMAT_FIXED_LENGTH = 8
STATE_ITEM_NAME = 'state'
INPUT_ITEM_NAME = 'train_input'
TMP_PREFIX = 'tmp_'
CHECKPOINT_PATTERN_RE = re.compile(rf'({CHECKPOINT_PREFIX})?[\d]+$')
TMP_CHECKPOINT_PATTERN_RE = re.compile(
    rf'{TMP_PREFIX}[\d]+.{CHECKPOINT_PREFIX}[\d]+$'
)
# Large value to disable flax-specific checkpoint management.
_MAX_CHECKPOINT_FLAX = 1000000
get_version_key = checkpoint_version.get_version_key
get_version = checkpoint_version.get_version

CheckpointType = checkpoint_types.CheckpointType
retrieve_checkpoint_type = checkpoint_types.retrieve_checkpoint_type

JTensorOrPartitionSpec = pytypes.JTensorOrPartitionSpec
PyTree = Any
AsyncCheckpointer = ocp.AsyncCheckpointer
Checkpointer = ocp.Checkpointer
COMMIT_SUCCESS_FILE = 'commit_success.txt'


def checkpoint_prefix(
    checkpoint_type: CheckpointType = CheckpointType.UNSPECIFIED,
) -> Optional[str]:
  """Checkpoint prefix, or None if no prefix is applied.

  The return type is optional to future-proof against instances where the
  prefix may be None.

  Args:
    checkpoint_type: CheckpointType.

  Returns:
    prefix or None if no prefix is applied.
  """
  del checkpoint_type
  return _CHECKPOINT_PREFIX


def checkpoint_name_fixed_length(
    checkpoint_type: CheckpointType = CheckpointType.UNSPECIFIED,
) -> Optional[int]:
  """Length of the fixed width step format, or None if not used."""
  return (
      None
      if checkpoint_type == CheckpointType.FLAX
      else _STEP_FORMAT_FIXED_LENGTH
  )


def is_checkpoint_asset(x: epath.Path) -> bool:
  """Determines whether path is a checkpoint."""
  return bool(CHECKPOINT_PATTERN_RE.match(x.name))


def is_tmp_checkpoint_asset(x: epath.Path) -> bool:
  """Determines whether a checkpoint is temporary."""
  # Would only match v0.0 checkpoints, without state/metadata subdirs.
  # This case should trigger very rarely.
  if bool(TMP_CHECKPOINT_PATTERN_RE.match(x.name)):
    return True
  # Very old format Flax checkpoint.
  if x.is_file():
    return False
  return ocp.utils.is_tmp_checkpoint(x)


def checkpoint_name(
    step: int,
    checkpoint_type: CheckpointType = CheckpointType.UNSPECIFIED,
    use_digit_step_subdirectory: bool = False,
) -> str:
  if use_digit_step_subdirectory:
    return str(step)
  if checkpoint_type == CheckpointType.FLAX:
    return f'{CHECKPOINT_PREFIX}{step}'
  else:
    return f'{CHECKPOINT_PREFIX}{step:08d}'


def make_checkpoint_step_dir(
    checkpoint_dir: epath.Path,
    step: int,
    checkpoint_type: CheckpointType = CheckpointType.UNSPECIFIED,
    use_digit_step_subdirectory: bool = False,
) -> epath.Path:
  """Returns a checkpoint step directory."""
  return checkpoint_dir / checkpoint_name(
      step,
      checkpoint_type=checkpoint_type,
      use_digit_step_subdirectory=use_digit_step_subdirectory,
  )


def get_step_from_checkpoint_asset(checkpoint_dir: epath.PathLike) -> int:
  checkpoint_dir = epath.Path(checkpoint_dir)
  # For supporting digit step-like subdirectories.
  if checkpoint_dir.name.isdigit():
    return int(checkpoint_dir.name)
  if is_tmp_checkpoint_asset(checkpoint_dir):
    return int(checkpoint_dir.suffix[len(CHECKPOINT_PREFIX) :])
  return int(checkpoint_dir.stem[len(CHECKPOINT_PREFIX) :])


def latest_checkpoint_if_exists(
    checkpoint_dir: epath.PathLike,
) -> Optional[epath.Path]:
  """Gets the path to the latest checkpoint if any.

  Use this method instead of latest_checkpoint() if you want to handle the case
  where no checkpoint exists (e.g., the caller waits for a checkpoint).

  Args:
    checkpoint_dir: The base directory from where to retrieve checkpoints.

  Returns:
    Path to the latest checkpoint or None if there is no checkpoint.
  """
  checkpoint_dir = epath.Path(checkpoint_dir)
  if not checkpoint_dir.exists():
    logging.info('Checkpoint dir \'%s\' does not exist.', checkpoint_dir)
    return None
  checkpoint_assets = [
      v
      for v in checkpoint_dir.iterdir()
      if is_checkpoint_asset(v) and not is_tmp_checkpoint_asset(v)
  ]
  if not checkpoint_assets:
    logging.info(
        'No non-temporary checkpoints found in dir: \'%s\'', checkpoint_dir)
    return None
  checkpoint_assets = sorted(
      checkpoint_assets, key=get_step_from_checkpoint_asset
  )
  return checkpoint_dir / checkpoint_assets[-1]


def latest_checkpoint(checkpoint_dir: epath.PathLike) -> epath.Path:
  """Gets the path to the latest checkpoint.

  Use this method instead of latest_checkpoint_if_exists() if checkpoint
  existence is expected.

  Args:
    checkpoint_dir: The base directory from where to retrieve checkpoints.

  Returns:
    Path to the latest checkpoint.

  Raises:
    ValueError: checkpoint_dir does not exist or no checkpoint files were found
    in it.
  """
  checkpoint_dir = epath.Path(checkpoint_dir)
  path = latest_checkpoint_if_exists(checkpoint_dir)
  if path is None:
    _raise_checkpoint_missing_error(checkpoint_dir)
  return path


def retrieve_latest_checkpoint_step_if_exists(
    checkpoint_dir: epath.Path,
) -> Optional[int]:
  """Retrieves the latest checkpoint step within the given directory if any.

  Use this method instead of retrieve_latest_checkpoint_step() if you want to
  handle the case where no checkpoint exists (e.g., the caller waits for a
  checkpoint). Note that this broadcasts the checkpoint step from host 0 to
  ensure that all processes get the exact same checkpoint step.

  Args:
    checkpoint_dir: The base directory from where to retrieve checkpoints.

  Returns:
    The latest checkpoint step as an integer or None if no checkpoint is found.
  """
  if not checkpoint_dir.exists():
    logging.info('Checkpoint dir \'%s\' does not exist.', checkpoint_dir)
    checkpoint_step = -1
  else:
    latest_checkpoint_path = latest_checkpoint_if_exists(checkpoint_dir)
    if latest_checkpoint_path is None:
      checkpoint_step = -1
    else:
      checkpoint_step = get_step_from_checkpoint_asset(latest_checkpoint_path)
      logging.info('Latest checkpoint step is %d', checkpoint_step)
  np_checkpoint_step = multihost_utils.broadcast_one_to_all(
      np.array(checkpoint_step)
  )
  multihost_utils.assert_equal(
      np_checkpoint_step, "checkpoint_steps across hosts don't match."
  )
  step = int(np_checkpoint_step.item())
  if step == -1:
    return None
  return step


def retrieve_latest_checkpoint_step(
    checkpoint_dir: epath.Path,
) -> int:
  """Returns the latest checkpoint step within the given directory.

  Use this method instead of retrieve_latest_checkpoint_step_if_exists() if
  checkpoint existence is expected. Note that this broadcasts the checkpoint
  step from host 0 to ensure that all processes get the exact same checkpoint
  step.

  Args:
    checkpoint_dir: The base directory from where to retrieve checkpoints.

  Raises:
    ValueError: checkpoint_dir does not exist or no checkpoint files were found
    in it.
  """
  step = retrieve_latest_checkpoint_step_if_exists(checkpoint_dir)
  if step is None:
    _raise_checkpoint_missing_error(checkpoint_dir)
  return step


def _raise_checkpoint_missing_error(checkpoint_dir: epath.Path):
  """Raise checkpoint missing error with helpful message."""
  if not checkpoint_dir.exists():
    raise ValueError(f'{checkpoint_dir=!r} does not exist')
  raise ValueError(
      f'No checkpoints were found in directory {checkpoint_dir=!r}'
  )
