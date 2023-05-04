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
import jax
from jax.experimental import multihost_utils
import numpy as np
import orbax.checkpoint
from paxml import checkpoint_types
from paxml import checkpoint_version
from praxis import pytypes


CHECKPOINT_PREFIX = 'checkpoint_'
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
AsyncCheckpointer = orbax.checkpoint.AsyncCheckpointer
Checkpointer = orbax.checkpoint.Checkpointer
COMMIT_SUCCESS_FILE = 'commit_success.txt'


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
  return orbax.checkpoint.utils.is_tmp_checkpoint(x)


def checkpoint_name(
    step: int,
    checkpoint_type: CheckpointType = CheckpointType.UNSPECIFIED,
) -> str:
  if checkpoint_type == CheckpointType.FLAX:
    return f'{CHECKPOINT_PREFIX}{step}'
  elif checkpoint_type == CheckpointType.GDA_VERSION_SUBDIR:
    return str(step)
  else:
    return f'{CHECKPOINT_PREFIX}{step:08d}'


def make_checkpoint_step_dir(
    checkpoint_dir: epath.Path,
    step: int,
    checkpoint_type: CheckpointType = CheckpointType.UNSPECIFIED,
) -> epath.Path:
  return checkpoint_dir / checkpoint_name(step, checkpoint_type=checkpoint_type)


def get_step_from_checkpoint_asset(checkpoint_dir: epath.PathLike) -> int:
  checkpoint_dir = epath.Path(checkpoint_dir)
  if checkpoint_types.is_gda_version_subdir(checkpoint_dir):
    return int(checkpoint_dir.name)
  if is_tmp_checkpoint_asset(checkpoint_dir):
    return int(checkpoint_dir.suffix[len(CHECKPOINT_PREFIX) :])
  return int(checkpoint_dir.stem[len(CHECKPOINT_PREFIX) :])


def latest_checkpoint(checkpoint_dir: epath.PathLike) -> Optional[epath.Path]:
  """Gets the path to the latest checkpoint.

  Args:
    checkpoint_dir: The base directory from where to retrieve checkpoints.

  Returns:
    Path to latest checkpoint or None if there is no checkpoint.
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


def retrieve_latest_checkpoint_step(
    checkpoint_dir: epath.Path,
) -> Optional[int]:
  """Retrieves the latest checkpoint step if any.

  Note that this broadcasts the checkpoint step from host 0 to ensure that all
  processes get the exact same checkpoint step.

  Args:
    checkpoint_dir: The base directory from where to retrieve checkpoints.

  Returns:
    The latest checkpoint step as an integer or None if no checkpoint is found.
  """
  if not checkpoint_dir.exists():
    logging.info('Checkpoint dir \'%s\' does not exist.', checkpoint_dir)
    checkpoint_step = -1
  else:
    latest_checkpoint_path = latest_checkpoint(checkpoint_dir)
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
