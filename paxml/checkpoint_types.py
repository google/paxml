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

"""Module defining possible checkpoint types and utility methods."""

from typing import Union
import enum
from etils import epath
from paxml import base_task
from praxis import pax_fiddle
from praxis import py_utils


@enum.unique
class CheckpointType(str, enum.Enum):
  """The type of the checkpointing format."""

  UNSPECIFIED = 'unspecified'
  FLAX = 'flax'
  GDA = 'gda'
  PERSISTENCE = 'persistence'


def retrieve_checkpoint_type(
    maybe_use_persistence_checkpointing,
    task: Union[base_task.BaseTask, pax_fiddle.Config[base_task.BaseTask]],
) -> CheckpointType:
  """Retrieves the CheckpointType given the input arguments."""
  using_pjit = task.model.mesh_shape is not None  # pytype: disable=attribute-error
  if using_pjit or py_utils.pmap_use_tensorstore():
    if maybe_use_persistence_checkpointing:
      return CheckpointType.PERSISTENCE
    else:
      return CheckpointType.GDA
  else:
    # pmap uses FLAX, Persistence-based or not.
    return CheckpointType.FLAX
