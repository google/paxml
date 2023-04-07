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

"""Stores current checkpoint version and version history."""

from typing import Optional

#
# Past versions:
# 1.2
# - State checkpoint uses Tensorstore's OCDBT format.
# - The state will consist of Tensorstore-managed files plus a 'checkpoint' file
#   managed by Orbax, which stores the PyTree structure.
#
# 1.1
# - Metadata has a new key 'train_state_metadata', which is a pytree of array
# metadata corresponding to the train state, including shape, dtype and
# is_masked_node for `TrainState.mdl_vars`.
#
# 1.0
# - Checkpoints folders are organized into per-step directories, where each has
# a subdirectory for every item.
# - The items are 'state' and 'metadata'.
# - Per-step metadata contains a version key.
#
# 0.0
# - Checkpoints do not have per-item directories.
# - Flax checkpoints may or may not be contained within a step directory. In
# other words, the msgpack file may be 'checkpoint_1' instead of
# 'checkpoint_1/checkpoint', where 'checkpoint' is the msgpack file.

# TODO(b/273803615) When rolled out globally, make _OCDBT_VERSION the standard
# version.
_OCDBT_VERSION: float = 1.2
_VERSION: float = 1.1
_VERSION_KEY: str = 'version'


def get_version(tensorstore_use_ocdbt: Optional[bool] = None) -> float:
  if tensorstore_use_ocdbt is None:
    raise ValueError('Must set the value of `tensorstore_use_ocdbt`.')
  if tensorstore_use_ocdbt:
    return _OCDBT_VERSION
  return _VERSION


def get_version_key() -> str:
  return _VERSION_KEY
