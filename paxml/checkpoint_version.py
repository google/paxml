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

"""Stores current checkpoint version and version history."""

#
# Past versions:
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

_VERSION: float = 1.0
_VERSION_KEY: str = 'version'


def get_version() -> float:
  return _VERSION


def get_version_key() -> str:
  return _VERSION_KEY
