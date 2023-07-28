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

"""Simple shims for Fiddle configs to reduce custom objects.

Inserting custom objects (anything that's not a Fiddle Buildable, list, tuple,
dict, enum, or primitive) into configuration can limit use of configs with
Fiddle's various tooling.

Therefore, we have a very simple helper function to convert primitive dicts
(which can live in the config) to NestedMap's (which we want after building the
config).
"""

from typing import Any, Dict

from praxis import pytypes


def make_nested_map(base: Dict[str, Any]) -> pytypes.NestedMap:
  """Converts a dict to a NestedMap.

  Args:
    base: The dict to convert.

  Returns:
    A NestedMap.
  """
  return pytypes.NestedMap(base)
