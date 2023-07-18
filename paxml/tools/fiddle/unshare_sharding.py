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

"""Copies sharding annotations, making them unshared.

Sharding doesn't matter for sharding annotations, but Fiddle maintains this
information in general for shared objects.
"""

from typing import Any, TypeVar

from fiddle import daglish
from paxml.tools.fiddle import remove_sharding


_T = TypeVar("_T")


def _deep_copy(value):
  """Returns a copy of a value, unsharing all sub-values.

  Unlike regular copy.deepcopy, this method removes all sharing of sub-objects
  in `value`. See deepcopy documentation. Given an input

  input = [shared, shared]

  where `shared` is any immutable variable,

  `copy.deepcopy(input)` returns `[shared_2, shared_2]`
  `_deep_copy(input)` returns `[unshared_3, unshared_4]`

  where usually shared == shared_2 == unshared_3 == unshared_4 (except custom
  equivalence operators). As indicated, the deepcopy input has the same object
  in both list positions, whereas this _deep_copy() produces different objects.

  Args:
    value: Value to copy.
  """
  return daglish.BasicTraversal.run(
      lambda x, state: state.map_children(x), value
  )


class CustomMemoizedTraversal(daglish.MemoizedTraversal):

  def apply(self, value: Any, state: Any) -> Any:
    result = super().apply(value, state)
    if remove_sharding._is_sharding_annotation(value, state.current_path):  # pylint: disable=protected-access
      del self.memo[id(value)]
    return result


def unshare_sharding(config: _T) -> _T:
  """Unshares sharding annotations.

  Args:
    config: Base configuration or structure of configuration.

  Returns:
    Config sharding annotations unshared.
  """

  def transform(value, state: daglish.State):
    if remove_sharding._is_sharding_annotation(value, state.current_path):  # pylint: disable=protected-access
      return _deep_copy(value)
    return state.map_children(value)

  return CustomMemoizedTraversal.run(transform, config)
