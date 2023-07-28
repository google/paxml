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

"""Wraps NestedMap instances into config sub-graphs that produce them.

Custom objects like NestedMap can cause problems, so we rewrite them into a
fdl.Config of a helper function.
"""

from fiddle import daglish
from paxml.experimental import nested_map_config_helper
from praxis import pax_fiddle
from praxis import pytypes


def wrap_nested_maps(config):
  """Wraps NestedMap instances into config sub-graphs that produce them."""

  def traverse(value, state: daglish.State):
    if isinstance(value, pytypes.NestedMap):
      value = pax_fiddle.Config(
          nested_map_config_helper.make_nested_map, base=dict(value)
      )
    return state.map_children(value)

  return daglish.MemoizedTraversal.run(traverse, config)
