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

"""Pass to remove sharding annotations from a model.

This can be useful for visualization, and code generation. Currently, the goal
is not perfection; for visualization we just want to remove enough to make
config more readable, and for code generation, sharding annotations are added
back in later (it just helps to factor them into a separate function).
"""

from typing import TypeVar

import fiddle as fdl
from fiddle import daglish
from praxis import base_layer
from praxis import pax_fiddle

_T = TypeVar("_T")

BASE_LAYER_SHARDING_FIELDS = {
    "weight_split_dims_mapping",
    "activation_split_dims_mapping",
}


SHARDING_TYPES = (
    base_layer.BaseLayer.WeightSharding,
    base_layer.BaseLayer.ActivationSharding,
)


def _is_sharding_annotation(value, path: daglish.Path) -> bool:
  """Returns whether the current value or path is for a sharding annotation."""
  if path:
    last_elt = path[-1]
    if (
        isinstance(last_elt, daglish.Attr)
        and last_elt.name in BASE_LAYER_SHARDING_FIELDS
    ):
      return True

  if isinstance(value, fdl.Buildable):
    fn_or_cls = fdl.get_callable(value)
    return isinstance(fn_or_cls, type) and issubclass(fn_or_cls, SHARDING_TYPES)

  return False


class RemoveSentinel:
  pass


_remove_sentinel = RemoveSentinel()


def remove_sharding(config: _T, replace_with_default: bool = False) -> _T:
  """Removes sharding annotations from a config.

  Args:
    config: Base configuration or structure of configuration.
    replace_with_default: Instead of just removing the sharding annotations,
      replace them with the default values.

  Returns:
    Config without sharding annotations.
  """

  def transform(value, state: daglish.State):
    value = state.map_children(value)
    if _is_sharding_annotation(value, state.current_path):
      return _remove_sentinel
    elif isinstance(value, fdl.Buildable):
      for name, sub_value in fdl.ordered_arguments(value).items():
        if sub_value is _remove_sentinel:
          delattr(value, name)
          if replace_with_default:
            default_obj = pax_fiddle.Config(fdl.get_callable(value))
            setattr(value, name, getattr(default_obj, name))
    return value

  return daglish.MemoizedTraversal.run(transform, config)
