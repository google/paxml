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

"""Tracer classes for the Pax-specific codegen."""

# Note: Fiddle and Pax devs are in collaboration; please generally do not import
# private libraries from Fiddle.

import dataclasses
import enum
from typing import Type, TypeVar

from fiddle import daglish
from paxml import base_experiment


@dataclasses.dataclass(frozen=True)
class BoolTracer:
  """Special-case tracer for bool values, since we can't inherit from bool."""

  name: str
  value: bool

  @property
  def __highlevel_name__(self):
    return self.name

  def __bool__(self):
    return self.value

  def __eq__(self, other):
    if isinstance(other, bool):
      return self.value == other
    return super().__eq__(other)


def make_tracer(
    name, value, allowed_types=(bool, int, float, str, list, tuple)
):
  """Wraps a value in a tracer object, that has a name."""
  typ = type(value)
  if not issubclass(typ, allowed_types):
    raise ValueError(
        f"Type {typ} is not allowed. If it seems to work with "
        "the subclassed tracers, please add it to allowed_types "
        "and write a unit test."
    )

  if typ == bool:
    return BoolTracer(name, value)
  wrapped = type(
      f"Wrapped{typ.__name__}_{name}", (typ,), {"__highlevel_name__": name}
  )
  return wrapped(value)


class TracerMixin:
  """Mixin that will wrap experiments' property accessors with traced versions.

  This means that when there are high-level settings as attributes on a
  BaseExperiment instance, a traced value will be returned. This can then be
  intercepted in code generation, resulting in partially abstracted code.
  """

  __trace_names__: set[str]

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    object.__setattr__(self, "__trace_names__", set())

  def __getattribute__(self, name):
    result = super().__getattribute__(name)
    if name in {"__trace_names__"}:
      return result
    elif isinstance(result, enum.Enum):
      return result
    elif isinstance(result, (bool, int, float, str, list, tuple)):
      self.__trace_names__.add(name)
      return make_tracer(name, result)
    else:
      return result


def make_subclass_mixin(
    experiment_cls: Type[base_experiment.BaseExperiment],
):
  """Creates a dynamic subclass of an experiment that adds TracerMixin."""
  if not issubclass(experiment_cls, base_experiment.BaseExperiment):
    raise TypeError("Please pass a subclass of BaseExperiment.")
  cls_name = experiment_cls.__name__
  return type(f"{cls_name}Traced", (experiment_cls, TracerMixin), {})


_T = TypeVar("_T")


def remove_tracers(root: _T) -> _T:
  def transform(value, state: daglish.State):
    if isinstance(value, BoolTracer):
      value = bool(value)
    elif hasattr(value, "__highlevel_name__"):
      value = type(value).__bases__[0](value)
    return state.map_children(value)

  return daglish.MemoizedTraversal.run(transform, root)
