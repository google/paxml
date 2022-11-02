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

"""Experiment utils."""
from typing import Any, Dict, List, Type

from paxml import base_experiment

BaseExperiment = base_experiment.BaseExperiment


def _full_cls_name(cls: Type[BaseExperiment]) -> str:
  """Full name of a class."""
  module = cls.__module__
  if module == 'builtins':
    return cls.__qualname__  # avoid outputs like 'builtins.str'
  return module + '.' + cls.__qualname__


def _short_cls_name(cls: Type[BaseExperiment]):
  return cls.__qualname__


def _get_mro(cls: Type[BaseExperiment]) -> List[Type[BaseExperiment]]:
  """Gets class mro."""
  try:
    # Gets all relevant classes in method-resolution order
    mro = list(cls.__mro__)
  except AttributeError:
    # If a class has no __mro__, then it's a built-in class
    def getmro(cls, recurse):
      mro = [cls]
      for base_cls in cls.__bases__:
        mro.extend(recurse(base_cls, recurse))
      return mro

    mro = getmro(cls, getmro)
    assert mro[-1] == object, mro

  return mro[:-1]


def _get_cls_vars(cls: Type[BaseExperiment]) -> Dict[str, Any]:
  """Returns cls vars."""
  res = {}

  def fn(clazz):
    res = {}
    for k, v in clazz.__dict__.items():
      if (callable(v) or isinstance(v, property) or
          # This is not correct strictly speaking, but a reasonable assumption.
          (k.startswith('__') and k.endswith('__'))):
        continue
      res[k] = v
    return res

  res = fn(cls)
  if cls == BaseExperiment:
    return res

  base_vars = fn(BaseExperiment)
  return {k: v for k, v in res.items() if k not in base_vars}


def _summarize_cls_vars(cls: Type[BaseExperiment]) -> Dict[str, Dict[str, Any]]:
  """Summarizes cls vars and their owners.

  Args:
    cls: a BaseExperiment cls.

  Returns:
    a nested dict of class_name -> {cls var name -> value}
  """
  mro = _get_mro(cls)

  name_to_val = {}
  name_to_owner = {}

  cls_vars_map = {}
  for c in mro:
    cls_vars_map[c] = _get_cls_vars(c)

  for cls in mro:
    for var_name, val in cls_vars_map[cls].items():
      if var_name not in name_to_owner:
        name_to_owner[var_name] = cls
      if var_name not in name_to_val:
        name_to_val[var_name] = val

  assert len(name_to_val) == len(name_to_owner), (name_to_val, name_to_owner)

  res = {}
  for n, c in name_to_owner.items():
    cname = _full_cls_name(c)
    if cname not in res:
      res[cname] = {}
    res[cname][n] = name_to_val[n]
  return res


def get_cls_vars_summary(cls: Type[BaseExperiment]) -> str:
  r"""Returns cls's class variables summary.

  Args:
    cls: a BaseExperiment class.

  Returns:
    A text string.

  This prints each class in the given cls' mro, along with their class
  variables.
  The goal is to help identify the live class variables in a Experiment in the
  Python mro, as well as which the "owning" Experiment class for each of them.

  With a concrete example below:

  ```
  class Exp1(BaseExperiment):
    a = 10

  class Exp2(Exp2):
    b = 20

  class Exp3(Exp2):
    c = 30
    a = 40
  ```

  get_cls_vars_summary(Exp3) will give the following output.

  ```
  Exp3:
    c = 30
    # Even that a is defined first in Exp1, it's value is overwritten by Exp3,
    # thus it's "owned" by Exp3.
    a = 40

  Exp2:
    b = 20

  Exp1:
    <empty>
  ```
  """
  mro = _get_mro(cls)
  res = _summarize_cls_vars(cls)

  def serialize_dict(d, indent=0):
    return '\n'.join(f'{" " * indent}{k}: {d[k]}' for k in d.keys())

  msgs = []
  for c in reversed(mro):
    cname = _full_cls_name(c)
    if cname not in res:
      continue
    msgs.append(f'{cname}:\n{serialize_dict(res[cname], indent=4)}\n')
  return '\n'.join(msgs)
