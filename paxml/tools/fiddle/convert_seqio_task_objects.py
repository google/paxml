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

"""Converts SeqIO task objects to configs referencing their name."""

from typing import TypeVar

from fiddle import daglish
from praxis import pax_fiddle
import seqio

_T = TypeVar("_T")


def convert_seqio_tasks(config: _T) -> _T:
  """Converts SeqIO Task objects within a config to more config-like objects.

  Currently, SeqIO tasks are identifiable by their name, which appears in a
  global registry. This is not an ideal pattern, but since it's hard to convert
  back from a Task object into a config producing that Task, using the name
  seems to be a reasonable strategy for now.

  Args:
    config: Fiddle config, or nested structure of configs.

  Returns:
    Version of config without SeqIO task instances.
  """

  def transform(value, state: daglish.State):
    if isinstance(value, seqio.Task):
      # Note: The following object has a `task_or_mixture_name` parameter
      # instead of a `name` parameter, effectively changing the config API
      # slightly.
      return pax_fiddle.Config(seqio.get_mixture_or_task, value.name)
    return state.map_children(value)

  return daglish.MemoizedTraversal.run(transform, config)
