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

"""Util for parsing experiment summary text files.

This util is intentionally in a separate lightweight module that can be used in
other libraries without adding heavy dependencies on other core paxml modules.
"""

import ast
from typing import Any

from absl import logging


def parse(cls_vars_summary: str) -> dict[str, Any]:
  """Parses a class variables summary into a dictionary of vars to values.

  Parses summaries created by experiment_utils.get_cls_vars_summary(). Values
  are left as strings if ast.literal_eval fails. For example, class instances
  will be strings.

  Args:
    cls_vars_summary: A summary in the format created by
      experiment_utils.get_cls_vars_summary().

  Returns:
    Dictionary of variable names and values.
  """
  cls_vars = {}
  lines = cls_vars_summary.splitlines()
  for line in lines:
    # Skip empty lines and class names
    if line.strip().endswith(':') or not line.strip():
      continue
    # Get name and value of each class variable
    keyval = [s.strip() for s in line.split(':', maxsplit=1)]
    if len(keyval) == 2:
      (key, val) = keyval
      try:
        cls_vars[key] = ast.literal_eval(val)
      except (ValueError, SyntaxError):
        # If unable to evaluate as literal, then store value as string
        cls_vars[key] = val
    else:
      logging.warning('Warning: Ignoring line with unexpected format: %s', line)
  return cls_vars
