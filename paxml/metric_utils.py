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

"""Utility functions for metric evaluation in Pax."""

from typing import Any, Dict, Union
from absl import logging

import clu.values as clu_values

from paxml import summary_utils
from praxis import py_utils
from praxis import pytypes


Metrics = pytypes.Metrics
NestedMap = py_utils.NestedMap
SummaryValueTypes = Union[
    clu_values.Scalar, clu_values.Image, clu_values.Text]


_VALUES_TO_SUMMARY_TYPE = {
    clu_values.Scalar: summary_utils.SummaryType.SCALAR,
    clu_values.Text: summary_utils.SummaryType.TEXT,
    clu_values.Image: summary_utils.SummaryType.IMAGE,
}


def _get_summary_type(
    metric_value: SummaryValueTypes) -> summary_utils.SummaryType:
  """Infers metric summary type from the metric value type."""
  if type(metric_value) not in _VALUES_TO_SUMMARY_TYPE:
    raise ValueError(f'Unknown metric value type: {type(metric_value)}.')
  return _VALUES_TO_SUMMARY_TYPE[type(metric_value)]


def compute_metric_values(metrics: Metrics) -> Dict[str, SummaryValueTypes]:
  """Given a dict of clu_metrics.Metric objects, returns their values.

  Args:
    metrics: A Dict[str, clu_metrics.Metric] objects with a compute_value()
      function implemented that returns either a clu_values.Value object,
      a Dict[str, clu_values.Value] objects, a Dict[str, List[clu_values.Value]]
      objects, or a List[clu_values.Value].
  Returns:
    metric_values: A flattened Dict[str, clu_values.Value] objects.
  """
  logging.info('Computing metric values.')
  metric_values = {}
  for metric_name, metric in metrics.items():
    logging.info('Computing metric %s', metric_name)
    metric_value = metric.compute_value()
    # compute_value can return either a scalar clu_values.Value object,
    # a Dict[str, clu_values.Value] objects, a Dict[str, List[clu_values.Value]]
    # objects, or a List[clu_values.Value] objects.
    if isinstance(metric_value, dict):
      for key, value in metric_value.items():
        summary_key = f'{metric_name}/{key}'
        if isinstance(value, (list, tuple)):
          for i, subval in enumerate(value):
            summary_key_i = f'{summary_key}_{i}'
            metric_values[summary_key_i] = subval
        else:
          metric_values[summary_key] = value
    elif isinstance(metric_value, (list, tuple)):
      for i, value in enumerate(metric_value):
        metric_values[f'{metric_name}/{metric_name}_{i}'] = value
    elif isinstance(
        metric_value, (clu_values.Scalar, clu_values.Image, clu_values.Text)):
      metric_values[f'{metric_name}'] = metric_value
    else:
      raise ValueError(
          'Unrecognized compute_value() output format for metric '
          f'{metric_name}: {type(metric_value)}.')
  return metric_values


def write_clu_metric_summaries(
    metric_values: Dict[str, SummaryValueTypes],
    step_i: int) -> None:
  """Given a dict of metric values, writes them out as tensorboard summaries.

  This is expected to be called under a summary context.

  Args:
    metric_values: A Dict[str, Any] objects with metric values. These values
      are one of the various clu_values.Value subtypes.
    step_i: An int representing the current step of decoding.
  """
  if not metric_values:
    return

  logging.info('Summarizing metrics.')
  for metric_name, metric_value in metric_values.items():
    logging.info('Summarizing metric %s', metric_name)
    summary_type = _get_summary_type(metric_value)
    summary_utils.write_summary_tensor(
        step_i, metric_name, metric_value.value, summary_type)
