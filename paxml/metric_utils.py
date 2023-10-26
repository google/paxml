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

"""Utility functions for metric evaluation in Pax."""

import numbers
import typing
from typing import Any, Mapping, Sequence, Tuple

from absl import logging
import clu.metrics as clu_metrics
import clu.values as clu_values
import jax
from jax import numpy as jnp
import numpy as np
# Internal platform import
from praxis import py_utils
from praxis import pytypes
import seqio

# internal runtime import


Metrics = pytypes.Metrics
WeightedScalar = pytypes.WeightedScalar
WeightedScalars = pytypes.WeightedScalars
WeightedScalarsList = pytypes.WeightedScalarsList

NestedMap = py_utils.NestedMap


def compute_metric_values(metrics: Metrics) -> dict[str, Any]:
  """Given a dict of clu_metrics.Metric objects, returns their values.

  Args:
    metrics: A dict[str, clu_metrics.Metric] objects with a compute_value()
      function implemented that returns either a clu_values.Value object, a
      dict[str, clu_values.Value] objects, a dict[str, List[clu_values.Value]]
      objects, or a List[clu_values.Value].

  Returns:
    metric_values: A flattened dict[str, clu_values.Value] objects.
  """
  logging.info('Computing metric values.')
  metric_values = {}
  for metric_name, metric in metrics.items():
    logging.info('Computing metric %s', metric_name)
    metric_value = metric.compute_value()
    # compute_value can return either a scalar clu_values.Value object,
    # a dict[str, clu_values.Value] objects, a dict[str, List[clu_values.Value]]
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
        metric_value,
        (
            clu_values.Scalar,
            clu_values.Image,
            clu_values.Text,
            clu_values.Histogram,
            clu_values.Audio,
        ),
    ):
      metric_values[f'{metric_name}'] = metric_value
    else:
      raise ValueError(
          'Unrecognized compute_value() output format for metric '
          f'{metric_name}: {type(metric_value)}.'
      )
  return metric_values


def is_scalar(v: Any) -> bool:
  """Returns True if input is a scalar."""
  scalar_types = [
      numbers.Number,
      np.ndarray,
      jnp.ndarray,
      jax.Array,
  ]
  # Internal scalar types
  return isinstance(v, tuple(scalar_types))


def is_weighted_scalar(v: Any) -> bool:
  """Returns True if input is a weighted scalar."""
  return (
      isinstance(v, tuple)
      and len(v) == 2
      and is_scalar(v[0])
      and is_scalar(v[1])
  )


def is_float_convertible(
    metric_value: numbers.Number | clu_values.Value | seqio.metrics.MetricValue,
):
  """Returns True if a metricv value is float convertible."""
  return (
      isinstance(metric_value, numbers.Number)
      or isinstance(metric_value, clu_values.Scalar)
      or isinstance(metric_value, seqio.metrics.Scalar)
      or is_weighted_scalar(metric_value)
      or (
          isinstance(metric_value, list)
          and all(is_weighted_scalar(v) for v in metric_value)
      )
  )


def as_float(
    metric_value: numbers.Number
    | clu_values.Scalar
    | seqio.metrics.Scalar
    | WeightedScalar
    | Sequence[WeightedScalar],
) -> float:
  """Returns the aggregated float value from heterogeneous metric value."""
  if is_weighted_scalar(metric_value):
    metric_value = [metric_value]

  if isinstance(metric_value, list):
    assert all(is_weighted_scalar(v) for v in metric_value), metric_value
    values = np.stack([x[0] for x in metric_value])
    weights = np.stack([x[1] for x in metric_value])
    return np.sum(values * weights) / np.sum(weights)
  if isinstance(metric_value, (clu_values.Scalar, seqio.metrics.Scalar)):
    return metric_value.value  # pytype: disable=bad-return-type  # numpy-scalars
  assert isinstance(metric_value, numbers.Number), metric_value
  return float(typing.cast(Any, metric_value))


def as_float_dict(
    metric_output: dict[str, Any]
    | WeightedScalars
    | WeightedScalarsList
    | Mapping[str, seqio.metrics.MetricValue | float],
    raise_on_non_float_convertible: bool = False,
) -> dict[str, float]:
  """Returns a float dict from heterogeneous metric output."""
  results = {}
  for k, v in metric_output.items():
    if not is_float_convertible(v):
      if raise_on_non_float_convertible:
        raise ValueError(f'Summary value cannot be converted to float: {v}.')
      continue
    results[k] = as_float(v)
  return results


def update_float_dict(
    target: dict[str, float],
    source: dict[str, float],
    prefix: str | None = None,
) -> dict[str, float]:
  """Inserts items from source dict to target dict with an optional prefix."""
  if prefix is None:
    target.update(source)
  else:
    for k, v in source.items():
      target[f'{prefix}/{k}'] = v
  return target


def merge_clu_metrics(metrics: Metrics, updated_metrics: Metrics) -> Metrics:
  """Merges updated metric data with existing metrics."""
  if metrics:
    if set(metrics.keys()) != set(updated_metrics.keys()):
      raise ValueError(
          'metrics and updated_metrics keys don`t match. '
          f'metrics keys: {metrics.keys()} '
          f'updated_metrics keys: {updated_metrics.keys()}'
      )

    for key in metrics:
      metrics[key] = metrics[key].merge(updated_metrics[key])
  else:
    metrics = updated_metrics
  return metrics


def extract_weighted_scalars_and_clu_metrics(
    metrics: dict[str, Any]
) -> Tuple[WeightedScalars, Metrics]:
  """Extracts weighted scalars and clu metrics from metrics dict.

  Args:
    metrics: Metrics data output by the model.

  Returns:
    A tuple of weighted scalars or clu.metrics. Only one of these will be
    returned by the model in its outputs, so one of the
    tuple elements will be an empty dictionary.
  """
  if isinstance(metrics, NestedMap):
    metric_values = metrics.Flatten()
  else:
    metric_values = metrics.values()

  for metric_value in metric_values:
    if is_weighted_scalar(metric_value):
      return metrics, {}
    elif isinstance(metric_value, clu_metrics.Metric):
      return {}, metrics
    else:
      raise TypeError(
          '`metrics` must be a `WeightedScalars` or `clu.Metrics`. Instead its'
          ' type is %s.'
          % type(metrics)
      )
  raise TypeError(
      '`metrics` must be a `WeightedScalars` or `clu.Metrics`. Instead its'
      ' type is %s.'
      % type(metrics)
  )
