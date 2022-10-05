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

import numbers
import typing
from typing import Any, Dict, Mapping, Optional, Sequence, Union

from absl import logging
import clu.values as clu_values
import jax
from jax import numpy as jnp
from jax.experimental import global_device_array
import numpy as np
from paxml import summary_utils
from praxis import py_utils
from praxis import pytypes
import seqio
from tensorflow.compat.v2 import summary as tf_summary

# internal runtime import


Metrics = pytypes.Metrics
WeightedScalar = pytypes.WeightedScalar
WeightedScalars = pytypes.WeightedScalars
WeightedScalarsList = pytypes.WeightedScalarsList

NestedMap = py_utils.NestedMap
SummaryValueTypes = Union[clu_values.Scalar, clu_values.Image, clu_values.Text]


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


def write_seqio_metric_summaries(seqio_metrics: Sequence[Mapping[str, Union[
    seqio.metrics.MetricValue, float]]], metric_name_prefix: str,
                                 step: int) -> None:
  """Write seqio metric as tensorboard summaries.

  Args:
    seqio_metrics: A sequence of Dict of str to seqio metric value or float.
    metric_name_prefix: A prefix added to metric name.
    step: An int. representing the current step.
  """
  for m_dict in seqio_metrics:
    for k, v in m_dict.items():
      metric_name = f'{metric_name_prefix}/{k}'
      if isinstance(v, seqio.metrics.Text):
        metric_str = (
            v.textdata.decode()
            if isinstance(v.textdata, bytes) else v.textdata)
        logging.info('Writing summary of %s with string value %s.', metric_name,
                     metric_str)
        tf_summary.text(metric_name, metric_str, step=step)
        continue

      if isinstance(v, seqio.metrics.Audio):
        logging.info('Writing summary of %s with audio.', metric_name)
        tf_summary.audio(
            metric_name,
            v.audiodata,
            v.sample_rate,
            step=step,
            max_outputs=v.max_outputs)
        continue

      if isinstance(v, seqio.metrics.Generic):
        tf_summary.write(metric_name, v.tensor, metadata=v.metadata, step=step)
        continue

      if isinstance(v, seqio.metrics.Scalar):
        v = float(v.value)
      else:
        v = float(v)
      logging.info('Writing summary of %s with value %.4f.', metric_name, v)
      summary_utils.write_summary_tensor(step, metric_name, v,
                                         summary_utils.SummaryType.SCALAR)


def is_scalar(v: Any) -> bool:
  """Returns True if input is a scalar."""
  return isinstance(v, (numbers.Number, np.ndarray, jnp.ndarray, global_device_array.GlobalDeviceArray, jax.Array))


def is_weighted_scalar(v: Any) -> bool:
  """Returns True if input is a weighted scalar."""
  return (isinstance(v, tuple) and len(v) == 2 and is_scalar(v[0]) and
          is_scalar(v[1]))


def is_float_convertible(metric_value: Union[numbers.Number, clu_values.Value,
                                             seqio.metrics.MetricValue]):
  """Returns True if a metricv value is float convertible."""
  return (isinstance(metric_value, numbers.Number) or
          isinstance(metric_value, clu_values.Scalar) or
          isinstance(metric_value, seqio.metrics.Scalar) or
          is_weighted_scalar(metric_value) or
          (isinstance(metric_value, list) and
           all(is_weighted_scalar(v) for v in metric_value)))


def as_float(
    metric_value: Union[numbers.Number, clu_values.Scalar, seqio.metrics.Scalar,
                        WeightedScalar, Sequence[WeightedScalar]]
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
    return metric_value.value
  assert isinstance(metric_value, numbers.Number), metric_value
  return float(typing.cast(Any, metric_value))


def as_float_dict(
    metric_output: Union[
        Dict[str, Union[SummaryValueTypes]],
        WeightedScalars,
        WeightedScalarsList,
        Mapping[str, Union[seqio.metrics.MetricValue, float]]],
    raise_on_non_float_convertible: bool = False) -> Dict[str, float]:
  """Returns a float dict from heterogeneous metric output."""
  results = {}
  for k, v in metric_output.items():
    if not is_float_convertible(v):
      if raise_on_non_float_convertible:
        raise ValueError(f'Summary value cannot be converted to float: {v}.')
      continue
    results[k] = as_float(v)
  return results


def update_float_dict(target: Dict[str, float],
                      source: Dict[str, float],
                      prefix: Optional[str] = None) -> Dict[str, float]:
  """Inserts items from source dict to target dict with an optional prefix."""
  if prefix is None:
    target.update(source)
  else:
    for k, v in source.items():
      target[f'{prefix}/{k}'] = v
  return target
