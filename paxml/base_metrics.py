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

"""A suite of metric classes to compute aggregate stats across TPU hosts."""

from __future__ import annotations

import abc
import collections
import dataclasses
import logging
from typing import Any, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from praxis import base_hyperparams
from praxis import base_layer
from praxis import lazy_loader
from praxis import pax_fiddle

# summary_utils is slow to import, so we do it lazily.
summary_utils = lazy_loader.LazyLoader(
    'summary_utils', globals(), 'paxml.summary_utils'
)

instantiate = base_hyperparams.instantiate
PMAP_PARALLEL_AXIS_NAME = base_layer.PMAP_PARALLEL_AXIS_NAME


def _pmap_aggregate_metrics(f, batch_metrics, metric_keys, reshard: bool):
  """Aggregate a dict of metrics over all replicas.

  Args:
    f: A callable that is used to aggregate the metrics across tpus().
      The function signature of f should following convention: f(value, weight)
        For example to compute the mean across TPU replicas
        def _pmap_mean(value, weight):
          sum_value = jax.lax.psum(value * weight, PMAP_PARALLEL_AXIS_NAME)
          sum_weight = jax.lax.psum(weight, PMAP_PARALLEL_AXIS_NAME)
          return (sum_value / (sum_weight + 1e-8), sum_weight)
    batch_metrics: dictionary of items to aggregate over.
    metric_keys: the set of keys to aggregate over. If None, will aggregate over
      all.
    reshard: boolean to indicate whether to reshard before aggregation.

  Returns:
    Aggregated across TPU version of the metrics dict.
  """

  # Reshard for sum over devices
  def _reshard(batch_metrics):
    reshard_metrics = type(batch_metrics)()
    for k, v in batch_metrics.items():
      value, weight = v
      assert weight.ndim == 0
      new_value = jnp.stack([jnp.array(value)] * jax.local_device_count())
      new_weight = jnp.ones(
          shape=(jax.local_device_count(),),
          dtype=weight.dtype) * weight / jax.local_device_count()
      reshard_metrics[k] = (new_value, new_weight)
    return reshard_metrics

  # aggregate across replicas
  def _aggregate(metrics_dict):
    metrics = type(metrics_dict)()
    for k, v in metrics_dict.items():
      if metric_keys and k not in metric_keys:
        continue
      value, weight = v
      metrics[k] = f(value, weight)
    return metrics

  if reshard:
    pmap_aggregate = jax.pmap(
        _aggregate, axis_name=PMAP_PARALLEL_AXIS_NAME, out_axes=None)
    return pmap_aggregate(_reshard(batch_metrics))
  else:
    return _aggregate(batch_metrics)


def _vmap_aggregate_metrics(f, metrics_dict):
  """Aggregate a dict of metrics over all recorded batches.

  Args:
    f: A Callable that computes the aggregate over a vector of metrics. For
      example to compute the mean, we sum over the input vector of weights
        def _vmap_mean(values, weights): sum_metric_weights = np.sum(weights)
          weighted_average = np.sum(values * weights) / sum_metric_weights
          return (weighted_average, sum_metric_weights
    metrics_dict: Dictionary of metrics each containing a vector of values and
      associated weights

  Returns:
      Aggregated metrics.
  """
  metrics = {}
  for k in metrics_dict:
    values = jnp.stack([metric[0] for metric in metrics_dict[k]])
    weights = jnp.stack([metric[1] for metric in metrics_dict[k]])
    metrics[k] = f(values, weights)
  return metrics


class BaseMetrics(
    base_hyperparams.FiddleBaseParameterizable, metaclass=abc.ABCMeta
):
  """Abstract base class for all metrics.

  There are two different usage patterns for BaseMetrics:

    aggregration over a whole dataset (to compute whole set metrics)
    aggregration under pmap (compute metrics across mini-batches
      across different data replicas)

   Usage pattern: Aggregation over a whole dataset:

      metrics_hparams = pax_fiddle.Config(SomeMetricsClass, xxx)
      aggregated_metrics = instantiate(metrics_hparams)

      for step_i in range(num_steps):
        batch_metric, xxx = train_step/eval_step(some_input)
        aggregated_metrics.store(batch_metric, reshard=True)

      aggregated_metrics.finalize() to aggregate the metrics
      aggregated_metrics.summarize() both aggregate the metrics and add them to
       tensorboard.

    The above usage pattern is mostly for aggregating metrics across multiple
    steps.

  Usage pattern: Aggregate under pmap:

    metrics_hparams = pax_fiddle.Config(SomeMetricsClass, xxx)
    metrics_helper = instantiate(metrics_hparams)

     def decode_step(some_input):
        batch_metric = compute some metrics
        batch_metric = metrics_helper.aggregate(batch_metrics, reshard=False)
  """
  _metrics: Any = dataclasses.field(init=False, repr=False)

  def __post_init__(self):
    self._metrics = collections.defaultdict(list)

  @abc.abstractmethod
  def aggregate(self, batch_metrics, reshard: Optional[bool] = False):
    """Aggregate metrics across TPUs.

    Args:
      batch_metrics: A dictionary of metrics each containing a Tuple of value,
        weight. Note the value may or may not be a scalar. The specific
        implementation of the BaseMetrics will compute accordingly.
      reshard: Optionally reshard the data. Used to compute aggregated metrics
        across hosts.

    Returns:
      The aggregated metrics.
    """
    pass

  @abc.abstractmethod
  def finalize(self):
    """Compute final metrics based on internal stored values."""
    pass

  def store(self, batch_metrics, reshard: Optional[bool] = False):
    if reshard:
      batch_metrics = self.aggregate(batch_metrics, reshard=True)
    for k in batch_metrics:
      self._metrics[k].append(batch_metrics[k])

  def summarize(self, step_i, prefix):
    metrics = self.finalize()
    for k, v in metrics.items():
      value, weight = v
      logging.info('  %s=%f (weight=%f)', k, value, weight)
      summary_utils.write_summary_tensor(
          step_i, f'{prefix}/{k}', value,
          summary_utils.SummaryType.AGGREGATE_SCALAR)
      summary_utils.write_summary_tensor(
          step_i, f'{prefix}/{k}-weight', weight,
          summary_utils.SummaryType.AGGREGATE_SCALAR)
    return metrics


class MeanMetrics(BaseMetrics):
  """Computes the mean of the metrics over devices.

  Attributes:
    metric_keys: List of metrics that will be aggregated and logged.
  """
  metric_keys: Optional[Sequence[str]] = None
  _metrics: Any = dataclasses.field(init=False, repr=False)

  def aggregate(self, batch_metrics, reshard: Optional[bool] = False):

    def _pmap_mean(value, weight):
      assert base_layer.is_running_under_pmap()
      sum_value = jax.lax.psum(
          value * weight, axis_name=PMAP_PARALLEL_AXIS_NAME)
      sum_weight = jax.lax.psum(weight, axis_name=PMAP_PARALLEL_AXIS_NAME)
      return (sum_value / (sum_weight + 1e-8), sum_weight)

    return _pmap_aggregate_metrics(
        _pmap_mean, batch_metrics, self.metric_keys, reshard
    )

  def finalize(self):
    """Finalize aggregation over all batches and returns the metrics."""

    def _vmap_mean(values, weights):
      sum_metric_weights = jnp.sum(weights)
      weighted_average = jnp.sum(values * weights, axis=0) / sum_metric_weights
      return (weighted_average, sum_metric_weights)

    metrics = _vmap_aggregate_metrics(_vmap_mean, self._metrics)
    self._metrics = collections.defaultdict(list)
    return metrics


class MaxMetrics(BaseMetrics):
  """Computes the max over sharded metrics.

  Attributes:
    metric_keys: List of metrics that will be aggregated and logged.
  """
  metric_keys: Optional[Sequence[str]] = None
  _metrics: Any = dataclasses.field(init=False, repr=False)

  def aggregate(self, batch_metrics, reshard: Optional[bool] = False):

    def _pmap_max(value, weight):
      assert base_layer.is_running_under_pmap()
      max_value = jax.lax.pmax(value, axis_name=PMAP_PARALLEL_AXIS_NAME)
      sum_weight = jax.lax.psum(weight, axis_name=PMAP_PARALLEL_AXIS_NAME)
      return (max_value, sum_weight)

    return _pmap_aggregate_metrics(
        _pmap_max, batch_metrics, self.metric_keys, reshard
    )

  def finalize(self):
    """Finalize aggregation over all batches and returns the metrics."""

    def _vmap_max(values, weights):
      sum_metric_weights = np.sum(weights, axis=0)
      max_value = np.max(values, axis=0)
      return (max_value, sum_metric_weights)

    metrics = _vmap_aggregate_metrics(_vmap_max, self._metrics)
    self._metrics = collections.defaultdict(list)
    return metrics


class HistogramMetrics(BaseMetrics):
  """Compute aggregate single scalar statistics over sharded batches.

  Attributes:
    histogram_key: Key which contains the histogram data.
  """
  histogram_key: Optional[str] = None

  def aggregate(self, batch_metrics, reshard: Optional[bool] = False):

    def _pmap_sum(value, weight):
      assert base_layer.is_running_under_pmap()
      value = jax.lax.psum(value, axis_name=PMAP_PARALLEL_AXIS_NAME)
      weight = jax.lax.psum(weight, axis_name=PMAP_PARALLEL_AXIS_NAME)
      return (value, weight)

    return _pmap_aggregate_metrics(
        _pmap_sum, batch_metrics, [self.histogram_key], reshard
    )

  def finalize(self):
    """Finalize aggregation over all batches and returns the metrics."""
    metrics = {}
    for k in self._metrics.keys():
      metric_values = np.stack([metric[0] for metric in self._metrics[k]])
      metric_weights = np.stack([metric[1] for metric in self._metrics[k]])
      sum_metric_weights = np.sum(metric_weights)
      histogram = np.sum(metric_values, axis=0)
      num_groups = histogram.shape[0] if histogram.ndim > 0 else 1
      normalizer = np.sum(histogram) / num_groups

      # [g, c]
      probs = histogram / jnp.maximum(normalizer, 1.0)
      log_probs = jnp.log(jnp.maximum(1.0e-30, probs))
      # [g]
      sum_plogp = jnp.sum(log_probs * probs, -1)
      pplx = jnp.mean(jnp.exp(-sum_plogp))
      entropy = jnp.log(pplx)

      metrics[k + '_pplx'] = (pplx, sum_metric_weights)
      metrics[k + '_entropy'] = (entropy, sum_metric_weights)

      onehot = jnp.greater(histogram, 0).astype(jnp.float32)
      avg_num_covered_words = jnp.mean(jnp.sum(onehot, -1))
      num_classes = histogram.shape[-1]
      metrics[k + '_coverage'] = (avg_num_covered_words / num_classes,
                                  sum_metric_weights)
    return metrics


class CompositeMetrics(BaseMetrics):
  """Compute aggregate single scalar statistics over sharded batches.

  Attributes:
    metrics_p: List of metrics that will be aggregated and logged.
  """
  metrics_p: Optional[Sequence[pax_fiddle.Config[BaseMetrics]]] = None
  metrics_calcs: Any = dataclasses.field(init=False, repr=False)

  def __post_init__(self):
    super().__post_init__()
    self.metrics_calcs = [instantiate(m) for m in self.metrics_p]

  def aggregate(self, batch_metrics, reshard: Optional[bool] = False):
    all_metrics = collections.defaultdict()
    for m in self.metrics_calcs:
      metrics = m.aggregate(batch_metrics, reshard)
      for k, v in metrics.items():
        all_metrics[k] = v
    return all_metrics

  def store(self, batch_metrics, reshard: Optional[bool] = False) -> None:
    for m in self.metrics_calcs:
      m.store(batch_metrics, reshard)

  def finalize(self):
    """Finalize aggregation over all batches and returns the metrics."""
    metrics = {}
    for m in self.metrics_calcs:
      finalized = m.finalize()
      for k, v in finalized.items():
        metrics[k] = v
    return metrics


class LossAggregator(base_hyperparams.FiddleBaseParameterizable):
  """Base class for aggregating loss metrics.

  LossAggregator is a helper class that is used to aggregate loss metrics across
  TPUs. First we normalize the loss using the loss_weights across shards

    loss_weight = per_shard_loss_weight / total_loss_weight_across_shards
    weighted_loss = loss * loss_weight

    This value is use for computing the gradient.

  Finally we compute the mean_loss across all shards for tracking purposes:

    mean_loss = average_loss_across_shards

  Attributes:
    loss_key: A string specifying which loss key should be aggregated and
      trained on. This key must be in the metrics dict (the first return of
      compute_loss).
  """

  loss_key: Optional[str] = None

  def aggregate(self, batch_metrics) -> Tuple[float, float, Union[float, None]]:
    """Computes the aggregated loss over shards.

    Args:
      batch_metrics: Input dictionary of metrics computed during model fprop.

    Returns:
      A Tuple containing the weighted loss for the shard, the mean loss
      across all shards, and the weight of the weighted loss (or `None`
      if no such weight is applicable).

    """
    loss_key = self.loss_key

    assert loss_key in batch_metrics
    loss, loss_weight = batch_metrics[loss_key]
    loss_weight = jax.lax.stop_gradient(loss_weight)

    if base_layer.is_running_under_pmap():
      # Renormalize loss weight by the total weight across all replicas.
      # This also takes care of dividing loss by num of data parallel
      # replicas.
      sum_weight = jax.lax.psum(loss_weight, axis_name=PMAP_PARALLEL_AXIS_NAME)
      sum_weight += 1e-8
      loss_weight /= sum_weight

      weighted_loss = loss * loss_weight
      mean_loss = jax.lax.psum(weighted_loss, axis_name=PMAP_PARALLEL_AXIS_NAME)
    else:
      weighted_loss = loss
      mean_loss = loss

    return weighted_loss, mean_loss, loss_weight


class MultiLossAggregator(LossAggregator):
  """Computes the mean of multiple losses.

  Attributes:
    loss_keys: A list of string keys specifying which loss keys should be
      aggregated and trained on. These keys must be in the metrics dict (the
      first return of compute_loss). Weights are renormalized across shards
      before computing the per key weighted_loss.
  """
  loss_keys: Optional[Sequence[str]] = None

  def aggregate(self, batch_metrics) -> Tuple[float, float, Union[float, None]]:

    total_weighted_loss = 0.0
    total_mean_loss = 0.0
    if base_layer.is_running_under_pmap():
      for key in self.loss_keys:
        assert key in batch_metrics

        loss, loss_weight = batch_metrics[key]
        loss_weight = jax.lax.stop_gradient(loss_weight)
        sum_weight = jax.lax.psum(
            loss_weight, axis_name=PMAP_PARALLEL_AXIS_NAME)
        sum_weight += 1e-8
        loss_weight /= sum_weight
        weighted_loss = loss * loss_weight
        mean_loss = jax.lax.psum(
            weighted_loss, axis_name=PMAP_PARALLEL_AXIS_NAME)
        total_weighted_loss += weighted_loss
        total_mean_loss += mean_loss

    else:
      for key in self.loss_keys:
        loss, loss_weight = batch_metrics[key]
        loss_weight = jax.lax.stop_gradient(loss_weight)

        total_weighted_loss += loss
        total_mean_loss += loss

    # Returns `None` for loss weight given that there is a loss weight
    # for each key, and no summary of them is particularly sensible.
    return total_weighted_loss, total_mean_loss, None
