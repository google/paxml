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

"""Tests for Paxml base_metrics."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from paxml import base_metrics
from praxis import base_layer
from praxis import pax_fiddle
from praxis import py_utils
from praxis import test_utils

NestedMap = py_utils.NestedMap
instantiate = base_layer.instantiate
PMAP_PARALLEL_AXIS_NAME = base_layer.PMAP_PARALLEL_AXIS_NAME


def _decode(feats, return_clu_loss_metrics=False):
  b, t, d = feats.shape
  mean_val = jnp.mean(feats)
  max_val = jnp.max(feats)
  min_val = jnp.min(feats)
  frames = jnp.reshape(feats, [b * t, d])
  val = jnp.where(frames > 0.8, size=b * t * d)
  hist = jnp.zeros([d])
  hist = hist.at[val[1]].add(1)
  hist = hist.at[0].set(1)
  nframes = jnp.array(b * t)
  if not return_clu_loss_metrics:
    metrics = {
        'mean': (mean_val, nframes),
        'max': (max_val, nframes),
        'min': (min_val, nframes),
        'hist': (hist, nframes),
        'loss': (mean_val, nframes),
        'loss_a': (mean_val + 10, nframes),
        'loss_b': (mean_val + 20, nframes),
    }
  else:
    metrics = {
        'loss': base_metrics.WeightedScalarCluMetric.create(mean_val, nframes),
    }
  return metrics


class BaseMetricsTest(test_utils.TestCase):

  def test_reshard_metrics(self):
    feats = jax.random.uniform(jax.random.PRNGKey(1234), [8, 10, 100, 128])

    mean_metrics_p = pax_fiddle.Config(
        base_metrics.MeanMetrics, metric_keys=['mean']
    )
    max_metrics_p = pax_fiddle.Config(
        base_metrics.MaxMetrics, metric_keys=['max']
    )
    hist_metrics_p = pax_fiddle.Config(
        base_metrics.HistogramMetrics, histogram_key='hist'
    )
    composite_p = base_metrics.CompositeMetrics.config(
        metrics_p=[mean_metrics_p, max_metrics_p, hist_metrics_p])
    composite = instantiate(composite_p)

    for i in range(8):
      batch_metrics = _decode(feats[i])
      composite.store(batch_metrics, reshard=True)
    metrics = composite.finalize()

    self.assertAllClose(metrics['mean'][0], jnp.mean(feats))
    self.assertAllClose(metrics['max'][0], jnp.max(feats))
    self.assertArraysEqual(metrics['hist_coverage'][0], jnp.array(1.0))

  def test_aggregate_metrics(self):
    feats = jax.random.uniform(jax.random.PRNGKey(1234), [1, 10, 100, 128])

    mean_metrics_p = pax_fiddle.Config(
        base_metrics.MeanMetrics, metric_keys=['mean']
    )
    max_metrics_p = pax_fiddle.Config(
        base_metrics.MaxMetrics, metric_keys=['max']
    )
    composite_p = base_metrics.CompositeMetrics.config(
        metrics_p=[mean_metrics_p, max_metrics_p])
    composite = instantiate(composite_p)

    def _decode_step(feats):
      metrics = _decode(feats)
      metrics = composite.aggregate(metrics)
      return metrics

    p_decode = jax.pmap(_decode_step, axis_name=PMAP_PARALLEL_AXIS_NAME)
    metrics = p_decode(feats)
    self.assertAllClose(metrics['mean'][0][0], jnp.mean(feats))
    self.assertAllClose(metrics['max'][0][0], jnp.max(feats))


class LossAggregatorTest(test_utils.TestCase):

  @parameterized.parameters(False, True)
  def test_loss_aggregate_metrics(self, use_clu_loss_metrics: bool):
    feats = jax.random.uniform(jax.random.PRNGKey(1234), [1, 10, 100, 128])

    loss_metrics_p = pax_fiddle.Config(
        base_metrics.LossAggregator, name='basic_loss', loss_key='loss'
    )
    loss_aggregator = instantiate(loss_metrics_p)

    def _decode_step(feats):
      metrics = _decode(feats, use_clu_loss_metrics)
      weighted_loss, mean_loss, loss_weight = loss_aggregator.aggregate(metrics)
      return weighted_loss, mean_loss, loss_weight

    p_decode = jax.pmap(_decode_step, axis_name=PMAP_PARALLEL_AXIS_NAME)
    weighted_loss, mean_loss, loss_weight = p_decode(feats)

    self.assertAllClose(weighted_loss, jnp.mean(feats))
    self.assertAllClose(mean_loss, jnp.mean(feats))
    self.assertAllClose(loss_weight,
                        jnp.mean(weighted_loss) / jnp.mean(mean_loss))

  # Note: this test isn't parameterized like the test above because
  # `metrics_aggregator` doesn't support aggregating `clu.Metrics`.
  def test_multiloss_aggregate_metrics(self):
    feats = jax.random.uniform(jax.random.PRNGKey(1234), [1, 10, 100, 128])

    loss_metrics_p = pax_fiddle.Config(
        base_metrics.MultiLossAggregator,
        name='multiloss',
        loss_keys=['loss_a', 'loss_b'],
    )
    loss_aggregator = instantiate(loss_metrics_p)

    metrics_p = pax_fiddle.Config(
        base_metrics.MeanMetrics,
        name='metrics',
        metric_keys=['loss_a', 'loss_b'],
    )
    metrics_aggregator = instantiate(metrics_p)

    def _decode_step(feats):
      metrics = _decode(feats)
      weighted_loss, mean_loss, loss_weight = loss_aggregator.aggregate(metrics)
      metrics = metrics_aggregator.aggregate(metrics)
      return metrics, weighted_loss, mean_loss, loss_weight

    p_decode = jax.pmap(_decode_step, axis_name=PMAP_PARALLEL_AXIS_NAME)
    metrics, weighted_loss, mean_loss, loss_weight = p_decode(feats)

    expected_loss = metrics['loss_a'][0][0] + metrics['loss_b'][0][0]

    self.assertAllClose(weighted_loss, expected_loss)
    self.assertAllClose(mean_loss, expected_loss)
    self.assertIsNone(loss_weight)


class WeightedScalarCluMetricTest(test_utils.TestCase):

  def test_weighted_scalar_metric(self):
    values1 = jnp.array([1, 2, 3])
    weights1 = jnp.array([0.1, 0.2, 0.3])
    metric1 = base_metrics.WeightedScalarCluMetric.create(
        weight=weights1,
        value=values1,
    )
    self.assertAllClose(
        metric1.compute(), jnp.average(values1, weights=weights1)
    )

    values2 = jnp.array([4, 5, 6])
    weights2 = jnp.array([0.4, 0.5, 0.6])
    metric2 = base_metrics.WeightedScalarCluMetric.create(
        weight=weights2,
        value=values2,
    )

    metric1 = metric1.merge(metric2)
    self.assertAllClose(
        metric1.compute(),
        jnp.average(
            jnp.concatenate([values1, values2]),
            weights=jnp.concatenate([weights1, weights2]),
        ),
    )

    values3 = jnp.array([7, 8, 9])
    weights3 = jnp.array([1.1, 2.2, 3.3])
    metric3 = base_metrics.WeightedScalarCluMetric.create(
        weight=weights3,
        value=values3,
    )

    metric1 = metric1.merge(metric3)
    self.assertAllClose(
        metric1.compute(),
        jnp.average(
            jnp.concatenate([values1, values2, values3]),
            weights=jnp.concatenate([weights1, weights2, weights3]),
        ),
    )


if __name__ == '__main__':
  absltest.main()
