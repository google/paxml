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

"""Tests for base_metrics."""

from absl.testing import absltest
import jax
import jax.numpy as jnp
from paxml import base_metrics
from praxis import base_layer
from praxis import py_utils
from praxis import test_utils

NestedMap = py_utils.NestedMap
instantiate = base_layer.instantiate
PMAP_PARALLEL_AXIS_NAME = base_layer.PMAP_PARALLEL_AXIS_NAME


def _decode(feats):
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
  metrics = {
      'mean': (mean_val, nframes),
      'max': (max_val, nframes),
      'min': (min_val, nframes),
      'hist': (hist, nframes),
      'loss': (mean_val, nframes),
      'loss_a': (mean_val + 10, nframes),
      'loss_b': (mean_val + 20, nframes),
  }
  return metrics


class BaseMetricsTest(test_utils.TestCase):

  def test_reshard_metrics(self):
    feats = jax.random.uniform(jax.random.PRNGKey(1234), [8, 10, 100, 128])

    mean_metrics_p = base_metrics.MeanMetrics.HParams(metric_keys=['mean'])
    max_metrics_p = base_metrics.MaxMetrics.HParams(metric_keys=['max'])
    hist_metrics_p = base_metrics.HistogramMetrics.HParams(histogram_key='hist')
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

    mean_metrics_p = base_metrics.MeanMetrics.HParams(metric_keys=['mean'])
    max_metrics_p = base_metrics.MaxMetrics.HParams(metric_keys=['max'])
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

  def test_loss_aggregate_metrics(self):
    feats = jax.random.uniform(jax.random.PRNGKey(1234), [1, 10, 100, 128])

    loss_metrics_p = base_metrics.LossAggregator.HParams(
        name='basic_loss', loss_key='loss')
    loss_aggregator = instantiate(loss_metrics_p)

    def _decode_step(feats):
      metrics = _decode(feats)
      weighted_loss, mean_loss, loss_weight = loss_aggregator.aggregate(metrics)
      return weighted_loss, mean_loss, loss_weight

    p_decode = jax.pmap(_decode_step, axis_name=PMAP_PARALLEL_AXIS_NAME)
    weighted_loss, mean_loss, loss_weight = p_decode(feats)

    self.assertAllClose(weighted_loss, jnp.mean(feats))
    self.assertAllClose(mean_loss, jnp.mean(feats))
    self.assertAllClose(loss_weight,
                        jnp.mean(weighted_loss) / jnp.mean(mean_loss))

  def test_multiloss_aggregate_metrics(self):
    feats = jax.random.uniform(jax.random.PRNGKey(1234), [1, 10, 100, 128])

    loss_metrics_p = base_metrics.MultiLossAggregator.HParams(
        name='multiloss', loss_keys=['loss_a', 'loss_b'])
    loss_aggregator = instantiate(loss_metrics_p)

    metrics_p = base_metrics.MeanMetrics.HParams(
        name='metrics', metric_keys=['loss_a', 'loss_b'])
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


if __name__ == '__main__':
  absltest.main()
