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

"""Tests for automl."""

import math
from typing import Optional
from absl.testing import absltest
from paxml import automl
from paxml import base_experiment
from paxml import base_task
from praxis import base_hyperparams
import pyglove as pg


instantiate = base_hyperparams.instantiate


class MetricTest(absltest.TestCase):
  """Tests for automl.Metric class."""

  def test_custom_type_metrics(self):
    m = automl.Metric.train_steps_per_second()
    self.assertEqual(m.pattern, '^train_steps_per_sec(:.+)?$')
    self.assertEqual(m.metric_type, automl.MetricType.CUSTOM)
    self.assertFalse(m.applies_to_multiple_datasets)
    self.assertIsNone(m.dataset_name)

    m = automl.Metric.eval_steps_per_second('sub-experiment1')
    self.assertEqual(m.pattern, '^eval_steps_per_sec:sub-experiment1$')
    self.assertEqual(m.metric_type, automl.MetricType.CUSTOM)
    self.assertFalse(m.applies_to_multiple_datasets)
    self.assertIsNone(m.dataset_name)

    m = automl.Metric.decode_steps_per_second()
    self.assertEqual(m.pattern, '^decode_steps_per_sec(:.+)?$')
    self.assertEqual(m.metric_type, automl.MetricType.CUSTOM)
    self.assertFalse(m.applies_to_multiple_datasets)
    self.assertIsNone(m.dataset_name)

    m = automl.Metric.num_params()
    self.assertEqual(m.pattern, '^num_params(:.+)?$')
    self.assertEqual(m.metric_type, automl.MetricType.CUSTOM)
    self.assertFalse(m.applies_to_multiple_datasets)
    self.assertIsNone(m.dataset_name)

  def test_train(self):
    m = automl.Metric.train('loss')
    self.assertEqual(m.metric_type, automl.MetricType.TRAIN_METRICS)
    self.assertFalse(m.applies_to_multiple_datasets)
    self.assertEqual(m.metric_name, 'loss')
    self.assertIsNone(m.dataset_name)
    self.assertEqual(m.pattern, '^train/loss(:.+)?$')

  def test_eval_train(self):
    m = automl.Metric.eval_train('log_pplx')
    self.assertEqual(m.metric_type, automl.MetricType.EVAL_TRAIN_METRICS)
    self.assertFalse(m.applies_to_multiple_datasets)
    self.assertEqual(m.metric_name, 'log_pplx')
    self.assertIsNone(m.dataset_name)
    self.assertEqual(m.pattern, '^eval_train/metrics/log_pplx(:.+)?$')

  def test_eval(self):
    m = automl.Metric.eval('total_loss')
    self.assertEqual(m.metric_type, automl.MetricType.EVAL_METRICS)
    self.assertTrue(m.applies_to_multiple_datasets)
    self.assertIsNone(m.dataset_name)
    self.assertEqual(m.metric_name, 'total_loss')
    self.assertEqual(m.pattern, '^eval_test_[^/]+/metrics/total_loss(:.+)?$')
    self.assertEqual(
        m.get_value({'eval_test_abc:xyz/metrics/total_loss': 0.1}), 0.1)
    self.assertEqual(
        m.get_value({'eval_test_abc:xyz/metrics/total_loss:x1': 0.1}), 0.1)
    self.assertEqual(
        m.get_values({
            'eval_test_abc:xyz/metrics/total_loss': 0.1,
            'eval_test_xyz:abc/metrics/total_loss': 0.2
        }), [0.1, 0.2])
    self.assertEqual(
        m.get_values({
            'eval_test_abc:xyz/metrics/total_loss': 0.1,
            'eval_test_abc:xyz/metrics/total_loss:x1': 0.2,
            'eval_test_xyz:abc/metrics/total_loss': 0.3,
            'eval_test_xyz:xyz/metrics/total_loss:x1': 0.4,
        }), [0.1, 0.2, 0.3, 0.4])

    m = automl.Metric.eval('total_loss', 'xyz')
    self.assertEqual(m.metric_type, automl.MetricType.EVAL_METRICS)
    self.assertEqual(m.metric_name, 'total_loss')
    self.assertEqual(m.dataset_name, 'xyz')
    self.assertEqual(m.pattern, '^eval_test_xyz/metrics/total_loss(:.+)?$')
    self.assertFalse(m.applies_to_multiple_datasets)
    self.assertEqual(
        m.get_values({
            'eval_test_abc/metrics/total_loss': 0.1,
            'eval_test_xyz/metrics/total_loss': 0.2
        }), [0.2])

    m = automl.Metric.eval('total_loss', sub_experiment_id='x2')
    self.assertEqual(m.metric_type, automl.MetricType.EVAL_METRICS)
    self.assertEqual(m.metric_name, 'total_loss')
    self.assertIsNone(m.dataset_name)
    self.assertEqual(m.pattern, '^eval_test_[^/]+/metrics/total_loss:x2$')
    self.assertTrue(m.applies_to_multiple_datasets)
    self.assertEqual(
        m.get_values({
            'eval_test_abc/metrics/total_loss:x2': 0.1,
            'eval_test_xyz/metrics/total_loss': 0.2
        }), [0.1])

  def test_eval_scoring(self):
    m = automl.Metric.eval_scoring('blue')
    self.assertEqual(m.metric_type, automl.MetricType.EVAL_SCORING_METRICS)
    self.assertTrue(m.applies_to_multiple_datasets)
    self.assertIsNone(m.dataset_name)
    self.assertEqual(m.metric_name, 'blue')
    self.assertEqual(m.pattern, '^eval_test_[^/]+/scoring_eval/blue(:.+)?$')
    self.assertEqual(
        m.get_value({'eval_test_abc:xyz/scoring_eval/blue': 0.1}), 0.1)
    self.assertEqual(
        m.get_values({
            'eval_test_abc:xyz/scoring_eval/blue': 0.1,
            'eval_test_xyz:abc/scoring_eval/blue': 0.2
        }), [0.1, 0.2])

    m = automl.Metric.eval_scoring('blue', 'xyz')
    self.assertEqual(m.metric_type, automl.MetricType.EVAL_SCORING_METRICS)
    self.assertEqual(m.metric_name, 'blue')
    self.assertEqual(m.dataset_name, 'xyz')
    self.assertEqual(m.pattern, '^eval_test_xyz/scoring_eval/blue(:.+)?$')
    self.assertFalse(m.applies_to_multiple_datasets)
    self.assertEqual(
        m.get_values({
            'eval_test_abc/scoring_eval/blue': 0.1,
            'eval_test_xyz/scoring_eval/blue': 0.2
        }), [0.2])

  def test_decode(self):
    m = automl.Metric.decode('num_decoded')
    self.assertEqual(m.metric_type, automl.MetricType.DECODE_METRICS)
    self.assertTrue(m.applies_to_multiple_datasets)
    self.assertIsNone(m.dataset_name)
    self.assertEqual(m.metric_name, 'num_decoded')
    self.assertEqual(m.pattern, '^decode_test_[^/]+/num_decoded(:.+)?$')
    self.assertEqual(m.get_value({'decode_test_abc:xyz/num_decoded': 1}), 1)
    self.assertEqual(
        m.get_values({
            'decode_test_abc:xyz/num_decoded': 1.,
            'decode_test_xyz:abc/num_decoded': 2.
        }), [1., 2.])

    m = automl.Metric.decode('num_decoded', 'xyz')
    self.assertEqual(m.metric_type, automl.MetricType.DECODE_METRICS)
    self.assertEqual(m.metric_name, 'num_decoded')
    self.assertEqual(m.dataset_name, 'xyz')
    self.assertEqual(m.pattern, '^decode_test_xyz/num_decoded(:.+)?$')
    self.assertFalse(m.applies_to_multiple_datasets)
    self.assertEqual(
        m.get_values({
            'decode_test_abc/num_decoded': 1.,
            'decode_test_xyz/num_decoded': 2.
        }), [2.])

  def test_case_insensitive(self):
    m = automl.Metric.decode('mAP/map')
    self.assertEqual(m.get_value({
        'decode_test_xyz.bcd/mAP/mAP': 1.,
    }), 1.)

  def test_value_aggregation(self):
    values = {
        'eval_test_abc/metrics/accuracy': 0.1,
        'eval_test_def/metrics/accuracy': 0.2,
        'eval_test_xyz/metrics/accuracy': 0.3,
    }
    m = automl.Metric.eval('accuracy')
    with self.assertRaisesRegex(
        ValueError, 'Found multple metrics that match .*'):
      m.get_value(values)

    self.assertEqual(
        automl.Metric.eval(
            'accuracy',
            aggregator=automl.MetricAggregator.MIN).get_value(values),
        0.1)
    self.assertEqual(
        automl.Metric.eval(
            'accuracy',
            aggregator=automl.MetricAggregator.MAX).get_value(values),
        0.3)
    self.assertAlmostEqual(
        automl.Metric.eval(
            'accuracy',
            aggregator=automl.MetricAggregator.AVERAGE).get_value(values),
        0.2)
    self.assertAlmostEqual(
        automl.Metric.eval(
            'accuracy',
            aggregator=automl.MetricAggregator.SUM).get_value(values),
        0.6)
    self.assertEqual(
        automl.Metric.eval(
            'accuracy',
            aggregator=lambda x: x[-1] - x[0] + x[1]).get_value(values), 0.4)
    with self.assertRaisesRegex(
        ValueError, 'Unsupported aggregator'):
      automl.Metric.eval('accuracy', aggregator='abc')


class SearchHParamsTest(absltest.TestCase):
  """Tests for search hyperparameters."""

  def test_hyperparameter_tuning(self):
    p = automl.hyperparameter_tuning(automl.Metric.eval('accuracy'))
    # Check algorithm cls for hyperparameter tuning.
    self.assertIs(p.search_algorithm.cls, automl.Sweeping)
    self.assertIs(p.search_reward.cls, automl.SingleObjective)
    self.assertEqual(p.search_reward.metric, automl.Metric.eval('accuracy'))
    self.assertEqual(p.search_reward.goal, 'maximize')
    self.assertIsNone(p.search_reward.reward_for_nan)
    self.assertEqual(p.max_num_trials, 100)
    self.assertIsNone(p.errors_to_skip)

  def test_neural_architecture_search_single_objective(self):
    p = automl.neural_architecture_search(automl.Metric.eval('accuracy'))
    self.assertIs(p.search_algorithm.cls, automl.RegularizedEvolution)
    self.assertIs(p.search_reward.cls, automl.SingleObjective)
    self.assertEqual(p.search_reward.metric, automl.Metric.eval('accuracy'))
    self.assertIsNone(p.search_reward.reward_for_nan)
    self.assertEqual(p.max_num_trials, 10000)
    self.assertIsNone(p.errors_to_skip)

  def test_neural_architecture_search_multi_objective(self):
    p = automl.neural_architecture_search([
        automl.Metric.eval('accuracy'),
        automl.Metric.train_steps_per_second()
    ],
                                          150,
                                          max_num_trials=6000)
    self.assertIs(p.search_algorithm.cls, automl.RegularizedEvolution)
    self.assertIs(p.search_reward.cls, automl.MultiObjective)
    self.assertEqual(p.search_reward.metrics, [
        automl.Metric.eval('accuracy'),
        automl.Metric.train_steps_per_second()
    ])
    self.assertEqual(p.search_reward.aggregator.cost_objective, 150)
    self.assertEqual(p.max_num_trials, 6000)

  def test_neural_architecture_search_multi_objective_aggregators(self):
    p = automl.neural_architecture_search([
        automl.Metric.eval('accuracy'),
        automl.Metric.train_steps_per_second()
    ],
                                          150,
                                          reward_type='tunas').search_reward
    self.assertIsInstance(p.aggregator, automl.TunasAbsolute.HParams)
    p = automl.neural_architecture_search([
        automl.Metric.eval('accuracy'),
        automl.Metric.train_steps_per_second()
    ],
                                          150,
                                          reward_type='mnas_hard').search_reward
    self.assertIsInstance(p.aggregator, automl.MnasHard.HParams)
    p = automl.neural_architecture_search([
        automl.Metric.eval('accuracy'),
        automl.Metric.train_steps_per_second()
    ],
                                          150,
                                          reward_type='mnas_soft').search_reward
    self.assertIsInstance(p.aggregator, automl.MnasSoft.HParams)
    with self.assertRaisesRegex(ValueError, 'Unsupported reward type'):
      automl.neural_architecture_search([
          automl.Metric.eval('accuracy'),
          automl.Metric.train_steps_per_second()
      ],
                                        150,
                                        reward_type='unsupported_type')


class SearchAlgorithmsTest(absltest.TestCase):
  """Tests for search algorithms."""

  def test_random_search(self):
    algorithm = instantiate(automl.RandomSearch.HParams(seed=1))
    self.assertTrue(pg.eq(algorithm(), pg.geno.Random(seed=1)))

  def test_sweeping(self):
    algorithm = instantiate(automl.Sweeping.HParams())
    self.assertTrue(pg.eq(algorithm(), pg.geno.Sweeping()))

  def test_regularized_evolution(self):
    algorithm = instantiate(
        automl.RegularizedEvolution.HParams(
            population_size=10, tournament_size=5))
    self.assertTrue(
        pg.eq(
            algorithm(),
            pg.evolution.regularized_evolution(
                mutator=pg.evolution.mutators.Uniform(),
                population_size=10,
                tournament_size=5)))


class EarlyStoppingPoliciesTest(absltest.TestCase):
  """Tests for early stopping policies."""

  def test_early_stop_by_value(self):
    policy = instantiate(automl.EarlyStoppingByValue.HParams(step_values=[
        (100, 0.5),
        (200, 0.7),
    ]))()
    self.assertIsInstance(policy, pg.early_stopping.StepWise)

  def test_early_stop_by_rank(self):
    policy = instantiate(automl.EarlyStoppingByRank.HParams(step_ranks=[
        (100, 0.5, 32),
        (200, 5, 0),
    ]))()
    self.assertIsInstance(policy, pg.early_stopping.StepWise)


class RewardsTest(absltest.TestCase):
  """Tests for common reward functions."""

  def test_single_objective(self):
    reward_fn = instantiate(automl.SingleObjective.HParams(
        metric=automl.Metric.eval('accuracy')))
    self.assertIsInstance(reward_fn, automl.SingleObjective)
    self.assertEqual(reward_fn({'eval_test_abc/metrics/accuracy': 0.9}, 0), 0.9)
    self.assertEqual(reward_fn.used_metrics, [automl.Metric.eval('accuracy')])
    self.assertTrue(math.isnan(
        reward_fn({'eval_test_abc/metrics/accuracy': math.nan}, 0)))

    reward_fn = instantiate(
        automl.SingleObjective.HParams(
            metric=automl.Metric.eval('accuracy'),
            goal='minimize',
            reward_for_nan=-1.0))
    self.assertIsInstance(reward_fn, automl.SingleObjective)
    self.assertEqual(
        reward_fn({'eval_test_abc/metrics/accuracy': 0.9}, 0), -0.9)
    self.assertEqual(
        reward_fn({'eval_test_abc/metrics/accuracy': math.nan}, 0), -1.0)

    with self.assertRaisesRegex(ValueError,
                                'Param `metric` should not be None'):
      _ = automl.SingleObjective.HParams()

    with self.assertRaisesRegex(ValueError,
                                'Param `goal` should be either .*'):
      _ = automl.SingleObjective.HParams(
          metric=automl.Metric.eval('accuracy'), goal='abc')

    with self.assertRaisesRegex(KeyError,
                                'Metric .* does not match with any metrics'):
      _ = reward_fn({'eval_test_abc/log_pplx': 0.1}, 0)

  def test_multi_objective(self):
    reward_fn = instantiate(automl.MultiObjective.HParams(
        metrics=[automl.Metric.eval('accuracy')]))
    self.assertIsInstance(reward_fn, automl.MultiObjective)
    self.assertEqual(reward_fn({'eval_test_abc/metrics/accuracy': 0.9}, 0), 0.9)
    self.assertEqual(reward_fn.used_metrics, [automl.Metric.eval('accuracy')])

    reward_fn = instantiate(automl.MultiObjective.HParams(
        metrics=[
            automl.Metric.eval('accuracy'),
            automl.Metric.train_steps_per_second()
        ],
        aggregator=automl.MnasHard.HParams(cost_objective=150),
        reward_for_nan=-1.0))
    self.assertIsInstance(reward_fn, automl.MultiObjective)
    self.assertEqual(reward_fn.used_metrics, [
        automl.Metric.eval('accuracy'),
        automl.Metric.train_steps_per_second()
    ])
    self.assertEqual(
        reward_fn(
            {
                'eval_test_abc/metrics/accuracy': 0.9,
                'train_steps_per_sec': 140
            }, 0), 0.9)
    self.assertEqual(
        reward_fn(
            {
                'eval_test_abc/metrics/accuracy': math.nan,
                'train_steps_per_sec': 140
            }, 0), -1.0)

    with self.assertRaisesRegex(ValueError,
                                'Param `metrics` must be provided.'):
      _ = automl.MultiObjective.HParams()

    with self.assertRaisesRegex(ValueError,
                                'Param `aggregator` must be provided.'):
      _ = automl.MultiObjective.HParams(metrics=[
          automl.Metric.eval('accuracy'),
          automl.Metric.train_steps_per_second()
      ])

    with self.assertRaisesRegex(KeyError,
                                'Metric .* does not match with any metrics'):
      _ = reward_fn({'cost': 0.1}, 0)


class MultiObjectiveAggregatorTest(absltest.TestCase):
  """Tests for multi-objective aggregators."""

  def test_tunas_abs(self):
    aggregator = instantiate(automl.TunasAbsolute.HParams(cost_objective=1))
    self.assertEqual(aggregator([2., 2.]), 1.93)

    with self.assertRaisesRegex(ValueError,
                                'Only two objectives are supported'):
      aggregator([2., 2., 2.])

    with self.assertRaisesRegex(ValueError,
                                'Param `cost_objective` must be provided.'):
      _ = automl.TunasAbsolute.HParams()

  def test_mnas_hard(self):
    aggregator = instantiate(automl.MnasHard.HParams(cost_objective=1))
    self.assertEqual(aggregator([2., 2.]), 1.9052759960878747)

  def test_mnas_soft(self):
    aggregator = instantiate(automl.MnasSoft.HParams(cost_objective=2.))
    self.assertEqual(aggregator([2., 1.]), 2.0994333672461347)


class CrossStepMetricAggregatorTest(absltest.TestCase):
  """Tests for cross-step metric aggregators."""

  def test_last_reported_metric_values(self):
    aggregator = instantiate(automl.LastReportedMetricValues.HParams())
    self.assertEqual(
        aggregator([
            (100, {'reward': 0.2, 'eval_test_abc/metrics/total_loss': 0.2}),
            (200, {'reward': 0.3, 'eval_test_abc/metrics/total_loss': 0.3}),
            (300, {'reward': 0.4, 'eval_test_abc/metrics/total_loss': 0.4}),
        ]),
        {'reward': 0.4, 'eval_test_abc/metrics/total_loss': 0.4})
    self.assertEqual(
        aggregator([
            (100,
             {'reward:1x': 0.1, 'eval_test_abc/metrics/total_loss:1x': 0.1}),
            (200,
             {'reward:1x': 0.2, 'eval_test_abc/metrics/total_loss:1x': 0.2}),
            (automl.SUB_EXPERIMENT_STEP_OFFSET + 100,
             {'reward:2x': 0.3, 'eval_test_abc/metrics/total_loss:2x': 0.3}),
            (automl.SUB_EXPERIMENT_STEP_OFFSET + 200,
             {'reward:2x': 0.4, 'eval_test_abc/metrics/total_loss:2x': 0.4}),
        ]),
        {
            'reward:1x': 0.2, 'eval_test_abc/metrics/total_loss:1x': 0.2,
            'reward:2x': 0.4, 'eval_test_abc/metrics/total_loss:2x': 0.4
        })

  def test_average_metric_values(self):
    aggregator = instantiate(automl.AverageMetricValues.HParams())
    self.assertEqual(
        aggregator([
            (100, {'reward': 0.2, 'eval_test_abc/metrics/total_loss': 0.2}),
            (200, {'reward': 0.3, 'eval_test_abc/metrics/total_loss': 0.3}),
        ]),
        {'reward': 0.25, 'eval_test_abc/metrics/total_loss': 0.25})
    self.assertEqual(
        aggregator([
            (100,
             {'reward:1x': 0.1, 'eval_test_abc/metrics/total_loss:1x': 0.1}),
            (200,
             {'reward:1x': 0.3, 'eval_test_abc/metrics/total_loss:1x': 0.3}),
            (automl.SUB_EXPERIMENT_STEP_OFFSET + 100,
             {'reward:2x': 0.5, 'eval_test_abc/metrics/total_loss:2x': 0.5}),
            (automl.SUB_EXPERIMENT_STEP_OFFSET + 200,
             {'reward:2x': 0.7, 'eval_test_abc/metrics/total_loss:2x': 0.7}),
        ]),
        {
            'reward:1x': 0.2, 'eval_test_abc/metrics/total_loss:1x': 0.2,
            'reward:2x': 0.6, 'eval_test_abc/metrics/total_loss:2x': 0.6
        })

  def test_average_metric_values_last_n(self):
    aggregator = instantiate(automl.AverageMetricValues.HParams(last_n=2))
    self.assertEqual(
        aggregator([
            (100, {'reward': 0.2, 'eval_test_abc/metrics/total_loss': 0.2}),
            (200, {'reward': 0.3, 'eval_test_abc/metrics/total_loss': 0.3}),
        ]),
        {'reward': 0.25, 'eval_test_abc/metrics/total_loss': 0.25})
    self.assertEqual(
        aggregator([
            (0,
             {'reward:1x': 1e9, 'eval_test_abc/metrics/total_loss:1x': 1e9}),
            (100,
             {'reward:1x': 0.1, 'eval_test_abc/metrics/total_loss:1x': 0.1}),
            (200,
             {'reward:1x': 0.3, 'eval_test_abc/metrics/total_loss:1x': 0.3}),
            (0,
             {'reward:2x': 1e9, 'eval_test_abc/metrics/total_loss:2x': 1e9}),
            (automl.SUB_EXPERIMENT_STEP_OFFSET + 100,
             {'reward:2x': 0.5, 'eval_test_abc/metrics/total_loss:2x': 0.5}),
            (automl.SUB_EXPERIMENT_STEP_OFFSET + 200,
             {'reward:2x': 0.7, 'eval_test_abc/metrics/total_loss:2x': 0.7}),
        ]),
        {
            'reward:1x': 0.2, 'eval_test_abc/metrics/total_loss:1x': 0.2,
            'reward:2x': 0.6, 'eval_test_abc/metrics/total_loss:2x': 0.6
        })

  def test_metrics_with_max_value(self):
    aggregator = instantiate(automl.MetricsWithMaxValue.HParams())
    self.assertEqual(
        aggregator([
            (100, {'reward': 0.1, 'eval_test_abc/metrics/total_loss': 0.3}),
            (200, {'reward': 0.3, 'eval_test_abc/metrics/total_loss': 0.2}),
            (300, {'reward': 0.2, 'eval_test_abc/metrics/total_loss': 0.1}),
        ]),
        {'reward': 0.3, 'eval_test_abc/metrics/total_loss': 0.2})

    aggregator = instantiate(automl.MetricsWithMaxValue.HParams(
        metric=automl.Metric.eval('total_loss')))
    self.assertEqual(
        aggregator([
            (100, {'reward': 0.1, 'eval_test_abc/metrics/total_loss': 0.3}),
            (200, {'reward': 0.3, 'eval_test_abc/metrics/total_loss': 0.2}),
            (300, {'reward': 0.2, 'eval_test_abc/metrics/total_loss': 0.1}),
        ]),
        {'reward': 0.1, 'eval_test_abc/metrics/total_loss': 0.3})

  def test_metrics_with_min_value(self):
    aggregator = instantiate(automl.MetricsWithMinValue.HParams())
    self.assertEqual(
        aggregator([
            (100, {'reward': 0.1, 'eval_test_abc/metrics/total_loss': 0.3}),
            (200, {'reward': 0.3, 'eval_test_abc/metrics/total_loss': 0.2}),
            (300, {'reward': 0.2, 'eval_test_abc/metrics/total_loss': 0.1}),
        ]),
        {'reward': 0.1, 'eval_test_abc/metrics/total_loss': 0.3})

    aggregator = instantiate(automl.MetricsWithMinValue.HParams(
        metric=automl.Metric.eval('total_loss')))
    self.assertEqual(
        aggregator([
            (100, {'reward': 0.1, 'eval_test_abc/metrics/total_loss': 0.3}),
            (200, {'reward': 0.3, 'eval_test_abc/metrics/total_loss': 0.2}),
            (300, {'reward': 0.2, 'eval_test_abc/metrics/total_loss': 0.1}),
        ]),
        {'reward': 0.2, 'eval_test_abc/metrics/total_loss': 0.1})


class EarlyStoppingErrorTest(absltest.TestCase):
  """Tests for early stopping error."""

  def test_skip(self):
    e = automl.EarlyStoppingError(skip_reason='Test stop')
    self.assertTrue(e.skip)
    self.assertEqual(e.skip_reason, 'Test stop')
    self.assertIsNone(e.step)
    self.assertIsNone(e.reward)
    self.assertIsNone(e.metrics)
    self.assertIsNone(e.checkpoint)

  def test_early_stop_without_skip(self):
    # Test stopping with providing final reward.
    e = automl.EarlyStoppingError(skip=False, step=1, reward=1.0)
    self.assertFalse(e.skip)
    self.assertIsNone(e.skip_reason)
    self.assertEqual(e.step, 1)
    self.assertEqual(e.reward, 1.0)
    self.assertIsNone(e.metrics)
    self.assertIsNone(e.checkpoint)

    # Test stopping with providing metrics.
    e = automl.EarlyStoppingError(
        skip=False, step=1, metrics={'accuracy': 1.0}, checkpoint_path='/path')
    self.assertFalse(e.skip)
    self.assertIsNone(e.skip_reason)
    self.assertEqual(e.step, 1)
    self.assertIsNone(e.reward)
    self.assertEqual(e.metrics, {'accuracy': 1.0})
    self.assertEqual(e.checkpoint, '/path')

    with self.assertRaisesRegex(
        ValueError, '`step` must be provided when `skip` is set to False'):
      _ = automl.EarlyStoppingError(skip=False)

    with self.assertRaisesRegex(
        ValueError,
        'At least one of `reward` and `metrics` should be provided'):
      _ = automl.EarlyStoppingError(skip=False, step=1)


class MyTask(base_task.BaseTask):
  """Task for testing purpose."""

  class HParams(base_task.BaseTask.HParams):
    learning_rate: Optional[float] = None
    batch_size: Optional[int] = None
    program_str: Optional[str] = None


class RegularExperiment(base_experiment.BaseExperiment):
  """Regular experiment."""

  LEARNING_RATE = 0.1
  BATCH_SIZE = 8
  PROGRAM_STR = 'foo'

  def task(self):
    return MyTask.HParams(
        learning_rate=self.LEARNING_RATE,
        batch_size=self.BATCH_SIZE,
        program_str=self.PROGRAM_STR)

  def datasets(self):
    return []


@pg.members([
    ('init_value', pg.typing.Str())
])
class VarString(pg.hyper.CustomHyper):

  def custom_decode(self, dna):
    return dna.value

  def first_dna(self):
    return pg.DNA(self.init_value)


class TuningExperiment(RegularExperiment):
  """Tuning experiment."""
  LEARNING_RATE = pg.floatv(0.0, 1.0, name='learning_rate')
  BATCH_SIZE = pg.oneof([8, 16, 32], name='batch_size')
  PROGRAM_STR = VarString(init_value='bar', name='program_str')


class TuningExperimentWithOverride(TuningExperiment):
  """Tuning experiment with modified learning_rate."""
  LEARNING_RATE = 0.1


class TuningExperimentWithoutHyperName(RegularExperiment):
  """Tuning experiment without specifying name for `pg.oneof`."""
  BATCH_SIZE = pg.oneof([8, 16, 32])


@automl.parameter_sweep()
class ParameterSweepingExperiment(TuningExperiment):
  """Parameter sweeping experiment."""


@automl.parameter_sweep(automl.Metric.train('total_loss'))
class ParameterSweepingWithReportingTrainLoss(TuningExperiment):
  """Parameter sweeping experiment with train loss as the objective."""


class ClassLevelHyperPrimitiveTest(absltest.TestCase):
  """Test class-level hyper primitives on experiment specifications."""

  def test_regular_experiment(self):
    """Test enable_class_level_hyper_primitives on regular experiment class."""
    context = pg.hyper.DynamicEvaluationContext()
    with context.collect():
      _ = RegularExperiment().task()
    self.assertEmpty(context.hyper_dict)

  def test_tuning_experiment(self):
    """Test enable_class_level_hyper_primitives on tuning experiment class."""
    context = pg.hyper.DynamicEvaluationContext()
    with context.collect():
      _ = TuningExperiment().task()
    self.assertEqual(
        context.hyper_dict, {
            'learning_rate': pg.floatv(0.0, 1.0, name='learning_rate'),
            'batch_size': pg.oneof([8, 16, 32], name='batch_size'),
            'program_str': VarString(init_value='bar', name='program_str')
        })

  def test_tuning_experiment_with_override(self):
    """Test enable_class_level_hyper_primitives on experiment with override."""
    context = pg.hyper.DynamicEvaluationContext()
    with context.collect():
      _ = TuningExperimentWithOverride().task()
    self.assertEqual(context.hyper_dict, {
        'batch_size': pg.oneof([8, 16, 32], name='batch_size'),
        'program_str': VarString(init_value='bar', name='program_str')
    })

  def test_tuning_experiment_without_hyper_name(self):
    """Test enable_class_level_hyper_primitives on bad experiment spec."""
    context = pg.hyper.DynamicEvaluationContext()
    with context.collect():
      _ = TuningExperimentWithoutHyperName().task()
    self.assertEqual(
        context.hyper_dict, {
            'BATCH_SIZE': pg.oneof([8, 16, 32], name='BATCH_SIZE')
        })


class AutoMLDecoratorsTest(absltest.TestCase):
  """Tests for AutoML decorators."""

  def test_parameter_sweep(self):
    search_hparams = ParameterSweepingExperiment().search()
    self.assertEqual(search_hparams.search_algorithm, automl.Sweeping.HParams())
    self.assertIsNone(search_hparams.search_reward)

    search_hparams = ParameterSweepingWithReportingTrainLoss().search()
    self.assertEqual(search_hparams.search_algorithm, automl.Sweeping.HParams())
    self.assertEqual(
        search_hparams.search_reward,
        automl.SingleObjective.HParams(
            metric=automl.Metric.train('total_loss')))


if __name__ == '__main__':
  absltest.main()
