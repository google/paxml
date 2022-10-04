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
from absl.testing import absltest
from clu import platform
from paxml import automl
from paxml import base_experiment
from paxml import trainer_lib
from paxml import tuning_lib

import pyglove as pg


class TuningExperiment(base_experiment.BaseExperiment):
  """A faked tuning experiment for testing."""

  LEARNING_RATE = pg.oneof([0.1, 0.01, 0.001])
  UNUSED_PARAM = pg.oneof(range(5))
  DATASET_PARAM1 = pg.oneof(['foo', 'bar'])
  DECODER_DATASET_PARAM1 = pg.oneof(['x', 'y', 'z'])

  def task(self):
    return {
        'learning_rate': self.LEARNING_RATE,
        'batch_size': pg.oneof([16, 32, 64], name='batch_size')
    }

  def datasets(self):
    return [{
        'dataset_param1': self.DATASET_PARAM1,
        'dataset_param2': pg.oneof(range(3), name='dataset_param2')
    }]

  def decoder_datasets(self):
    return [{
        'decoder_dataset_param1':
            self.DECODER_DATASET_PARAM1,
        'decoder_dataset_param2':
            pg.oneof(range(3), name='decoder_dataset_param2')
    }]

  def search(self):
    return automl.SearchHParams(
        search_algorithm=automl.RandomSearch.HParams(seed=1),
        search_reward=automl.SingleObjective.HParams(
            metric=automl.Metric.eval('reward')),
        cross_step_metric_aggregator=automl.AverageMetricValues.HParams(),
        max_num_trials=10)


class TuningWithSubExperiments(TuningExperiment):
  """A faked tuning experiment with multiple sub-experiments."""

  def sub_experiments(self) -> dict[str, type[base_experiment.BaseExperiment]]:
    def _scale_experiment(multiplier):
      class ScaledExperiment(self.__class__):

        @property
        def LEARNING_RATE(self):
          return super().LEARNING_RATE * multiplier
      return ScaledExperiment
    return {f'{i}x': _scale_experiment(i) for i in [1, 2, 4, 8]}

  def search(self):
    search_hparams = super().search()
    # Use the sum metric across different sub-experiments as the reward.
    search_hparams.search_reward.metric = automl.Metric.eval(
        'reward', aggregator=automl.MetricAggregator.SUM)
    return search_hparams


def run_experiment(experiment_config: base_experiment.BaseExperiment,
                   work_unit: platform.WorkUnit,
                   job_log_dir: str,
                   early_stopping_fn: trainer_lib.EarlyStoppingFn):
  del work_unit, job_log_dir
  task_p = experiment_config.task()
  _ = experiment_config.datasets()
  _ = experiment_config.decoder_datasets()
  reward = task_p['learning_rate'] * task_p['batch_size'] * 1
  if reward > 5:
    reward = math.nan
  # Report measurements at step 1 and step 2.
  early_stopping_fn({'eval_test_abc/metrics/reward': reward},
                    trainer_lib.RunningMode.EVAL, 1, False)
  early_stopping_fn({'eval_test_abc/metrics/reward': reward * 3},
                    trainer_lib.RunningMode.EVAL, 2, True)


class TuningLibTest(absltest.TestCase):
  """Tests for tuning_lib."""

  def test_search_space(self):
    search_space = tuning_lib.get_search_space(TuningExperiment())
    self.assertEqual(search_space.hyper_dict, pg.Dict({
        'LEARNING_RATE': pg.oneof([0.1, 0.01, 0.001], name='LEARNING_RATE'),
        'batch_size': pg.oneof([16, 32, 64], name='batch_size'),
        'DATASET_PARAM1': pg.oneof(['foo', 'bar'], name='DATASET_PARAM1'),
        'dataset_param2': pg.oneof(range(3), name='dataset_param2'),
        'DECODER_DATASET_PARAM1': pg.oneof(['x', 'y', 'z'],
                                           name='DECODER_DATASET_PARAM1'),
        'decoder_dataset_param2': pg.oneof(range(3),
                                           name='decoder_dataset_param2')
    }))

  def test_tune(self):
    job_log_dir = absltest.get_default_test_tmpdir()
    tuning_lib.tune(run_experiment, TuningExperiment(), platform.work_unit(),
                    job_log_dir, study='local1', max_num_trials=5)
    result = pg.tuning.poll_result('local1')
    self.assertLen(result.trials, 5)
    self.assertEqual([t.infeasible for t in result.trials],
                     [True, False, False, False, False])
    # We use the average of the metrics across steps as the final measurement.
    self.assertEqual([t.final_measurement.reward for t in result.trials],
                     [0.0, 0.32 * 2, 3.2 * 2, 0.32 * 2, 1.6 * 2])
    # We added an extra measurement for the final report, with final step + 1.
    self.assertEqual([t.final_measurement.step for t in result.trials],
                     [0, 3, 3, 3, 3])

  def test_tune_with_subexperiments(self):
    job_log_dir = absltest.get_default_test_tmpdir()
    tuning_lib.tune(run_experiment, TuningWithSubExperiments(),
                    platform.work_unit(), job_log_dir,
                    study='local2', max_num_trials=5)
    result = pg.tuning.poll_result('local2')
    self.assertLen(result.trials, 5)
    self.assertEqual([t.infeasible for t in result.trials],
                     [True, False, True, False, True])
    self.assertEqual(result.trials[1].final_measurement.metrics, {
        'eval_test_abc/metrics/reward:1x': 0.64,
        'eval_test_abc/metrics/reward:2x': 1.28,
        'eval_test_abc/metrics/reward:4x': 2.56,
        'eval_test_abc/metrics/reward:8x': 5.12
    })
    self.assertEqual(result.trials[3].final_measurement.metrics, {
        'eval_test_abc/metrics/reward:1x': 0.64,
        'eval_test_abc/metrics/reward:2x': 1.28,
        'eval_test_abc/metrics/reward:4x': 2.56,
        'eval_test_abc/metrics/reward:8x': 5.12
    })
    # We use the average of the metrics across steps as the final measurement.
    self.assertEqual([t.final_measurement.reward for t in result.trials],
                     # Used aggregator='max' for computing the reward.
                     [0.0, 0.64 + 1.28 + 2.56 + 5.12,
                      0.0, 0.64 + 1.28 + 2.56 + 5.12, 0.0])


if __name__ == '__main__':
  absltest.main()
