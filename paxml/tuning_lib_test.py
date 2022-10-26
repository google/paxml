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
from typing import Dict, Type
from absl.testing import absltest
from clu import platform
from etils import epath
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


class TuningWithEarlyStopping(TuningExperiment):
  """A faked tuning experiment with population-wise early stopping."""

  def search(self):
    search_hparams = super().search()
    search_hparams.early_stopping = automl.EarlyStoppingByValue.HParams(
        step_values=[
            # Watch metrics at step 5, values below 1.0 will be set as
            # infeasible.
            (1, 1.0)
        ],
        metric=automl.Metric.eval('reward'))
    return search_hparams


class TuningWithSubExperiments(TuningExperiment):
  """A faked tuning experiment with multiple sub-experiments."""

  def sub_experiments(self) -> Dict[str, Type[base_experiment.BaseExperiment]]:
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
                   work_unit: platform.WorkUnit, job_log_dir: epath.Path,
                   early_stopping_fn: trainer_lib.EarlyStoppingFn):
  del work_unit, job_log_dir
  task_p = experiment_config.task()
  _ = experiment_config.datasets()
  _ = experiment_config.decoder_datasets()
  reward = task_p['learning_rate'] * task_p['batch_size'] * 1
  if reward > 5:
    reward = math.nan
  # Report measurements at step 1 and step 2.
  early_stopping_fn({
      'eval_test_abc/metrics/reward': reward,
      'unused_metric': 0.0
  }, trainer_lib.RunningMode.EVAL, 1, False)
  early_stopping_fn({
      'eval_test_abc/metrics/reward': reward * 3,
      'unused_metric': 1.0
  }, trainer_lib.RunningMode.EVAL, 2, True)


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
    job_log_dir = epath.Path(absltest.get_default_test_tmpdir())
    tuning_lib.tune(run_experiment, TuningExperiment(), platform.work_unit(),
                    job_log_dir, study='local1', max_num_trials=5)
    result = pg.tuning.poll_result('local1')
    self.assertLen(result.trials, 5)
    self.assertEqual([t.infeasible for t in result.trials],
                     [True, False, False, False, False])
    # We use the average of the metrics across steps as the final measurement.
    self.assertEqual([t.final_measurement.reward for t in result.trials],
                     [0.0, 0.32 * 2, 3.2 * 2, 0.32 * 2, 1.6 * 2])
    # Make sure unused metric does not appear in reported metrics.
    self.assertEqual(result.trials[1].final_measurement.metrics, {
        'eval_test_abc/metrics/reward': 0.64,
    })
    # We added an extra measurement for the final report, with final step + 1.
    self.assertEqual([t.final_measurement.step for t in result.trials],
                     [0, 3, 3, 3, 3])

  def test_tune_with_early_stopping_policy(self):
    job_log_dir = epath.Path(absltest.get_default_test_tmpdir())
    tuning_lib.tune(run_experiment, TuningWithEarlyStopping(),
                    platform.work_unit(),
                    job_log_dir, study='local2', max_num_trials=5)
    result = pg.tuning.poll_result('local2')
    self.assertLen(result.trials, 5)
    # We use the average of the metrics across steps as the final measurement.
    reward_at_step0 = (
        lambda t: t.measurements[0].reward if t.measurements else None)
    self.assertEqual([reward_at_step0(t) for t in result.trials],
                     [None, 0.32, 3.2, 0.32, 1.6])
    self.assertEqual([len(t.measurements) for t in result.trials],
                     [0, 1, 3, 1, 3])
    self.assertEqual([t.infeasible for t in result.trials],
                     [True, True, False, True, False])
    # We use the average of the metrics across steps as the final measurement.
    self.assertEqual([t.final_measurement.reward for t in result.trials],
                     [0.0, 0.0, 3.2 * 2, 0.0, 1.6 * 2])

  def test_tune_with_subexperiments(self):
    job_log_dir = epath.Path(absltest.get_default_test_tmpdir())
    tuning_lib.tune(run_experiment, TuningWithSubExperiments(),
                    platform.work_unit(), job_log_dir,
                    study='local3', max_num_trials=5)
    result = pg.tuning.poll_result('local3')
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

  def test_is_last_checkpoint(self):
    # Training step has reached.
    self.assertTrue(
        tuning_lib.is_last_checkpoint(
            trainer_lib.RunningMode.TRAIN,
            global_step=1000,
            num_train_steps=1000,
            eval_interval_steps=10,
            decode_interval_steps=10,
            save_interval_steps=100))
    # Training step has not reached.
    self.assertFalse(
        tuning_lib.is_last_checkpoint(
            trainer_lib.RunningMode.TRAIN,
            global_step=999,
            num_train_steps=1000,
            eval_interval_steps=10,
            decode_interval_steps=10,
            save_interval_steps=100))
    # Training step has not reached, but there is not a next eval.
    self.assertTrue(
        tuning_lib.is_last_checkpoint(
            trainer_lib.RunningMode.TRAIN | trainer_lib.RunningMode.EVAL,
            global_step=900,
            num_train_steps=1000,
            eval_interval_steps=110,
            decode_interval_steps=10,
            save_interval_steps=10))
    # Training step has not reached, but there is not a next checkpoint.
    self.assertTrue(
        tuning_lib.is_last_checkpoint(
            trainer_lib.RunningMode.EVAL,
            global_step=900,
            num_train_steps=1000,
            eval_interval_steps=10,
            decode_interval_steps=10,
            save_interval_steps=400))
    # Training step has not reached, but there is not a next decode.
    self.assertTrue(
        tuning_lib.is_last_checkpoint(
            trainer_lib.RunningMode.TRAIN | trainer_lib.RunningMode.DECODE,
            global_step=900,
            num_train_steps=1000,
            eval_interval_steps=10,
            decode_interval_steps=110,
            save_interval_steps=10))
    # Training step has not reached, but there is not a next checkpoint.
    self.assertTrue(
        tuning_lib.is_last_checkpoint(
            trainer_lib.RunningMode.DECODE,
            global_step=900,
            num_train_steps=1000,
            eval_interval_steps=10,
            decode_interval_steps=10,
            save_interval_steps=400))


def get_trial_dirname(search_space_fun,
                      trial_id: int,
                      dna: pg.DNA,
                      root_dir: str = 'root') -> epath.Path:
  search_space = pg.hyper.trace(search_space_fun, require_hyper_name=True)
  dirname_generator = tuning_lib.TrialDirectoryNameGenerator(
      epath.Path(root_dir), search_space.dna_spec)
  dna.use_spec(search_space.dna_spec)
  return dirname_generator.dirname(trial_id, dna)


class TrialDirnameTest(absltest.TestCase):
  """Tests for trial dirname."""

  def test_trial_with_named_values(self):
    def _fn():
      return pg.oneof([1, 2], name='x') + len(
          pg.oneof(['a', 'b', 'c'], name='y'))

    self.assertEqual(
        str(get_trial_dirname(_fn, 1, pg.DNA([0, 0]))), 'root/1/x=1|y=a')

  def test_trial_with_named_decisions(self):
    def _fn():
      # ? is not allowed in path, so we use the decision index for 'y'.
      return pg.oneof([1, 2], name='x') + len(
          pg.oneof(['a', 'b?', 'c'], name='y'))

    self.assertEqual(
        str(get_trial_dirname(_fn, 1, pg.DNA([0, 0]))), 'root/1/x=1|y=(0)')

  def test_trial_with_decision_values(self):
    def _fn():
      # The total length of name strings exceeds 64.
      # Thus we only include values.
      r = 0
      for i in range(10):
        r += pg.oneof([1, 2, 3], name=f'decision_{i}')
      return r
    self.assertEqual(
        str(get_trial_dirname(_fn, 1, pg.DNA([0 for _ in range(10)]))),
        'root/1/1|1|1|1|1|1|1|1|1|1')

  def test_trial_with_format_literal(self):
    class Foo:
      pass

    class Bar:
      pass

    def _fn():
      return (pg.oneof([Foo, Bar], name='x')(), pg.floatv(0.0, 1.0, name='y'))
    self.assertEqual(
        str(get_trial_dirname(_fn, 1, pg.DNA([0, 0.0025112]))),
        'root/1/x=Foo|y=2.511e-03')

  def test_trial_with_custom_hyper(self):
    class CustomHyper(pg.hyper.CustomHyper):

      def custom_decode(self, dna: pg.geno.DNA):
        return dna.value

      def first_dna(self):
        return pg.DNA('abc')

    def _fn():
      # ? is not allowed in path, so we use the decision index for 'y'.
      return pg.oneof([1, 2], name='x') + len(CustomHyper(name='y'))

    self.assertEqual(
        str(get_trial_dirname(_fn, 1, pg.DNA([0, 'xyz']))),
        'root/1/x=1|y=(CUSTOM)')


if __name__ == '__main__':
  absltest.main()
