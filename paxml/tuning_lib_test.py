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

"""Tests for automl."""

import dataclasses
import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

from absl.testing import absltest
from clu import platform
from etils import epath
from paxml import automl
from paxml import base_experiment
from paxml import trainer_lib
from paxml import tuning_lib
from praxis import base_hyperparams
from praxis import pax_fiddle
import pyglove as pg


class StopWithLowerMetric(automl.BaseReward):
  metric: Optional[automl.Metric] = None
  threshold: Union[float, List[Tuple[int, float]]] = 0.0
  skip: bool = False
  reward_replacement: Optional[float] = None
  metrics_replacement: Optional[Dict[str, float]] = None

  def get_threshold(self, global_step: int) -> float:
    if isinstance(self.threshold, float):
      return self.threshold
    for i, (step, value) in enumerate(self.threshold):
      if step <= global_step and (
          i == len(self.threshold) - 1 or self.threshold[i + 1][0] > global_step
      ):
        return value
    return 0.0

  def __call__(self, metrics_dict: Dict[str, float], global_step: int) -> float:
    reward = self.metric.get_value(metrics_dict)
    if reward < self.get_threshold(global_step):
      if self.skip:
        raise automl.EarlyStoppingError(
            skip=self.skip,
            skip_reason='Trial skipped due to lower metric value.',
            step=global_step,
        )
      else:
        raise automl.EarlyStoppingError(
            skip=False,
            step=global_step,
            reward=self.reward_replacement,
            metrics=self.metrics_replacement,
        )
    return reward

  @property
  def used_metrics(self):
    return [self.metric]


class MockTask(pg.Dict):

  def to_text(self) -> str:
    return 'MOCK_TASK_CONFIG'


class MockDataset(base_hyperparams.FiddleBaseParameterizable):
  dataset_param1: Optional[str] = None
  dataset_param2: Optional[Callable[[], int]] = None
  is_training: bool = False
  param1: Any = dataclasses.field(init=False, repr=False)
  param2: Any = dataclasses.field(init=False, repr=False)

  def __post_init__(self):
    self.param1 = self.dataset_param1
    if callable(self.dataset_param2):
      self.param2 = self.dataset_param2()


class TuningExperiment(base_experiment.BaseExperiment):
  """A faked tuning experiment for testing."""

  LEARNING_RATE = pg.oneof([0.1, 0.01, 0.001])
  UNUSED_PARAM = pg.oneof(range(5))
  DATASET_PARAM1 = pg.oneof(['foo', 'bar'])
  DECODER_DATASET_PARAM1 = pg.oneof(['x', 'y', 'z'])

  def task(self):
    return MockTask(
        learning_rate=self.LEARNING_RATE,
        batch_size=pg.oneof([16, 32, 64], name='batch_size'),
        train=pg.Dict(
            eval_interval_steps=100,
            decode_interval_steps=100))

  def datasets(self):
    # NOTE(daiyip): `dataset_param2` shall appear in the search
    # space: though its evaluation is delayed until instantiation,
    # datasets will be instantiated during search space inspection.
    return [
        pax_fiddle.Config(
            MockDataset,
            dataset_param1=self.DATASET_PARAM1,
            dataset_param2=lambda: pg.oneof(range(3), name='dataset_param2'),
            is_training=True,
        )
    ]

  def decoder_datasets(self):
    # NOTE(daiyip): `decoder_dataset_param2` shall NOT appear in the search
    # space: its evaluation is delayed until instantiation, and decoder
    # datasets are not instantiated during search space inspection.
    return [
        pax_fiddle.Config(
            MockDataset,
            dataset_param1=self.DECODER_DATASET_PARAM1,
            dataset_param2=(
                lambda: pg.oneof(range(3), name='decoder_dataset_param2')
            ),
        )
    ]

  def search(self):
    return automl.SearchHParams(
        search_algorithm=pax_fiddle.Config(automl.RandomSearch, seed=1),
        search_reward=pax_fiddle.Config(
            automl.SingleObjective, metric=automl.Metric.eval('reward')
        ),
        cross_step_metric_aggregator=pax_fiddle.Config(
            automl.AverageMetricValues
        ),
        max_num_trials=10,
        enable_dataset_tuning=True,
    )


class TuningExperimentDatasetTuningDisabled(TuningExperiment):

  def search(self):
    return automl.SearchHParams(
        search_algorithm=pax_fiddle.Config(automl.RandomSearch, seed=1),
        search_reward=pax_fiddle.Config(
            automl.SingleObjective, metric=automl.Metric.eval('reward')
        ),
        cross_step_metric_aggregator=pax_fiddle.Config(
            automl.AverageMetricValues
        ),
        max_num_trials=10,
        enable_dataset_tuning=False,
    )


class TuningWithTrainMetricsOnly(TuningExperiment):
  """Tuning experiment based on train metrics only."""

  def search(self):
    search_p = super().search()
    search_p.search_reward.metric = automl.Metric.train_steps_per_second()
    return search_p


class TuningWithDecodeMetricsOnly(TuningExperiment):
  """Tuning experiment based on train metrics only."""

  def search(self):
    search_p = super().search()
    search_p.search_reward.metric = automl.Metric.decode('f1')
    return search_p


@automl.parameter_sweep()
class ParameterSweepingWithCartesianProduct(base_experiment.BaseExperiment):
  """Parameter sweep the search space."""

  LEARNING_RATE = pg.oneof([0.1, 0.01, 0.001])
  DATASET_PARAM1 = pg.oneof(['foo', 'bar'])

  def task(self):
    return MockTask(
        learning_rate=self.LEARNING_RATE,
        batch_size=pg.oneof([16, 32, 64], name='batch_size'))

  def datasets(self):
    return [pax_fiddle.Config(MockDataset, dataset_param1=self.DATASET_PARAM1)]


@automl.parameter_sweep([
    ('LEARNING_RATE', 'BATCH_SIZE', 'DATASET_PARAM1'),
    (0.1, 16, 'foo'),
    (0.01, 32, 'foo'),
    # Sparse specification will override the `pg.oneof` definition in class
    # attribute, so its value can be different (e.g. 128 for batch_size).
    (0.1, 128, 'bar'),
])
class ParameterSweepingWithCustomCombinations(
    ParameterSweepingWithCartesianProduct):
  """Parameter sweep with sparse combinations."""

  BATCH_SIZE = pg.oneof([16, 32, 64])

  def task(self):
    return MockTask(
        learning_rate=self.LEARNING_RATE,
        # Sparse parameter sweep does not support decision points defined
        # within function body, thus we make it a class attribute.
        batch_size=self.BATCH_SIZE)


class TuningWithBadSearchSpace(TuningExperiment):
  """A bad search space which has `oneof` on-the-fly without name."""

  def task(self):
    return MockTask(
        learning_rate=self.LEARNING_RATE,
        # Bad decision point: a `pg.oneof` created on the fly without
        # specifying a name.
        batch_size=pg.oneof([16, 32, 64]))


class TuningWithPerTrialEarlyStopping(TuningExperiment):
  """A faked tuning experiment with per-trial early stopping."""

  def search(self):
    search_hparams = super().search()
    search_hparams.search_reward = pax_fiddle.Config(
        StopWithLowerMetric,
        metric=automl.Metric.eval('reward'),
        threshold=1.0,
        skip=True,
    )
    return search_hparams


class TuningWithTreatingEarlyStoppedTrailsAsDone(TuningExperiment):
  """A faked tuning experiment for treating early stopped trials as done."""

  def search(self):
    search_hparams = super().search()
    search_hparams.search_reward = pax_fiddle.Config(
        StopWithLowerMetric,
        metric=automl.Metric.eval('reward'),
        threshold=[(10, 0.5), (20, 2.0)],
        skip=True,
    )
    search_hparams.treats_early_stopped_trials_as_done = True
    return search_hparams


class TuningWithPerTrialEarlyStoppingAndRewardReplacement(TuningExperiment):
  """A faked experiment with per-trial early stopping and reward replacement."""

  def search(self):
    search_hparams = super().search()
    search_hparams.search_reward = pax_fiddle.Config(
        StopWithLowerMetric,
        metric=automl.Metric.eval('reward'),
        threshold=1.0,
        skip=False,
        reward_replacement=-1.0,
    )
    return search_hparams


class TuningWithPerTrialEarlyStoppingAndMetricsReplacement(TuningExperiment):
  """A faked experiment with per-trial early stopping and reward replacement."""

  def search(self):
    search_hparams = super().search()
    search_hparams.search_reward = pax_fiddle.Config(
        StopWithLowerMetric,
        metric=automl.Metric.eval('reward'),
        threshold=1.0,
        skip=False,
        metrics_replacement={'eval_test_dev/metrics/reward': 100.0},
    )
    return search_hparams


class TuningWithPopulationWiseEarlyStopping(TuningExperiment):
  """A faked tuning experiment with population-wise early stopping."""

  def search(self):
    search_hparams = super().search()
    search_hparams.early_stopping = pax_fiddle.Config(
        automl.EarlyStoppingByValue,
        step_values=[
            # Watch metrics at step 10, values below 1.0 will be set as
            # infeasible.
            (10, 1.0)
        ],
        metric=automl.Metric.eval('reward'),
    )
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
  # Report eval metrics at step 10 with possible early stopping.
  if tuning_lib.should_early_stop(
      early_stopping_fn,
      global_step=10,
      is_last_ckpt=False,
      eval_metrics=tuning_lib.EvalMetrics(
          input_names=['abc'],
          metrics_list=[dict(reward=reward)],
          steps_per_sec=1.0)):
    return
  # Check early stopping without reporting metrics at step 15.
  if tuning_lib.should_early_stop(
      early_stopping_fn,
      global_step=15,
      is_last_ckpt=False):
    return
  # Report both eval and decode metrics at step 20 with possible early stopping.
  if tuning_lib.should_early_stop(
      early_stopping_fn,
      global_step=20,
      is_last_ckpt=True,
      eval_metrics=tuning_lib.EvalMetrics(
          input_names=['abc'],
          metrics_list=[dict(reward=reward * 3)],
          steps_per_sec=1.0),
      decode_metrics=tuning_lib.DecodeMetrics(
          input_names=['abc'],
          metrics_list=[dict(reward=reward * 3)],
          steps_per_sec=1.0)):
    return


def run_experiment_with_train_metrics_only(
    experiment_config: base_experiment.BaseExperiment,
    work_unit: platform.WorkUnit, job_log_dir: epath.Path,
    early_stopping_fn: trainer_lib.EarlyStoppingFn):
  del work_unit, job_log_dir
  _ = experiment_config.task()
  _ = experiment_config.datasets()
  _ = experiment_config.decoder_datasets()

  _ = tuning_lib.should_early_stop(
      early_stopping_fn,
      global_step=10,
      train_steps_per_sec=2.5,
      is_last_ckpt=False)

  _ = tuning_lib.should_early_stop(
      early_stopping_fn,
      global_step=20,
      train_steps_per_sec=3.0,
      is_last_ckpt=True)


def run_experiment_without_reporting_metrics(
    experiment_config: base_experiment.BaseExperiment,
    work_unit: platform.WorkUnit, job_log_dir: epath.Path,
    early_stopping_fn: trainer_lib.EarlyStoppingFn):
  del work_unit, job_log_dir
  _ = experiment_config.task()
  _ = experiment_config.datasets()
  _ = experiment_config.decoder_datasets()

  # Reach to the final step without reporting any metrics.
  _ = tuning_lib.should_early_stop(
      early_stopping_fn,
      global_step=10,
      is_last_ckpt=True)


class GetSearchSpaceTest(absltest.TestCase):
  """Tests for `tuning_lib.get_search_space`."""

  def test_joint_space_from_class_attributes_and_runtime_call(self):
    search_space = tuning_lib.get_search_space(TuningExperiment())
    self.assertEqual(search_space.hyper_dict, pg.Dict({
        'LEARNING_RATE': pg.oneof([0.1, 0.01, 0.001], name='LEARNING_RATE'),
        'batch_size': pg.oneof([16, 32, 64], name='batch_size'),
        'DATASET_PARAM1': pg.oneof(['foo', 'bar'], name='DATASET_PARAM1'),
        'dataset_param2': pg.oneof(range(3), name='dataset_param2'),
        'DECODER_DATASET_PARAM1': pg.oneof(['x', 'y', 'z'],
                                           name='DECODER_DATASET_PARAM1'),
    }))
    self.assertEqual(search_space.dna_spec.space_size, 3 * 3 * 2 * 3 * 3)

  def test_parameter_sweep_space_with_cartesian_product(self):
    search_space = tuning_lib.get_search_space(
        ParameterSweepingWithCartesianProduct())
    self.assertEqual(search_space.hyper_dict, pg.Dict({
        'LEARNING_RATE': pg.oneof([0.1, 0.01, 0.001], name='LEARNING_RATE'),
        'batch_size': pg.oneof([16, 32, 64], name='batch_size'),
        'DATASET_PARAM1': pg.oneof(['foo', 'bar'], name='DATASET_PARAM1'),
    }))
    self.assertEqual(search_space.dna_spec.space_size, 3 * 3 * 2)

  def test_parameter_sweep_space_with_custom_combinations(self):
    search_space = tuning_lib.get_search_space(
        ParameterSweepingWithCustomCombinations())
    self.assertEqual(search_space.hyper_dict, pg.Dict({
        automl.COMBINED_DECISION_ATTR: pg.oneof([
            (0.1, 16, 'foo'),
            (0.01, 32, 'foo'),
            (0.1, 128, 'bar'),
        ], name=automl.COMBINED_DECISION_ATTR)
    }))
    self.assertEqual(search_space.dna_spec.space_size, 3)

  def test_bad_custom_combination_space(self):

    @automl.parameter_sweep([
        ('LEARNING_RATE', 'DATASET_PARAM1'),
        (0.1, 'foo'),
        (0.2, 'bar'),
    ])
    class CustomCombinationWithOnTheFlyDecisionPoints(
        ParameterSweepingWithCartesianProduct):
      pass

    with self.assertRaisesRegex(ValueError, 'Found extra tuning parameters'):
      tuning_lib.get_search_space(CustomCombinationWithOnTheFlyDecisionPoints())

  def test_bad_search_space(self):
    with self.assertRaisesRegex(
        ValueError, '\'name\' must be specified for hyper primitive'):
      tuning_lib.get_search_space(TuningWithBadSearchSpace())


class TuneTest(absltest.TestCase):
  """Tests for `tuning_lib.tune`."""

  def test_tune(self):
    job_log_dir = epath.Path(absltest.get_default_test_tmpdir())
    tuning_lib.tune(run_experiment, TuningExperiment(), platform.work_unit(),
                    job_log_dir, study='local_basic', max_num_trials=5)
    result = pg.tuning.poll_result('local_basic')
    self.assertLen(result.trials, 5)
    self.assertEqual([t.infeasible for t in result.trials],
                     [True, False, False, False, False])
    # We use the average of the metrics across steps as the final measurement.
    self.assertEqual([t.final_measurement.reward for t in result.trials],
                     [0.0, 0.32 * 2, 1.6 * 2, 0.64 * 2, 0.64 * 2])
    # Make sure unused metric does not appear in reported metrics.
    self.assertEqual(result.trials[1].final_measurement.metrics, {
        'eval_test_abc/metrics/reward': 0.64,
    })
    # We added an extra measurement for the final report, with final step + 1.
    self.assertEqual([t.final_measurement.step for t in result.trials],
                     [0, 21, 21, 21, 21])
    # Make sure experiment config is saved as trial metadata.
    actual = result.trials[0].metadata.get('experiment_config')
    actual['config']['']['datasets'][0] = 'MOCK_DATASET_CONFIG'
    actual['config']['']['decoder_datasets'][0] = 'MOCK_DATASET_CONFIG'
    actual['config']['']['task'] = 'MOCK_TASK_CONFIG'
    self.assertEqual(
        actual,
        {
            'format_version': 1.0,
            'source': 'pax',
            'config': {
                '': pg.Dict(
                    datasets=['MOCK_DATASET_CONFIG'],
                    decoder_datasets=['MOCK_DATASET_CONFIG'],
                    task='MOCK_TASK_CONFIG',
                )
            },
        },
    )

  def test_parameter_sweep_with_catesian_product(self):
    search_space = tuning_lib.get_search_space(
        ParameterSweepingWithCartesianProduct())
    job_log_dir = epath.Path(absltest.get_default_test_tmpdir())
    tuning_lib.tune(run_experiment, ParameterSweepingWithCartesianProduct(),
                    platform.work_unit(), job_log_dir,
                    study='local_hp_sweep_cartesian')
    result = pg.tuning.poll_result('local_hp_sweep_cartesian')
    self.assertLen(result.trials, search_space.dna_spec.space_size)

  def test_parameter_sweep_with_custom_combinations(self):
    search_space = tuning_lib.get_search_space(
        ParameterSweepingWithCustomCombinations())
    self.assertEqual(search_space.dna_spec.space_size, 3)
    job_log_dir = epath.Path(absltest.get_default_test_tmpdir())
    tuning_lib.tune(run_experiment, ParameterSweepingWithCustomCombinations(),
                    platform.work_unit(), job_log_dir,
                    study='local_hp_sweep_custom')
    result = pg.tuning.poll_result('local_hp_sweep_custom')
    self.assertLen(result.trials, search_space.dna_spec.space_size)

  def test_tune_with_train_metrics_only(self):
    job_log_dir = epath.Path(absltest.get_default_test_tmpdir())
    tuning_lib.tune(run_experiment_with_train_metrics_only,
                    TuningWithTrainMetricsOnly(),
                    platform.work_unit(),
                    job_log_dir,
                    study='local_train_metrics_only',
                    max_num_trials=2)
    result = pg.tuning.poll_result('local_train_metrics_only')
    self.assertLen(result.trials, 2)
    self.assertEqual([t.status for t in result.trials], ['COMPLETED'] * 2)
    self.assertEqual([t.infeasible for t in result.trials], [False] * 2)
    # There will be two measurements, one reported at the last step, and
    # one aggregated by cross-step metric aggregator.
    self.assertEqual([len(t.measurements) for t in result.trials], [2] * 2)
    self.assertEqual([t.final_measurement.reward for t in result.trials],
                     [3.0] * 2)

  def test_tune_without_reporting_metrics(self):
    job_log_dir = epath.Path(absltest.get_default_test_tmpdir())
    tuning_lib.tune(run_experiment_without_reporting_metrics,
                    TuningExperiment(),
                    platform.work_unit(),
                    job_log_dir,
                    study='local_no_metrics_reported',
                    max_num_trials=5)
    result = pg.tuning.poll_result('local_no_metrics_reported')
    self.assertLen(result.trials, 5)
    self.assertEqual([t.status for t in result.trials], ['COMPLETED'] * 5)
    self.assertEqual([t.infeasible for t in result.trials], [True] * 5)

  def test_tune_with_treating_early_stopped_trials_as_done(self):
    job_log_dir = epath.Path(absltest.get_default_test_tmpdir())
    tuning_lib.tune(run_experiment,
                    TuningWithTreatingEarlyStoppedTrailsAsDone(),
                    platform.work_unit(),
                    job_log_dir,
                    study='local_treating_early_stopped_trials_as_done',
                    max_num_trials=5)
    result = pg.tuning.poll_result(
        'local_treating_early_stopped_trials_as_done')
    self.assertLen(result.trials, 5)
    self.assertEqual([t.status for t in result.trials], ['COMPLETED'] * 5)
    self.assertEqual([t.infeasible for t in result.trials],
                     [True, True, False, False, False])
    final_rewards = [t.final_measurement.reward if t.final_measurement else 0
                     for t in result.trials]
    self.assertEqual(final_rewards, [0.0, 0.0, 3.2, 0.64, 0.64])

  def test_tune_with_per_trial_early_stopping(self):
    job_log_dir = epath.Path(absltest.get_default_test_tmpdir())
    tuning_lib.tune(run_experiment,
                    TuningWithPerTrialEarlyStopping(),
                    platform.work_unit(),
                    job_log_dir,
                    study='local_per_trial_early_stopping',
                    max_num_trials=5)
    result = pg.tuning.poll_result('local_per_trial_early_stopping')
    self.assertLen(result.trials, 5)
    reward_at_step0 = (
        lambda t: t.measurements[0].reward if t.measurements else None)
    self.assertEqual([reward_at_step0(t) for t in result.trials],
                     [None, None, 1.6, None, None])
    self.assertEqual([len(t.measurements) for t in result.trials],
                     [0, 0, 3, 0, 0])
    self.assertEqual([t.infeasible for t in result.trials],
                     [True, True, False, True, True])

  def test_tune_with_per_trial_early_stopping_and_reward_replacement(self):
    job_log_dir = epath.Path(absltest.get_default_test_tmpdir())
    tuning_lib.tune(
        run_experiment,
        TuningWithPerTrialEarlyStoppingAndRewardReplacement(),
        platform.work_unit(),
        job_log_dir,
        study='local_per_trial_early_stopping_and_reward_replacement',
        max_num_trials=5)
    result = pg.tuning.poll_result(
        'local_per_trial_early_stopping_and_reward_replacement')
    self.assertLen(result.trials, 5)
    reward_at_step0 = (
        lambda t: t.measurements[0].reward if t.measurements else None)
    self.assertEqual([reward_at_step0(t) for t in result.trials],
                     [None, -1.0, 1.6, -1.0, -1.0])
    self.assertEqual([len(t.measurements) for t in result.trials],
                     [0, 1, 3, 1, 1])
    self.assertEqual([t.infeasible for t in result.trials],
                     [True, False, False, False, False])

  def test_tune_with_per_trial_early_stopping_and_metrics_replacement(self):
    job_log_dir = epath.Path(absltest.get_default_test_tmpdir())
    tuning_lib.tune(
        run_experiment,
        TuningWithPerTrialEarlyStoppingAndMetricsReplacement(),
        platform.work_unit(),
        job_log_dir,
        study='local_per_trial_early_stopping_and_metrics_replacement',
        max_num_trials=5)
    result = pg.tuning.poll_result(
        'local_per_trial_early_stopping_and_metrics_replacement')
    self.assertLen(result.trials, 5)
    reward_at_step0 = (
        lambda t: t.measurements[0].reward if t.measurements else None)
    self.assertEqual([reward_at_step0(t) for t in result.trials],
                     [None, 100., 1.6, 100., 100.0])
    self.assertEqual([t.final_measurement.reward for t in result.trials],
                     [0., 100., 3.2, 100., 100.0])
    self.assertEqual([len(t.measurements) for t in result.trials],
                     [0, 1, 3, 1, 1])
    self.assertEqual([t.infeasible for t in result.trials],
                     [True, False, False, False, False])

  def test_tune_with_population_wise_early_stopping_policy(self):
    job_log_dir = epath.Path(absltest.get_default_test_tmpdir())
    tuning_lib.tune(run_experiment,
                    TuningWithPopulationWiseEarlyStopping(),
                    platform.work_unit(),
                    job_log_dir,
                    study='local_population_wise_early_stopping',
                    max_num_trials=5)
    result = pg.tuning.poll_result('local_population_wise_early_stopping')
    self.assertLen(result.trials, 5)
    # We use the average of the metrics across steps as the final measurement.
    reward_at_step0 = (
        lambda t: t.measurements[0].reward if t.measurements else None)
    self.assertEqual([reward_at_step0(t) for t in result.trials],
                     [None, 0.32, 1.6, 0.64, 0.64])
    self.assertEqual([len(t.measurements) for t in result.trials],
                     [0, 1, 3, 1, 1])
    self.assertEqual([t.infeasible for t in result.trials],
                     [True, True, False, True, True])
    # We use the average of the metrics across steps as the final measurement.
    self.assertEqual([t.final_measurement.reward for t in result.trials],
                     [0.0, 0.0, 1.6 * 2, 0.0, 0.0])

  def test_tune_with_subexperiments(self):
    job_log_dir = epath.Path(absltest.get_default_test_tmpdir())
    tuning_lib.tune(run_experiment,
                    TuningWithSubExperiments(),
                    platform.work_unit(),
                    job_log_dir,
                    study='local_subexperiments',
                    max_num_trials=5)
    result = pg.tuning.poll_result('local_subexperiments')
    self.assertLen(result.trials, 5)
    self.assertEqual([t.infeasible for t in result.trials],
                     [True, False, True, True, True])
    self.assertEqual(result.trials[1].final_measurement.metrics, {
        'eval_test_abc/metrics/reward:1x': 0.64,
        'eval_test_abc/metrics/reward:2x': 1.28,
        'eval_test_abc/metrics/reward:4x': 2.56,
        'eval_test_abc/metrics/reward:8x': 5.12
    })
    # We use the average of the metrics across steps as the final measurement.
    self.assertEqual([t.final_measurement.reward for t in result.trials],
                     # Used aggregator='max' for computing the reward.
                     [0.0, 0.64 + 1.28 + 2.56 + 5.12, 0.0, 0.0, 0.0])

  def test_incorrectly_left_dataset_search_disabled(self):
    job_log_dir = epath.Path(absltest.get_default_test_tmpdir())
    with self.assertRaisesRegex(
        ValueError, 'Hyper primitive .* is not defined'
    ):
      tuning_lib.tune(
          run_experiment,
          TuningExperimentDatasetTuningDisabled(),
          platform.work_unit(),
          job_log_dir,
          study='local_incorrectly_not_searching_dataset_params',
          max_num_trials=5,
      )

  def test_bad_running_mode(self):
    job_log_dir = epath.Path(absltest.get_default_test_tmpdir())
    with self.assertRaisesRegex(
        ValueError,
        'Tuning uses training metrics but the reporting role is \'eval\''):
      tuning_lib.tune(run_experiment,
                      TuningWithTrainMetricsOnly(),
                      platform.work_unit(),
                      job_log_dir,
                      study='local_bad_running_mode1',
                      running_mode='eval')

    with self.assertRaisesRegex(
        ValueError,
        'Tuning uses decode metrics but the reporting role is \'eval\''):
      tuning_lib.tune(run_experiment,
                      TuningWithDecodeMetricsOnly(),
                      platform.work_unit(),
                      job_log_dir,
                      study='local_bad_running_mode2',
                      running_mode='eval')

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
    # Training step has not reached, but there is not a next checkpoint and
    # `train_to_end` is set to True.
    self.assertFalse(
        tuning_lib.is_last_checkpoint(
            trainer_lib.RunningMode.DECODE,
            global_step=900,
            num_train_steps=1000,
            eval_interval_steps=10,
            decode_interval_steps=10,
            save_interval_steps=400,
            train_to_end=True))


def get_trial_dirname(search_space_fun,
                      trial_id: int,
                      dna: pg.DNA,
                      combined_parameter_names: Optional[List[str]] = None,
                      root_dir: str = 'root') -> epath.Path:
  search_space = pg.hyper.trace(search_space_fun, require_hyper_name=True)
  dirname_generator = tuning_lib.TrialDirectoryNameGenerator(
      epath.Path(root_dir), search_space, combined_parameter_names)
  dna.use_spec(search_space.dna_spec)
  with search_space.apply(dna):
    return dirname_generator.dirname(trial_id)


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
        str(get_trial_dirname(_fn, 1, pg.DNA([0, 0]))), 'root/1/x=1|y=a')

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

  def test_trial_with_making_path_friendly(self):

    @pg.members([('x', pg.typing.Any())])
    class A(pg.Object):
      pass

    def _fn():
      return (pg.oneof(['a/b/c  ???', '"???\t  e_f-12/3\\4"'], name='x'),
              pg.oneof([A([A(A(1)), 1]), A([A(2)])], name='y'))
    self.assertEqual(
        str(get_trial_dirname(_fn, 1, pg.DNA([1, 0]))),
        'root/1/x=e_f1234|y=A(x={0=A(x=A(x=1)),1=1})')

  def test_trial_with_class_values(self):
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
