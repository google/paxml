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

"""AutoML utility library for PAX."""

import abc
import collections
import dataclasses
import math
import typing
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, Union

from paxml import automl_interfaces
# Placeholder for importing Google-internal tuning modules.
from praxis import base_hyperparams
from praxis import pax_fiddle
import pyglove as pg


BaseAlgorithm = automl_interfaces.BaseAlgorithm
BaseEarlyStoppingPolicy = automl_interfaces.BaseEarlyStoppingPolicy
BaseReward = automl_interfaces.BaseReward
CrossStepMetricAggregator = automl_interfaces.CrossStepMetricAggregator
SearchHParams = automl_interfaces.SearchHParams
MetricType = automl_interfaces.MetricType
Metric = automl_interfaces.Metric
MetricAggregator = automl_interfaces.MetricAggregator


# Aliases for Google-internal symbols.


# An AutoML experiment can have multiple sub-experiments enabled by
# `SubExperimentFactory`, which transforms an experiment config into
# a dictionary of sub-experiment id (str) to their configs.
# Since Vizier does not natively support merging metrics at a given step,
# we need to report metrics from sub-experiments at non-overlapping steps.
# To do so, we add a reporting step offset for each sub-experiment, and use
# this constant to compute the step space. This essentially means:
#
# When there are 3 sub-experiments in a tuning experiment:
#   the 1st sub-experiment starts its step 0 at step 0;
#   the 2nd sub-experiment starts its step 0 at step 1000_000_000;
#   the 3rd sub-experiment starts its step 0 at step 2000_000_000.
SUB_EXPERIMENT_STEP_OFFSET = 1000_000_000

#
# Common search hyperparameters.
#


def hyperparameter_tuning(
    metric: Metric,
    max_num_trials: int = 100,
    goal: str = 'maximize',
    *,
    enable_dataset_tuning: bool = False,
    errors_to_skip: Optional[
        List[Union[Type[Exception], Tuple[Type[Exception], str]]]
    ] = None,
    cross_step_metric_aggregator: Optional[
        pax_fiddle.Config[CrossStepMetricAggregator]
    ] = None,
    early_stopping: Optional[pax_fiddle.Config[BaseEarlyStoppingPolicy]] = None,
    reward_for_nan: Optional[float] = None,
) -> SearchHParams:
  """Returns a common search config for hyper-parameter tuning.

  Args:
    metric: The metric to optimize.
    max_num_trials: Max number of trials for tuning.
    goal: 'maximize' or 'minimize'.
    enable_dataset_tuning: Whether to enable dataset tuning.
    errors_to_skip: An optional field to specify on what errors the trial should
      be skipped. It's in the form of a list of (ExceptionType) or
      (ExceptionType, regexForError). For example, if users specify:
      `[RuntimeError, (Exception, 'XLACompilation.*')]`, the trails that
      RuntimeError or errors that match 'XLACompilation.*' will be treated as to
      skip.
    cross_step_metric_aggregator: An optional cross-step metric aggregator
      config indicating how metrics will be aggregated at the end of the search
      for computing the reward. If None, the last reported metrics will be used.
    early_stopping: An optional population-wise early stopping policy. If None,
      no population-wise early stopping policy will be used, though users still
      can raise `automl.EarlyStoppingError` to early terminate a a single trial
      during training/evaluation.
    reward_for_nan: An optional float used as the reward when metric value is
      NaN. If not specified, the reward will remain NaN so the trial will be
      skipped by the search algorithm.

  Returns:
    A search config object.
  """
  return SearchHParams(
      # Use Sweeping for hyperparameter tuning.
      search_algorithm=pax_fiddle.Config(Sweeping),
      search_reward=pax_fiddle.Config(
          SingleObjective,
          metric=metric,
          goal=goal,
          reward_for_nan=reward_for_nan,
      ),
      early_stopping=early_stopping,
      max_num_trials=max_num_trials,
      errors_to_skip=errors_to_skip,
      cross_step_metric_aggregator=cross_step_metric_aggregator,
      treats_early_stopped_trials_as_done=True,
      enable_dataset_tuning=enable_dataset_tuning,
  )


def neural_architecture_search(
    metrics: Union[Metric, Sequence[Metric]],
    cost_objective: Optional[float] = None,
    reward_type: str = 'tunas',
    exponent: float = -0.07,
    max_num_trials: int = 10000,
    errors_to_skip: Optional[List[
        Union[Type[Exception], Tuple[Type[Exception], str]]]] = None,
    cross_step_metric_aggregator: Optional[
        pax_fiddle.Config[CrossStepMetricAggregator]] = None,
    early_stopping: Optional[
        pax_fiddle.Config[BaseEarlyStoppingPolicy]] = None,
    reward_for_nan: Optional[float] = None
    ) -> SearchHParams:
  """Search params for Neural Architecture Search."""

  if isinstance(metrics, Metric):
    metrics = [metrics]

  if len(metrics) == 1:
    reward = pax_fiddle.Config(
        SingleObjective, metric=metrics[0], reward_for_nan=reward_for_nan
    )
  elif len(metrics) == 2:
    if cost_objective is None:
      raise ValueError('cost objective must be provided.')
    if reward_type == 'tunas':
      aggregator_cls = TunasAbsolute
    elif reward_type == 'mnas_hard':
      aggregator_cls = MnasHard
    elif reward_type == 'mnas_soft':
      aggregator_cls = MnasSoft
    else:
      raise ValueError('Unsupported reward type %r.' % reward_type)

    reward = pax_fiddle.Config(
        MultiObjective,
        metrics=metrics,
        aggregator_tpl=pax_fiddle.Config(
            aggregator_cls, cost_objective=cost_objective, exponent=exponent
        ),
        reward_for_nan=reward_for_nan,
    )
  else:
    raise ValueError('Only 1 or 2 metrics are supported.')

  return SearchHParams(
      search_algorithm=pax_fiddle.Config(RegularizedEvolution),
      search_reward=reward,
      early_stopping=early_stopping,
      max_num_trials=max_num_trials,
      errors_to_skip=errors_to_skip,
      cross_step_metric_aggregator=cross_step_metric_aggregator,
  )


#
# Concrete search algorithms.
#


class RandomSearch(BaseAlgorithm):
  """Random search.

  Comparing to the VizierBuiltin('RANDOM_SEARCH'), PyGlove's random search
  supports `pg.manyof` with constraints.

  Attributes:
    seed: Seed of the Random search.
  """
  seed: Optional[int] = None

  def __call__(self):
    return pg.geno.Random(seed=self.seed)


class Sweeping(BaseAlgorithm):
  """Sweeping all possible combinations.

  Comparing to the VizierBuiltin('GRID_SEARCH'), PyGlove's sweeping algorithm
  supports `pg.manyof` with constraints.
  However, it does not support `pg.floatv`.
  """

  def __call__(self):
    return pg.geno.Sweeping()


class RegularizedEvolution(BaseAlgorithm):
  """Regularized evolution.

  Reference:
  https://arxiv.org/abs/1802.01548.

  Attributes:
    mutator: Mutator to use.
    population_size: Population size.
    tournament_size: Tournament size.
    seed: Random seed.
  """
  mutator: pg.evolution.Mutator = pg.evolution.mutators.Uniform()  # pytype: disable=annotation-type-mismatch
  population_size: int = 100
  tournament_size: int = 10
  seed: Optional[int] = None

  def __call__(self):
    return pg.evolution.regularized_evolution(
        self.mutator,
        population_size=self.population_size,
        tournament_size=self.tournament_size,
        seed=self.seed,
    )


#
# Concrete search rewards.
#


class SingleObjective(BaseReward):
  """Single objective reward.

  Attributes:
    metric: The key of metric whose value will be used as reward.
    goal: Defines how the metric should be optimized. Acceptable values are
      'maximize' or 'minimize'.
    reward_for_nan: An optional float used as the reward when metric value is
      NaN. If not specified, the reward will remain NaN so the trial will be
      skipped by the search algorithm.
  """
  metric: Optional[Metric] = None
  goal: str = 'maximize'
  reward_for_nan: Optional[float] = None

  def __post_init__(self):
    super().__post_init__()
    if self.metric is None:
      raise ValueError('Param `metric` should not be None.')
    if self.goal not in ['maximize', 'minimize']:
      raise ValueError(
          "Param `goal` should be either 'maximize' or 'minimize'."
      )

  def __call__(self, metrics_dict: Dict[str, float], global_step: int) -> float:
    del global_step
    assert self.metric is not None
    reward = self.metric.get_value(metrics_dict)

    if self.goal == 'minimize':
      reward *= -1
    if self.reward_for_nan is not None and math.isnan(reward):
      reward = self.reward_for_nan
    return reward

  @property
  def used_metrics(self) -> Sequence[Metric]:
    assert self.metric is not None
    return [self.metric]


class MultiObjectiveAggregator(
    base_hyperparams.FiddleBaseParameterizable, metaclass=abc.ABCMeta
):
  """Base class for multi objective aggregators."""

  @abc.abstractmethod
  def __call__(self, values: Sequence[float]) -> Union[float, complex]:
    """Aggregate multiple values into a single value."""


class MultiObjective(BaseReward):
  """Multi-objective reward.

  Attributes:
    metrics: The keys of metric whose value will be used as reward.
    aggregator_tpl: Multi-objective aggregator for coupling multiple values into
      a single float value.
    goal: Defines how the metric should be optimized. Acceptable values are
      'maximize' or 'minimize'.
    reward_for_nan: An optional float used as the reward when metric value is
      NaN. If not specified, the reward will remain NaN so the trial will be
      skipped by the search algorithm.
  """
  metrics: Optional[Sequence[Metric]] = None
  aggregator_tpl: Optional[pax_fiddle.Config[MultiObjectiveAggregator]] = None
  goal: str = 'maximize'
  reward_for_nan: Optional[float] = None
  _aggregator: Any = dataclasses.field(init=False, repr=False)

  def __post_init__(self):
    if not self.metrics:
      raise ValueError('Param `metrics` must be provided.')

    if len(self.metrics) > 1 and self.aggregator_tpl is None:
      raise ValueError('Param `aggregator` must be provided.')
    super().__post_init__()
    if self.aggregator_tpl is not None:
      self._aggregator = self.aggregator_tpl.Instantiate()

  def __call__(self, metrics_dict: Dict[str, float], global_step: int) -> float:
    del global_step
    metric_values = [m.get_value(metrics_dict) for m in self.metrics]
    if self.reward_for_nan is not None and any(
        math.isnan(m) for m in metric_values
    ):
      return self.reward_for_nan
    if len(metric_values) == 1:
      reward = metric_values[0]
    else:
      assert self._aggregator is not None
      reward = self._aggregator(metric_values)
    if self.goal == 'minimize':
      reward *= -1
    return reward

  @property
  def used_metrics(self) -> Sequence[Metric]:
    assert self.metrics is not None
    return self.metrics


class WeightedSumAggregator(MultiObjectiveAggregator):
  """Weighted sum multiple objectives.

  Attributes:
    weights: A sequence of float as the weights for the objectives to optimize.
      Its value does not need to sum to 1.
  """

  weights: Optional[Sequence[float]] = None
  _sum_of_weights: Any = dataclasses.field(init=False, repr=False)

  def __post_init__(self):
    super().__post_init__()
    weights = self.weights
    weights = typing.cast(Sequence[float], weights)
    if not weights or sum([abs(w) for w in weights]) == 0:
      raise ValueError(f'Invalid value for `weights`: {weights}')
    self._sum_of_weights = sum([abs(w) for w in weights]) * 1.0

  def __call__(self, values: Sequence[float]) -> Union[float, complex]:
    """Aggregate multiple values into a single value."""
    if len(values) != len(self.weights):
      raise ValueError(
          f'The length of weights ({self.weights}) does not match '
          f'the length of objective values {values!r}.'
      )
    return (
        sum([w * v for w, v in zip(self.weights, values)])
        / self._sum_of_weights
    )


def weighted_sum_reward(
    metrics_and_weights: Sequence[Tuple[Metric, float]],
    goal: str = 'maximize',
    reward_for_nan: Optional[float] = None,
) -> pax_fiddle.Config[MultiObjective]:
  """Returns a reward by weighted summing multiple metrics."""
  metrics = [m for m, _ in metrics_and_weights]
  weights = [w for _, w in metrics_and_weights]
  return pax_fiddle.Config(
      MultiObjective,
      metrics=metrics,
      goal=goal,
      aggregator_tpl=pax_fiddle.Config(WeightedSumAggregator, weights=weights),
      reward_for_nan=reward_for_nan,
  )


class TwoObjectiveAggregator(MultiObjectiveAggregator):
  """Base class for two-objective aggregator.

  Attributes:
    cost_objective: A float value as cost objective.
    exponent: A float exponent controlling the trade-off between quality and
      cost. The more negative this exponent is, the more heavily the reward will
      penalize this model with cost larger than cost objective.
  """
  cost_objective: Optional[float] = None
  exponent: float = -0.07

  def __post_init__(self):
    super().__post_init__()
    if self.cost_objective is None:
      raise ValueError('Param `cost_objective` must be provided.')

  def __call__(self, values: Sequence[float]) -> Union[float, complex]:
    """Aggregate multiple values into a single value."""
    if len(values) != 2:
      raise ValueError('Only two objectives are supported. Encountered: %r' %
                       values)
    return self.aggregate(*values)

  @abc.abstractmethod
  def aggregate(self, quality: float, cost: float) -> Union[float, complex]:
    """Aggregate quality and cost into a single value."""


class TunasAbsolute(TwoObjectiveAggregator):
  """TuNAS absolute reward.

  This aggregator promotes models with absolute cost towards objective.

  Formula: reward = accuracy + exponent * abs(cost / objective - 1)
  This is the absolute reward scheme described in the TuNAS paper.

  https://arxiv.org/abs/2008.06120
  """

  def aggregate(self, quality: float, cost: float) -> Union[float, complex]:
    """Aggregate quality and cost into a single value."""
    cost_ratio = cost / self.cost_objective
    cost_adjustment = self.exponent * abs(cost_ratio - 1)
    return quality + cost_adjustment


class MnasHard(TwoObjectiveAggregator):
  """Mnas hard reward.

  This aggregator penalizes models with larger cost than objective.

  Formula: accuracy * min((cost / objective) ^ exponent, 1)
  This is the hard exponential reward scheme described in the MNAS paper.
  https://arxiv.org/pdf/1807.11626.pdf
  """

  def aggregate(self, quality: float, cost: float) -> Union[float, complex]:
    """Aggregate quality and cost into a single value."""
    cost_ratio = cost / self.cost_objective
    cost_adjustment = min(pow(cost_ratio, self.exponent), 1.0)
    return quality * cost_adjustment


class MnasSoft(TwoObjectiveAggregator):
  """Mnas soft reward.

  This aggregator soft balances between accuracy and cost towards a cost
  objective.

  Formula: accuracy * (cost / objective) ^ exponent
  This is the soft exponential reward scheme described in the MNAS paper:
  https://arxiv.org/pdf/1807.11626.pdf
  """

  def aggregate(self, quality: float, cost: float) -> Union[float, complex]:
    """Aggregate quality and cost into a single value."""
    cost_ratio = cost / self.cost_objective
    cost_adjustment = pow(cost_ratio, self.exponent)
    return quality * cost_adjustment


#
# Common metric aggregators across multiple steps.
#


class MultiSubExperimentCrossStepMetricAggregator(CrossStepMetricAggregator):
  """Metric aggregator with sub-experiment support."""

  def __call__(
      self, metrics_across_steps: Sequence[Tuple[int, Dict[str, float]]]
      ) -> Dict[str, float]:
    merged_metrics_across_steps = self._merge_metrics(metrics_across_steps)
    return self.call(merged_metrics_across_steps)

  def _merge_metrics(
      self, metrics_across_steps: Sequence[Tuple[int, Dict[str, float]]]
      ) -> Sequence[Tuple[int, Dict[str, float]]]:
    """Merges metrics from sub-experiments."""
    merged_metrics_across_steps = collections.defaultdict(dict)
    for step, metrics in metrics_across_steps:
      step = step % SUB_EXPERIMENT_STEP_OFFSET
      merged_metrics_across_steps[step].update(metrics)
    return [(s, m) for s, m in merged_metrics_across_steps.items()]

  @abc.abstractmethod
  def call(
      self, merged_metrics_across_steps: Sequence[Tuple[int, Dict[str, float]]]
      ) -> Dict[str, float]:
    """Aggregates metrics from merged metrics from multiple sub-experiments."""


class LastReportedMetricValues(MultiSubExperimentCrossStepMetricAggregator):
  """Returns the last reported metrics."""

  def call(
      self, merged_metrics_across_steps: Sequence[Tuple[int, Dict[str, float]]]
      ) -> Dict[str, float]:
    """Returns an aggregated metric dict from metrics from multiple steps."""
    return merged_metrics_across_steps[-1][1]


class AverageMetricValues(MultiSubExperimentCrossStepMetricAggregator):
  """Returns the average values of per-step metrics.

  Attributes:
    last_n: If not None, then only the `last_n` values will be used in the
      metric average. If None, all values are used.
  """
  last_n: Optional[int] = None

  def call(
      self, merged_metrics_across_steps: Sequence[Tuple[int, Dict[str, float]]]
      ) -> Dict[str, float]:
    """Returns an aggregated metric dict from metrics from multiple steps."""
    accumulated_metrics = collections.defaultdict(list)
    for _, step_metrics in merged_metrics_across_steps:
      for k, v in step_metrics.items():
        accumulated_metrics[k].append(v)

    metrics = {}
    for k, v in accumulated_metrics.items():
      if self.last_n is not None:
        metrics[k] = sum(v[-self.last_n :]) / len(v[-self.last_n :])
      else:
        metrics[k] = sum(v) / len(v)
    return metrics


class MetricsWithMaxValue(MultiSubExperimentCrossStepMetricAggregator):
  """Returns the step metrics which has the max value on a metric.

  Attributes:
    metric: An optional metric against whom to choose the max value. If None,
      the comparison is against the reward.
  """
  metric: Optional[Metric] = None

  def call(
      self, merged_metrics_across_steps: Sequence[Tuple[int, Dict[str, float]]]
      ) -> Dict[str, float]:
    """Returns an aggregated metric dict from metrics from multiple steps."""
    metric = self.metric or Metric('reward')
    max_i, max_value = None, None
    # TODO(daiyip): current code assumes that all metrics are reported at
    # the same step, which might not be the case if we allow multiple processes
    # to report the measurement. We may need to revisit the implementation once
    # multi-role reporting is supported.
    for i, (_, step_metrics) in enumerate(merged_metrics_across_steps):
      v = metric.get_value(step_metrics)
      if max_value is None or v >= max_value:
        max_i, max_value = i, v
    return merged_metrics_across_steps[max_i][1]


class MetricsWithMinValue(MultiSubExperimentCrossStepMetricAggregator):
  """Returns the step metrics which has the min value on an metric.

  Attributes:
    metric: An optional metric against whom to choose the max value. If None,
      the comparison is against the reward.
  """
  metric: Optional[Metric] = None

  def call(
      self, merged_metrics_across_steps: Sequence[Tuple[int, Dict[str, float]]]
      ) -> Dict[str, float]:
    """Returns an aggregated metric dict from metrics from multiple steps."""
    metric = self.metric or Metric('reward')
    min_i, min_value = None, None
    for i, (_, step_metrics) in enumerate(merged_metrics_across_steps):
      v = metric.get_value(step_metrics)
      if min_value is None or v <= min_value:
        min_i, min_value = i, v
    return merged_metrics_across_steps[min_i][1]

#
# Population-wiise early stopping policies.
#


class EarlyStoppingByValue(BaseEarlyStoppingPolicy):
  """Early stopping based on the absolute value of a metric at a step.

  Attributes:
    step_values: A list of tuples for defining gating rules: (step, threshold
      value).
    metric: Metric to watch. If None, it watches the reward at the step.
    maximize: If True, value below the threshold will be stopped. Otherwise
      values above the threshold.
  """
  step_values: Optional[List[Tuple[int, float]]] = None
  metric: Optional[Metric] = None
  maximize: bool = True

  def __call__(self) -> pg.early_stopping.StepWise:
    def metric_to_watch(m: pg.tuning.Measurement) -> float:
      metric = self.metric
      if metric is None:
        return m.reward
      return metric.get_value(m.metrics)
    return pg.early_stopping.early_stop_by_value(
        step_values=self.step_values,
        metric=metric_to_watch,
        maximize=self.maximize,
    )()


class EarlyStoppingByRank(BaseEarlyStoppingPolicy):
  """Early stopping based on the rank of a metric at a step.

  Attributes:
    step_ranks: A list of tuples for defining gating rules: (step, threshold
      rank or threshold percentage, min_histogram_size).
    metric: Metric to watch. If None, it watches the reward at the step.
    maximize: If True, the sorting for computing the rank is from the largest to
      the smallest, otherwise will be the smallest to the largest.
  """
  step_ranks: Optional[List[Tuple[int, Union[int, float], int]]] = None
  metric: Optional[Metric] = None
  maximize: bool = True

  def __call__(self) -> pg.early_stopping.StepWise:
    def metric_to_watch(m: pg.tuning.Measurement) -> float:
      metric = self.metric
      if metric is None:
        return m.reward
      return metric.get_value(m.metrics)
    return pg.early_stopping.early_stop_by_rank(
        step_ranks=self.step_ranks,
        metric=metric_to_watch,
        maximize=self.maximize,
    )()


#
# Exception used to early stop a trial.
#


class EarlyStoppingError(BaseException):
  """Early stopping signal which can be thrown from the program."""

  def __init__(self,
               skip: bool = True,
               skip_reason: Optional[str] = None,
               step: Optional[int] = None,
               reward: Optional[float] = None,
               metrics: Optional[Dict[str, float]] = None,
               checkpoint_path: Optional[str] = None):
    """Constructor.

    Args:
      skip: If True, marks current trial as infeasible.
      skip_reason: Reason to skip. Applicable when `skip` is set to True.
      step: At which step the early stopping is triggered. Required when `skip`
        is set to False.
      reward: Final reward to report. Applicable when `skip` is set to False. If
        provided, it will be used as the final reward value. Otherwise a reward
        will be computed from the `metrics` based on the reward function for the
        search.
      metrics: Metrics to report for current trial. Applicable when `skip` is
        set to False. Either `final_objective` or `metrics` or both should be
        provided when `skip` is set to False.
      checkpoint_path: Checkpoint path used for current trial.
    """
    if not skip:
      if step is None:
        raise ValueError('`step` must be provided when `skip` is set to False.')
      if reward is None and metrics is None:
        raise ValueError(
            'At least one of `reward` and `metrics` should be provided '
            'when `skip` is set to False.')

    super().__init__(
        'Early stopping signal, which should be caught by the tuning loop.')

    self.skip = skip
    self.skip_reason = skip_reason
    self.step = step
    self.reward = reward
    self.metrics = metrics
    self.checkpoint = checkpoint_path


#
# Decorators for parameter sweeping.
#

COMBINED_DECISION_ATTR = 'PARAMETER_SWEEP_COMBINATION'
COMBINED_DECISION_POINT_NAMES = 'COMBINED_DECISION_POINT_NAMES'


def parameter_sweep(
    combinations: Optional[List[Tuple[Any, ...]]] = None,
    *,
    metric: Optional[Metric] = None,
    goal: str = 'maximize') -> Callable[[Type[Any]], Type[Any]]:
  """Returns a decorator for enabling parameter sweeping on a PAX experiment.

  Args:
    combinations: A list of tuples representing parameter combinations to
      sweep. If None, the cartesian product from all `pg.oneof` will be swept.
      When specified, the first row (`combinations[0]`) shall be a tuple of
      strings, which are the names of the class attributes that will be swept.
      Then it follows with one or multiple rows - each provides a combination
      of parameter values for the header row, representing a single trial.

      For example:

        ```
        @automl.parameter_sweep([
            ('LEARNING_RATE', 'HIDDEN_DIMS'),
            (0.1, 32),
            (0.01, 64),
            (0.001, 128),
        ])
        class MySweepingExperiment(MyExperiment):
          pass
        ```

    metric: Metric to use as trial objective. If None, trial objective will be
      a constant 0, and users will be able to see all metrics in Vizier
      dashboard.
    goal: Goal for optimization. By default, it's to maximize the metric.

  Returns:
    A new experiment class that is derived from the user class.
  """
  if combinations is not None:
    if not isinstance(combinations, list) or len(combinations) < 2:
      raise ValueError(
          f'`combinations` must be a list (of tuples) with at least two items. '
          f'Encountered: {combinations}.')
    num_attrs = None
    for i, row in enumerate(combinations):
      if not isinstance(row, tuple) or (
          num_attrs is not None and len(row) != num_attrs):
        raise ValueError(
            f'`combinations` must be a list of tuples of equal length. '
            f'Encountered bad row {row!r} (index={i}) from {combinations!r}')
      num_attrs = len(row)
      if num_attrs == 0:
        raise ValueError(
            f'`combinations` must have at least 1 columns. '
            f'Encountered: {combinations!r}')
      if i == 0:
        if not all(isinstance(name, str) for name in row):
          raise ValueError(
              f'The first row of `combinations` must be a list '
              f'of class attribute names. Encountered: {row!r}')

  if metric is None:
    search_reward = None
  else:
    search_reward = pax_fiddle.Config(SingleObjective, metric=metric, goal=goal)

  def decorator(cls):
    class _ParameterSweeping(cls):

      def search(self):
        del self
        return SearchHParams(
            search_algorithm=pax_fiddle.Config(Sweeping),
            search_reward=search_reward,
            treats_early_stopped_trials_as_done=True,
            train_to_end=True,
            # Consider making toggleable b/285879603
            enable_dataset_tuning=True,
        )

    new_cls = _ParameterSweeping
    # Create a COMBINATION property and use it to set HP attributes' values.
    if combinations:
      attr_names = combinations[0]
      assert attr_names
      def create_getter(i: int):
        def _getter(self):
          return getattr(self, COMBINED_DECISION_ATTR)[i]
        return _getter
      for i, attr_name in enumerate(attr_names):
        if not hasattr(cls, attr_name):
          raise ValueError(f'Attribute {attr_name!r} does not exist in {cls!r}')
        setattr(cls, attr_name, property(create_getter(i)))

      setattr(new_cls, COMBINED_DECISION_ATTR, pg.oneof(combinations[1:]))
      setattr(new_cls, COMBINED_DECISION_POINT_NAMES, attr_names)
      automl_interfaces.enable_class_level_hyper_primitives(new_cls)

    setattr(new_cls, '__name__', cls.__name__)
    setattr(new_cls, '__module__', cls.__module__)
    setattr(new_cls, '__qualname__', cls.__qualname__)
    return new_cls

  return decorator
