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

"""AutoML utility library for PAX."""

import abc
import collections
import inspect
import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, Union
from paxml import automl_interfaces
# Placeholder for importing Google-internal tuning modules.
from praxis import base_hyperparams
import pyglove as pg


BaseParameterizable = base_hyperparams.BaseParameterizable
InstantiableHyperParams = base_hyperparams.InstantiableHyperParams

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
    errors_to_skip: Optional[List[
        Union[Type[Exception], Tuple[Type[Exception], str]]]] = None,
    cross_step_metric_aggregator: Optional[
        CrossStepMetricAggregator.HParams] = None,
    early_stopping: Optional[
        BaseEarlyStoppingPolicy.HParams] = None,
    reward_for_nan: Optional[float] = None) -> SearchHParams:
  """Returns a common search HParams for hyper-parameter tuning.

  Args:
    metric: The metric to optimize.
    max_num_trials: Max number of trials for tuning.
    goal: 'maximize' or 'minimize'.
    errors_to_skip: An optional field to specify on what errors the trial
      should be skipped. It's in the form of a list of (ExceptionType) or
      (ExceptionType, regexForError). For example, if users specify:
      `[RuntimeError, (Exception, 'XLACompilation.*')]`, the trails that
      RuntimeError or errors that match 'XLACompilation.*' will be treated as
      to skip.
    cross_step_metric_aggregator: An optional cross-step metric aggregator
      hparams indicating how metrics will be aggregated at the end of the search
      for computing the reward. If None, the last reported metrics will be used.
    early_stopping: An optional population-wise early stopping policy.
      If None, no population-wise early stopping policy will be used, though
      users still can raise `automl.EarlyStoppingError` to early terminate a
      a single trial during training/evaluation.
    reward_for_nan: An optional float used as the reward when metric value is
      NaN. If not specified, the reward will remain NaN so the trial will be
      skipped by the search algorithm.

  Returns:
    A search HParams object.
  """
  return SearchHParams(
      # Use Sweeping for hyperparameter tuning.
      search_algorithm=Sweeping.HParams(),
      search_reward=SingleObjective.HParams(
          metric=metric, goal=goal, reward_for_nan=reward_for_nan),
      early_stopping=early_stopping,
      max_num_trials=max_num_trials,
      errors_to_skip=errors_to_skip,
      cross_step_metric_aggregator=cross_step_metric_aggregator)


def neural_architecture_search(
    metrics: Union[Metric, Sequence[Metric]],
    cost_objective: Optional[float] = None,
    reward_type: str = 'tunas',
    exponent: float = -0.07,
    max_num_trials: int = 10000,
    errors_to_skip: Optional[List[
        Union[Type[Exception], Tuple[Type[Exception], str]]]] = None,
    cross_step_metric_aggregator: Optional[
        CrossStepMetricAggregator.HParams] = None,
    early_stopping: Optional[
        BaseEarlyStoppingPolicy.HParams] = None,
    reward_for_nan: Optional[float] = None
    ) -> SearchHParams:
  """Search params for Neural Architecture Search."""

  if isinstance(metrics, Metric):
    metrics = [metrics]

  if len(metrics) == 1:
    reward = SingleObjective.HParams(
        metric=metrics[0], reward_for_nan=reward_for_nan)
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

    reward = MultiObjective.HParams(
        metrics=metrics,
        aggregator=aggregator_cls.HParams(
            cost_objective=cost_objective, exponent=exponent),
        reward_for_nan=reward_for_nan)
  else:
    raise ValueError('Only 1 or 2 metrics are supported.')

  return SearchHParams(
      search_algorithm=RegularizedEvolution.HParams(),
      search_reward=reward,
      early_stopping=early_stopping,
      max_num_trials=max_num_trials,
      errors_to_skip=errors_to_skip,
      cross_step_metric_aggregator=cross_step_metric_aggregator)


#
# Concrete search algorithms.
#


class RandomSearch(BaseAlgorithm):
  """Random search.

  Comparing to the VizierBuiltin('RANDOM_SEARCH'), PyGlove's random search
  supports `pg.manyof` with constraints.
  """

  class HParams(BaseAlgorithm.HParams):
    """Hyperparameters for RandomSearch.

    Attributes:
      seed: Seed of the Random search.
    """
    seed: Optional[int] = None

  def __call__(self):
    return pg.geno.Random(seed=self._hparams.seed)


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
  """

  class HParams(BaseAlgorithm.HParams):
    """Hyperparameters for PPO.

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
        self._hparams.mutator,
        population_size=self._hparams.population_size,
        tournament_size=self._hparams.tournament_size,
        seed=self._hparams.seed)


#
# Concrete search rewards.
#


class SingleObjective(BaseReward):
  """Single objective reward."""

  class HParams(BaseReward.HParams):
    """Hyperparameters for SingleObjective.

    Attributes:
      metric_key: The key of metric whose value will be used as reward.
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
            'Param `goal` should be either \'maximize\' or \'minimize\'.')

  def __call__(self, metrics_dict: Dict[str, float], global_step: int) -> float:
    del global_step
    reward = self._hparams.metric.get_value(metrics_dict)

    if self._hparams.goal == 'minimize':
      reward *= -1
    if self._hparams.reward_for_nan is not None and math.isnan(reward):
      reward = self._hparams.reward_for_nan
    return reward

  @property
  def used_metrics(self) -> Sequence[Metric]:
    assert self._hparams.metric is not None
    return [self._hparams.metric]


class MultiObjectiveAggregator(BaseParameterizable, metaclass=abc.ABCMeta):
  """Base class for multi objective aggregators."""

  class HParams(InstantiableHyperParams):
    """Hyperparameters for MultiObjectiveAggregator."""

  @abc.abstractmethod
  def __call__(self, values: Sequence[float]) -> float:
    """Aggregate multiple values into a single value."""


class MultiObjective(BaseReward):
  """Multi-objective reward."""

  class HParams(BaseReward.HParams):
    """Hyperparameters for SingleObjective.

    Attributes:
      metric_keys: The keys of metric whose value will be used as reward.
      aggregator: Multi-objective aggregator for coupling multiple values into a
        single float value.
      reward_for_nan: An optional float used as the reward when metric value is
        NaN. If not specified, the reward will remain NaN so the trial will be
        skipped by the search algorithm.
    """
    metrics: Optional[Sequence[Metric]] = None
    aggregator: Optional[MultiObjectiveAggregator.HParams] = None
    reward_for_nan: Optional[float] = None

    def __post_init__(self):
      super().__post_init__()
      if not self.metrics:
        raise ValueError('Param `metrics` must be provided.')

      if len(self.metrics) > 1 and self.aggregator is None:
        raise ValueError('Param `aggregator` must be provided.')

  def __init__(self, hparams: HParams):
    super().__init__(hparams)
    if self._hparams.aggregator is not None:
      self._aggregator = self._hparams.aggregator.Instantiate()

  def __call__(self, metrics_dict: Dict[str, float], global_step: int) -> float:
    del global_step
    metric_values = [m.get_value(metrics_dict) for m in self._hparams.metrics]
    if (self._hparams.reward_for_nan is not None
        and any(math.isnan(m) for m in metric_values)):
      return self._hparams.reward_for_nan
    if len(metric_values) == 1:
      return metric_values[0]
    assert self._aggregator is not None
    return self._aggregator(metric_values)

  @property
  def used_metrics(self) -> Sequence[Metric]:
    assert self._hparams.metrics is not None
    return self._hparams.metrics


class TwoObjectiveAggregator(MultiObjectiveAggregator):
  """Base class for two-objective aggregator."""

  class HParams(MultiObjectiveAggregator.HParams):
    """Hyperparameters for SingleObjective.

    Attributes:
      cost_objective: A float value as cost objective.
      exponent: A float exponent controlling the trade-off between quality and
        cost. The more negative this exponent is, the more heavily the reward
        will penalize this model with cost larger than cost objective.
    """
    cost_objective: Optional[float] = None
    exponent: float = -0.07

    def __post_init__(self):
      super().__post_init__()
      if self.cost_objective is None:
        raise ValueError('Param `cost_objective` must be provided.')

  def __call__(self, values: Sequence[float]) -> float:
    """Aggregate multiple values into a single value."""
    if len(values) != 2:
      raise ValueError('Only two objectives are supported. Encountered: %r' %
                       values)
    return self.aggregate(*values)

  @abc.abstractmethod
  def aggregate(self, quality: float, cost: float) -> float:
    """Aggregate quality and cost into a single value."""


class TunasAbsolute(TwoObjectiveAggregator):
  """TuNAS absolute reward.

  This aggregator promotes models with absolute cost towards objective.

  Formula: reward = accuracy + exponent * abs(cost / objective - 1)
  This is the absolute reward scheme described in the TuNAS paper.

  https://arxiv.org/abs/2008.06120
  """

  def aggregate(self, quality: float, cost: float) -> float:
    """Aggregate quality and cost into a single value."""
    cost_ratio = cost / self._hparams.cost_objective
    cost_adjustment = self._hparams.exponent * abs(cost_ratio - 1)
    return quality + cost_adjustment


class MnasHard(TwoObjectiveAggregator):
  """Mnas hard reward.

  This aggregator penalizes models with larger cost than objective.

  Formula: accuracy * min((cost / objective) ^ exponent, 1)
  This is the hard exponential reward scheme described in the MNAS paper.
  https://arxiv.org/pdf/1807.11626.pdf
  """

  def aggregate(self, quality: float, cost: float) -> float:
    """Aggregate quality and cost into a single value."""
    cost_ratio = cost / self._hparams.cost_objective
    cost_adjustment = min(pow(cost_ratio, self._hparams.exponent), 1.)
    return quality * cost_adjustment


class MnasSoft(TwoObjectiveAggregator):
  """Mnas soft reward.

  This aggregator soft balances between accuracy and cost towards a cost
  objective.

  Formula: accuracy * (cost / objective) ^ exponent
  This is the soft exponential reward scheme described in the MNAS paper:
  https://arxiv.org/pdf/1807.11626.pdf
  """

  def aggregate(self, quality: float, cost: float) -> float:
    """Aggregate quality and cost into a single value."""
    cost_ratio = cost / self._hparams.cost_objective
    cost_adjustment = pow(cost_ratio, self._hparams.exponent)
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
  """Returns the average values of per-step metrics."""

  class HParams(MultiSubExperimentCrossStepMetricAggregator.HParams):
    """Hyperparameters for AverageMetricValues.

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
      if self._hparams.last_n is not None:
        metrics[k] = sum(v[-self._hparams.last_n:]) / len(
            v[-self._hparams.last_n:])
      else:
        metrics[k] = sum(v) / len(v)
    return metrics


class MetricsWithMaxValue(MultiSubExperimentCrossStepMetricAggregator):
  """Returns the step metrics which has the max value on a metric."""

  class HParams(MultiSubExperimentCrossStepMetricAggregator.HParams):
    """Hyperparameters for ValueWithMax.

    Attributes:
      metric: An optional metric against whom to choose the max value.
        If None, the comparison is against the reward.
    """
    metric: Optional[Metric] = None

  def call(
      self, merged_metrics_across_steps: Sequence[Tuple[int, Dict[str, float]]]
      ) -> Dict[str, float]:
    """Returns an aggregated metric dict from metrics from multiple steps."""
    metric = self._hparams.metric or Metric('reward')
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
  """Returns the step metrics which has the min value on an metric."""

  class HParams(MultiSubExperimentCrossStepMetricAggregator.HParams):
    """Hyperparameters for ValueWithMax.

    Attributes:
      metric: An optional metric against whom to choose the max value.
        If None, the comparison is against the reward.
    """
    metric: Optional[Metric] = None

  def call(
      self, merged_metrics_across_steps: Sequence[Tuple[int, Dict[str, float]]]
      ) -> Dict[str, float]:
    """Returns an aggregated metric dict from metrics from multiple steps."""
    metric = self._hparams.metric or Metric('reward')
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
  """Early stopping based on the absolute value of a metric at a step."""

  class HParams(BaseEarlyStoppingPolicy.HParams):
    """Hyperparameters for value-based early stopping policy.

    Attributes:
      step_values: A list of tuples for defining gating rules:
        (step, threshold value).
      metric: Metric to watch. If None, it watches the reward at the step.
      maximize: If True, value below the threshold will be stopped. Otherwise
        values above the threshold.
    """
    step_values: Optional[List[Tuple[int, float]]] = None
    metric: Optional[Metric] = None
    maximize: bool = True

  def __call__(self) -> pg.early_stopping.StepWise:
    def metric_to_watch(m: pg.tuning.Measurement) -> float:
      metric = self._hparams.metric
      if metric is None:
        return m.reward
      return metric.get_value(m.metrics)
    return pg.early_stopping.early_stop_by_value(
        step_values=self._hparams.step_values,
        metric=metric_to_watch,
        maximize=self._hparams.maximize)()


class EarlyStoppingByRank(BaseEarlyStoppingPolicy):
  """Early stopping based on the rank of a metric at a step."""

  class HParams(BaseEarlyStoppingPolicy.HParams):
    """Hyperparameters for rank-based early stopping policy.

    Attributes:
      step_ranks: A list of tuples for defining gating rules:
        (step, threshold rank or threshold percentage, min_histogram_size).
      metric: Metric to watch. If None, it watches the reward at the step.
      maximize: If True, the sorting for computing the rank is from the largest
        to the smallest, otherwise will be the smallest to the largest.
    """
    step_ranks: Optional[List[Tuple[int, Union[int, float], int]]] = None
    metric: Optional[Metric] = None
    maximize: bool = True

  def __call__(self) -> pg.early_stopping.StepWise:
    def metric_to_watch(m: pg.tuning.Measurement) -> float:
      metric = self._hparams.metric
      if metric is None:
        return m.reward
      return metric.get_value(m.metrics)
    return pg.early_stopping.early_stop_by_rank(
        step_ranks=self._hparams.step_ranks,
        metric=metric_to_watch,
        maximize=self._hparams.maximize)()


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


def enable_class_level_hyper_primitives(cls: Type[Any]) -> None:
  """Enable class-level hypers for a BaseExperiment subclass."""

  def create_hyper_property(name: str, hyper: pg.hyper.HyperPrimitive):
    attr_name = f'_PROPERTY_{name}'
    hyper_kwargs = dict(hyper.sym_init_args)
    if 'name' not in hyper_kwargs or hyper_kwargs['name'] is None:
      hyper_kwargs['name'] = name

    def getter(x):
      if hasattr(x, attr_name):
        return getattr(x, attr_name)
      return hyper.__class__(**hyper_kwargs)  # pytype: disable=not-instantiable

    def setter(x, v):
      setattr(x, attr_name, v)

    return property(getter, setter)

  for name, hyper in inspect.getmembers(
      cls, lambda x: isinstance(x, pg.hyper.HyperPrimitive)):
    setattr(cls, name, create_hyper_property(name, hyper))


#
# Decorators for parameter sweeping.
#


def parameter_sweep(
    metric: Optional[Metric] = None,
    goal: str = 'maximize') -> Callable[[Type[Any]], Type[Any]]:
  """Returns a decorator for enabling parameter sweeping on a PAX experiment.

  Args:
    metric: Metric to use as trial objective. If None, trial objective will be
      0, and users will be able to see all metrics in Vizier dashboard.
    goal: Goal for optimization. By default, it's to maximize the metric.

  Returns:
    A callable that adds the `search` method to the experiment class.
  """
  if metric is None:
    search_reward = None
  else:
    search_reward = SingleObjective.HParams(metric=metric, goal=goal)
  def decorator(cls):
    def search(self):
      del self
      return SearchHParams(
          search_algorithm=Sweeping.HParams(),
          search_reward=search_reward)
    setattr(cls, 'search', search)
    return cls
  return decorator

