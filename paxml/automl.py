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
import inspect
from typing import Any, Dict, Optional, Sequence, Type, Union
from paxml import automl_interfaces
# Placeholder for importing Google-internal tuning modules.
from praxis import base_hyperparams
import pyglove as pg


BaseParameterizable = base_hyperparams.BaseParameterizable
InstantiableHyperParams = base_hyperparams.InstantiableHyperParams

BaseAlgorithm = automl_interfaces.BaseAlgorithm
BaseReward = automl_interfaces.BaseReward
SearchHParams = automl_interfaces.SearchHParams

# Aliases for Google-internal symbols.

#
# Common search hyperparameters.
#


def hyperparameter_tuning(metric_key: str,
                          max_num_trials: int = 100,
                          goal: str = 'maximize') -> SearchHParams:
  """Returns a common search HParams for hyper-parameter tuning.

  Args:
    metric_key: The metric key to optimize. It should be in format of
      '<metric_name>' or '<dataset_name>/<metric_name>' when there is more
      than one dataset for evaluation.
    max_num_trials: Max number of trials for tuning.
    goal: 'maximize' or 'minimize'.

  Returns:
    A search HParams object.
  """
  return SearchHParams(
      # Use Sweeping for hyperparameter tuning.
      search_algorithm=Sweeping.HParams(),
      search_reward=SingleObjective.HParams(
          metric_key=metric_key, goal=goal),
      max_num_trials=max_num_trials)


def neural_architecture_search(metric_key_or_keys: Union[str, Sequence[str]],
                               cost_objective: Optional[float] = None,
                               reward_type: str = 'tunas',
                               exponent: float = -0.07,
                               max_num_trials: int = 10000) -> SearchHParams:
  """Search params for Neural Architecture Search."""

  if isinstance(metric_key_or_keys, str):
    metric_key_or_keys = [metric_key_or_keys]

  if len(metric_key_or_keys) == 1:
    reward = SingleObjective.HParams(metric_key=metric_key_or_keys[0])
  elif len(metric_key_or_keys) == 2:
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
        metric_keys=metric_key_or_keys,
        aggregator=aggregator_cls.HParams(
            cost_objective=cost_objective, exponent=exponent))
  else:
    raise ValueError('Only 1 or 2 metrics are supported.')

  return SearchHParams(
      search_algorithm=RegularizedEvolution.HParams(),
      search_reward=reward,
      max_num_trials=max_num_trials)


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
    """
    metric_key: Optional[str] = None
    goal: str = 'maximize'

    def __post_init__(self):
      super().__post_init__()
      if self.metric_key is None:
        raise ValueError('Param `metric_key` should not be None.')
      if self.goal not in ['maximize', 'minimize']:
        raise ValueError(
            'Param `goal` should be either \'maximize\' or \'minimize\'.')

  def __call__(self, metrics_dict: Dict[str, float], global_step: int) -> float:
    del global_step
    if self._hparams.metric_key not in metrics_dict:
      raise ValueError('Metric %r does not exist. Available keys are: %r' %
                       (self._hparams.metric_key, list(metrics_dict.keys())))
    reward = metrics_dict[self._hparams.metric_key]
    if self._hparams.goal == 'minimize':
      reward *= -1
    return reward


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
    """
    metric_keys: Optional[Sequence[str]] = None
    aggregator: Optional[MultiObjectiveAggregator.HParams] = None

    def __post_init__(self):
      super().__post_init__()
      if not self.metric_keys:
        raise ValueError('Param `metric_keys` must be provided.')

      if len(self.metric_keys) > 1 and self.aggregator is None:
        raise ValueError('Param `aggregator` must be provided.')

  def __init__(self, hparams: HParams):
    super().__init__(hparams)
    if self._hparams.aggregator is not None:
      self._aggregator = self._hparams.aggregator.Instantiate()

  def __call__(self, metrics_dict: Dict[str, float], global_step: int) -> float:
    del global_step
    for metric in self._hparams.metric_keys:
      if metric not in metrics_dict:
        raise ValueError('Metric %r does not exist. Available keys are: %r' %
                         (metric, list(metrics_dict.keys())))
    if len(self._hparams.metric_keys) == 1:
      return metrics_dict[self._hparams.metric_keys[0]]
    assert self._aggregator is not None
    return self._aggregator(
        [metrics_dict[metric] for metric in self._hparams.metric_keys])


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
    if hyper.name is None:
      raise ValueError(
          f'Missing the \'name\' argument for the value of {name!r}: {hyper!r}')
    kwargs = {'hints': hyper.hints, 'name': hyper.name}
    if isinstance(hyper, pg.hyper.OneOf):
      factory = lambda: pg.oneof(hyper.candidates, **kwargs)
    elif isinstance(hyper, pg.hyper.ManyOf):
      factory = lambda: pg.manyof(  # pylint: disable=g-long-lambda
          hyper.num_choices,
          hyper.candidates,
          distinct=hyper.choices_distinct,
          sorted=hyper.choices_sorted,
          **kwargs)
    elif isinstance(hyper, pg.hyper.Float):
      factory = lambda: pg.floatv(hyper.min_value, hyper.max_value, **kwargs)
    else:
      raise NotImplementedError(f'Not supported hyper primitive: {hyper!r}.')

    attr_name = f'_PROPERTY_{name}'

    def getter(x):
      if hasattr(x, attr_name):
        return getattr(x, attr_name)
      return factory()

    def setter(x, v):
      setattr(x, attr_name, v)

    return property(getter, setter)

  for name, hyper in inspect.getmembers(
      cls, lambda x: isinstance(x, pg.hyper.HyperPrimitive)):
    setattr(cls, name, create_hyper_property(name, hyper))
