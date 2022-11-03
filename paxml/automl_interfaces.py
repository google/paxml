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

"""Interfaces for AutoML for PAX."""

import abc
import dataclasses
import enum
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, Union
from praxis import base_hyperparams
import pyglove as pg


BaseHyperParams = base_hyperparams.BaseHyperParams
BaseParameterizable = base_hyperparams.BaseParameterizable
MetricAggregationFn = Callable[[Sequence[float]], float]


class BaseAlgorithm(BaseParameterizable, metaclass=abc.ABCMeta):
  """Base class for search algorithms."""

  @abc.abstractmethod
  def __call__(self) -> pg.DNAGenerator:
    """Returns a PyGlove DNAGenerator."""


class BaseEarlyStoppingPolicy(BaseParameterizable, metaclass=abc.ABCMeta):
  """Base class for population-wise early stopping policy."""

  @abc.abstractmethod
  def __call__(self) -> pg.tuning.EarlyStoppingPolicy:
    """Returns a PyGlove early stopping policy."""


class BaseReward(BaseParameterizable, metaclass=abc.ABCMeta):
  """Base class for reward functions."""

  @abc.abstractmethod
  def __call__(self, metrics_dict: Dict[str, float], global_step: int) -> float:
    """Returns a float value as reward from a dict of metrics."""

  @property
  @abc.abstractmethod
  def used_metrics(self) -> Sequence['Metric']:
    """Returns `automl.Metric` objects used for computing current reward."""


class CrossStepMetricAggregator(BaseParameterizable, metaclass=abc.ABCMeta):
  """Aggregator for gathering metrics across multiple steps."""

  @abc.abstractmethod
  def __call__(
      self,
      metrics_across_steps: Sequence[Tuple[int, Dict[str, float]]]
      ) -> Dict[str, float]:
    """Aggregates metrics across multiple steps.

    Args:
      metrics_across_steps: A sequence of tuple (step, metrics).

    Returns:
      An aggregated metric dict used for final reward computing.
    """


# To avoid introducing dependency on base_experiment,
# we use Any as its PyType annotation for now.
BaseExperiment = Any


class SearchHParams(BaseHyperParams):
  """Hyperparameters for an AutoML search.

  Attributes:
    search_algorithm: Hyperparameters for search algorithm.
    search_reward: Hyperparameters for search reward.
      If None, 0 will be used as objective, which shall be used only
      for grid search or random search.
    early_stopping: An optional population-wise early stopping policy.
      If None, no population-wise early stopping policy will be used, though
      users still can raise `automl.EarlyStoppingError` to early terminate a
      a single trial during training/evaluation.
    cross_step_metric_aggregator: Hyperparameters for cross-step metric
      aggregator. If None, `automl.LastReportedMetricValues` will be used.
    max_num_trials: Max number of trials for the search. If None, there is no
      limit.
    errors_to_skip: An optional field to specify on what errors the trial
      should be skipped. It's in the form of a list of (ExceptionType) or
      (ExceptionType, regexForError). For example, if users specify:
      `[RuntimeError, (Exception, 'XLACompilation.*')]`, the trails that
      RuntimeError or errors that match 'XLACompilation.*' will be treated as
      to skip.
  """
  search_algorithm: Optional[BaseAlgorithm.HParams] = None
  search_reward: Optional[BaseReward.HParams] = None
  early_stopping: Optional[BaseEarlyStoppingPolicy.HParams] = None
  cross_step_metric_aggregator: Optional[
      CrossStepMetricAggregator.HParams] = None
  max_num_trials: int = None
  errors_to_skip: Optional[List[
      Union[Type[Exception], Tuple[Type[Exception], str]]]] = None


class MetricType(enum.Enum):
  """Metric type for AutoML search."""
  CUSTOM = 0
  TRAIN_METRICS = 1
  EVAL_TRAIN_METRICS = 2
  EVAL_METRICS = 3
  EVAL_SCORING_METRICS = 4
  DECODE_METRICS = 5

  @classmethod
  def metric_schema(cls, metric_type: 'MetricType'):
    """Returns the metric schema in tuple (category, section)."""
    if metric_type == MetricType.CUSTOM:
      return ('', '')
    elif metric_type == MetricType.TRAIN_METRICS:
      return ('train', '')
    elif metric_type == MetricType.EVAL_TRAIN_METRICS:
      return ('eval_train', 'metrics')
    elif metric_type == MetricType.EVAL_METRICS:
      return ('eval_test', 'metrics')
    elif metric_type == MetricType.EVAL_SCORING_METRICS:
      return ('eval_test', 'scoring_eval')
    elif metric_type == MetricType.DECODE_METRICS:
      return ('decode_test', '')
    else:
      assert False, 'Should never happen'

  @classmethod
  def applies_to_multiple_datasets(cls, metric_type: 'MetricType'):
    """Returns True if a metric can be applied to multiple datasets."""
    return metric_type in _MULTI_DATASET_METRIC_TYPES


_MULTI_DATASET_METRIC_TYPES = frozenset([
    MetricType.EVAL_METRICS,
    MetricType.EVAL_SCORING_METRICS,
    MetricType.DECODE_METRICS,
])


class MetricAggregator(str, enum.Enum):
  """Builtin metric aggregator."""
  MAX = 0
  MIN = 1
  AVERAGE = 2
  SUM = 3


@dataclasses.dataclass
class Metric:
  """Representing a metric for tuning.

  Attributes:
    metric_type: Type of the tuning metric.
    metric_name: Name of the metric.
    dataset_name: Dataset name. If None, the metric is either not
      dataset-specific or there is only one dataset for that metric type so the
      name can be omitted.
    sub_experiment_id: Sub-experiment ID. If None, the metric will match
      all metrics from all sub-experiments.
    aggregator: Optional metric aggregation at a single step, which
      will be used to obtain a single value from multiple matched metric items
      based on current Metric specification. It can be a value from enum
      `MetricAggregator` or a callable object. If None, `Metric.get_value` will
      raise error when there are multiple matched items.
  """
  metric_name: str
  metric_type: MetricType = MetricType.CUSTOM
  dataset_name: Optional[str] = None
  sub_experiment_id: Optional[str] = None
  aggregator: Optional[Union[
      MetricAggregator,
      Callable[[Sequence[float]], float]]] = None

  def __post_init__(self):
    self._metric_key_regex = re.compile(self.pattern, re.IGNORECASE)
    if self.aggregator is None:
      self._aggregator = None
    elif self.aggregator == MetricAggregator.MAX:
      self._aggregator = max
    elif self.aggregator == MetricAggregator.MIN:
      self._aggregator = min
    elif self.aggregator == MetricAggregator.AVERAGE:
      self._aggregator = lambda xs: sum(xs) / len(xs)
    elif self.aggregator == MetricAggregator.SUM:
      self._aggregator = sum
    elif callable(self.aggregator):
      self._aggregator = self.aggregator
    else:
      raise ValueError(
          f'Unsupported aggregator {self.aggregator}. '
          f'Expecting a value from `automl.MetricAggregator` enum'
          f'or Callable[[Sequence[float]], float].')

  @property
  def pattern(self):
    """Returns key pattern."""
    category, section = MetricType.metric_schema(self.metric_type)
    prefix = ''
    if MetricType.applies_to_multiple_datasets(self.metric_type):
      dataset_pattern = self.dataset_name or '[^/]+'
      assert category, category
      prefix = f'{category}_{dataset_pattern}/'
    elif category:
      prefix = f'{category}/'
    if section:
      prefix = f'{prefix}{section}/'

    if self.sub_experiment_id is None:
      suffix = '(:.+)?'
    else:
      suffix = f':{self.sub_experiment_id}'
    return f'^{prefix}{self.metric_name}{suffix}$'

  @property
  def applies_to_multiple_datasets(self):
    """Returns True if current metric is dataset specific."""
    return (MetricType.applies_to_multiple_datasets(self.metric_type) and
            self.dataset_name is None)

  def match_items(self, metric_dict: Dict[str,
                                          float]) -> List[Tuple[str, float]]:
    """Gets matched items of current metric from a metric dict."""
    return [(k, v)
            for k, v in metric_dict.items()
            if self._metric_key_regex.match(k)]

  def get_values(self, metric_dict: Dict[str, float]) -> List[float]:
    """Gets the value of current metric from a metric dict."""
    return [v for k, v in self.match_items(metric_dict)]

  def get_value(self, metric_dict: Dict[str, float]) -> float:
    """Gets the only value for current metric from a metric dict."""
    items = list(self.match_items(metric_dict))
    if not items:
      raise KeyError(
          f'Metric {self.pattern!r} does not match with any metrics. '
          f'Available metrics are: {list(metric_dict.keys())}.')
    if len(items) != 1:
      if self._aggregator is not None:
        return self._aggregator([m[1] for m in items])
      raise ValueError(
          f'Found multple metrics that match {self.pattern!r} while '
          f'aggregator is not specified: {items}.')
    return items[0][1]

  # Class method for creating custom metric types.
  @classmethod
  def train_steps_per_second(
      cls,
      sub_experiment_id: Optional[str] = None,
      aggregator: Optional[
          Union[str, Callable[[Sequence[float]], float]]] = None) -> 'Metric':
    """Returns metric for training steps per second."""
    return Metric('train_steps_per_sec',
                  sub_experiment_id=sub_experiment_id,
                  aggregator=aggregator)

  @classmethod
  def eval_steps_per_second(
      cls,
      sub_experiment_id: Optional[str] = None,
      aggregator: Optional[
          Union[str, Callable[[Sequence[float]], float]]] = None) -> 'Metric':
    """Returns metric for evaluation steps per second."""
    return Metric('eval_steps_per_sec',
                  sub_experiment_id=sub_experiment_id,
                  aggregator=aggregator)

  @classmethod
  def decode_steps_per_second(
      cls,
      sub_experiment_id: Optional[str] = None,
      aggregator: Optional[Union[str, MetricAggregationFn]] = None) -> 'Metric':
    """Returns metric for `decode_steps_per_second`."""
    return Metric('decode_steps_per_sec',
                  sub_experiment_id=sub_experiment_id,
                  aggregator=aggregator)

  @classmethod
  def num_params(
      cls,
      sub_experiment_id: Optional[str] = None,
      aggregator: Optional[Union[str, MetricAggregationFn]] = None) -> 'Metric':
    """Returns metric for `num_params`."""
    return Metric('num_params',
                  sub_experiment_id=sub_experiment_id,
                  aggregator=aggregator)

  # Class methods for creating eval metric types.
  @classmethod
  def train(cls,
            metric_name: str,
            sub_experiment_id: Optional[str] = None,
            aggregator: Optional[Union[str, MetricAggregationFn]] = None
            ) -> 'Metric':
    """Returns a metric from evaluation on the training dataset."""
    return Metric(metric_name,
                  MetricType.TRAIN_METRICS,
                  sub_experiment_id=sub_experiment_id,
                  aggregator=aggregator)

  @classmethod
  def eval_train(cls,
                 metric_name: str,
                 sub_experiment_id: Optional[str] = None,
                 aggregator: Optional[Union[str, MetricAggregationFn]] = None
                 ) -> 'Metric':
    """Returns a metric from evaluation on the training dataset."""
    return Metric(metric_name,
                  MetricType.EVAL_TRAIN_METRICS,
                  sub_experiment_id=sub_experiment_id,
                  aggregator=aggregator)

  @classmethod
  def eval(cls,
           metric_name: str,
           dataset_name: Optional[str] = None,
           sub_experiment_id: Optional[str] = None,
           aggregator: Optional[Union[str, MetricAggregationFn]] = None
           ) -> 'Metric':
    """Returns a metric from evaluation on the test dataset."""
    return Metric(metric_name,
                  MetricType.EVAL_METRICS,
                  dataset_name,
                  sub_experiment_id=sub_experiment_id,
                  aggregator=aggregator)

  @classmethod
  def eval_scoring(cls,
                   metric_name: str,
                   dataset_name: Optional[str] = None,
                   sub_experiment_id: Optional[str] = None,
                   aggregator: Optional[Union[str, MetricAggregationFn]] = None
                   ) -> 'Metric':
    """Returns a metric from evaluation on the test dataset."""
    return Metric(metric_name,
                  MetricType.EVAL_SCORING_METRICS,
                  dataset_name,
                  sub_experiment_id=sub_experiment_id,
                  aggregator=aggregator)

  @classmethod
  def decode(cls,
             metric_name: str,
             dataset_name: Optional[str] = None,
             sub_experiment_id: Optional[str] = None,
             aggregator: Optional[Union[str, MetricAggregationFn]] = None
             ) -> 'Metric':
    """Returns a metric or processed metric from a decode dataset."""
    return Metric(metric_name,
                  MetricType.DECODE_METRICS,
                  dataset_name,
                  sub_experiment_id=sub_experiment_id,
                  aggregator=aggregator)
