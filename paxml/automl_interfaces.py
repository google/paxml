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
from typing import Dict, List, Optional, Tuple, Type, Union
from praxis import base_hyperparams
import pyglove as pg


BaseHyperParams = base_hyperparams.BaseHyperParams
BaseParameterizable = base_hyperparams.BaseParameterizable


class BaseAlgorithm(BaseParameterizable, metaclass=abc.ABCMeta):
  """Base class for search algorithms."""

  @abc.abstractmethod
  def __call__(self) -> pg.DNAGenerator:
    """Returns a PyGlove DNAGenerator."""


class BaseReward(BaseParameterizable, metaclass=abc.ABCMeta):
  """Base class for reward functions."""

  @abc.abstractmethod
  def __call__(self, metrics_dict: Dict[str, float], global_step: int) -> float:
    """Returns a float value as reward from a dict of metrics."""


class SearchHParams(BaseHyperParams):
  """Hyperparameters for an AutoML search.

  Attributes:
    search_algorithm: Hyperparameters for search algorithm.
    search_reward: Hyperparameters for search reward.
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


@dataclasses.dataclass
class Metric:
  """Representing a metric for tuning.

  Attributes:
    metric_type: Type of the tuning metric.
    metric_name: Name of the metric.
    dataset_name: Dataset name. If None, the metric is either not
      dataset-specific or there is only one dataset for that metric type so the
      name can be omitted.
  """
  metric_name: str
  metric_type: MetricType = MetricType.CUSTOM
  dataset_name: Optional[str] = None

  def __post_init__(self):
    self._metric_key_regex = re.compile(self.pattern, re.IGNORECASE)

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
    return f'^{prefix}{self.metric_name}$'

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
      raise ValueError(
          f'Found multple metrics that match {self.pattern!r}: {items}.')
    return items[0][1]

  # Class method for creating custom metric types.
  @classmethod
  def train_steps_per_second(cls) -> 'Metric':
    """Returns metric for training steps per second."""
    return Metric('train_steps_per_sec')

  @classmethod
  def eval_steps_per_second(cls) -> 'Metric':
    """Returns metric for evaluation steps per second."""
    return Metric('eval_steps_per_sec')

  @classmethod
  def decode_steps_per_second(cls) -> 'Metric':
    """Returns metric for `decode_steps_per_second`."""
    return Metric('decode_steps_per_sec')

  @classmethod
  def num_params(cls) -> 'Metric':
    """Returns metric for `num_params`."""
    return Metric('num_params')

  # Class methods for creating eval metric types.
  @classmethod
  def train(cls, metric_name: str) -> 'Metric':
    """Returns a metric from evaluation on the training dataset."""
    return Metric(metric_name, MetricType.TRAIN_METRICS)

  @classmethod
  def eval_train(cls, metric_name: str) -> 'Metric':
    """Returns a metric from evaluation on the training dataset."""
    return Metric(metric_name, MetricType.EVAL_TRAIN_METRICS)

  @classmethod
  def eval(cls,
           metric_name: str,
           dataset_name: Optional[str] = None) -> 'Metric':
    """Returns a metric from evaluation on the test dataset."""
    return Metric(metric_name, MetricType.EVAL_METRICS, dataset_name)

  @classmethod
  def eval_scoring(cls,
                   metric_name: str,
                   dataset_name: Optional[str] = None) -> 'Metric':
    """Returns a metric from evaluation on the test dataset."""
    return Metric(metric_name, MetricType.EVAL_SCORING_METRICS, dataset_name)

  @classmethod
  def decode(cls,
             metric_name: str,
             dataset_name: Optional[str] = None) -> 'Metric':
    """Returns a metric or processed metric from a decode dataset."""
    return Metric(metric_name, MetricType.DECODE_METRICS, dataset_name)
