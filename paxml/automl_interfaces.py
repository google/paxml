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
from typing import Dict, Optional
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
  """
  search_algorithm: Optional[BaseAlgorithm.HParams] = None
  search_reward: Optional[BaseReward.HParams] = None
  max_num_trials: int = None

