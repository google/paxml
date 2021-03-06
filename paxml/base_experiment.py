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

"""Definition of a ML experiment.

Specifically, BaseExperiment encapsulates all the hyperparameters related
to a specific ML experiment.
"""

import abc
from typing import List, Type, TypeVar
from paxml import automl
from paxml import base_task
from praxis import base_input

_BaseExperimentT = TypeVar('_BaseExperimentT', bound='BaseExperiment')
BaseExperimentT = Type[_BaseExperimentT]


class BaseExperiment(metaclass=abc.ABCMeta):
  """Encapsulates the hyperparameters of an experiment."""

  # p.is_training on each input param is used to determine whether
  # the dataset is used for training or eval.
  # All training and eval datasets must have unique names.
  @abc.abstractmethod
  def datasets(self) -> List[base_input.BaseInput.HParams]:
    """Returns the list of dataset parameters."""

  def training_dataset(self) -> base_input.BaseInput.HParams:
    """Returns the tentatively unique training split.

    Raises a ValueError exception if there is no training split or there are
    multiple of them.
    """
    training_splits = [s for s in self.datasets() if s.is_training]
    if not training_splits:
      raise ValueError(
          'Could not find any training split dataset in this experiment '
          'config (`{self.datasets()}`).')
    if len(training_splits) > 1:
      raise ValueError(
          'Found multiple training split datasets in this experiment '
          'config (`{self.datasets()}`).')
    return training_splits[0]

  # Optional. Returns a list of datasets to be decoded.
  # When specified, all decoder datasets must have unique names.
  def decoder_datasets(self) -> List[base_input.BaseInput.HParams]:
    """Returns the list of dataset parameters for decoder."""
    return []

  @abc.abstractmethod
  def task(self) -> base_task.BaseTask.HParams:
    """Returns the task parameters."""

  def search(self) -> automl.SearchHParams:
    """Returns the parameters for AutoML search."""
    raise NotImplementedError(
        'Please implement `search` method for your experiment for tuning.')

  def __init_subclass__(cls):
    """Modifications to the subclasses."""
    automl.enable_class_level_hyper_primitives(cls)


def create_input_specs_provider(
    experiment: BaseExperiment) -> base_input.DatasetInputSpecsProvider.HParams:
  """Creates a DatasetInputSpecsProvider from an experiment configuration."""
  input_p = experiment.training_dataset()
  return base_input.DatasetInputSpecsProvider.HParams(input_p=input_p)
