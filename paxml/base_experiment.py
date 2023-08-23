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

"""Definition of a ML experiment.

Specifically, BaseExperiment encapsulates all the hyperparameters related
to a specific ML experiment.
"""

import abc
from typing import Dict, List, Optional, Type, TypeVar
from paxml import automl_interfaces
from paxml import base_executor
from paxml import base_task
from paxml import decode_programs
from paxml import partitioning
from paxml import programs
from praxis import base_input
from praxis import pax_fiddle


_BaseExperimentT = TypeVar('_BaseExperimentT', bound='BaseExperiment')
BaseExperimentT = Type[_BaseExperimentT]


class BaseExperiment(metaclass=abc.ABCMeta):
  """Encapsulates the hyperparameters of an experiment."""

  # p.is_training on each input param is used to determine whether
  # the dataset is used for training or eval.
  # All training and eval datasets must have unique names.
  @abc.abstractmethod
  def datasets(self) -> List[pax_fiddle.Config[base_input.BaseInput]]:
    """Returns the list of dataset parameters."""

  def training_dataset(self) -> pax_fiddle.Config[base_input.BaseInput]:
    """Returns the tentatively unique training split.

    Raises a ValueError exception if there is no training split or there are
    multiple of them.
    """
    training_splits = [s for s in self.datasets() if s.is_training]
    if not training_splits:
      raise ValueError(
          'Could not find any training split dataset in this experiment '
          f'config (`{self.datasets()}`).')
    if len(training_splits) > 1:
      raise ValueError(
          'Found multiple training split datasets in this experiment '
          f'config (`{self.datasets()}`).')
    return training_splits[0]

  def eval_datasets(self) -> List[pax_fiddle.Config[base_input.BaseInput]]:
    """Returns the list of dataset parameters for evaluation."""
    return [dataset for dataset in self.datasets() if not dataset.is_training]

  # Optional. Returns a list of datasets to be decoded.
  # When specified, all decoder datasets must have unique names.
  def decoder_datasets(self) -> List[pax_fiddle.Config[base_input.BaseInput]]:
    """Returns the list of dataset parameters for decoder."""
    return []

  @abc.abstractmethod
  def task(self) -> pax_fiddle.Config[base_task.BaseTask]:
    """Returns the task parameters."""

  def get_input_specs_provider_params(
      self,
  ) -> pax_fiddle.Config[base_input.BaseInputSpecsProvider]:
    """Returns the hparams of the input specs provider.

    By default, it retrieves the input specs from the training input pipeline
    (hence, required to exist). But the method can be overridden in derived
    classes to return a different input specs provider, which directly
    returns the specs.

    Returns:
      An InputSpecsProvider instance.

    Raises:
      A ValueError if there is no training set. In this case, the user must
      override this method to provide the input specs for model weight
      initialization.
    """
    # TODO(b/236417790): Make this method fully abstract and enforce users to
    # provide input specs.
    input_p = self.training_dataset()
    return pax_fiddle.Config(
        base_input.DatasetInputSpecsProvider, input_p=input_p
    )

  def validate(self) -> None:
    """Validates the experiment config but raises if misconfigured."""
    return

  def search(self) -> automl_interfaces.SearchHParams:
    """Returns the parameters for AutoML search."""
    raise NotImplementedError(
        'Please implement `search` method for your experiment for tuning.')

  def sub_experiments(self) -> Dict[str, Type['BaseExperiment']]:
    """Creates sub-experiments for joint tuning.

    A PAX experiment can have multiple sub-experiments during tuning, which
    will be included in a single trial and run in sequence. Each sub-experiment
    is described by an ID (str) and a `BaseExperiment` subclass, therefore,
    PAX users can include multiple PAX experiments in the same tuning task and
    use their metrics to compute tuning rewards. Please note that when a PAX
    experiment class is included as a sub-experiment of other experiment, its
    own sub-experiments will not be included. Users can also programmatically
    create new classes based on current class, by overriding class attributes
    or overriding its method.

    Returns:
      A dict of sub-experiment ID to sub-experiment class.
    """
    return {'': self.__class__}

  def partitioner(self) -> Optional[partitioning.Partitioner]:
    """Returns the partitioner to use for partitioning model functions.

    Returns:
      A Partitioner instance or None, in which case a default partitioner will
      be created based on the task settings.
    """
    return None

  def train_program(self) -> programs.BaseTrainProgram:
    """Returns the train program to use for training the model."""
    return programs.SingleTaskTrainProgram()

  def eval_programs(self) -> List[programs.BaseEvalProgram]:
    """Returns the list of eval programs to use for model evaluation."""
    return [
        programs.SingleTaskEvalProgram(input_p)
        for input_p in self.eval_datasets()
    ]

  def decode_programs(self) -> List[decode_programs.SingleTaskDecodeProgram]:
    """Returns the list of decode_programs to use for model decode."""
    decode_program_list = [
        decode_programs.SingleTaskDecodeProgram(input_p)
        for input_p in self.decoder_datasets()
    ]
    return decode_program_list

  def executor(self) -> Optional[base_executor.BaseExecutor]:
    """Returns the executor to use to run the programs.

    Returns:
      A BaseExecutor instance or None, in which case a default executor will be
      used.
    """
    # TODO(laigd): return the default executor instead of None to make it
    # consistent with e.g. train_program.
    return None

  def __init_subclass__(cls):
    """Modifications to the subclasses."""
    automl_interfaces.enable_class_level_hyper_primitives(cls)
