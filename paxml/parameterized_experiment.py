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

"""Defines a parameterizable BaseExperiment subclass."""

from typing import List, Optional, Sequence

from paxml import base_experiment
from paxml import base_task
from praxis import base_input
from praxis import pax_fiddle


class ParameterizedExperiment(base_experiment.BaseExperiment):
  """A parameterizable `BaseExperiment` subclass.

  This class provides a basic "parameterizable" way to construct a
  `BaseExperiment` instance: instead of subclassing `BaseExperiment` and
  overriding methods that return `pax_fiddle.Config` objects corresponding to
  the different config components defining the experiment (task, datasets, etc),
  those components are passed directly as parameters.

  Passing experiment components as parameters enables straightforward end-to-end
  Fiddle configuration of an experiment setup. For example:

      def minimal_experiment():
        model_cfg = ...
        train_dataset_cfg = ...
        eval_dataset_cfg = ...
        return pax_fiddle.Config(
            ParameterizedExperiment,
            task=pax_fiddle.Config(tasks_lib.SingleTask, model=model_cfg),
            training_dataset=train_dataset_cfg,
            eval_datasets=[eval_dataset_cfg],
        )

  This approach is compatible with Fiddle's command line flag integration (see
  https://github.com/google/fiddle/blob/main/docs/flags_code_lab.md for detailed
  documentation). For instance, it is straightforward to write a "fiddler" to
  adapt the configuration for running a single-step smoke test:

      def adapt_for_testing(cfg: pax_fiddle.Config[ParameterizedExperiment]):
        cfg.task.train.num_train_steps = 1
        cfg.task.train.summary_interval_steps = 1
        cfg.task.train.eval_interval_steps = 1
        cfg.task.train.save_interval_steps = 1

        for ds in cfg.eval_datasets:
          ds.eval_loop_num_batches = 1

  Together, this enables launching and overriding settings for an experiment
  without requiring use of the Pax experiment registry. In this example, this
  would be done by passing `--fdl_config=some.module.minimal_experiment` and
  `--fiddler=some.module.adapt_for_testing`.

  Note that unlike `BaseExperiment`, `ParameterizableExperiment` explicitly
  separates the training dataset and evaluation datasets (instead of allowing
  them to be specified as a combined collection). Additionally, not all methods
  from `BaseExperiment` are currently parameterizable via this class (e.g.,
  `search`, `validate`, `sub_experiments`). More may be added in the future.
  """

  def __init__(
      self,
      *,
      task: pax_fiddle.Config[base_task.BaseTask],
      training_dataset: Optional[
          pax_fiddle.Config[base_input.BaseInput]
      ] = None,
      eval_datasets: Sequence[pax_fiddle.Config[base_input.BaseInput]] = (),
      decoder_datasets: Sequence[pax_fiddle.Config[base_input.BaseInput]] = (),
      input_specs_provider: Optional[
          pax_fiddle.Config[base_input.BaseInputSpecsProvider]
      ] = None,
  ):
    """Initializes a `ParameterizedExpermiment` instance.

    Some basic validation is performed to ensure that the `is_training`
    parameter is appropriately set on the provided datasets.

    Args:
      task: The config for the task.
      training_dataset: The training dataset config to use.
      eval_datasets: A sequence of dataset configs for evaluation. If not
        provided, defaults to an empty sequence.
      decoder_datasets: A sequence of dataset configs for decoding. If not
        provided, defaults to an empty sequence.
      input_specs_provider: The config for a `BaseInputSpecsProvider` subclass.
        If not provided, a `DatasetInputSpecsProvider` will be created using the
        training dataset.
    """
    if training_dataset is not None and not training_dataset.is_training:
      raise ValueError(
          f"The training dataset with name {training_dataset.name!r} must have"
          " `is_training` set to `True`."
      )
    for eval_dataset in eval_datasets:
      if eval_dataset.is_training:
        raise ValueError(
            f"The evaluation dataset with name {eval_dataset.name!r} must have"
            " `is_training` set to `False`."
        )
    for decoder_dataset in decoder_datasets:
      if decoder_dataset.is_training:
        raise ValueError(
            f"The decoder dataset with name {decoder_dataset.name!r} must have"
            " `is_training` set to `False`."
        )
    self._task = task
    self._training_dataset = training_dataset
    self._eval_datasets = list(eval_datasets)
    self._decoder_datasets = list(decoder_datasets)
    self._input_specs_provider = input_specs_provider
    super().__init__()

  def task(self) -> pax_fiddle.Config[base_task.BaseTask]:
    """Returns the task config."""
    return self._task

  def datasets(self) -> List[pax_fiddle.Config[base_input.BaseInput]]:
    """Returns the union of training and eval dataset configs."""
    return (
        [self._training_dataset] if self._training_dataset else []
    ) + self.eval_datasets()

  def training_dataset(self) -> pax_fiddle.Config[base_input.BaseInput]:
    """Returns the training dataset config. Raises an error if it is `None`."""
    if self._training_dataset is None:
      raise ValueError("No training dataset was provided.")
    return self._training_dataset

  def eval_datasets(self) -> List[pax_fiddle.Config[base_input.BaseInput]]:
    """Returns the list of evaluation dataset configs."""
    return self._eval_datasets

  def decoder_datasets(self) -> List[pax_fiddle.Config[base_input.BaseInput]]:
    """Returns the list of decoder dataset configs."""
    return self._decoder_datasets

  def get_input_specs_provider_params(
      self,
  ) -> pax_fiddle.Config[base_input.BaseInputSpecsProvider]:
    """Returns the input specs provider config.

    If an input specs provider config was not given, creates a default config
    using the training dataset.
    """
    if self._input_specs_provider is None:
      return super().get_input_specs_provider_params()
    else:
      return self._input_specs_provider
