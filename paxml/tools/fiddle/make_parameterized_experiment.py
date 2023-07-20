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

"""Makes a parameterized experiment from a legacy BaseExperiment subclass.

TODO(b/292000357): Add unit tests. This is currently decently well integration
tested through codegen_test.py
"""

from typing import Optional, Type

from paxml import base_experiment
from paxml import parameterized_experiment
from paxml.tools.fiddle import config_normalization
from praxis import pax_fiddle


def from_legacy(
    experiment_cls: Type[base_experiment.BaseExperiment],
    *,
    normalizer: Optional[
        config_normalization.ConfigNormalizer
    ] = config_normalization.default_normalizer(),
    has_train_dataset: bool = True,
    has_input_specs_provider: bool = True,
) -> pax_fiddle.Config[parameterized_experiment.ParameterizedExperiment]:
  """Returns a ParameterizedExperiment config from a legacy experiment.

  Args:
    experiment_cls: Subclass of BaseExperiment.
    normalizer: Object that will normalize the output configuration. Pass None
      if you don't want any normalization.
    has_train_dataset: Whether the experiment has a training dataset. Usually
      this is true, but some experiments may be inference-only, and calling
      their .training_dataset() method might raise an error. Set this to False
      in those cases.
    has_input_specs_provider: Likewise, usually it's safe to leave this as its
      default (True), but for occasional situations like testing, it may be
      reasonable to disable.
  """
  experiment: base_experiment.BaseExperiment = experiment_cls()

  # Get the task configuration, modulo any changes.
  task_config = experiment.task()

  dataset_configs = experiment.datasets()
  eval_datasets = [
      dataset_config
      for dataset_config in dataset_configs
      if not dataset_config.is_training
  ]
  decoder_datasets = experiment.decoder_datasets()
  if not isinstance(decoder_datasets, list):
    decoder_datasets = list(decoder_datasets)
  overall_config = pax_fiddle.Config(
      parameterized_experiment.ParameterizedExperiment,
      task=task_config,
      eval_datasets=eval_datasets,
  )
  if has_train_dataset:
    overall_config.training_dataset = experiment.training_dataset()
  if has_input_specs_provider:
    overall_config.input_specs_provider = (
        experiment.get_input_specs_provider_params()
    )
  if decoder_datasets:
    overall_config.decoder_datasets = decoder_datasets

  # Now run normalization, and return the result.
  return normalizer(overall_config) if normalizer else overall_config
