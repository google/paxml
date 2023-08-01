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

"""Helper for testing the import and construction of experiment configs."""

import re
from typing import List

from absl.testing import absltest
import jax
from paxml import base_task
from praxis import base_hyperparams
from praxis import base_input
from praxis import base_model
from praxis import pax_fiddle
from praxis import py_utils
import pyglove as pg


instantiate = base_hyperparams.instantiate


class ExperimentImportsTestHelper(absltest.TestCase):
  """Helper class for testing experiment configurations."""

  _PREFIX = 'lingvo.'

  def _test_one_experiment_params(self, registry, name):
    """Performs basic checks on an experiment configuration."""
    if name.startswith(self._PREFIX):
      name = name[len(self._PREFIX):]

    experiment_params = registry.get(name)()

    task_p = experiment_params.task()
    task = instantiate(task_p)
    self.assertIsInstance(task, base_task.BaseTask)

    tags: List[str] = registry.get_registry_tags(name)

    dataset_splits = (experiment_params.datasets()
                      + experiment_params.decoder_datasets())
    # Registered experiment configurations must have at least a dataset split.
    self.assertNotEmpty(dataset_splits)
    for s in dataset_splits:
      self.assertIsInstance(
          s,
          (
              pax_fiddle.Config,
          ),
      )

    # Note: Creating the input generator may require data access. Only do it
    # for explicitly allowed experiments for now.
    if 'smoke_test_abstract_init' in tags:
      input_specs_provider = instantiate(
          experiment_params.get_input_specs_provider_params())
      self.assertNotIsInstance(
          input_specs_provider, base_input.DatasetInputSpecsProvider,
          'Please only tag experiments with smoke_test_abstract_init if '
          'they implement an input specs provider that doesn\'t require '
          'initialization of the training pipeline.')
      input_specs = input_specs_provider.get_input_specs()
      model: base_model.BaseModel = task.model  # pytype: disable=attribute-error
      # TODO(pax-dev): Add better/cleaner API to identify pmap vs. pjit models
      # (and check for dcn_mesh_shape too).
      if (hasattr(model, 'ici_mesh_shape') and
          model.ici_mesh_shape is not None):
        input_specs = jax.tree_map(py_utils.get_global_input_shape_dtype,
                                   input_specs)
      model.abstract_init_with_metadata(input_specs)

  @classmethod
  def create_test_methods_for_all_registered_experiments(
      cls,
      registry,
      task_regexes=None,
      exclude_regexes=None,
      include_only_regexes=None):
    """Programmatically defines test methods for each registered experiment."""
    task_regexes = task_regexes or []
    include_only_regexes = include_only_regexes or []
    exclude_regexes = exclude_regexes or []
    experiment_names = list(registry.get_all().keys())
    print(f'Creating tests for {task_regexes}, excluding {exclude_regexes}')
    valid_experiments = []
    for experiment_name in sorted(experiment_names):
      if not any([re.search(regex, experiment_name) for regex in task_regexes]):
        print(f'Skipping tests for registered experiment {experiment_name}')
        continue
      if include_only_regexes and not any(
          [re.search(regex, experiment_name)
           for regex in include_only_regexes]):
        print(f'Skipping tests for experiment {experiment_name}, since '
              'include_only_regexes was provided and this does not match.')
        continue
      if any([re.search(regex, experiment_name) for regex in exclude_regexes]):
        print('Explicitly excluding tests for registered experiment '
              f'{experiment_name}')
        continue
      valid_experiments.append(experiment_name)

      def _test(self, name=experiment_name):
        # `pg.hyper.trace` provides the default decision choices for tunable
        # hyperparameters that is placeheld with `pg.oneof`, `pg.floatv`, etc.
        # For non-AutoML experiments, it's a non-op.
        def _test_fn():
          self._test_one_experiment_params(registry, name)  # pylint: disable=protected-access
        pg.hyper.trace(_test_fn, require_hyper_name=True)
      setattr(cls,
              'test_experiment_params_%s' % experiment_name.replace('.', '_'),
              _test)
    print(f'Created {len(valid_experiments)} tests: {valid_experiments}')
    return len(valid_experiments)
