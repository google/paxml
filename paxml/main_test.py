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

from paxml import base_experiment
from paxml import experiment_registry
from paxml import main
from absl.testing import absltest


class FakeExperimentClassForTest(base_experiment.BaseExperiment):
  pass


class MainTest(absltest.TestCase):

  def test_get_experiment_failure_to_import_module(self):
    with self.assertRaisesRegex(
        ValueError,
        'Could not find experiment'
        ' `fake_module_for_paxml_main_test.Experiment9876` because could not'
        ' import module `fake_module_for_paxml_main_test`',
    ):
      # my_module is not a module that exists in this test.
      _ = main.get_experiment('fake_module_for_paxml_main_test.Experiment9876')

  def test_get_experiment_failure_to_find_experiment_in_module(self):
    with self.assertRaisesRegex(
        ValueError,
        'Could not find experiment `builtins.Experiment9876`.\n'
        'Registered experiments are: {}',
    ):
      # Module builtins is guaranteed to exist, but there's no corresponding
      # experiment in the builtins module.
      _ = main.get_experiment('builtins.Experiment9876')

  def test_get_experiment_success(self):
    try:
      experiment_registry.register(FakeExperimentClassForTest)

      actual = main.get_experiment(
          FakeExperimentClassForTest.__module__
          + '.'
          + FakeExperimentClassForTest.__qualname__
      )

      expected = FakeExperimentClassForTest
      self.assertEqual(actual, expected)

    finally:
      # Reset registry to empty.
      experiment_registry._ExperimentRegistryHelper._registry = {}


if __name__ == '__main__':
  absltest.main()
