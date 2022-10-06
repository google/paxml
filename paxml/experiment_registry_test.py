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

"""Tests for experiment_registry."""
from absl.testing import absltest
from paxml import base_experiment
from paxml import experiment_registry
from paxml.tasks.test import synthetic  # pylint: disable=unused-import
from praxis import layers


@experiment_registry.register
class DummyExperiment(base_experiment.BaseExperiment):

  def datasets(self):
    return []

  def task(self):
    act_p = layers.Identity.HParams()
    return act_p


@experiment_registry.register()
class SharedNameExperiment(DummyExperiment):
  pass


# This tests that explicit re-register works.
@experiment_registry.register(allow_overwrite=True)
class SharedNameExperiment(DummyExperiment):  # pylint: disable=function-redefined
  pass


@experiment_registry.register(allow_overwrite=True, tags=['foo_tag'])
class TaggedA(DummyExperiment):
  pass


@experiment_registry.register(allow_overwrite=True, tags=['foo_tag'])
class TaggedB(DummyExperiment):
  pass


class ExperimentRegistryTest(absltest.TestCase):

  def test_get(self):
    # Module name is `__main__` when registering locally like here.
    dummy_experiment_cls = experiment_registry.get('__main__.DummyExperiment')
    dummy_experiment = dummy_experiment_cls()
    self.assertEmpty(dummy_experiment.datasets())
    self.assertIsNotNone(dummy_experiment.task())
    dummy_experiment_cls2 = experiment_registry.get('DummyExperiment')
    self.assertEqual(dummy_experiment_cls2, dummy_experiment_cls)

    self.assertIsNone(experiment_registry.get('DummyExperimentNotDefined'))

  def test_secondary_keys(self):
    classes = set()
    classes.add(experiment_registry.get('test.synthetic.SyntheticClassifier'))
    classes.add(
        experiment_registry.get('tasks.test.synthetic.SyntheticClassifier'))
    classes.add(
        experiment_registry.get(
            'paxml.tasks.test.synthetic.SyntheticClassifier'))
    classes.add(experiment_registry.get('synthetic.SyntheticClassifier'))
    classes.add(experiment_registry.get('SyntheticClassifier'))
    # custom secondary keys
    classes.add(experiment_registry.get('test.synthetic.SyntheticClassifier'))
    print(classes)
    self.assertLen(classes, 1)
    self.assertNotIn(None, classes)
    classes.add(experiment_registry.get('not_a_file_name.SyntheticClassifier'))
    self.assertLen(classes, 2)
    self.assertIn(None, classes)

    dummy_experiment_cls = experiment_registry.get('SyntheticClassifier')
    dummy_experiment = dummy_experiment_cls()
    self.assertEmpty(dummy_experiment.datasets())
    self.assertIsNotNone(dummy_experiment.task())

  def test_register_overwrite(self):
    classes = set()
    classes.add(experiment_registry.get('DummyExperiment'))
    self.assertLen(classes, 1)
    self.assertNotIn(None, classes)

    class DummyExperiment(synthetic.SyntheticClassifier):  # pylint: disable=redefined-outer-name
      pass

    with self.assertRaises(ValueError):
      experiment_registry.register(DummyExperiment)
    self.assertLen(classes, 1)
    self.assertNotIn(None, classes)

    experiment_registry.register(DummyExperiment, allow_overwrite=True)
    # re-registering returns the updated class.
    classes.add(experiment_registry.get('DummyExperiment'))
    self.assertLen(classes, 2)
    self.assertNotIn(None, classes)

    experiment_registry.register(DummyExperiment, allow_overwrite=True)
    classes.add(experiment_registry.get('DummyExperiment'))
    self.assertLen(classes, 2)

  def test_duplicate_secondary_keys(self):
    with self.assertRaises(ValueError):
      experiment_registry.get('SharedNameExperiment')
    classes = set()
    classes.add(experiment_registry.get('__main__.SharedNameExperiment'))
    classes.add(experiment_registry.get('synthetic.SharedNameExperiment'))
    self.assertLen(classes, 2)
    self.assertNotIn(None, classes)

  def test_get_tags(self):
    collected = []
    for key in experiment_registry.get_all():
      tags = experiment_registry.get_registry_tags(key)
      if 'foo_tag' in tags:
        collected.append(key)
        self.assertRegex(key, '.*(TaggedA|TaggedB)')
    self.assertLen(collected, 2)


if __name__ == '__main__':
  absltest.main()
