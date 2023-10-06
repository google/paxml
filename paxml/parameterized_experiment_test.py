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

"""Provides tests for parameterized_experiment."""

from absl.testing import absltest

import jax
import jax.numpy as jnp

from paxml import parameterized_experiment
from paxml import tasks_lib

from praxis import base_input
from praxis import pax_fiddle
from praxis import py_utils
from praxis import test_utils
from praxis.layers import models


class ExampleInputSpecsProvider(base_input.BaseInputSpecsProvider):

  def get_input_specs(self) -> base_input.NestedShapeDtypeStruct:
    return py_utils.NestedMap.FromNestedDict({
        'ids': jax.ShapeDtypeStruct(shape=[32, 1024], dtype=jnp.int32),
        'labels': jax.ShapeDtypeStruct(shape=[32, 1024], dtype=jnp.int32),
    })


def example_experiment_cfg():
  return pax_fiddle.Config(
      parameterized_experiment.ParameterizedExperiment,
      task=pax_fiddle.Config(
          tasks_lib.SingleTask,
          model=pax_fiddle.Config(models.ClassificationMLPModel),
      ),
      train_datasets=[
          pax_fiddle.Config(
              base_input.BaseInput, name='train', is_training=True
          )
      ],
      eval_datasets=[pax_fiddle.Config(base_input.BaseInput, name='eval')],
      decode_datasets=[pax_fiddle.Config(base_input.BaseInput, name='decode')],
      input_specs_provider=pax_fiddle.Config(ExampleInputSpecsProvider),
  )


class ParameterizedExperimentTest(test_utils.TestCase):

  def test_can_build_parameterized_experiment(self):
    experiment_cfg = example_experiment_cfg()
    self.assertIsInstance(experiment_cfg.task, pax_fiddle.Config)
    experiment = pax_fiddle.build(experiment_cfg)
    self.assertEqual(experiment.task(), experiment_cfg.task)
    self.assertEqual(experiment.train_datasets(), experiment_cfg.train_datasets)
    self.assertEqual(experiment.eval_datasets(), experiment_cfg.eval_datasets)
    self.assertEqual(
        experiment.datasets(),
        experiment_cfg.train_datasets + experiment_cfg.eval_datasets,
    )
    self.assertEqual(
        experiment.decode_datasets(), experiment_cfg.decode_datasets
    )
    self.assertEqual(
        experiment.get_input_specs_provider_params(),
        experiment_cfg.input_specs_provider,
    )

  def test_training_dataset_with_is_training_false_throws_error(self):
    experiment_cfg = example_experiment_cfg()
    experiment_cfg.train_datasets[0].is_training = False
    expected_msg = (
        r"The training dataset with name 'train' must have"
        r' `is_training` set to `True`\.'
    )
    with self.assertRaisesRegex(ValueError, expected_msg):
      pax_fiddle.build(experiment_cfg)

  def test_datasets_without_train_datasets(self):
    experiment_cfg = example_experiment_cfg()
    del experiment_cfg.train_datasets
    experiment = pax_fiddle.build(experiment_cfg)
    self.assertEqual(experiment.datasets(), experiment_cfg.eval_datasets)

  def test_datasets_without_eval_datasets(self):
    experiment_cfg = example_experiment_cfg()
    del experiment_cfg.eval_datasets
    experiment = pax_fiddle.build(experiment_cfg)
    self.assertEqual(experiment.datasets(), experiment_cfg.train_datasets)

  def test_datasets_without_train_or_eval_datasets(self):
    experiment_cfg = example_experiment_cfg()
    del experiment_cfg.train_datasets
    del experiment_cfg.eval_datasets
    experiment = pax_fiddle.build(experiment_cfg)
    self.assertEmpty(experiment.datasets())

  def test_eval_dataset_with_is_training_true_throws_error(self):
    experiment_cfg = example_experiment_cfg()
    experiment_cfg.eval_datasets[0].is_training = True
    expected_msg = (
        r"The evaluation dataset with name 'eval' must have"
        r' `is_training` set to `False`\.'
    )
    with self.assertRaisesRegex(ValueError, expected_msg):
      pax_fiddle.build(experiment_cfg)

  def test_decode_dataset_with_is_training_true_throws_error(self):
    experiment_cfg = example_experiment_cfg()
    experiment_cfg.decode_datasets[0].is_training = True
    expected_msg = (
        r"The decode dataset with name 'decode' must have"
        r' `is_training` set to `False`\.'
    )
    with self.assertRaisesRegex(ValueError, expected_msg):
      pax_fiddle.build(experiment_cfg)

  def test_train_datasets_fallback(self):
    experiment_cfg = example_experiment_cfg()
    del experiment_cfg.train_datasets
    experiment = pax_fiddle.build(experiment_cfg)
    self.assertEmpty(experiment.train_datasets())

  def test_eval_datasets_fallback(self):
    experiment_cfg = example_experiment_cfg()
    del experiment_cfg.eval_datasets
    experiment = pax_fiddle.build(experiment_cfg)
    self.assertEmpty(experiment.eval_datasets())

  def test_decode_datasets_fallback(self):
    experiment_cfg = example_experiment_cfg()
    del experiment_cfg.decode_datasets
    experiment = pax_fiddle.build(experiment_cfg)
    self.assertEmpty(experiment.decode_datasets())

  def test_input_specs_provider_fallback(self):
    experiment_cfg = example_experiment_cfg()
    del experiment_cfg.input_specs_provider
    experiment = pax_fiddle.build(experiment_cfg)
    input_specs_provider_params = experiment.get_input_specs_provider_params()
    self.assertIsInstance(input_specs_provider_params, pax_fiddle.Config)
    self.assertEqual(
        pax_fiddle.get_callable(input_specs_provider_params),
        base_input.DatasetInputSpecsProvider,
    )
    self.assertEqual(
        input_specs_provider_params.input_p, experiment_cfg.train_datasets[0]
    )


if __name__ == '__main__':
  absltest.main()
