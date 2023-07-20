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

"""Tests for config_normalization.

For now, we don't test every combination of flags.
"""

from absl.testing import absltest
import fiddle as fdl
from paxml import parameterized_experiment
from paxml.tools.fiddle import config_normalization
from paxml.tools.fiddle import test_fixtures
from praxis import pax_fiddle


class ConfigNormalizationTest(absltest.TestCase):

  def test_lowlevel_config(self):
    task_config = (
        test_fixtures.SampleExperimentWithSharedShardingAnnotations().task()
    )
    config = pax_fiddle.Config(
        parameterized_experiment.ParameterizedExperiment,
        task=task_config,
    )
    self.assertIs(
        config.task.model.sublayers[0].activation_split_dims_mapping,
        config.task.model.sublayers[1].activation_split_dims_mapping,
    )
    normalizer_config = (
        config_normalization.default_normalizer().lowlevel_config()
    )
    normalizer_config.task_normalizer.unshare_sharding_config = False
    normalizer = fdl.build(normalizer_config)
    normalized = normalizer(config)
    self.assertIs(
        normalized.task.model.sublayers[0].activation_split_dims_mapping,
        normalized.task.model.sublayers[1].activation_split_dims_mapping,
    )

  def test_default_normalizer(self):
    task_config = (
        test_fixtures.SampleExperimentWithSharedShardingAnnotations().task()
    )
    config = pax_fiddle.Config(
        parameterized_experiment.ParameterizedExperiment,
        task=task_config,
    )
    self.assertIs(
        config.task.model.sublayers[0].activation_split_dims_mapping,
        config.task.model.sublayers[1].activation_split_dims_mapping,
    )
    normalized = config_normalization.default_normalizer()(config)
    self.assertIsNot(
        normalized.task.model.sublayers[0].activation_split_dims_mapping,
        normalized.task.model.sublayers[1].activation_split_dims_mapping,
    )

  def test_aggressive_normalizer_remove_sharding(self):
    task_config = (
        test_fixtures.SampleExperimentWithSharedShardingAnnotations().task()
    )
    config = pax_fiddle.Config(
        parameterized_experiment.ParameterizedExperiment,
        task=task_config,
    )
    normalized = config_normalization.aggressive_normalizer()(config)
    self.assertNotIn(
        "activation_split_dims_mapping",
        fdl.ordered_arguments(normalized.task.model.sublayers[0]),
    )


if __name__ == "__main__":
  absltest.main()
