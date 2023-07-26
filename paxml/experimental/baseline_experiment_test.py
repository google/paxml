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

import dataclasses
from typing import Any, Optional

from absl.testing import absltest
from paxml import parameterized_experiment
from paxml import tasks_lib
from paxml.experimental import baseline_experiment
from praxis import base_model
from praxis import pax_fiddle


def foo(z):
  return 2 * z


class SampleModel(base_model.BaseModel):
  my_setting: int = 1
  bar_tpl: Optional[pax_fiddle.Config[Any]] = None


@dataclasses.dataclass
class MyExperiment(baseline_experiment.BaselineExperiment):
  foo_setting: int = 4123

  def experiment_fixture(
      self,
  ) -> pax_fiddle.Config[parameterized_experiment.ParameterizedExperiment]:
    task = pax_fiddle.PaxConfig(
        tasks_lib.SingleTask, model=self.model_fixture()
    )
    return pax_fiddle.PaxConfig(
        parameterized_experiment.ParameterizedExperiment,
        task=task,
        eval_datasets=[],
    )

  def model_fixture(self):
    return pax_fiddle.PaxConfig(SampleModel, my_setting=self.foo_setting)


@dataclasses.dataclass
class ExperimentWithConfigSetting(MyExperiment):
  """This is NOT a recommended pattern, but we want to make sure it works."""

  foo_setting: int = 4123
  bar_tpl: Optional[pax_fiddle.Config[Any]] = None

  def model_fixture(self):
    return pax_fiddle.PaxConfig(
        SampleModel, my_setting=self.foo_setting, bar_tpl=self.bar_tpl
    )


class BaselineExperimentTest(absltest.TestCase):

  def test_make_experiment(self):
    built = MyExperiment.make_experiment(foo_setting=1)
    self.assertIsInstance(
        built, parameterized_experiment.ParameterizedExperiment
    )
    self.assertEqual(built.task().model.my_setting, 1)

  def test_highlevel_config(self):
    highlevel_config = MyExperiment.highlevel_config()
    self.assertIsInstance(highlevel_config, pax_fiddle.Config)
    highlevel_config.foo_setting = 7
    built = pax_fiddle.build(highlevel_config)
    self.assertIsInstance(
        built, parameterized_experiment.ParameterizedExperiment
    )
    self.assertEqual(built.task().model.my_setting, 7)

  def test_lowlevel_config(self):
    lowlevel_config = MyExperiment.lowlevel_config()
    self.assertIsInstance(lowlevel_config, pax_fiddle.Config)
    lowlevel_config.task.model.my_setting = 7
    built = pax_fiddle.build(lowlevel_config)
    self.assertIsInstance(
        built, parameterized_experiment.ParameterizedExperiment
    )
    self.assertEqual(built.task().model.my_setting, 7)

  def test_lowlevel_config_with_highlevel_settings(self):
    config = MyExperiment.lowlevel_config(foo_setting=1)
    self.assertIsInstance(config, pax_fiddle.Config)
    built = pax_fiddle.build(config)
    self.assertIsInstance(
        built, parameterized_experiment.ParameterizedExperiment
    )
    self.assertEqual(built.task().model.my_setting, 1)

  def test_lower(self):
    highlevel_config = MyExperiment.highlevel_config()
    highlevel_config.foo_setting = 7
    lowlevel_config = baseline_experiment.lower(highlevel_config)
    self.assertIsInstance(lowlevel_config, pax_fiddle.Config)
    self.assertEqual(lowlevel_config.task.model.my_setting, 7)
    built = pax_fiddle.build(lowlevel_config)
    self.assertIsInstance(
        built, parameterized_experiment.ParameterizedExperiment
    )
    self.assertEqual(built.task().model.my_setting, 7)

  def test_lower_with_highlevel_setting_from_config(self):
    foo_subconfig = pax_fiddle.Config(foo, z=9)
    lowlevel_config = MyExperiment.lowlevel_config(foo_setting=foo_subconfig)
    self.assertIsInstance(lowlevel_config, pax_fiddle.Config)
    self.assertEqual(lowlevel_config.task.model.my_setting, 18)
    built = pax_fiddle.build(lowlevel_config)
    self.assertIsInstance(
        built, parameterized_experiment.ParameterizedExperiment
    )
    self.assertEqual(built.task().model.my_setting, 18)

  def test_lower_with_template(self):
    bar_tpl = pax_fiddle.Config(foo, z=9)
    lowlevel_config = ExperimentWithConfigSetting.lowlevel_config(
        bar_tpl=bar_tpl
    )
    self.assertIsInstance(lowlevel_config, pax_fiddle.Config)
    self.assertIsInstance(lowlevel_config.task.model.bar_tpl, pax_fiddle.Config)
    self.assertEqual(lowlevel_config.task.model.bar_tpl.z, 9)

  def test_lower_error(self):
    with self.assertRaisesRegex(
        ValueError,
        "Expected config of BaselineExperiment.make_experiment.*got.*PaxConfig",
    ):
      baseline_experiment.lower(MyExperiment.lowlevel_config())


if __name__ == "__main__":
  absltest.main()
