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

"""Tests for codegen."""

import dataclasses

from absl.testing import absltest
from absl.testing import parameterized
from paxml import base_experiment
from paxml import base_task
from paxml import tasks_lib
from paxml.tools.fiddle import codegen_tracer
from praxis import base_input
from praxis import base_model
from praxis import pax_fiddle
from praxis.layers import checkpoint_policy


class SampleModel(base_model.BaseModel):
  my_setting: int = 1
  derived_setting: int = 2


class SampleExperiment(base_experiment.BaseExperiment):
  FOO_SETTING = 4123

  def datasets(self) -> list[pax_fiddle.Config[base_input.BaseInput]]:
    return []

  @property
  def derived_setting(self):
    return self.FOO_SETTING * 2

  def task(self) -> pax_fiddle.Config[base_task.BaseTask]:
    return pax_fiddle.Config(
        tasks_lib.SingleTask,
        model=pax_fiddle.Config(
            SampleModel,
            my_setting=self.FOO_SETTING,
            derived_setting=self.derived_setting,
        ),
    )


class SameValueDifferentParameters(base_experiment.BaseExperiment):
  """Tests the case where we have multiple names for the same value.

  It's not infeasible to that some networks will have num_heads == num_layers.
  So let's make sure the tracers work properly in this case.
  """

  A = 105
  B = 105
  C = 105
  D = 105

  def demo_config(self):
    return [self.A, self.B, self.D, self.C]

  def datasets(self) -> list[pax_fiddle.Config[base_input.BaseInput]]:
    return []

  def task(self) -> pax_fiddle.Config[base_task.BaseTask]:
    raise NotImplementedError()


class CodegenTracerTest(parameterized.TestCase):

  def test_tracing(self):
    experiment_cls = codegen_tracer.make_subclass_mixin(SampleExperiment)
    experiment = experiment_cls()
    task = experiment.task()
    self.assertEqual(
        getattr(task.model.my_setting, "__highlevel_name__"), "FOO_SETTING"
    )
    self.assertEqual(task.model.my_setting, 4123)

    self.assertEqual(
        getattr(task.model.derived_setting, "__highlevel_name__"),
        "derived_setting",
    )
    self.assertEqual(task.model.derived_setting, 8246)

  def test_tracing_same_value(self):
    # See SameValueDifferentParameters docstring.
    experiment_cls = codegen_tracer.make_subclass_mixin(
        SameValueDifferentParameters
    )
    experiment = experiment_cls()
    config = experiment.demo_config()
    self.assertEqual(config[0].__highlevel_name__, "A")
    self.assertEqual(config[1].__highlevel_name__, "B")
    self.assertEqual(config[2].__highlevel_name__, "D")
    self.assertEqual(config[3].__highlevel_name__, "C")
    self.assertEqual(config, [105, 105, 105, 105])

  @parameterized.named_parameters(
      {
          "testcase_name": "bool",
          "values": [True, False],
          "cast_fn": bool,
          "instance_cls": (bool, codegen_tracer.BoolTracer),
      },
      {
          "testcase_name": "int",
          "values": [-1, 0, 1],
          "cast_fn": int,
          "instance_cls": int,
      },
      {
          "testcase_name": "float",
          "values": [-1, 0, 1.1, 1e-10],
          "cast_fn": float,
          "instance_cls": (float, int),
      },
      {
          "testcase_name": "str",
          "values": ["foo", "😀", ""],
          "cast_fn": str,
          "instance_cls": str,
      },
      {
          "testcase_name": "list",
          "values": [[], [[]], [0, 1, 2, 3]],
          "cast_fn": list,
          "instance_cls": list,
      },
      {
          "testcase_name": "tuple",
          "values": [(), ((),), (0, 1, 2, 3)],
          "cast_fn": tuple,
          "instance_cls": tuple,
      },
  )
  def test_make_tracer(self, values, cast_fn, instance_cls):
    for value in values:
      tracer = codegen_tracer.make_tracer("foo", value)
      self.assertEqual(tracer.__highlevel_name__, "foo")
      self.assertEqual(tracer, value)
      self.assertEqual(value, tracer)
      self.assertEqual(cast_fn(tracer), value)
      self.assertEqual(not tracer, not value)
      self.assertIsInstance(tracer, instance_cls)

      unwrapped = codegen_tracer.remove_tracers(tracer)
      self.assertEqual(unwrapped, value)
      self.assertIs(type(unwrapped), type(value))
      self.assertIs(
          type(codegen_tracer.remove_tracers([tracer])[0]), type(value)
      )
      self.assertIs(
          type(codegen_tracer.remove_tracers({"foo": tracer})["foo"]),
          type(value),
      )

  def test_unwrap_tracer_of_tracer(self):
    sub_value = codegen_tracer.make_tracer("sub_value", 1)
    value = codegen_tracer.make_tracer("wrapped_value", [sub_value])

    unwrapped = codegen_tracer.remove_tracers(value)
    self.assertEqual(unwrapped, value)
    self.assertIs(type(unwrapped), list)
    self.assertIs(type(unwrapped[0]), int)

  def test_exclude_enums_for_now(self):
    @dataclasses.dataclass
    class Foo(codegen_tracer.TracerMixin):
      cp: checkpoint_policy.AutodiffCheckpointType

    values = [
        checkpoint_policy.AutodiffCheckpointType.SAVE_NOTHING,
        checkpoint_policy.AutodiffCheckpointType.SAVE_EVERYTHING,
    ]
    for value in values:
      traced = Foo(cp=value)
      self.assertEqual(traced.cp, value)
      self.assertIsInstance(traced.cp, checkpoint_policy.AutodiffCheckpointType)


if __name__ == "__main__":
  absltest.main()
