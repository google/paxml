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

"""Tests for codegen_highlevel_parameterization."""

from absl.testing import absltest
from fiddle._src.codegen.auto_config import ir_printer
from paxml.tools.fiddle import codegen
from paxml.tools.fiddle import codegen_highlevel_parameterization
from paxml.tools.fiddle import codegen_tracer


class CodegenTest(absltest.TestCase):

  def test_highlevel_parameterization_transform(self):
    init_pass = codegen.InitTask()
    codegen_pass = codegen_highlevel_parameterization.HighlevelParameterization(
        lowercasing=False
    )
    tracer_obj = codegen_tracer.make_tracer("tracer_name", 1)
    task = init_pass(tracer_obj)
    self.assertIs(codegen_pass(task), task)
    self.assertEqual(
        ir_printer.format_expr(task.top_level_call.fn.output_value),
        "self.tracer_name",
    )

  def test_highlevel_parameterization_transforms_keys(self):
    init_pass = codegen.InitTask()
    codegen_pass = codegen_highlevel_parameterization.HighlevelParameterization(
        lowercasing=False
    )
    tracer_obj = codegen_tracer.make_tracer("tracer_foo", 1)
    tracer_obj_2 = codegen_tracer.make_tracer("tracer_bar", 2)
    task = init_pass({tracer_obj: [1, 2, 3], 0: 10, tracer_obj_2: [4, 5, 6]})
    self.assertIs(codegen_pass(task), task)
    converted = task.top_level_call.fn.output_value

    with self.subTest("order_preservation"):
      self.assertEqual(list(converted.values()), [[1, 2, 3], 10, [4, 5, 6]])

    with self.subTest("tracer_conversion"):
      self.assertEqual(list(converted.keys())[0].attribute, "tracer_foo")
      self.assertEqual(list(converted.keys())[2].attribute, "tracer_bar")


if __name__ == "__main__":
  absltest.main()
