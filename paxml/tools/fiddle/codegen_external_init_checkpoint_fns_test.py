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

"""Tests for codegen_external_init_checkpoint_fns."""

from absl.testing import absltest
from fiddle._src.codegen.auto_config import init_task
from fiddle._src.codegen.auto_config import ir_printer
from fiddle.experimental import visualize
from paxml.tools.fiddle import codegen_external_init_checkpoint_fns
from paxml.tools.fiddle import codegen_pax_code_ir
from paxml.tools.fiddle import test_fixtures


class InitCheckpointRulesFromOtherTaskTest(absltest.TestCase):

  def test_creates_calls(self):
    config = test_fixtures.SampleExperimentWithInitFromCheckpointRules().task()
    config = visualize.with_defaults_trimmed(config, remove_deep_defaults=True)
    task = init_task.init_task(config)
    task = codegen_pax_code_ir.PaxCodegenTask(
        original_config=task.original_config,
        top_level_call=task.top_level_call,
    )
    codegen_pass = (
        codegen_external_init_checkpoint_fns.InitCheckpointRulesFromOtherTask()
    )
    task = codegen_pass(
        task,
        init_checkpoint_experiments={
            "/path/to/my/checkpoint": (
                test_fixtures.SampleExperimentWithInputSpecsProvider
            )
        },
    )

    debug_str = ir_printer.format_task(task)
    self.assertIn(
        "task_p=call:<call:<test_fixtures."
        "SampleExperimentWithInputSpecsProvider(*[[]],"
        " **{})>.task(*[[]], **{})>",
        debug_str,
    )
    self.assertIn(
        "input_specs_provider_p=call:<call:<test_fixtures."
        "SampleExperimentWithInputSpecsProvider(*[[]],"
        " **{})>.get_input_specs_provider_params(*[[]], **{})>",
        debug_str,
    )

  def test_errors_unused_rule(self):
    config = test_fixtures.SampleExperiment().task()
    config = visualize.with_defaults_trimmed(config, remove_deep_defaults=True)
    task = init_task.init_task(config)
    task = codegen_pax_code_ir.PaxCodegenTask(
        original_config=task.original_config,
        top_level_call=task.top_level_call,
    )
    codegen_pass = (
        codegen_external_init_checkpoint_fns.InitCheckpointRulesFromOtherTask()
    )
    with self.assertRaisesRegex(
        ValueError, r"Didn't encounter.*path/to/my/checkpoint"
    ):
      codegen_pass(
          task,
          init_checkpoint_experiments={
              "/path/to/my/checkpoint": (
                  test_fixtures.SampleExperimentWithInputSpecsProvider
              )
          },
      )

  def test_errors_unmatched_rule(self):
    config = test_fixtures.SampleExperimentWithInitFromCheckpointRules().task()
    config = visualize.with_defaults_trimmed(config, remove_deep_defaults=True)
    task = init_task.init_task(config)
    task = codegen_pax_code_ir.PaxCodegenTask(
        original_config=task.original_config,
        top_level_call=task.top_level_call,
    )
    codegen_pass = (
        codegen_external_init_checkpoint_fns.InitCheckpointRulesFromOtherTask()
    )
    with self.assertRaisesRegex(
        ValueError, r"No task for checkpoint /path/to/my/checkpoint"
    ):
      codegen_pass(
          task,
          init_checkpoint_experiments={},
      )


if __name__ == "__main__":
  absltest.main()
