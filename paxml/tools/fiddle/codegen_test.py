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

import re
import textwrap

from absl.testing import absltest
from fiddle._src.codegen.auto_config import ir_printer
from paxml.tools.fiddle import codegen
from paxml.tools.fiddle import codegen_tracer
from paxml.tools.fiddle import test_fixtures


def _wrap_matched_line(m: re.Match[str]) -> str:
  return (
      m.group(0)
      if len(m.group(0)) < 80
      else m.group(1) + "\n" + " " * 8 + m.group(2)
  )


def _update_expected_text(code: str) -> str:
  indented = re.sub(r"\s+$", "", textwrap.indent(code, " " * 4))
  wrapped = re.sub(r"([^\n]+.{,70})[ ]+(.+)", _wrap_matched_line, indented)
  return (
      'PLEASE UPDATE THE EXPECTED CODE TO:\n\n    expected = """\n'
      f'{wrapped}\n    """'
  )


class CodegenTest(absltest.TestCase):

  def test_highlevel_parameterization_transform(self):
    init_pass = codegen.InitTask()
    codegen_pass = codegen.HighlevelParameterization(lowercasing=False)
    tracer_obj = codegen_tracer.make_tracer("tracer_name", 1)
    task = init_pass(tracer_obj)
    self.assertIs(codegen_pass(task), task)
    self.assertEqual(
        ir_printer.format_expr(task.top_level_call.fn.output_value),
        "self.tracer_name",
    )

  def test_highlevel_parameterization_transforms_keys(self):
    init_pass = codegen.InitTask()
    codegen_pass = codegen.HighlevelParameterization(lowercasing=False)
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

  def test_codegen(self):
    code = codegen.codegen_baseline_from_legacy(
        test_fixtures.SampleExperiment,
        has_train_dataset=False,
        has_input_specs_provider=False,
    )
    expected = """
    import dataclasses
    from paxml import parameterized_experiment
    from paxml import tasks_lib
    from paxml.tools.fiddle import test_fixtures
    from praxis import pax_fiddle


    @dataclasses.dataclass(frozen=True)
    class Experiment:
      foo_setting: int = 4123
      derived_setting: int = 8246
      derived_list_setting: list = [4123, 8246]

      def experiment_fixture(self):
        task = pax_fiddle.PaxConfig(tasks_lib.SingleTask,
            model=self.model_fixture())
        return pax_fiddle.PaxConfig(parameterized_experiment.ParameterizedExperiment,
            task=task, eval_datasets=[])

      def model_fixture(self):
        return pax_fiddle.PaxConfig(test_fixtures.SampleModel,
            my_setting=self.foo_setting, derived_setting=self.derived_setting,
            derived_list_setting=self.derived_list_setting)
    """
    self.assertEqual(
        code.split(), expected.split(), msg=_update_expected_text(code)
    )

  def test_codegen_sharding_unfactored(self):
    code = codegen.codegen_baseline_from_legacy(
        test_fixtures.SampleShardedExperiment,
        factor_out_sharding_annotations=False,
        has_train_dataset=False,
        has_input_specs_provider=False,
    )
    expected = """
    import dataclasses
    from paxml import parameterized_experiment
    from paxml import tasks_lib
    from paxml.tools.fiddle import test_fixtures
    from praxis import base_layer
    from praxis import pax_fiddle


    @dataclasses.dataclass(frozen=True)
    class Experiment:
      foo_setting: int = 4123
      derived_setting: int = 8246
      derived_list_setting: list = [4123, 8246]

      def experiment_fixture(self):
        task = pax_fiddle.PaxConfig(tasks_lib.SingleTask,
            model=self.model_fixture())
        return pax_fiddle.PaxConfig(parameterized_experiment.ParameterizedExperiment,
            task=task, eval_datasets=[])

      def model_fixture(self):
        return pax_fiddle.PaxConfig(test_fixtures.SampleModel,
            activation_split_dims_mapping=pax_fiddle.PaxConfig(base_layer.BaseLayer.ActivationSharding,
            out=['foo_axis', 'bar_axis']),
            my_setting=self.foo_setting,
            derived_setting=self.derived_setting,
            derived_list_setting=self.derived_list_setting)
    """
    self.assertEqual(
        code.split(), expected.split(), msg=_update_expected_text(code)
    )

  def test_codegen_sharding_factored(self):
    code = codegen.codegen_baseline_from_legacy(
        test_fixtures.SampleShardedExperiment,
        has_train_dataset=False,
        has_input_specs_provider=False,
    )
    expected = """
    import dataclasses
    from paxml import parameterized_experiment
    from paxml import tasks_lib
    from paxml.tools.fiddle import test_fixtures
    from praxis import pax_fiddle


    @dataclasses.dataclass(frozen=True)
    class Experiment:
      foo_setting: int = 4123
      derived_setting: int = 8246
      derived_list_setting: list = [4123, 8246]

      def experiment_fixture(self):
        task = pax_fiddle.PaxConfig(tasks_lib.SingleTask,
            model=self.model_fixture())
        return pax_fiddle.PaxConfig(parameterized_experiment.ParameterizedExperiment,
            task=task, eval_datasets=[])

      def model_fixture(self):
        return pax_fiddle.PaxConfig(test_fixtures.SampleModel,
            my_setting=self.foo_setting, derived_setting=self.derived_setting,
            derived_list_setting=self.derived_list_setting)

    def shard_model_config(model_config):
      original_model_config_activation_split_dims_mapping = model_config.activation_split_dims_mapping
      original_model_config_activation_split_dims_mapping.out = ['foo_axis',
          'bar_axis']
    """
    self.assertEqual(
        code.split(), expected.split(), msg=_update_expected_text(code)
    )

  def test_codegen_with_datasets(self):
    code = codegen.codegen_baseline_from_legacy(
        test_fixtures.SampleExperimentWithDatasets,
        has_input_specs_provider=False,
    )
    expected = """
    import dataclasses
    from paxml import parameterized_experiment
    from paxml import tasks_lib
    from paxml.tools.fiddle import test_fixtures
    from praxis import base_input
    from praxis import pax_fiddle


    @dataclasses.dataclass(frozen=True)
    class Experiment:
      foo_setting: int = 4123
      derived_setting: int = 8246
      derived_list_setting: list = [4123, 8246]

      def experiment_fixture(self):
        task = pax_fiddle.PaxConfig(tasks_lib.SingleTask,
            model=self.model_fixture())
        return pax_fiddle.PaxConfig(parameterized_experiment.ParameterizedExperiment,
            task=task, training_dataset=self.training_dataset_fixture(),
            eval_datasets=self.eval_datasets_fixture())

      def model_fixture(self):
        return pax_fiddle.PaxConfig(test_fixtures.SampleModel,
        my_setting=self.foo_setting, derived_setting=self.derived_setting,
            derived_list_setting=self.derived_list_setting)

      def training_dataset_fixture(self):
        return pax_fiddle.PaxConfig(base_input.BaseInput, batch_size=1024,
            is_training=True)

      def eval_datasets_fixture(self):
        return [pax_fiddle.PaxConfig(base_input.BaseInput, batch_size=128,
            is_training=False)]
    """
    self.assertEqual(
        code.split(), expected.split(), msg=_update_expected_text(code)
    )

  def test_codegen_with_input_specs_provider(self):
    code = codegen.codegen_baseline_from_legacy(
        test_fixtures.SampleExperimentWithInputSpecsProvider,
        has_train_dataset=False,
    )
    expected = """
    import dataclasses
    from paxml import parameterized_experiment
    from paxml import tasks_lib
    from paxml.tools.fiddle import test_fixtures
    from praxis import pax_fiddle


    @dataclasses.dataclass(frozen=True)
    class Experiment:
      foo_setting: int = 4123
      derived_setting: int = 8246
      derived_list_setting: list = [4123, 8246]

      def experiment_fixture(self):
        task = pax_fiddle.PaxConfig(tasks_lib.SingleTask,
            model=self.model_fixture())
        return pax_fiddle.PaxConfig(parameterized_experiment.ParameterizedExperiment,
            task=task, eval_datasets=[],
            input_specs_provider=self.input_specs_provider_fixture())

      def model_fixture(self):
        return pax_fiddle.PaxConfig(test_fixtures.SampleModel,
            my_setting=self.foo_setting, derived_setting=self.derived_setting,
            derived_list_setting=self.derived_list_setting)

      def input_specs_provider_fixture(self):
        return pax_fiddle.PaxConfig(test_fixtures.SampleInputSpecsProvider)
    """
    self.assertEqual(
        code.split(), expected.split(), msg=_update_expected_text(code)
    )

  def test_codegen_with_inlined_init_from_checkpoint_rules(self):
    code = codegen.codegen_baseline_from_legacy(
        test_fixtures.SampleExperimentWithInitFromCheckpointRules,
        has_train_dataset=False,
        has_input_specs_provider=False,
    )
    expected = """
    import dataclasses
    from paxml import parameterized_experiment
    from paxml import tasks_lib
    from paxml.tools.fiddle import test_fixtures
    from praxis import pax_fiddle


    @dataclasses.dataclass(frozen=True)
    class Experiment:
      foo_setting: int = 4123
      derived_setting: int = 8246
      derived_list_setting: list = [4123, 8246]

      def experiment_fixture(self):
        train = pax_fiddle.PaxConfig(tasks_lib.SingleTask.Train,
            init_from_checkpoint_rules=self.init_from_checkpoint_rules_fixture())
        task = pax_fiddle.PaxConfig(tasks_lib.SingleTask,
            model=self.model_fixture(), train=train)
        return pax_fiddle.PaxConfig(parameterized_experiment.ParameterizedExperiment,
            task=task, eval_datasets=[])

      def model_fixture(self):
        return pax_fiddle.PaxConfig(test_fixtures.SampleModel,
            my_setting=self.foo_setting, derived_setting=self.derived_setting,
            derived_list_setting=self.derived_list_setting)

      def init_from_checkpoint_rules_fixture(self):
        task_p = pax_fiddle.PaxConfig(tasks_lib.SingleTask,
            model=pax_fiddle.PaxConfig(test_fixtures.SampleModel,
            my_setting=4123, derived_setting=8246, derived_list_setting=[4123,
            8246]))
        checkpoint_loading_rules = pax_fiddle.PaxConfig(tasks_lib.CheckpointLoadingRules,
            task_p=task_p, load_rules=[('(.*)', '{}')], safe_load=True,
            ignore_rules=[], step=532000,
            input_specs_provider_p=pax_fiddle.PaxConfig(test_fixtures.SampleInputSpecsProvider))
        return {'/path/to/my/checkpoint': checkpoint_loading_rules}
    """
    self.assertEqual(
        code.split(), expected.split(), msg=_update_expected_text(code)
    )

  def test_codegen_referenced_init_from_checkpoint_rules(self):
    code = codegen.codegen_baseline_from_legacy(
        test_fixtures.SampleExperimentWithInitFromCheckpointRules,
        has_train_dataset=False,
        has_input_specs_provider=False,
        init_checkpoint_experiments={
            "/path/to/my/checkpoint": (
                test_fixtures.SampleExperimentWithInputSpecsProvider
            )
        },
    )
    expected = """
    import dataclasses
    from paxml import parameterized_experiment
    from paxml import tasks_lib
    from paxml.tools.fiddle import test_fixtures
    from praxis import pax_fiddle


    @dataclasses.dataclass(frozen=True)
    class Experiment:
      foo_setting: int = 4123
      derived_setting: int = 8246
      derived_list_setting: list = [4123, 8246]

      def experiment_fixture(self):
        train = pax_fiddle.PaxConfig(tasks_lib.SingleTask.Train,
            init_from_checkpoint_rules=self.init_from_checkpoint_rules_fixture())
        task = pax_fiddle.PaxConfig(tasks_lib.SingleTask,
            model=self.model_fixture(), train=train)
        return pax_fiddle.PaxConfig(parameterized_experiment.ParameterizedExperiment,
            task=task, eval_datasets=[])

      def model_fixture(self):
        return pax_fiddle.PaxConfig(test_fixtures.SampleModel,
            my_setting=self.foo_setting, derived_setting=self.derived_setting,
            derived_list_setting=self.derived_list_setting)

      def init_from_checkpoint_rules_fixture(self):
        base = test_fixtures.SampleExperimentWithInputSpecsProvider()
        checkpoint_loading_rules = pax_fiddle.PaxConfig(tasks_lib.CheckpointLoadingRules,
            task_p=base.task(), load_rules=[('(.*)', '{}')], safe_load=True,
            ignore_rules=[], step=532000,
            input_specs_provider_p=base.get_input_specs_provider_params())
        return {'/path/to/my/checkpoint': checkpoint_loading_rules}
    """
    self.assertEqual(
        code.split(), expected.split(), msg=_update_expected_text(code)
    )

  def test_codegen_additional_sub_fixtures(self):
    def _sub_fixtures(config):
      rules = config.task.train.init_from_checkpoint_rules
      return {"pretrain_task_fixture": rules["/path/to/my/checkpoint"].task_p}

    code = codegen.codegen_baseline_from_legacy(
        test_fixtures.SampleExperimentWithInitFromCheckpointRules,
        has_train_dataset=False,
        has_input_specs_provider=False,
        additional_sub_fixtures=_sub_fixtures,
    )
    # FIXME(b/289289423): Update this output when multiply-nested subfixtures
    # are better supported.
    expected = """
    import dataclasses
    from paxml import parameterized_experiment
    from paxml import tasks_lib
    from paxml.tools.fiddle import test_fixtures
    from praxis import pax_fiddle

    @dataclasses.dataclass(frozen=True)
    class Experiment:
      foo_setting: int = 4123
      derived_setting: int = 8246
      derived_list_setting: list = [4123, 8246]

      def experiment_fixture(self):
        model = pax_fiddle.PaxConfig(test_fixtures.SampleModel, my_setting=4123,
        derived_setting=8246, derived_list_setting=[4123, 8246])
        derived_list_setting = [4123, 8246] train =
        pax_fiddle.PaxConfig(tasks_lib.SingleTask.Train,
        init_from_checkpoint_rules=self.init_from_checkpoint_rules_fixture(model=model,
        derived_list_setting=derived_list_setting)) task =
        pax_fiddle.PaxConfig(tasks_lib.SingleTask, model=self.model_fixture(),
        train=train) return
        pax_fiddle.PaxConfig(parameterized_experiment.ParameterizedExperiment,
        task=task, eval_datasets=[])

      def model_fixture(self):
        return pax_fiddle.PaxConfig(test_fixtures.SampleModel,
        my_setting=self.foo_setting, derived_setting=self.derived_setting,
        derived_list_setting=self.derived_list_setting)

      def pretrain_task_fixture(
        self, model, derived_list_setting):
        return pax_fiddle.PaxConfig(tasks_lib.SingleTask, model=model)

      def init_from_checkpoint_rules_fixture(
        self, model, derived_list_setting):
        checkpoint_loading_rules =
        pax_fiddle.PaxConfig(tasks_lib.CheckpointLoadingRules,
        task_p=self.pretrain_task_fixture(model=model,
        derived_list_setting=derived_list_setting), load_rules=[('(.*)', '{}')],
        safe_load=True, ignore_rules=[], step=532000,
        input_specs_provider_p=pax_fiddle.PaxConfig(test_fixtures.SampleInputSpecsProvider))
        return {'/path/to/my/checkpoint': checkpoint_loading_rules}
    """
    self.assertEqual(
        code.split(), expected.split(), msg=_update_expected_text(code)
    )


if __name__ == "__main__":
  absltest.main()
