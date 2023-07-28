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
import fiddle as fdl
from paxml.tools.fiddle import codegen
from paxml.tools.fiddle import test_fixtures
import seqio

# pylint: disable=g-inconsistent-quotes


def _wrap_matched_line(m: re.Match[str]) -> str:
  return (
      m.group(0)
      if len(m.group(0)) < 80
      else m.group(1) + "\n" + " " * 8 + m.group(2)
  )


def _update_expected_text(code: str) -> str:
  indented = re.sub(r"\s+$", "", textwrap.indent(code, " " * 4))
  wrapped = re.sub(r"([^\n]+.{,70})[ ]+(.+)", _wrap_matched_line, indented)
  wrapped = wrapped.replace("  \n", "\n")
  return (
      'PLEASE UPDATE THE EXPECTED CODE TO:\n\n    expected = """\n'
      f'{wrapped}\n    """'
  )


def _codegen_arbitrary_object(
    config: fdl.Config,
    *,
    sub_fixtures=None,
    init_checkpoint_experiments=None,
    model_sharding_diff=None,
    **kwargs,
):
  if sub_fixtures is None:
    sub_fixtures = {}
  codegen_obj = fdl.build(codegen.code_generator_config())
  return codegen_obj(
      config,
      sub_fixtures=sub_fixtures,
      init_checkpoint_experiments=init_checkpoint_experiments,
      model_sharding_diff=model_sharding_diff,
      **kwargs,
  ).code


def module_docstring(name: str) -> str:
  return (
      '"""'
      + codegen._DEFAULT_MODULE_DOCSTRING.format(base_experiment_name=name)
      + '\n"""\n'
  )


class CodegenExamplesTest(absltest.TestCase):
  """Tests output from codegen on smaller non-experiment objects."""

  def test_seqio_import(self):
    config = fdl.Config(
        seqio.SentencePieceVocabulary,
        sentencepiece_model_file="/path/to/vocab.txt",
    )
    code = _codegen_arbitrary_object(config)
    expected = module_docstring("Experiment") + '''
    import dataclasses
    import fiddle as fdl
    from paxml.experimental import baseline_experiment
    import seqio


    @dataclasses.dataclass
    class Experiment(baseline_experiment.BaselineExperiment):
      """Experiment definition for Experiment."""

      def config_fixture(self) -> fdl.Config[seqio.SentencePieceVocabulary]:
        return fdl.Config(seqio.SentencePieceVocabulary,
            sentencepiece_model_file='/path/to/vocab.txt')
    '''
    self.assertEqual(
        code.split(), expected.split(), msg=_update_expected_text(code)
    )


class CodegenErrorsTest(absltest.TestCase):

  def test_friendly_message_custom_objects(self):
    with self.assertRaisesRegex(
        ValueError,
        r"Custom objects.*Custom"
        r" objects:\n.*task.model.my_setting.*test_fixtures.*task",
    ):
      codegen.codegen_baseline_from_legacy(
          test_fixtures.SampleExperimentCustomObject,
          has_train_dataset=False,
          has_input_specs_provider=False,
      )


class CodegenOutputsTest(absltest.TestCase):
  """Tests entire output for codegen."""

  def test_codegen(self):
    # Note: This output is copied in `test_fixtures.py` for derived fixture
    # tests. Please make sure to update it there if it changes.
    code = codegen.codegen_baseline_from_legacy(
        test_fixtures.SampleExperiment,
        has_train_dataset=False,
        has_input_specs_provider=False,
    )
    expected = module_docstring("SampleExperiment") + '''
    import dataclasses
    from paxml.experimental import baseline_experiment
    from paxml import parameterized_experiment
    from paxml import tasks_lib
    from paxml.tools.fiddle import test_fixtures
    from praxis import pax_fiddle


    @dataclasses.dataclass
    class SampleExperimentNewBaseline(baseline_experiment.BaselineExperiment):
      """Experiment definition for SampleExperiment."""
      foo_setting: int = 4123
      derived_setting: int = 8246
      derived_list_setting: list = dataclasses.field(default_factory=lambda:
          [4123, 8246])

      def experiment_fixture(self) ->
          pax_fiddle.PaxConfig[parameterized_experiment.ParameterizedExperiment]:
        """Returns configuration for the entire experiment."""
        task = pax_fiddle.PaxConfig(tasks_lib.SingleTask,
            model=self.model_fixture())
        return pax_fiddle.PaxConfig(parameterized_experiment.ParameterizedExperiment,
            task=task, eval_datasets=[])

      def model_fixture(self) ->
          pax_fiddle.PaxConfig[test_fixtures.SampleModel]:
        """Returns configuration for the model."""
        return pax_fiddle.PaxConfig(test_fixtures.SampleModel,
            my_setting=self.foo_setting, derived_setting=self.derived_setting,
            derived_list_setting=self.derived_list_setting)
    '''
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
    expected = module_docstring("SampleShardedExperiment") + '''
    import dataclasses
    from paxml.experimental import baseline_experiment
    from paxml import parameterized_experiment
    from paxml import tasks_lib
    from paxml.tools.fiddle import test_fixtures
    from praxis import base_layer
    from praxis import pax_fiddle


    @dataclasses.dataclass
    class SampleShardedExperimentNewBaseline(baseline_experiment.BaselineExperiment):
      """Experiment definition for SampleShardedExperiment."""
      foo_setting: int = 4123
      derived_setting: int = 8246
      derived_list_setting: list = dataclasses.field(default_factory=lambda:
          [4123, 8246])

      def experiment_fixture(self) ->
          pax_fiddle.PaxConfig[parameterized_experiment.ParameterizedExperiment]:
        """Returns configuration for the entire experiment."""
        task = pax_fiddle.PaxConfig(tasks_lib.SingleTask,
            model=self.model_fixture())
        return pax_fiddle.PaxConfig(parameterized_experiment.ParameterizedExperiment,
            task=task, eval_datasets=[])

      def model_fixture(self) ->
          pax_fiddle.PaxConfig[test_fixtures.SampleModel]:
        """Returns configuration for the model."""
        return pax_fiddle.PaxConfig(test_fixtures.SampleModel,
            activation_split_dims_mapping=pax_fiddle.PaxConfig(base_layer.BaseLayer.ActivationSharding,
            out=['foo_axis', 'bar_axis']),
            my_setting=self.foo_setting,
            derived_setting=self.derived_setting,
            derived_list_setting=self.derived_list_setting)
    '''
    self.assertEqual(
        code.split(), expected.split(), msg=_update_expected_text(code)
    )

  def test_codegen_shared_sharding_unfactored(self):
    code = codegen.codegen_baseline_from_legacy(
        test_fixtures.SampleExperimentWithSharedShardingAnnotations,
        factor_out_sharding_annotations=False,
        has_train_dataset=False,
        has_input_specs_provider=False,
    )
    expected = (
        module_docstring("SampleExperimentWithSharedShardingAnnotations") + '''
    import dataclasses
    from paxml.experimental import baseline_experiment
    from paxml import parameterized_experiment
    from paxml import tasks_lib
    from paxml.tools.fiddle import test_fixtures
    from praxis import base_layer
    from praxis import pax_fiddle

    @dataclasses.dataclass
    class SampleExperimentWithSharedShardingAnnotationsNewBaseline(baseline_experiment.BaselineExperiment):
      """Experiment definition for SampleExperimentWithSharedShardingAnnotations."""
      foo_setting: int = 4123
      derived_setting: int = 8246
      derived_list_setting: list = dataclasses.field(default_factory=lambda:
          [4123, 8246])

      def experiment_fixture(self) ->
          pax_fiddle.PaxConfig[parameterized_experiment.ParameterizedExperiment]:
        """Returns configuration for the entire experiment."""
        task = pax_fiddle.PaxConfig(tasks_lib.SingleTask,
            model=self.model_fixture())
        return pax_fiddle.PaxConfig(parameterized_experiment.ParameterizedExperiment,
            task=task, eval_datasets=[])

      def model_fixture(self) ->
          pax_fiddle.PaxConfig[test_fixtures.SampleModel]:
        """Returns configuration for the model."""
        sublayers = [pax_fiddle.PaxConfig(test_fixtures.TestLayer,
            activation_split_dims_mapping=pax_fiddle.PaxConfig(base_layer.BaseLayer.ActivationSharding,
            out=['foo_axis', 'bar_axis'])),
            pax_fiddle.PaxConfig(test_fixtures.TestLayer,
            activation_split_dims_mapping=pax_fiddle.PaxConfig(base_layer.BaseLayer.ActivationSharding,
            out=['foo_axis', 'bar_axis']))]
        return pax_fiddle.PaxConfig(test_fixtures.SampleModel,
            activation_split_dims_mapping=pax_fiddle.PaxConfig(base_layer.BaseLayer.ActivationSharding,
            out=['foo_axis', 'bar_axis']), my_setting=self.foo_setting,
            derived_setting=self.derived_setting,
            derived_list_setting=self.derived_list_setting,
            sublayers=sublayers)
    '''
    )
    self.assertEqual(
        code.split(), expected.split(), msg=_update_expected_text(code)
    )

  def test_codegen_sharding_factored(self):
    code = codegen.codegen_baseline_from_legacy(
        test_fixtures.SampleShardedExperiment,
        has_train_dataset=False,
        has_input_specs_provider=False,
    )
    expected = module_docstring("SampleShardedExperiment") + '''
    import dataclasses
    from paxml.experimental import baseline_experiment
    from paxml import parameterized_experiment
    from paxml import tasks_lib
    from paxml.tools.fiddle import test_fixtures
    from praxis import pax_fiddle


    @dataclasses.dataclass
    class SampleShardedExperimentNewBaseline(baseline_experiment.BaselineExperiment):
      """Experiment definition for SampleShardedExperiment."""
      foo_setting: int = 4123
      derived_setting: int = 8246
      derived_list_setting: list = dataclasses.field(default_factory=lambda:
          [4123, 8246])

      def experiment_fixture(self) ->
          pax_fiddle.PaxConfig[parameterized_experiment.ParameterizedExperiment]:
        """Returns configuration for the entire experiment."""
        task = pax_fiddle.PaxConfig(tasks_lib.SingleTask,
            model=self.model_fixture())
        return pax_fiddle.PaxConfig(parameterized_experiment.ParameterizedExperiment,
            task=task, eval_datasets=[])

      def model_fixture(self) ->
          pax_fiddle.PaxConfig[test_fixtures.SampleModel]:
        """Returns configuration for the model."""
        model_config = pax_fiddle.PaxConfig(test_fixtures.SampleModel,
            my_setting=self.foo_setting, derived_setting=self.derived_setting,
            derived_list_setting=self.derived_list_setting)
        shard_model_config(model_config)
        return model_config

    def shard_model_config(model_config):
      model_config.activation_split_dims_mapping.out = ['foo_axis', 'bar_axis']
    '''
    self.assertEqual(
        code.split(), expected.split(), msg=_update_expected_text(code)
    )

  def test_codegen_shared_sharding_factored(self):
    code = codegen.codegen_baseline_from_legacy(
        test_fixtures.SampleExperimentWithSharedShardingAnnotations,
        has_train_dataset=False,
        has_input_specs_provider=False,
    )
    expected = (
        module_docstring("SampleExperimentWithSharedShardingAnnotations") + '''
    import dataclasses
    from paxml.experimental import baseline_experiment
    from paxml import parameterized_experiment
    from paxml import tasks_lib
    from paxml.tools.fiddle import test_fixtures
    from praxis import pax_fiddle


    @dataclasses.dataclass
    class SampleExperimentWithSharedShardingAnnotationsNewBaseline(baseline_experiment.BaselineExperiment):
      """Experiment definition for SampleExperimentWithSharedShardingAnnotations."""
      foo_setting: int = 4123
      derived_setting: int = 8246
      derived_list_setting: list = dataclasses.field(default_factory=lambda:
          [4123, 8246])

      def experiment_fixture(self) ->
          pax_fiddle.PaxConfig[parameterized_experiment.ParameterizedExperiment]:
        """Returns configuration for the entire experiment."""
        task = pax_fiddle.PaxConfig(tasks_lib.SingleTask,
            model=self.model_fixture())
        return pax_fiddle.PaxConfig(parameterized_experiment.ParameterizedExperiment,
            task=task, eval_datasets=[])

      def model_fixture(self) ->
          pax_fiddle.PaxConfig[test_fixtures.SampleModel]:
        """Returns configuration for the model."""
        model_config = pax_fiddle.PaxConfig(test_fixtures.SampleModel,
            my_setting=self.foo_setting, derived_setting=self.derived_setting,
            derived_list_setting=self.derived_list_setting,
            sublayers=[pax_fiddle.PaxConfig(test_fixtures.TestLayer),
            pax_fiddle.PaxConfig(test_fixtures.TestLayer)])
        shard_model_config(model_config)
        return model_config

    def shard_model_config(model_config):
      model_config.activation_split_dims_mapping.out = ['foo_axis', 'bar_axis']
      model_config.sublayers[0].activation_split_dims_mapping.out = ['foo_axis',
        'bar_axis']
      model_config.sublayers[1].activation_split_dims_mapping.out = ['foo_axis',
        'bar_axis']
    '''
    )
    self.assertEqual(
        code.split(), expected.split(), msg=_update_expected_text(code)
    )

  def test_codegen_with_datasets(self):
    code = codegen.codegen_baseline_from_legacy(
        test_fixtures.SampleExperimentWithDatasets,
        has_input_specs_provider=False,
    )
    expected = module_docstring("SampleExperimentWithDatasets") + '''
    import dataclasses
    from paxml.experimental import baseline_experiment
    from paxml import parameterized_experiment
    from paxml import tasks_lib
    from paxml.tools.fiddle import test_fixtures
    from praxis import base_input
    from praxis import pax_fiddle


    @dataclasses.dataclass
    class SampleExperimentWithDatasetsNewBaseline(baseline_experiment.BaselineExperiment):
      """Experiment definition for SampleExperimentWithDatasets."""
      foo_setting: int = 4123
      derived_setting: int = 8246
      derived_list_setting: list = dataclasses.field(default_factory=lambda:
          [4123, 8246])

      def experiment_fixture(self) ->
          pax_fiddle.PaxConfig[parameterized_experiment.ParameterizedExperiment]:
        """Returns configuration for the entire experiment."""
        task = pax_fiddle.PaxConfig(tasks_lib.SingleTask,
            model=self.model_fixture())
        return pax_fiddle.PaxConfig(parameterized_experiment.ParameterizedExperiment,
            task=task, training_dataset=self.training_dataset_fixture(),
            eval_datasets=self.eval_datasets_fixture())

      def model_fixture(self) ->
          pax_fiddle.PaxConfig[test_fixtures.SampleModel]:
        """Returns configuration for the model."""
        return pax_fiddle.PaxConfig(test_fixtures.SampleModel,
            my_setting=self.foo_setting, derived_setting=self.derived_setting,
            derived_list_setting=self.derived_list_setting)

      def training_dataset_fixture(self) ->
          pax_fiddle.PaxConfig[base_input.BaseInput]:
        """Returns configuration for the training dataset."""
        return pax_fiddle.PaxConfig(base_input.BaseInput, batch_size=1024,
            is_training=True)

      def eval_datasets_fixture(self) ->
          list[pax_fiddle.PaxConfig[base_input.BaseInput]]:
        """Returns configuration for eval datasets."""
        return [pax_fiddle.PaxConfig(base_input.BaseInput, batch_size=128)]
    '''
    self.assertEqual(
        code.split(), expected.split(), msg=_update_expected_text(code)
    )

  def test_codegen_with_decoder_datasets(self):
    code = codegen.codegen_baseline_from_legacy(
        test_fixtures.SampleExperimentWithDecoderDatasets,
        has_input_specs_provider=False,
        has_train_dataset=False,
    )
    expected = module_docstring("SampleExperimentWithDecoderDatasets") + '''
    import dataclasses
    from paxml.experimental import baseline_experiment
    from paxml import parameterized_experiment
    from paxml import tasks_lib
    from paxml.tools.fiddle import test_fixtures
    from praxis import base_input
    from praxis import pax_fiddle


    @dataclasses.dataclass
    class SampleExperimentWithDecoderDatasetsNewBaseline(baseline_experiment.BaselineExperiment):
      """Experiment definition for SampleExperimentWithDecoderDatasets."""
      foo_setting: int = 4123
      derived_setting: int = 8246
      derived_list_setting: list = dataclasses.field(default_factory=lambda:
          [4123, 8246])

      def experiment_fixture(self) ->
          pax_fiddle.PaxConfig[parameterized_experiment.ParameterizedExperiment]:
        """Returns configuration for the entire experiment."""
        task = pax_fiddle.PaxConfig(tasks_lib.SingleTask,
            model=self.model_fixture())
        return pax_fiddle.PaxConfig(parameterized_experiment.ParameterizedExperiment,
            task=task, eval_datasets=[],
            decoder_datasets=self.decoder_datasets_fixture())

      def model_fixture(self) ->
          pax_fiddle.PaxConfig[test_fixtures.SampleModel]:
        """Returns configuration for the model."""
        return pax_fiddle.PaxConfig(test_fixtures.SampleModel,
            my_setting=self.foo_setting, derived_setting=self.derived_setting,
            derived_list_setting=self.derived_list_setting)

      def decoder_datasets_fixture(self) ->
          list[pax_fiddle.PaxConfig[base_input.BaseInput]]:
        """Returns configuration for decoder datasets."""
        return [pax_fiddle.PaxConfig(base_input.BaseInput, batch_size=256)]
    '''
    self.assertEqual(
        code.split(), expected.split(), msg=_update_expected_text(code)
    )

  def test_codegen_with_input_specs_provider(self):
    code = codegen.codegen_baseline_from_legacy(
        test_fixtures.SampleExperimentWithInputSpecsProvider,
        has_train_dataset=False,
    )
    expected = module_docstring("SampleExperimentWithInputSpecsProvider") + '''
    import dataclasses
    from paxml.experimental import baseline_experiment
    from paxml import parameterized_experiment
    from paxml import tasks_lib
    from paxml.tools.fiddle import test_fixtures
    from praxis import pax_fiddle


    @dataclasses.dataclass
    class SampleExperimentWithInputSpecsProviderNewBaseline(baseline_experiment.BaselineExperiment):
      """Experiment definition for SampleExperimentWithInputSpecsProvider."""
      foo_setting: int = 4123
      derived_setting: int = 8246
      derived_list_setting: list = dataclasses.field(default_factory=lambda:
          [4123, 8246])

      def experiment_fixture(self) ->
          pax_fiddle.PaxConfig[parameterized_experiment.ParameterizedExperiment]:
        """Returns configuration for the entire experiment."""
        task = pax_fiddle.PaxConfig(tasks_lib.SingleTask,
            model=self.model_fixture())
        return pax_fiddle.PaxConfig(parameterized_experiment.ParameterizedExperiment,
            task=task, eval_datasets=[],
            input_specs_provider=self.input_specs_provider_fixture())

      def model_fixture(self) ->
          pax_fiddle.PaxConfig[test_fixtures.SampleModel]:
        """Returns configuration for the model."""
        return pax_fiddle.PaxConfig(test_fixtures.SampleModel,
            my_setting=self.foo_setting, derived_setting=self.derived_setting,
            derived_list_setting=self.derived_list_setting)

      def input_specs_provider_fixture(self) ->
          pax_fiddle.PaxConfig[test_fixtures.SampleInputSpecsProvider]:
        """Returns configuration for the input specs provider."""
        return pax_fiddle.PaxConfig(test_fixtures.SampleInputSpecsProvider)
    '''
    self.assertEqual(
        code.split(), expected.split(), msg=_update_expected_text(code)
    )

  def test_codegen_with_inlined_init_from_checkpoint_rules(self):
    code = codegen.codegen_baseline_from_legacy(
        test_fixtures.SampleExperimentWithInitFromCheckpointRules,
        has_train_dataset=False,
        has_input_specs_provider=False,
    )
    expected = (
        module_docstring("SampleExperimentWithInitFromCheckpointRules") + '''
    import dataclasses
    from paxml.experimental import baseline_experiment
    from paxml import parameterized_experiment
    from paxml import tasks_lib
    from paxml.tools.fiddle import test_fixtures
    from praxis import pax_fiddle


    @dataclasses.dataclass
    class SampleExperimentWithInitFromCheckpointRulesNewBaseline(baseline_experiment.BaselineExperiment):
      """Experiment definition for SampleExperimentWithInitFromCheckpointRules."""
      foo_setting: int = 4123
      derived_setting: int = 8246
      derived_list_setting: list = dataclasses.field(default_factory=lambda:
          [4123, 8246])

      def experiment_fixture(self) ->
          pax_fiddle.PaxConfig[parameterized_experiment.ParameterizedExperiment]:
        """Returns configuration for the entire experiment."""
        train = pax_fiddle.PaxConfig(tasks_lib.SingleTask.Train,
            init_from_checkpoint_rules=self.init_from_checkpoint_rules_fixture())
        task = pax_fiddle.PaxConfig(tasks_lib.SingleTask,
            model=self.model_fixture(), train=train)
        return pax_fiddle.PaxConfig(parameterized_experiment.ParameterizedExperiment,
            task=task, eval_datasets=[])

      def model_fixture(self) ->
          pax_fiddle.PaxConfig[test_fixtures.SampleModel]:
        """Returns configuration for the model."""
        return pax_fiddle.PaxConfig(test_fixtures.SampleModel,
            my_setting=self.foo_setting, derived_setting=self.derived_setting,
            derived_list_setting=self.derived_list_setting)

      def init_from_checkpoint_rules_fixture(self) -> dict[str, Any]:
        """Returns configuration for checkpoint initialization rules."""
        task_p = pax_fiddle.PaxConfig(tasks_lib.SingleTask,
            model=pax_fiddle.PaxConfig(test_fixtures.SampleModel,
            my_setting=4123, derived_setting=8246, derived_list_setting=[4123,
            8246]))
        checkpoint_loading_rules = pax_fiddle.PaxConfig(tasks_lib.CheckpointLoadingRules,
            task_p=task_p, load_rules=[('(.*)', '{}')], safe_load=True,
            ignore_rules=[], step=532000,
            input_specs_provider_p=pax_fiddle.PaxConfig(test_fixtures.SampleInputSpecsProvider))
        return {'/path/to/my/checkpoint': checkpoint_loading_rules}
    '''
    )
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
    expected = (
        module_docstring("SampleExperimentWithInitFromCheckpointRules") + '''
    import dataclasses
    from paxml.experimental import baseline_experiment
    from paxml import parameterized_experiment
    from paxml import tasks_lib
    from paxml.tools.fiddle import test_fixtures
    from praxis import pax_fiddle


    @dataclasses.dataclass
    class SampleExperimentWithInitFromCheckpointRulesNewBaseline(baseline_experiment.BaselineExperiment):
      """Experiment definition for SampleExperimentWithInitFromCheckpointRules."""
      foo_setting: int = 4123
      derived_setting: int = 8246
      derived_list_setting: list = dataclasses.field(default_factory=lambda:
          [4123, 8246])

      def experiment_fixture(self) ->
          pax_fiddle.PaxConfig[parameterized_experiment.ParameterizedExperiment]:
        """Returns configuration for the entire experiment."""
        train = pax_fiddle.PaxConfig(tasks_lib.SingleTask.Train,
            init_from_checkpoint_rules=self.init_from_checkpoint_rules_fixture())
        task = pax_fiddle.PaxConfig(tasks_lib.SingleTask,
            model=self.model_fixture(), train=train)
        return pax_fiddle.PaxConfig(parameterized_experiment.ParameterizedExperiment,
            task=task, eval_datasets=[])

      def model_fixture(self) -> pax_fiddle.PaxConfig[test_fixtures.SampleModel]:
        """Returns configuration for the model."""
        return pax_fiddle.PaxConfig(test_fixtures.SampleModel,
            my_setting=self.foo_setting, derived_setting=self.derived_setting,
            derived_list_setting=self.derived_list_setting)

      def init_from_checkpoint_rules_fixture(self) -> dict[str, Any]:
        """Returns configuration for checkpoint initialization rules."""
        base = test_fixtures.SampleExperimentWithInputSpecsProvider()
        checkpoint_loading_rules = pax_fiddle.PaxConfig(tasks_lib.CheckpointLoadingRules,
            task_p=base.task(), load_rules=[('(.*)', '{}')], safe_load=True,
            ignore_rules=[], step=532000,
            input_specs_provider_p=base.get_input_specs_provider_params())
        return {'/path/to/my/checkpoint': checkpoint_loading_rules}
    '''
    )
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
        fixture_docstrings={
            "pretrain_task_fixture": (
                "Returns configuration for the pretraining task."
            )
        },
    )
    # FIXME(b/289289423): Update this output when multiply-nested subfixtures
    # are better supported.
    expected = (
        module_docstring("SampleExperimentWithInitFromCheckpointRules") + '''
    import dataclasses
    from paxml.experimental import baseline_experiment
    from paxml import parameterized_experiment
    from paxml import tasks_lib
    from paxml.tools.fiddle import test_fixtures
    from praxis import pax_fiddle

    @dataclasses.dataclass
    class SampleExperimentWithInitFromCheckpointRulesNewBaseline(baseline_experiment.BaselineExperiment):
      """Experiment definition for SampleExperimentWithInitFromCheckpointRules."""
      foo_setting: int = 4123
      derived_setting: int = 8246
      derived_list_setting: list = dataclasses.field(default_factory=lambda:
          [4123, 8246])

      def experiment_fixture(self) ->
          pax_fiddle.PaxConfig[parameterized_experiment.ParameterizedExperiment]:
        """Returns configuration for the entire experiment."""
        train = pax_fiddle.PaxConfig(tasks_lib.SingleTask.Train,
        init_from_checkpoint_rules=self.init_from_checkpoint_rules_fixture())
        task = pax_fiddle.PaxConfig(tasks_lib.SingleTask, model=self.model_fixture(),
        train=train)
        return pax_fiddle.PaxConfig(parameterized_experiment.ParameterizedExperiment, task=task,
        eval_datasets=[])

      def model_fixture(self) -> pax_fiddle.PaxConfig[test_fixtures.SampleModel]:
        """Returns configuration for the model."""
        return pax_fiddle.PaxConfig(test_fixtures.SampleModel, my_setting=self.foo_setting, derived_setting=self.derived_setting,
        derived_list_setting=self.derived_list_setting)

      def pretrain_task_fixture(self) -> pax_fiddle.PaxConfig[tasks_lib.SingleTask]:
        """Returns configuration for the pretraining task."""
        return pax_fiddle.PaxConfig(tasks_lib.SingleTask, model=pax_fiddle.PaxConfig(test_fixtures.SampleModel, my_setting=4123, derived_setting=8246, derived_list_setting=[4123,
        8246]))

      def init_from_checkpoint_rules_fixture(self) -> dict[str, Any]:
        """Returns configuration for checkpoint initialization rules."""
        checkpoint_loading_rules = pax_fiddle.PaxConfig(tasks_lib.CheckpointLoadingRules, task_p=self.pretrain_task_fixture(), load_rules=[('(.*)', '{}')], safe_load=True, ignore_rules=[], step=532000,
        input_specs_provider_p=pax_fiddle.PaxConfig(test_fixtures.SampleInputSpecsProvider))
        return {'/path/to/my/checkpoint': checkpoint_loading_rules}
    '''
    )
    self.assertEqual(
        code.split(), expected.split(), msg=_update_expected_text(code)
    )

  def test_codegen_experiment_from_diff(self):
    code = codegen.codegen_experiment_diff(
        test_fixtures.SampleDerivedExperiment,
        baseline=test_fixtures.SampleExperimentNewBaseline,
        has_train_dataset=False,
        has_input_specs_provider=False,
    )
    expected = module_docstring("SampleDerivedExperiment") + '''
    from paxml.tools.fiddle import test_fixtures

    @dataclasses.dataclass
    class SampleDerivedExperimentNewExperiment(test_fixtures.SampleExperimentNewBaseline):
    """Experiment definition for SampleDerivedExperiment."""

      def experiment_fixture(self):
        config = super().experiment_fixture()
        config.task.model.my_setting = 4217
        return config
    '''
    self.assertEqual(
        code.split(), expected.split(), msg=_update_expected_text(code)
    )

  def test_codegen_experiment_highlevel_only(self):
    code = codegen.codegen_experiment_diff(
        test_fixtures.SampleDerivedExperimentHighlevel,
        baseline=test_fixtures.SampleExperimentNewBaseline,
        has_train_dataset=False,
        has_input_specs_provider=False,
    )
    expected = module_docstring("SampleDerivedExperimentHighlevel") + '''
    from paxml.tools.fiddle import test_fixtures

    @dataclasses.dataclass
    class SampleDerivedExperimentHighlevelNewExperiment(test_fixtures.SampleExperimentNewBaseline):
      """Experiment definition for SampleDerivedExperimentHighlevel."""
      # Overrides to existing high-level settings.
      foo_setting: int = 4217
      derived_setting: int = 8434
      derived_list_setting: list = dataclasses.field(default_factory=lambda:
          [4217, 8434])
    '''
    self.assertEqual(
        code.split(), expected.split(), msg=_update_expected_text(code)
    )


if __name__ == "__main__":
  absltest.main()
