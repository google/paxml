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

"""Pax-specific Fiddle codegen APIs."""

# Note: Fiddle and Pax devs are in collaboration; please generally do not import
# private libraries from Fiddle.

import dataclasses
from typing import Any, Callable, Dict, Optional, Type

import fiddle as fdl
from fiddle import daglish
from fiddle import diffing
from fiddle import selectors
from fiddle._src.codegen.auto_config import code_ir
from fiddle._src.codegen.auto_config import ir_to_cst
from fiddle.codegen import codegen
from fiddle.codegen import codegen_diff
from fiddle.codegen.auto_config import experimental_top_level_api
import libcst as cst
from libcst import matchers
from paxml import base_experiment
from paxml import parameterized_experiment
from paxml.tools.fiddle import codegen_external_init_checkpoint_fns
from paxml.tools.fiddle import codegen_pax_code_ir
from paxml.tools.fiddle import codegen_tracer
from paxml.tools.fiddle import config_normalization
from paxml.tools.fiddle import make_parameterized_experiment
from paxml.tools.fiddle import remove_sharding
from paxml.tools.fiddle import unshare_sharding
from praxis import pax_fiddle


@dataclasses.dataclass(frozen=True)
class InitTask(experimental_top_level_api.InitTask):
  """Parameterizes fixtures by a highlevel settings object."""

  def __call__(self, value: Any) -> Any:
    result = super().__call__(value)
    modified = codegen_pax_code_ir.PaxCodegenTask(
        original_config=result.original_config,
        top_level_call=result.top_level_call,
    )
    modified.import_manager.add_by_name("dataclasses")
    return modified


@dataclasses.dataclass(frozen=True)
class HighlevelParameterization(experimental_top_level_api.CodegenPass):
  """Parameterizes fixtures by a highlevel settings object."""

  lowercasing: bool = False

  def __call__(self, task: Any, **pass_kwargs: Any) -> Any:
    assert isinstance(task, codegen_pax_code_ir.PaxCodegenTask)
    all_fns = task.top_level_call.all_fixture_functions()

    def process_fn(fn: code_ir.FixtureFunction):
      self_name = code_ir.Name("self", is_generated=False)
      fn.parameters.insert(0, code_ir.Parameter(self_name))

      def add_self_to_calls(value, state: daglish.State):
        value = state.map_children(value)

        # Convert calls to sub-fixtures like model_fixture() to
        # self.model_fixture().
        if isinstance(value, code_ir.SymbolOrFixtureCall):
          if isinstance(value.symbol_expression, code_ir.FixtureReference):
            value.symbol_expression = code_ir.AttributeExpression(
                base=code_ir.VariableReference(self_name),
                attribute=value.symbol_expression.name.value,
            )

        # Convert any instances of highlevel variables to relevant expressions.
        elif hasattr(value, "__highlevel_name__"):
          attribute_name = value.__highlevel_name__
          if self.lowercasing:
            attribute_name = attribute_name.lower()
          task.highlevel_accesses[attribute_name] = value
          return code_ir.AttributeExpression(
              base=code_ir.VariableReference(self_name),
              attribute=attribute_name,
          )

        # Process dict keys too (normally ignored by daglish).
        if isinstance(value, dict):
          converted = {}
          for key, sub_value in value.items():
            if hasattr(key, "__highlevel_name__"):
              attribute_name = key.__highlevel_name__
              if self.lowercasing:
                attribute_name = attribute_name.lower()
              task.highlevel_accesses[attribute_name] = key
              key = code_ir.AttributeExpression(
                  base=code_ir.VariableReference(self_name),
                  attribute=attribute_name,
              )
            converted[key] = sub_value
          return converted

        return value

      fn.replace_with(daglish.MemoizedTraversal.run(add_self_to_calls, fn))

    for fn in all_fns:
      process_fn(fn)

    return task


@dataclasses.dataclass(frozen=True)
class ModelShardingDiff:
  diff: diffing.Diff
  old: pax_fiddle.Config[Any]


def _sharding_diff(
    experiment_cls: Type[base_experiment.BaseExperiment],
    unshare_sharding_config: bool = True,
) -> Optional[ModelShardingDiff]:
  """Returns a diff that will re-add sharding to a model.

  The diff is calculated by removing the sharding from the model for the "diff
  base", and using the original model (with sharding) as the diff
  right-hand-side.

  Args:
    experiment_cls: The class of the experiment to generate a diff for.
    unshare_sharding_config: Whether to unshare sharding configurations. Please
      see codegen_baseline_from_legacy() for details.
  """
  sharding_diff_rhs = experiment_cls().task().model
  if unshare_sharding_config:
    sharding_diff_rhs = unshare_sharding.unshare_sharding(sharding_diff_rhs)
  sharding_diff_lhs = remove_sharding.remove_sharding(
      sharding_diff_rhs, replace_with_default=True
  )
  diff = diffing.build_diff(sharding_diff_lhs, sharding_diff_rhs)
  return ModelShardingDiff(diff, sharding_diff_lhs) if diff.changes else None


@dataclasses.dataclass(frozen=True)
class MakeShardingFiddler(experimental_top_level_api.CodegenPass):
  """Modified LibCST conversion that doesn't emit imports."""

  PASS_INPUT_KWARGS = ["model_sharding_diff"]

  def __call__(
      self,
      task: Any,
      model_sharding_diff: Optional[ModelShardingDiff],
  ) -> Any:
    assert isinstance(task, codegen_pax_code_ir.PaxCodegenTask)
    if model_sharding_diff:
      assert isinstance(model_sharding_diff, ModelShardingDiff)
      task.sharding_diff_module = codegen_diff.fiddler_from_diff(
          model_sharding_diff.diff,
          old=model_sharding_diff.old,
          func_name="shard_model_config",
          param_name="model_config",
          import_manager=task.import_manager,
          variable_naming="short",
      )
    return task


@dataclasses.dataclass(frozen=True)
class _AddShardingCall:
  model_fixture_name: str
  add_sharding_function: str


@dataclasses.dataclass(frozen=True)
class IrToCst(experimental_top_level_api.CodegenPass):
  """Modified LibCST conversion that doesn't emit imports."""

  class_name: str = "Experiment"
  add_sharding_call: Optional[_AddShardingCall] = None

  def _add_sharding_call(self, fn_code: cst.FunctionDef) -> cst.FunctionDef:
    """Adds a call to the sharding function within the model fixture.

    This would be a little prettier as an IR pass, but we don't currently have
    an attribute in code_ir.FixtureFunction for mutation-based functions, and it
    would probably be nice not to worry about it for the rest of codegen. So
    this is just an add-on CST-modifying function for now.

    Args:
      fn_code: Function definition for the model fixture.

    Returns:
      Modified version of fn_code.
    """
    assert isinstance(
        fn_code.body, cst.IndentedBlock
    ), f"Expected an IndentedBlock body, got a {type(fn_code.body)}"
    body = fn_code.body.body
    if not matchers.matches(
        body[-1], matchers.SimpleStatementLine([matchers.Return()])
    ):
      raise ValueError(
          "Expected a single return line at the end of the model fixture."
      )
    # TODO(b/291758766): use namer to avoid unlikely collisions.
    var_name = cst.Name("model_config")
    shard_model_name = cst.Name(self.add_sharding_call.add_sharding_function)  # pytype: disable=attribute-error
    last_lines = [
        cst.Assign([cst.AssignTarget(var_name)], body[-1].body[0].value),
        cst.Expr(cst.Call(shard_model_name, args=[cst.Arg(var_name)])),
        cst.Return(var_name),
    ]
    body = body[:-1] + [cst.SimpleStatementLine([line]) for line in last_lines]
    return fn_code.with_changes(body=fn_code.body.with_changes(body=body))

  def __call__(self, task: Any) -> Any:
    assert isinstance(task, codegen_pax_code_ir.PaxCodegenTask)

    module_body = list(task.import_manager.sorted_import_lines())

    # Add highlevel accesses as attributes.
    class_body = []
    for name, value in task.highlevel_accesses.items():
      types = [
          ("bool", (bool, codegen_tracer.BoolTracer)),
          ("int", int),
          # `int` will never take effect, but included here for completeness
          ("float", (float, int)),
          ("str", str),
          ("list", list),
          ("tuple", tuple),
      ]
      type_name = "Any"
      for candidate_name, candidate_typ in types:
        if isinstance(value, candidate_typ):
          type_name = candidate_name
          break

      # Convert the value to its primitive form.
      value = codegen_tracer.remove_tracers(value)

      class_body.append(
          cst.SimpleStatementLine([
              cst.AnnAssign(
                  target=cst.Name(name),
                  annotation=cst.Annotation(cst.Name(type_name)),
                  value=ir_to_cst.code_for_expr(value),
              ),
          ])
      )

    # Add main fixtures.
    for fn in task.top_level_call.all_fixture_functions():
      fn_code = ir_to_cst.code_for_fn(fn, task=task)
      if (
          self.add_sharding_call
          and fn.name.value == self.add_sharding_call.model_fixture_name
      ):
        fn_code = self._add_sharding_call(fn_code)

      # The main codegen produces two newlines because it's emitting
      # module-level functions, but generally we only want one for class
      # methods.
      fn_code = fn_code.with_changes(
          leading_lines=[cst.EmptyLine(newline=cst.Newline())]
      )
      class_body.append(fn_code)

    # Create the class with these attributes and fixtures.
    module_body.append(
        cst.ClassDef(
            name=cst.Name(self.class_name),
            body=cst.IndentedBlock(body=class_body),
            leading_lines=[
                cst.EmptyLine(newline=cst.Newline()),
                cst.EmptyLine(newline=cst.Newline()),
            ],
            decorators=[
                cst.Decorator(
                    cst.parse_expression("dataclasses.dataclass(frozen=True)")
                )
            ],
        )
    )

    # Add fiddler for sharding, if it is set.
    if task.sharding_diff_module:
      matcher = matchers.SaveMatchedNode(
          matchers.FunctionDef(matchers.Name("shard_model_config")),
          "shard_model_fn",
      )
      matches = matchers.extractall(task.sharding_diff_module, matcher)
      assert (
          len(matches) == 1
      ), "If sharding_diff_module is present, should include fn"
      module_body.append(matches[0]["shard_model_fn"])

    return cst.Module(body=module_body, default_indent="  ")


@dataclasses.dataclass
class PaxExpressionIsComplex:
  """Pax-specific expression complexity logic."""

  base: Callable[[Any], bool]

  def __call__(self, node):
    return self.base(node)


def _get_pass_idx(codegen_config, cls) -> int:
  for i, codegen_pass in enumerate(codegen_config.passes):
    if issubclass(fdl.get_callable(codegen_pass), cls):
      return i
  raise ValueError(f"Could not find codegen pass {cls}")


def code_generator_config(
    top_level_fixture_name: str = "config_fixture",
    max_expression_complexity: int | None = None,
    include_history: bool = False,
    debug_print: bool = False,
    lowercase_highlevel_settings: bool = True,
    class_name: str = "Experiment",
    init_checkpoint_experiments_strict: bool = True,
    add_sharding_call: Optional[_AddShardingCall] = None,
):
  """Returns a code generator object.

  Args:
    top_level_fixture_name: Name of the top-level fixture.
    max_expression_complexity: Breaks complex expressions into variables for
      readability.
    include_history: Whether history should be included. These currently appear
      as trailing comments in the field of Buildable's.
    debug_print: Whether to use the IR printer to print intermediate
      representations as various passes run to generate code.
    lowercase_highlevel_settings: Lowercase the high-level variable names.
    class_name: Name of the high-level experiment class.
    init_checkpoint_experiments_strict: Whether to enforce that
      init_checkpoint_experiments contains entries for all items in
      init_from_checkpoint_rules, if it is provided.
    add_sharding_call: Modify the model fixture to call an add sharding method
      before returning.
  """
  config = codegen.code_generator.as_buildable(
      top_level_fixture_name=top_level_fixture_name,
      max_expression_complexity=max_expression_complexity,
      include_history=include_history,
      debug_print=debug_print,
  )

  # Replace InitTask with our custom one, that creates a wrapped task object.
  (init_task_pass,) = selectors.select(
      config, experimental_top_level_api.InitTask
  )
  fdl.update_callable(init_task_pass, InitTask)

  # Transform init_from_checkpoint_rules. We need to do this after sub-fixture
  # extraction though, so that the IDs of objects don't change.
  sub_fixtures_idx = _get_pass_idx(
      config, experimental_top_level_api.TransformSubFixtures
  )
  config.passes.insert(
      sub_fixtures_idx + 1,
      fdl.Config(
          codegen_external_init_checkpoint_fns.InitCheckpointRulesFromOtherTask,
          strict=init_checkpoint_experiments_strict,
      ),
  )

  # Replace expression complexity logic with Pax-specific one.
  (complex_pass,) = selectors.select(
      config, experimental_top_level_api.MoveComplexNodesToVariables
  )
  complex_pass.is_complex = fdl.Config(
      PaxExpressionIsComplex, base=complex_pass.is_complex
  )

  # Add the core highlevel parameterization pass. This adds a `self` parameter
  # and abstracts certain expressions using tracers, linking them to high-level
  # settings.
  move_shared_idx = _get_pass_idx(
      config, experimental_top_level_api.MoveSharedNodesToVariables
  )
  config.passes.insert(
      move_shared_idx,
      fdl.Config(
          HighlevelParameterization, lowercasing=lowercase_highlevel_settings
      ),
  )

  # Insert the sharding pass before the IrToCst pass.
  ir_to_cst_idx = _get_pass_idx(config, experimental_top_level_api.IrToCst)
  config.passes.insert(ir_to_cst_idx, fdl.Config(MakeShardingFiddler))

  # Replace IrToCst pass with our custom one that emits a class.
  (ir_to_cst_pass,) = selectors.select(
      config, experimental_top_level_api.IrToCst
  )
  fdl.update_callable(ir_to_cst_pass, IrToCst)
  ir_to_cst_pass.class_name = class_name
  ir_to_cst_pass.add_sharding_call = add_sharding_call

  return config


def codegen_baseline_from_legacy(
    experiment_cls: Type[base_experiment.BaseExperiment],
    *,
    factor_out_sharding_annotations: bool = True,
    unshare_sharding_config: bool = True,
    remove_defaults: bool = True,
    lowercase_highlevel_settings: bool = True,
    has_train_dataset: bool = True,
    has_input_specs_provider: bool = True,
    init_checkpoint_experiments: Optional[
        Dict[str, Optional[Type[base_experiment.BaseExperiment]]]
    ] = None,
    init_checkpoint_experiments_strict: bool = True,
    additional_sub_fixtures: Optional[
        Callable[
            [
                pax_fiddle.Config[
                    parameterized_experiment.ParameterizedExperiment
                ]
            ],
            Dict[str, Any],
        ]
    ] = None,
):
  """Generates code for a baseline configuration, from a legacy BaseExperiment.

  Primitive parameters and lists/tuples from the experiment, typically expressed
  as class variables e.g.

  class Foo:
    EMBED_DIM = 12

  are pulled into high-level parameters

  Args:
    experiment_cls: Class used for the experiment.
    factor_out_sharding_annotations: Whether to remove sharding annotations.
    unshare_sharding_config: Whether to unshare values in sharding
      configuration. Fiddle generally retains information when mutables like
      lists or sub-config objects are shared, but for sharding annotations this
      shouldn't matter; only the values matter. Generated code is generally
      prettier when you unshare sharding configs. However, if you later write a
      test asserting equality with the original config, please make sure to run
      unshare_sharding.unshare_sharding() on the original config.
    remove_defaults: Whether to remove default values. Often with Pax configs,
      dataclass field defaulting magic means that you get large, expanded
      templates that may actually be unused or equal to their default values.
    lowercase_highlevel_settings: Lowercase the high-level variable names.
      Generally this is recommended, since it is more PEP-8 compliant, as
      high-level attributes become fields on a class. However, in the case where
      there may be name conflicts (names distinguished only by capitalization),
      or when the user intentionally wants to distinguish high and low level
      settings by captialization, this can be set to False.
    has_train_dataset: Whether the configuration has a training dataset. If not,
      then the resulting ParameterizedExperiment config's train_dataset field
      will not be populated.
    has_input_specs_provider: Whether the experiment has an input specs provider
      defined. Please set to False if you are using the (slower) one predefined
      by the dataset.
    init_checkpoint_experiments: Dictionary mapping checkpoint path to the
      experiment used to initialize it. For example, {"/path/to/my/checkpoint":
      my_pretrain_model.PretrainedModelExperiment}. This is useful for avoiding
      inlining *too* much code into a baseline, which might make it less
      readable.
    init_checkpoint_experiments_strict: Whether to check that the checkpoint
      experiments are provided for all init_from_checkpoint_rules entries. This
      only applies if `init_checkpoint_experiments` is provided.
    additional_sub_fixtures: Optional callable for producing additional
      sub-fixtures. It is generally recommended to choose a granularity of
      sub-fixtures so that experiments can override parts of a baseline without
      doing much mutation. This function will receive the root
      ParameterizedExperiment configuration, and should produce a dict with keys
      as additional sub-fixture names, and values as sub-config objects.

  Returns:
    Generated code.
  """
  normalizer = config_normalization.ConfigNormalizer(
      remove_sharding_annotations=factor_out_sharding_annotations,
      unshare_sharding_config=unshare_sharding_config,
      remove_defaults=remove_defaults,
      convert_seqio_task_objects=True,
  )
  overall_config = make_parameterized_experiment.from_legacy(
      experiment_cls=codegen_tracer.make_subclass_mixin(experiment_cls),
      normalizer=normalizer,
      has_train_dataset=has_train_dataset,
      has_input_specs_provider=has_input_specs_provider,
  )

  # If factor_out_sharding_annotations is set, save some A/B configs to diff the
  # sharding annotations. This is a bit disjointed because we want to compute it
  # before removing nested/deep field defaults.
  if factor_out_sharding_annotations:
    model_sharding_diff = _sharding_diff(
        experiment_cls, unshare_sharding_config=unshare_sharding_config
    )
  else:
    model_sharding_diff = None

  add_sharding_call = None
  if model_sharding_diff is not None:
    add_sharding_call = _AddShardingCall("model_fixture", "shard_model_config")
  codegen_config = code_generator_config(
      top_level_fixture_name="experiment_fixture",
      class_name=f"{experiment_cls.__name__}_NewBaseline",
      max_expression_complexity=6,
      lowercase_highlevel_settings=lowercase_highlevel_settings,
      init_checkpoint_experiments_strict=init_checkpoint_experiments_strict,
      add_sharding_call=add_sharding_call,
      debug_print=False,
  )
  codegen_obj = fdl.build(codegen_config)

  # Set up sub-fixtures. Which are present will depend on the experiment. For
  # most real-world experiments, we expect all of these to be present.
  sub_fixtures = {
      "model_fixture": overall_config.task.model,
  }
  if has_train_dataset:
    sub_fixtures["training_dataset_fixture"] = overall_config.training_dataset
  if overall_config.eval_datasets:
    sub_fixtures["eval_datasets_fixture"] = overall_config.eval_datasets
  if has_input_specs_provider:
    sub_fixtures["input_specs_provider_fixture"] = (
        overall_config.input_specs_provider
    )
  if additional_sub_fixtures:
    sub_fixtures.update(additional_sub_fixtures(overall_config))
  if overall_config.decoder_datasets:
    sub_fixtures["decoder_datasets_fixture"] = overall_config.decoder_datasets
  try:
    init_from_checkpoint_rules = (
        overall_config.task.train.init_from_checkpoint_rules
    )
  except (AttributeError, ValueError):
    pass
  else:
    sub_fixtures["init_from_checkpoint_rules_fixture"] = (
        init_from_checkpoint_rules
    )

  return codegen_obj(
      overall_config,
      sub_fixtures=sub_fixtures,
      model_sharding_diff=model_sharding_diff,
      init_checkpoint_experiments=init_checkpoint_experiments,
  ).code
