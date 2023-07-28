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

import copy
import dataclasses
import inspect
from typing import Any, Callable, Dict, List, Optional, Type

import fiddle as fdl
from fiddle import daglish
from fiddle import diffing
from fiddle import selectors
from fiddle import validation
from fiddle._src.codegen.auto_config import code_ir
from fiddle._src.codegen.auto_config import ir_to_cst
from fiddle.codegen import codegen
from fiddle.codegen import codegen_diff
from fiddle.codegen.auto_config import experimental_top_level_api
from fiddle.experimental import visualize
import fiddle.extensions.jax
import fiddle.extensions.seqio
import libcst as cst
from libcst import matchers
from paxml import base_experiment
from paxml import parameterized_experiment
from paxml.experimental import baseline_experiment
from paxml.tools.fiddle import codegen_external_init_checkpoint_fns
from paxml.tools.fiddle import codegen_highlevel_parameterization
from paxml.tools.fiddle import codegen_pax_code_ir
from paxml.tools.fiddle import codegen_sharding
from paxml.tools.fiddle import codegen_tracer
from paxml.tools.fiddle import config_normalization
from paxml.tools.fiddle import make_parameterized_experiment
from praxis import pax_fiddle


fiddle.extensions.jax.enable()
fiddle.extensions.seqio.enable()


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


def _make_default_factory(node: cst.CSTNode) -> cst.Expr:
  return cst.Call(
      func=cst.parse_expression("dataclasses.field"),
      args=[
          cst.Arg(
              value=cst.Lambda(
                  params=cst.Parameters([]),
                  body=node,
              ),
              keyword=cst.Name("default_factory"),
              equal=cst.AssignEqual(
                  whitespace_before=cst.SimpleWhitespace(value=""),
                  whitespace_after=cst.SimpleWhitespace(value=""),
              ),
          )
      ],
  )


def _class_attributes(highlevel_settings: Dict[str, Any]) -> List[cst.CSTNode]:
  """Returns CST nodes for dataclass fields, given highlevel settings."""
  result = []
  for name, value in highlevel_settings.items():
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

    if daglish.is_immutable(value):
      value = ir_to_cst.code_for_expr(value)
    else:
      value = _make_default_factory(ir_to_cst.code_for_expr(value))

    result.append(
        cst.SimpleStatementLine([
            cst.AnnAssign(
                target=cst.Name(name),
                annotation=cst.Annotation(cst.Name(type_name)),
                value=value,
            ),
        ])
    )
  return result


def _make_docstring(docstring: str, *, indent: int) -> cst.SimpleStatementLine:
  docstring = inspect.cleandoc(docstring)
  if not ('"""' in docstring or "'''" in docstring):
    docstring = docstring.strip()
    if "\n" in docstring:
      docstring = docstring + "\n" + (" " * indent)
    docstring = '"""' + docstring + '"""'
  return cst.SimpleStatementLine(
      body=[cst.Expr(cst.SimpleString(value=docstring))]
  )


def _make_class_def(
    name: str,
    bases: List[cst.Arg],
    body: List[cst.BaseStatement],
    docstring: Optional[str] = None,
) -> cst.ClassDef:
  """Makes a LibCST class definition."""
  if docstring:
    body = [_make_docstring(docstring, indent=2), *body]
  return cst.ClassDef(
      name=cst.Name(name),
      bases=bases,
      body=cst.IndentedBlock(body=body),
      leading_lines=[
          cst.EmptyLine(newline=cst.Newline()),
          cst.EmptyLine(newline=cst.Newline()),
      ],
      decorators=[cst.Decorator(cst.parse_expression("dataclasses.dataclass"))],
  )


def _comment_line(comment: str) -> cst.EmptyLine:
  return cst.EmptyLine(
      indent=True,
      comment=cst.Comment(value=f"# {comment}"),
  )


def _classmethod_make_experiment_fn() -> cst.FunctionDef:
  node = cst.parse_module("""
class UnusedClass:  # For whitespace/newlines/indentation.

  @classmethod
  def make_experiment(cls, **kwargs):
    return cls(**kwargs).experiment_fixture()
""")
  return _extract_function_def(node, "make_experiment")


def _highlevel_config_fn(import_manager: ..., cls_name: str) -> cst.FunctionDef:
  """Generates the highlevel_config function."""
  pax_fiddle_config = import_manager.add(pax_fiddle.Config)
  node = cst.parse_module(f"""
import unused_module  # For whitespace/newlines


def highlevel_config():
  return {pax_fiddle_config}({cls_name}.make)
""")
  return _extract_function_def(node, "highlevel_config")


def _lower_fn(import_manager: ...) -> cst.FunctionDef:
  """Generates the function to lower highlevel to lowlevel configs."""
  pax_fiddle_build = import_manager.add(pax_fiddle.build)
  fdl_ordered_arguments = import_manager.add(fdl.ordered_arguments)
  fdl_get_callable = import_manager.add(fdl.get_callable)
  node = cst.parse_module(f"""
import unused_module  # For whitespace/newlines


def lower(highlevel_config):
  kwargs = {pax_fiddle_build}({fdl_ordered_arguments}(highlevel_config))
  exp_cls = {fdl_get_callable}(highlevel_config)
  return exp_cls(**kwargs).experiment_fixture()
""")
  return _extract_function_def(node, "lower")


def _lowlevel_config_fn() -> cst.FunctionDef:
  """Generates the lowlevel_config function."""
  node = cst.parse_module("""
import unused_module  # For whitespace/newlines


def lowlevel_config():
  return lower(highlevel_config())
""")
  return _extract_function_def(node, "lowlevel_config")


def _extract_function_def(node: cst.CSTNode, name: str) -> cst.FunctionDef:
  matcher = matchers.SaveMatchedNode(
      matchers.FunctionDef(matchers.Name(name)),
      "matched_fn",
  )
  matches = matchers.extractall(node, matcher)
  if len(matches) != 1:
    raise ValueError("Found multiple functions named {name}")
  return matches[0]["matched_fn"]


_DEFAULT_MODULE_DOCSTRING = """Baseline experiment for the {base_experiment_name} model.

This file was generated using the following code:

from paxml.tools.fiddle import codegen
...

And the following manual cleanups:

 * ...
"""


@dataclasses.dataclass(frozen=True)
class IrToCst(experimental_top_level_api.CodegenPass):
  """Modified LibCST conversion that doesn't emit imports."""

  class_name: str = "Experiment"
  add_sharding_call: Optional[codegen_sharding.AddShardingCall] = None
  add_boilerplate: bool = True
  fixture_docstrings: Optional[Dict[str, str]] = None

  def __call__(self, task: Any) -> Any:
    fixture_docstrings = self.fixture_docstrings or {}
    assert isinstance(task, codegen_pax_code_ir.PaxCodegenTask)

    # Add highlevel accesses as attributes.
    class_body = _class_attributes(task.highlevel_accesses)

    # Add main fixtures.
    for fn in task.top_level_call.all_fixture_functions():
      fn_code = ir_to_cst.code_for_fn(fn, task=task)
      if (
          self.add_sharding_call
          and fn.name.value == self.add_sharding_call.model_fixture_name
      ):
        fn_code = self.add_sharding_call(fn_code)

      # The main codegen produces two newlines because it's emitting
      # module-level functions, but generally we only want one for class
      # methods.
      fn_code = fn_code.with_changes(
          leading_lines=[cst.EmptyLine(newline=cst.Newline())]
      )

      # Add the docstring if provided.
      docstring = fixture_docstrings.get(fn.name.value, None)
      if docstring and isinstance(fn_code.body, cst.IndentedBlock):
        fn_code = fn_code.with_changes(
            body=fn_code.body.with_changes(
                body=[_make_docstring(docstring, indent=4), *fn_code.body.body]
            )
        )

      class_body.append(fn_code)

    # Create the class with these attributes and fixtures.
    class_base_expression = task.import_manager.add(
        baseline_experiment.BaselineExperiment
    )
    name = self.class_name.replace("NewBaseline", "")
    module_body = [
        _make_class_def(
            self.class_name,
            [cst.Arg(cst.parse_expression(class_base_expression))],
            class_body,
            docstring=f"Experiment definition for {name}.",
        )
    ]

    # Add fiddler for sharding, if it is set.
    if task.sharding_diff_module:
      module_body.append(
          _extract_function_def(task.sharding_diff_module, "shard_model_config")
      )

    # Add imports last, since the import manager is updated.
    module_body[0:0] = task.import_manager.sorted_import_lines()

    # Add the module docstring.
    module_body.insert(
        0,
        _make_docstring(
            _DEFAULT_MODULE_DOCSTRING.format(base_experiment_name=name),
            indent=0,
        ),
    )

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
    add_sharding_call: Optional[codegen_sharding.AddShardingCall] = None,
    fixture_docstrings: Optional[Dict[str, str]] = None,
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
    fixture_docstrings: Docstrings for fixtures.
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
  if max_expression_complexity is not None:
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
          codegen_highlevel_parameterization.HighlevelParameterization,
          lowercasing=lowercase_highlevel_settings,
      ),
  )

  # Insert the sharding pass before the IrToCst pass.
  ir_to_cst_idx = _get_pass_idx(config, experimental_top_level_api.IrToCst)
  config.passes.insert(
      ir_to_cst_idx, fdl.Config(codegen_sharding.MakeShardingFiddler)
  )

  # Replace IrToCst pass with our custom one that emits a class.
  (ir_to_cst_pass,) = selectors.select(
      config, experimental_top_level_api.IrToCst
  )
  fdl.update_callable(ir_to_cst_pass, IrToCst)
  ir_to_cst_pass.class_name = class_name
  ir_to_cst_pass.add_sharding_call = add_sharding_call
  ir_to_cst_pass.fixture_docstrings = fixture_docstrings

  return config


_DEFAULT_FIXTURE_DOCSTRINGS = {
    "experiment_fixture": "Returns configuration for the entire experiment.",
    "model_fixture": "Returns configuration for the model.",
    "input_specs_provider_fixture": (
        "Returns configuration for the input specs provider."
    ),
    "eval_datasets_fixture": "Returns configuration for eval datasets.",
    "decoder_datasets_fixture": "Returns configuration for decoder datasets.",
    "training_dataset_fixture": (
        "Returns configuration for the training dataset."
    ),
    "init_from_checkpoint_rules_fixture": (
        "Returns configuration for checkpoint initialization rules."
    ),
}


def codegen_baseline_from_legacy(
    experiment_cls: Type[base_experiment.BaseExperiment],
    *,
    has_train_dataset: bool = True,
    has_input_specs_provider: bool = True,
    unshare_sharding_config: bool = True,
    remove_defaults: bool = True,
    remove_eval_datasets: bool = False,
    remove_decoder_datasets: bool = False,
    factor_out_sharding_annotations: bool = True,
    lowercase_highlevel_settings: bool = True,
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
    fixture_docstrings: Optional[Dict[str, str]] = None,
    init_checkpoint_experiments: Optional[
        Dict[str, Optional[Type[base_experiment.BaseExperiment]]]
    ] = None,
    init_checkpoint_experiments_strict: bool = True,
) -> str:
  """Generates code for a baseline configuration, from a legacy BaseExperiment.

  Primitive parameters and lists/tuples from the experiment, typically expressed
  as class variables e.g.

  class Foo:
    EMBED_DIM = 12

  are pulled into high-level parameters

  Args:
    experiment_cls: Class used for the experiment.
    has_train_dataset: Whether the configuration has a training dataset. If not,
      then the resulting ParameterizedExperiment config's train_dataset field
      will not be populated.
    has_input_specs_provider: Whether the experiment has an input specs provider
      defined. Please set to False if you are using the (slower) one predefined
      by the dataset.
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
    remove_eval_datasets: Whether to remove/clear eval_datasets, even if they
      exist.
    remove_decoder_datasets: Whether to remove/clear decoder_datasets, even if
      they exist.
    factor_out_sharding_annotations: Whether to remove sharding annotations.
    lowercase_highlevel_settings: Lowercase the high-level variable names.
      Generally this is recommended, since it is more PEP-8 compliant, as
      high-level attributes become fields on a class. However, in the case where
      there may be name conflicts (names distinguished only by capitalization),
      or when the user intentionally wants to distinguish high and low level
      settings by captialization, this can be set to False.
    additional_sub_fixtures: Optional callable for producing additional
      sub-fixtures. It is generally recommended to choose a granularity of
      sub-fixtures so that experiments can override parts of a baseline without
      doing much mutation. This function will receive the root
      ParameterizedExperiment configuration, and should produce a dict with keys
      as additional sub-fixture names, and values as sub-config objects.
    fixture_docstrings: Docstrings to be generated for fixtures. Defaults will
      be provided, so usually you only need this if you provide
      `additional_sub_fixtures`. But you can override them, or set them to an
      empty string to remove them.
    init_checkpoint_experiments: Dictionary mapping checkpoint path to the
      experiment used to initialize it. For example, {"/path/to/my/checkpoint":
      my_pretrain_model.PretrainedModelExperiment}. This is useful for avoiding
      inlining *too* much code into a baseline, which might make it less
      readable.
    init_checkpoint_experiments_strict: Whether to check that the checkpoint
      experiments are provided for all init_from_checkpoint_rules entries. This
      only applies if `init_checkpoint_experiments` is provided.

  Returns:
    Generated code.
  """
  fixture_docstrings = fixture_docstrings or {}
  fixture_docstrings = {**_DEFAULT_FIXTURE_DOCSTRINGS, **fixture_docstrings}
  normalizer = config_normalization.ConfigNormalizer(
      remove_sharding_annotations=factor_out_sharding_annotations,
      unshare_sharding_config=unshare_sharding_config,
      remove_defaults=remove_defaults,
      remove_eval_datasets=remove_eval_datasets,
      remove_decoder_datasets=remove_decoder_datasets,
      convert_seqio_task_objects=True,
  )
  overall_config = make_parameterized_experiment.from_legacy(
      experiment_cls=codegen_tracer.make_subclass_mixin(experiment_cls),
      normalizer=normalizer,
      has_train_dataset=has_train_dataset,
      has_input_specs_provider=has_input_specs_provider,
  )
  # Check for custom objects, but without tracers.
  validation.check_no_custom_objects(
      make_parameterized_experiment.from_legacy(
          experiment_cls=experiment_cls,
          normalizer=normalizer,
          has_train_dataset=has_train_dataset,
          has_input_specs_provider=has_input_specs_provider,
      )
  )

  # If factor_out_sharding_annotations is set, save some A/B configs to diff the
  # sharding annotations. This is a bit disjointed because we want to compute it
  # before removing nested/deep field defaults.
  if factor_out_sharding_annotations:
    model_sharding_diff = codegen_sharding.sharding_diff(
        experiment_cls, unshare_sharding_config=unshare_sharding_config
    )
  else:
    model_sharding_diff = None

  add_sharding_call = None
  if model_sharding_diff is not None:
    add_sharding_call = codegen_sharding.AddShardingCall(
        "model_fixture", "shard_model_config"
    )
  codegen_config = code_generator_config(
      top_level_fixture_name="experiment_fixture",
      class_name=f"{experiment_cls.__name__}NewBaseline",
      max_expression_complexity=6,
      lowercase_highlevel_settings=lowercase_highlevel_settings,
      init_checkpoint_experiments_strict=init_checkpoint_experiments_strict,
      add_sharding_call=add_sharding_call,
      debug_print=False,
      fixture_docstrings=fixture_docstrings,
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


def codegen_experiment_diff(
    experiment_cls: Type[base_experiment.BaseExperiment],
    *,
    baseline: Type[Any],
    unshare_sharding_config: bool = True,
    remove_defaults: bool = True,
    lowercase_highlevel_settings: bool = None,
    has_train_dataset: bool = False,
    has_input_specs_provider: bool = False,
):
  """Generates an experiment subclass from a main class.

  Note, this is not the most recommended way of expressing experiments. If you
  can factor your configuration into granular sub-fixtures, and then just
  override one of those, it can be easier to read.

  But using the output of this should be fine / still within style
  recommendations.

  Args:
    experiment_cls: Legacy experiment class deriving from BaseExperiment.
    baseline: Type of the new baseline generated; usually output from
      codegen_baseline_from_legacy().
    unshare_sharding_config: Whether to unshare values in sharding
      configuration. Enter the same value as for the baseline configuratoin
    remove_defaults: Whether to remove default values. Enter the same value as
      for the baseline configuration.
    lowercase_highlevel_settings: Lowercase the high-level variable names. If
      not provided, this will be inferred from the baseline configuration.
    has_train_dataset: Whether the configuration has a training dataset. If not,
      then the resulting ParameterizedExperiment config's train_dataset field
      will not be populated.
    has_input_specs_provider: Whether the experiment has an input specs provider
      defined. Please set to False if you are using the (slower) one predefined
      by the dataset.

  Returns:
    Generated code.
  """
  baseline_field_names = [field.name for field in dataclasses.fields(baseline)]

  # Auto-detect lowercasing from the baseline.
  if lowercase_highlevel_settings is None:
    lowercase_highlevel_settings = all(
        name.lower() == name for name in baseline_field_names
    )

  experiment_instance = experiment_cls()
  highlevel_settings = {}
  for key in set(dir(experiment_instance)) | set(dir(experiment_cls)):
    value = getattr(experiment_instance, key)
    if key.startswith("_") and key not in baseline_field_names:
      continue  # Skip attributes like __module__.
    if isinstance(value, (bool, int, float, str, list, tuple)):
      key = key.lower() if lowercase_highlevel_settings else key
      highlevel_settings[key] = value

  highlevel_overrides = {}
  baseline_instance = baseline()
  for field in dataclasses.fields(baseline):
    base_value = getattr(baseline_instance, field.name)
    if base_value != highlevel_settings[field.name]:
      highlevel_overrides[field.name] = highlevel_settings[field.name]
    else:
      highlevel_settings.pop(field.name)
  # Note: highlevel_settings now contains only new settings.

  # Create a version of `baseline` with only highlevel settings overridden.
  experiment_highlevel_only = baseline(**copy.deepcopy(highlevel_overrides))

  diff_lhs = experiment_highlevel_only.experiment_fixture()
  if remove_defaults:
    diff_lhs = visualize.with_defaults_trimmed(
        diff_lhs, remove_deep_defaults=True
    )
  normalizer = config_normalization.ConfigNormalizer(
      remove_sharding_annotations=False,
      unshare_sharding_config=unshare_sharding_config,
      remove_defaults=remove_defaults,
      convert_seqio_task_objects=True,
  )
  diff_rhs = make_parameterized_experiment.from_legacy(
      experiment_cls=experiment_cls,
      normalizer=normalizer,
      has_train_dataset=has_train_dataset,
      has_input_specs_provider=has_input_specs_provider,
  )

  diff = diffing.build_diff(diff_lhs, diff_rhs)
  import_manager = code_ir._init_import_manager()  # pylint: disable=protected-access
  class_body = _class_attributes(highlevel_overrides)
  if class_body:
    class_body[0] = class_body[0].with_changes(
        leading_lines=[
            _comment_line("Overrides to existing high-level settings."),
        ]
    )
  new_attributes = _class_attributes(highlevel_settings)
  if new_attributes:
    new_attributes[0] = new_attributes[0].with_changes(
        leading_lines=[_comment_line("New high-level settings.")]
    )

  if diff.changes:
    diff_module = codegen_diff.fiddler_from_diff(
        diff,
        old=diff_lhs,
        func_name="experiment_fixture",
        param_name="config",
        import_manager=import_manager,
        variable_naming="short",
    )
    diff_fn = _extract_function_def(diff_module, "experiment_fixture")
    new_params = [
        cst.Param(cst.Name("self")),
        *diff_fn.params.params[1:],
    ]
    new_body = [
        cst.parse_statement("config = super().experiment_fixture()"),
        *diff_fn.body.body,
        cst.parse_statement("return config"),
    ]
    diff_fn = diff_fn.with_changes(
        body=diff_fn.body.with_changes(body=new_body),
        params=diff_fn.params.with_changes(params=new_params),
        leading_lines=[cst.EmptyLine(newline=cst.Newline())],
    )
    class_body.append(diff_fn)

  class_name = f"{experiment_cls.__name__}NewExperiment"
  class_def = _make_class_def(
      class_name,
      [cst.Arg(cst.parse_expression(import_manager.add(baseline)))],
      class_body,
      docstring=f"Experiment definition for {experiment_cls.__name__}.",
  )

  module_body = [class_def]

  # Add imports last, since the import manager was updated with the baseline.
  module_body[0:0] = import_manager.sorted_import_lines()

  # Add the module docstring.
  module_body.insert(
      0,
      _make_docstring(
          _DEFAULT_MODULE_DOCSTRING.format(
              base_experiment_name=experiment_cls.__name__
          ),
          indent=0,
      ),
  )

  return cst.Module(body=module_body, default_indent="  ").code
