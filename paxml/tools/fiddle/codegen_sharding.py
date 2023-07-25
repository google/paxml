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

"""Sharding-specific features of the codegen."""

import dataclasses
from typing import Any, Optional, Type

from fiddle import diffing
from fiddle.codegen import codegen_diff
from fiddle.codegen.auto_config import experimental_top_level_api
import libcst as cst
from libcst import matchers
from paxml import base_experiment
from paxml.tools.fiddle import codegen_pax_code_ir
from paxml.tools.fiddle import remove_sharding
from paxml.tools.fiddle import unshare_sharding
from praxis import pax_fiddle


@dataclasses.dataclass(frozen=True)
class ModelShardingDiff:
  diff: diffing.Diff
  old: pax_fiddle.Config[Any]


def sharding_diff(
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
class AddShardingCall:
  """Helper which adds a call to shard_model_config() in model_fixture()."""

  model_fixture_name: str  # usually model_fixture()
  add_sharding_function: str  # usually shard_model_config()

  def __call__(self, fn_code: cst.FunctionDef) -> cst.FunctionDef:
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
    shard_model_name = cst.Name(self.add_sharding_function)
    last_lines = [
        cst.Assign([cst.AssignTarget(var_name)], body[-1].body[0].value),
        cst.Expr(cst.Call(shard_model_name, args=[cst.Arg(var_name)])),
        cst.Return(var_name),
    ]
    body = body[:-1] + [cst.SimpleStatementLine([line]) for line in last_lines]
    return fn_code.with_changes(body=fn_code.body.with_changes(body=body))
