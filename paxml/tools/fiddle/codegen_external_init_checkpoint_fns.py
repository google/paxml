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

"""Codegen pass that will replace init_from_checkpoint_rules with concise refs.

While we generally want to "flatten" experiment definitions, since checkpoint
loading rules tend to reference other experiments (at least their tasks and
input specs), including all of that would likely be too verbose.
"""

import dataclasses
from typing import Any, Dict, Optional, Type

import fiddle as fdl
from fiddle import daglish
from fiddle._src.codegen.auto_config import code_ir
from fiddle._src.codegen.auto_config import import_manager_wrapper
from fiddle.codegen.auto_config import experimental_top_level_api
from paxml import base_experiment
from paxml import tasks_lib
from paxml.tools.fiddle import codegen_pax_code_ir


@dataclasses.dataclass(frozen=True)
class InitCheckpointRulesFromOtherTask(experimental_top_level_api.CodegenPass):
  """Makes init_checkpoint_rules refer to another task.

  In current Pax design, init_from_checkpoint_rules is a dict that maps
  checkpoint paths to load rules, and those load rules refer to the task and
  input specs of another model. This can cause an explosion in config size, and
  we think generally inlining these tasks is not worth it.
  """

  PASS_INPUT_KWARGS = ["init_checkpoint_experiments"]
  strict: bool = True

  def __call__(
      self,
      task: Any,
      init_checkpoint_experiments: Optional[
          Dict[str, Optional[Type[base_experiment.BaseExperiment]]]
      ],
  ) -> Any:
    assert isinstance(task, codegen_pax_code_ir.PaxCodegenTask)
    if init_checkpoint_experiments is None:
      return task

    unseen = set(init_checkpoint_experiments)

    def traverse(value, state: daglish.State):
      if not isinstance(value, fdl.Buildable):
        return state.map_children(value)

      fn_or_cls = fdl.get_callable(value)
      if not issubclass(fn_or_cls, tasks_lib.CheckpointLoadingRules):
        return state.map_children(value)

      if not state.current_path:
        raise ValueError(
            "Didn't expect to encounter top-level CheckpointLoadingRules."
        )
      if not isinstance(state.current_path[-1], daglish.Key):
        raise ValueError(
            f"Unexpected path to checkpoint loading rules: {state.current_path}"
        )
      key = state.current_path[-1].key

      if key not in init_checkpoint_experiments:
        if self.strict:
          raise ValueError(
              f"No task for checkpoint {key}. If you provide"
              " init_checkpoint_experiments, please provide an entry for this"
              " one, or set init_checkpoint_experiments_strict=False."
          )
        else:
          return value

      # Replace task and model with refs.
      experiment_symbol = import_manager_wrapper.add(
          init_checkpoint_experiments[key], task.import_manager
      )
      experiment_expr = code_ir.SymbolOrFixtureCall(
          symbol_expression=experiment_symbol,
          positional_arg_expressions=[],
          arg_expressions={},
      )
      task_call = code_ir.SymbolOrFixtureCall(
          symbol_expression=code_ir.AttributeExpression(
              experiment_expr, "task"
          ),
          positional_arg_expressions=[],
          arg_expressions={},
      )
      input_specs_call = code_ir.SymbolOrFixtureCall(
          symbol_expression=code_ir.AttributeExpression(
              experiment_expr, "get_input_specs_provider_params"
          ),
          positional_arg_expressions=[],
          arg_expressions={},
      )
      unseen.remove(key)
      return fdl.copy_with(
          value,
          task_p=task_call,
          input_specs_provider_p=input_specs_call,
      )

    for fn in task.top_level_call.all_fixture_functions():
      fn.replace_with(daglish.MemoizedTraversal.run(traverse, fn))
    if unseen and self.strict:
      raise ValueError(
          "Didn't encounter these load_from_checkpoint rules in the config:"
          f" {unseen}. Perhaps the keys in the experiment have diverged from"
          " the init_checkpoint_experiments dict you provided? It's also"
          " possible that instead of providing"
          " pax_fiddle.Config(CheckpointLoadingRules, ...) in your config, you"
          " accidentally directly inserted a CheckpointLoadingRules()"
          " instance. Please refactor this (it should be a simple"
          " substitution, and will help make other Fiddle functions like"
          " visualization and serialization work)."
      )
    return task
