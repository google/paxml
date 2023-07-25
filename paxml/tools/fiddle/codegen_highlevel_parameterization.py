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

"""Codegen pass to trace high-level settings and make those useable."""

import dataclasses
from typing import Any

from fiddle import daglish
from fiddle._src.codegen.auto_config import code_ir
from fiddle.codegen.auto_config import experimental_top_level_api
from paxml.tools.fiddle import codegen_pax_code_ir


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
