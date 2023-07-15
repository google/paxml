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

"""A few small code IR nodes for Pax."""

import dataclasses
from typing import Any, Optional

from fiddle._src.codegen.auto_config import code_ir
import libcst as cst


@dataclasses.dataclass
class PaxCodegenTask(code_ir.CodegenTask):
  """CodegenTask that tracks a few extra bits of state.

  Attributes:
    highlevel_accesses: Accesses to high-level settings. These become fields on
      the generated class.
    sharding_diff_module: CST module containing a fiddler that will re-add
      sharding to a model. Factoring our fixtures into ones that generate an
      unsharded module, and a function that re-adds the sharding can be more
      readable. (You can disable this in `codegen.py`.)
  """

  highlevel_accesses: dict[str, Any] = dataclasses.field(default_factory=dict)
  sharding_diff_module: Optional[cst.Module] = None
