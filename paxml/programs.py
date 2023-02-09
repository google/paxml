# coding=utf-8
# Copyright 2022 Google LLC.
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

"""The basic program concept that encapsulates a per-step runnable."""
import dataclasses
from typing import Any, Dict, Optional, Protocol

import jax
from paxml import train_states
from praxis import pytypes


TrainState = train_states.TrainState


@dataclasses.dataclass
class ProgramOutput:
  # The train_state that's potentially modified by the program.
  # For example, a train program is expected to update the state to reflect
  # optimizer updates, while a eval program is expected to keep the state as is.
  state: TrainState
  # Auxilary dictionary that contains any information that program intends to
  # feedback to outer loop.
  aux: Dict[str, Any]


class Program(Protocol):
  """The basic interface for a program."""

  def run_step(
      self,
      train_state: TrainState,
      prng_key: Optional[jax.random.KeyArray],
      inputs: pytypes.NestedJTensor,
  ) -> ProgramOutput:
    """Runs a single step on the program."""
    ...
