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

"""Program executor that drives the training/evaluation loops."""

import abc
from typing import Any, Sequence

from etils import epath
from paxml import decode_programs as decode_program_lib
from paxml import partitioning
from paxml import programs
from paxml import tasks_lib
from paxml import trainer_lib
from praxis import base_input
from praxis import pax_fiddle


class BaseExecutor(metaclass=abc.ABCMeta):

  @abc.abstractmethod
  def setup(
      self,
      jax_task: tasks_lib.SingleTask,
      job_log_dir: epath.Path,
      checkpointer: Any,
      partitioner: partitioning.Partitioner,
      input_specs_provider: base_input.BaseInputSpecsProvider,
      # TODO(laigd): encapsulate train_input_p in train_program.
      train_input_p: pax_fiddle.Config[base_input.BaseInput],
      train_program: programs.BaseTrainProgram,
      eval_programs: Sequence[programs.BaseEvalProgram],
      decode_programs: Sequence[decode_program_lib.SingleTaskDecodeProgram],
      # TODO(laigd): this shouldn't be part of the executor API, consider adding
      # a dedicated executor for auto-tuning and get rid of this instead.
      early_stopping_fn: trainer_lib.EarlyStoppingFn | None,
      exit_after_ondemand_checkpoint: bool = False,
  ) -> None:
    """Sets up the programs and the executor."""

  @abc.abstractmethod
  def start(self) -> None:
    """Start executing the programs."""
