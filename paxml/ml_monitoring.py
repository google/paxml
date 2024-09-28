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

"""ML Monitoring for PAX."""

import contextlib
import enum


class MlEvent(enum.Enum):
  """ML events to be recorded."""

  INITIALIZE_BACKEND = enum.auto()
  INITIALIZE_SETUP = enum.auto()
  MAIN_LOOP = enum.auto()
  TRAIN_STEP = enum.auto()
  EVAL_STEP = enum.auto()
  DECODE_STEP = enum.auto()


class EventBoundary(enum.Enum):
  """Event boundary to be recorded."""

  START = enum.auto()
  END = enum.auto()


def record_step_number(step_number: int):
  """Records the step number."""
  pass


def record_event_boundary(event: MlEvent, boundary: EventBoundary, **kwargs):
  """Records the event boundary."""
  pass


@contextlib.contextmanager
def ml_event_logger(event: MlEvent, **kwargs):
  yield
