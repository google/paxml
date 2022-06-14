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

"""Base class for all tasks.

A model solely consists of the network, while a task combines one or several
models with one or several learners/optimizers.
"""

from __future__ import annotations

import abc

from praxis import base_hyperparams


class BaseTask(base_hyperparams.BaseParameterizable, metaclass=abc.ABCMeta):
  """Abstract base class for all tasks."""

  def __init__(self, hparams: BaseTask.HParams) -> None:
    """Constructor.

    Args:
      hparams: The dataclasses-like instance used to configure this class
        instance.
    """
    assert hparams.name, ('Task params for %s must have a "name"' %
                          self.__class__.__name__)
    super().__init__(hparams)
