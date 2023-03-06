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

"""Test model configuration using synthetic data."""

from paxml import base_experiment
from paxml import experiment_registry
from praxis import layers
from praxis import pax_fiddle


@experiment_registry.register
class SyntheticClassifier(base_experiment.BaseExperiment):
  # TODO(shafey): Implement a real test model.

  def datasets(self):
    return []

  def task(self):
    act_p = pax_fiddle.Config(layers.Identity)
    return act_p


@experiment_registry.register
class SharedNameExperiment(SyntheticClassifier):
  pass
