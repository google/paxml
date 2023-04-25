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
"""Configs for benchmarking decoder-only models on C4 dataset."""

from absl import logging
import jax
from jax import numpy as jnp
from paxml import base_experiment
from paxml import experiment_registry
from paxml import tasks_lib
from paxml.tasks.lm import model_params
from paxml.tasks.lm.params import lm_cloud
from paxml.tasks.lm.params import c4
from praxis import layers

@experiment_registry.register
class C4Spmd1BAdam4ReplicasMultislice(c4.C4Spmd1BAdam4Replicas):
  """GPT-3 config with 1B params. Model Parameters: 
  Global batch size = 1 * 4 * 1 * 32 = 128"""
 
  CHECKPOINT_POLICY = layers.AutodiffCheckpointType.SAVE_NOTHING
  DCN_MESH_SHAPE = [2, 1, 1]


@experiment_registry.register
class C4Spmd1BAdam4ReplicasMultisliceLimitSteps(C4Spmd1BAdam4ReplicasMultislice):
  def task(self) -> tasks_lib.SingleTask.HParams:
    task_p = super().task()
    task_p.train.num_train_steps = 300
    return task_p


@experiment_registry.register
class C4Spmd2BAdam4ReplicasMultislice(c4.C4Spmd2BAdam4Replicas):
  """GPT-3 config with 2B params. Model Parameters: 
  Global batch size = 1 * 4 * 1 * 32 = 128"""
 
  CHECKPOINT_POLICY = layers.AutodiffCheckpointType.SAVE_NOTHING
  DCN_MESH_SHAPE = [2, 1, 1]


@experiment_registry.register
class C4Spmd16BAdam32ReplicasMultislice(c4.C4Spmd16BAdam32Replicas):
  """GPT-3 config with 16B params.
  Model Parameters: Global batch size = 1 * 2 * 16 * 32 = 1024.
  """

  CHECKPOINT_POLICY = layers.AutodiffCheckpointType.SAVE_NOTHING
  DCN_MESH_SHAPE = [2, 1, 1]


@experiment_registry.register
class C4Spmd32BAdam64ReplicasMultislice(c4.C4Spmd32BAdam64Replicas):
  """GPT-3 config with 32B params.
  Model Parameters: Global batch size = 1 * 16 * 4 * 16 = 1024.
  """

  CHECKPOINT_POLICY = layers.AutodiffCheckpointType.SAVE_NOTHING
  DCN_MESH_SHAPE = [2, 1, 1]
