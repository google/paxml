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
class C4Spmd22BAdamMaxText(c4.C4SpmdAdam):
  """GPT-3 config with 22B params. Model Parameters: 
  Global batch size = 1 * 4 * 1 * 32 = 128"""
 
  NUM_LAYERS = 16
  MODEL_DIMS = 2048
  HIDDEN_DIMS = MODEL_DIMS * 4
  NUM_HEADS = 8
  DIMS_PER_HEAD = 256
  PERCORE_BATCH_SIZE = 32
  MAX_SEQ_LEN = 1024
  VOCAB_SIZE = 327680.

  FPROP_DTYPE = jnp.bfloat16
  USE_REPEATED_LAYER = True
  SUMMARY_INTERVAL_STEPS = 10


@experiment_registry.register
class C4Spmd22BAdam1xv4_128(c4.C4SpmdAdam):
  """GPT-3 config with 22B params. Model Parameters: 
  Global batch size = 1 * 64 * 1 * 16 = 1024"""
  ICI_MESH_SHAPE = [1, 64, 1]
  PERCORE_BATCH_SIZE = 16


@experiment_registry.register
class C4Spmd22BAdam2xv4_128(C4Spmd22BAdam1xv4_128):
  """GPT-3 config with 22B params. Model Parameters: 
  Global batch size = 2* 1 * 64 * 1 * 16 = 2048"""
  DCN_MESH_SHAPE = [2, 1, 1]


@experiment_registry.register
class C4Spmd22BAdam4xv4_128(C4Spmd22BAdam1xv4_128):
  """GPT-3 config with 22B params. Model Parameters: 
  Global batch size = 4 * 1 * 64 * 1 * 16 = 4096"""
  DCN_MESH_SHAPE = [4, 1, 1]


@experiment_registry.register
class C4Spmd22BAdam1xv4_384(c4.C4SpmdAdam):
  """GPT-3 config with 22B params. Model Parameters: 
  Global batch size = 1 * 64 * 1 * 16 = 1024"""
  ICI_MESH_SHAPE = [1, 192, 1]
  NUM_LAYERS = 8
  MODEL_DIMS = 3072
  HIDDEN_DIMS = MODEL_DIMS * 4
  PERCORE_BATCH_SIZE = 10


@experiment_registry.register
class C4Spmd22BAdam2xv4_384(C4Spmd22BAdam1xv4_384):
  """GPT-3 config with 22B params. Model Parameters: 
  Global batch size = 2* 1 * 64 * 1 * 16 = 2048"""
  DCN_MESH_SHAPE = [2, 1, 1]
