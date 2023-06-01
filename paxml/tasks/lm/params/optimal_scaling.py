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

"""Decoder-only language model configurations with Chinchilla-like scaling."""

from paxml import experiment_registry
from paxml import tasks_lib
from paxml.tasks.lm.params.lm_cloud import LmCloudSpmd
from praxis import layers
from praxis import pax_fiddle


class OptimalScaling(LmCloudSpmd):
  """Decoder-only language model configurations with Chinchilla-like scaling."""

  CHECKPOINT_POLICY = layers.AutodiffCheckpointType.SAVE_DOT_WITH_NO_BATCH_DIM

  # subclasses override these
  PERCORE_BATCH_SIZE = None
  NUM_LAYERS = None

  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    # pylint: disable=invalid-name
    assert self.NUM_LAYERS
    self.MODEL_DIMS = self.NUM_LAYERS * 128
    self.HIDDEN_DIMS = self.MODEL_DIMS * 4
    # pylint: enable=invalid-name
    return super().task()


@experiment_registry.register
class OptimalScaling2x2x1(OptimalScaling):
  NUM_LAYERS = 28
  PERCORE_BATCH_SIZE = 16
  ICI_MESH_SHAPE = [1, 4, 1]


@experiment_registry.register
class OptimalScaling2x2x2(OptimalScaling):
  NUM_LAYERS = 32
  PERCORE_BATCH_SIZE = 8
  ICI_MESH_SHAPE = [1, 8, 1]


@experiment_registry.register
class OptimalScaling2x2x4(OptimalScaling):
  NUM_LAYERS = 36
  PERCORE_BATCH_SIZE = 8
  ICI_MESH_SHAPE = [1, 16, 1]


@experiment_registry.register
class OptimalScaling2x4x4(OptimalScaling):
  NUM_LAYERS = 40
  PERCORE_BATCH_SIZE = 8
  ICI_MESH_SHAPE = [1, 32, 1]


@experiment_registry.register
class OptimalScaling4x4x4(OptimalScaling):
  NUM_LAYERS = 45
  PERCORE_BATCH_SIZE = 8
  ICI_MESH_SHAPE = [1, 64, 1]


@experiment_registry.register
class OptimalScaling4x4x8(OptimalScaling):
  NUM_LAYERS = 50
  PERCORE_BATCH_SIZE = 4
  ICI_MESH_SHAPE = [1, 128, 1]


@experiment_registry.register
class OptimalScaling4x8x8(OptimalScaling):
  NUM_LAYERS = 56
  PERCORE_BATCH_SIZE = 2
  ICI_MESH_SHAPE = [1, 64, 4]


@experiment_registry.register
class OptimalScaling4x8x16(OptimalScaling):
  NUM_LAYERS = 64
  PERCORE_BATCH_SIZE = 2
  ICI_MESH_SHAPE = [1, 128, 4]


@experiment_registry.register
class OptimalScaling4x16x16(OptimalScaling):
  NUM_LAYERS = 64
  PERCORE_BATCH_SIZE = 1
  ICI_MESH_SHAPE = [1, 256, 4]


@experiment_registry.register
class OptimalScaling4x16x32(OptimalScaling):
  NUM_LAYERS = 64
  PERCORE_BATCH_SIZE = 1
  ICI_MESH_SHAPE = [1, 512, 4]


@experiment_registry.register
class OptimalScaling4x24x32(OptimalScaling):
  NUM_LAYERS = 64
  PERCORE_BATCH_SIZE = 1
  ICI_MESH_SHAPE = [1, 768, 4]
