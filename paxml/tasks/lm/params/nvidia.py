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

"""Language Model configurations on the T5/C4 dataset. Contributed by NVIDIA."""

import math
from jax import numpy as jnp
from paxml import experiment_registry
from paxml import tasks_lib
from paxml.tasks.lm.params import c4
from paxml.tasks.lm.params import lm_cloud
from praxis import base_layer
from praxis import layers
from praxis import optimizers
from praxis import pax_fiddle
from praxis import schedules
from praxis.layers import gpu_fast_attention


WeightInit = base_layer.WeightInit


@experiment_registry.register
class NVIDIA1_3B(c4.TransformerLmSpmdAdam, lm_cloud.SyntheticDataset):
  """Pipelined Transformer using Adam optimizer."""

  USE_FLASH_ATTENTION = False
  USE_TRITON_LAYER_NORM = False

  USE_REPEATED_LAYER = True

  DCN_MESH_SHAPE = [1, 1, 1]
  ICI_MESH_SHAPE = [16, 1, 1]

  PERCORE_BATCH_SIZE = 4

  NUM_LAYERS = 24
  NUM_HEADS = 32
  DIMS_PER_HEAD = 64
  MODEL_DIMS = 2048
  HIDDEN_DIMS = 8192

  MAX_SEQ_LEN = 2048
  VOCAB_SIZE = 51200
  ACTIVATION_CLS = layers.GELU
  USE_GATED_ACTIVATION = False
  PACKED_INPUT = True

  FPROP_DTYPE = jnp.bfloat16

  TRAINABLE_POSITION_EMB = True
  TRAINABLE_PE_MAX_SEQ_LEN = MAX_SEQ_LEN

  INIT_STD = 0.023
  SOFTMAX_INIT_STD = 0.023

  # optimizer related
  LEARNING_RATE = 6e-4
  ADAM_BETA1 = 0.9
  ADAM_BETA2 = 0.95
  ADAM_EPSILON = 1e-8
  ADAM_EPSILON_ROOT = 0.0
  CLIP_GRADIENT_NORM_TO_VALUE = 1.0
  CLIP_THRESHOLD = 1.0

  # Learning rate schedule
  LR_SCHEDULE = 'linear_rampup_cosine_decay'
  LR_COS_WARMUP = 0
  LR_COS_DECAY_START = 1
  LR_COS_DECAY_END = 500000
  LR_COS_MIN_RATIO = 0.1
  LR_COS_MAX = 1.0
  USE_ADAFACTOR = False

  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    """Returns the task parameters."""
    task_p = super().task()
    # Disable all summaries.
    task_p.summary_verbosity = 0
    # (Practically) disable summary writes.
    task_p.train.save_interval_steps = 100000

    model_p = task_p.model
    model_p.params_init = WeightInit.Gaussian(self.INIT_STD)

    if self.USE_FLASH_ATTENTION:
      layer_p = (
          model_p.lm_tpl.stacked_transformer_tpl.pipeline_stage.transformer_layer_params_tpl
      )
      # Use Triton flash attention.
      assert layer_p.tr_atten_tpl.cls == layers.DotProductAttention
      fused_tr_atten_tpl = pax_fiddle.Config(
          gpu_fast_attention.GpuTritonFusedDotProductAttention,
      )
      fused_tr_atten_tpl.copy_fields_from(layer_p.tr_atten_tpl)
      layer_p.tr_atten_tpl = fused_tr_atten_tpl

    # Use Triton Layer Norm.
    if self.USE_TRITON_LAYER_NORM:
      assert layer_p.ln_tpl.cls == layers.LayerNorm
      fused_ln_tpl = pax_fiddle.Config(
          gpu_fast_attention.GpuTritonFusedLayerNorm,
      )
      fused_ln_tpl.copy_fields_from(layer_p.ln_tpl)
      layer_p.ln_tpl = fused_ln_tpl

    scale = self.SOFTMAX_INIT_STD
    if not scale:
      scale = 1.0 / math.sqrt(self.MODEL_DIMS)
    softmax_init = WeightInit.Gaussian(scale)

    lp = task_p.train.learner
    lp.loss_name = 'total_loss'
    if self.USE_ADAFACTOR:
      lp.optimizer = pax_fiddle.Config(
          optimizers.ShardedAdafactor,
          decay_method='adam',
          beta1=self.ADAM_BETA1,
          decay_adam=0.99,
          weight_decay=self.WEIGHT_DECAY,
          clip_gradient_norm_to_value=self.CLIP_GRADIENT_NORM_TO_VALUE,
      )
    else:
      lp.optimizer = pax_fiddle.Config(
          optimizers.Adam,
          beta1=self.ADAM_BETA1,
          beta2=self.ADAM_BETA2,
          weight_decay=self.WEIGHT_DECAY,
          epsilon=self.ADAM_EPSILON,
          epsilon_root=self.ADAM_EPSILON_ROOT,
          clip_gradient_norm_to_value=self.CLIP_GRADIENT_NORM_TO_VALUE,
          clip_threshold=self.CLIP_THRESHOLD,
      )
    lp.optimizer.learning_rate = self.LEARNING_RATE

    lp.optimizer.lr_schedule = pax_fiddle.Config(
        schedules.LinearRampupCosineDecay,
        warmup_steps=self.LR_COS_WARMUP,
        decay_start=self.LR_COS_DECAY_START,
        decay_end=self.LR_COS_DECAY_END,
        min_ratio=self.LR_COS_MIN_RATIO,
        max=self.LR_COS_MAX,
    )
    return task_p


@experiment_registry.register
class NVIDIA5B(c4.TransformerLmSpmdPipelineAdam, lm_cloud.SyntheticDataset):
  """Pipelined Transformer using Adam optimizer."""

  USE_FLASH_ATTENTION = False
  USE_TRITON_LAYER_NORM = False

  USE_REPEATED_LAYER = False
  DCN_MESH_SHAPE = [1, 1, 1, 1]
  ICI_MESH_SHAPE = [4, 2, 1, 2]
  NUM_STAGES = 4

  MICROBATCH_SIZE = 2
  PERCORE_BATCH_SIZE = 1

  NUM_LAYERS = 24
  NUM_HEADS = 32
  DIMS_PER_HEAD = 128
  MODEL_DIMS = 4096
  HIDDEN_DIMS = 16384

  MAX_SEQ_LEN = 2048
  VOCAB_SIZE = 51200
  ACTIVATION_CLS = layers.GELU
  USE_GATED_ACTIVATION = False
  PACKED_INPUT = True

  FPROP_DTYPE = jnp.bfloat16

  TRAINABLE_POSITION_EMB = True
  TRAINABLE_PE_MAX_SEQ_LEN = MAX_SEQ_LEN

  INIT_STD = 0.023
  SOFTMAX_INIT_STD = 0.023

  # optimizer related
  LEARNING_RATE = 6e-4
  ADAM_BETA1 = 0.9
  ADAM_BETA2 = 0.95
  ADAM_EPSILON = 1e-8
  ADAM_EPSILON_ROOT = 0.0
  CLIP_GRADIENT_NORM_TO_VALUE = 1.0
  CLIP_THRESHOLD = 1.0

  # Learning rate schedule
  LR_SCHEDULE = 'linear_rampup_cosine_decay'
  LR_COS_WARMUP = 0
  LR_COS_DECAY_START = 1
  LR_COS_DECAY_END = 500000
  LR_COS_MIN_RATIO = 0.1
  LR_COS_MAX = 1.0
  USE_ADAFACTOR = False

  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    """Returns the task parameters."""
    task_p = super().task()
    task_p.train.save_interval_steps = 100000

    model_p = task_p.model
    model_p.params_init = WeightInit.Gaussian(self.INIT_STD)

    if self.USE_FLASH_ATTENTION:
      layer_p = (
          model_p.lm_tpl.stacked_transformer_tpl.pipeline_stage.transformer_layer_params_tpl
      )
      # Use Triton flash attention.
      assert layer_p.tr_atten_tpl.cls == layers.DotProductAttention
      fused_tr_atten_tpl = pax_fiddle.Config(
          gpu_fast_attention.GpuTritonFusedDotProductAttention,
      )
      fused_tr_atten_tpl.copy_fields_from(layer_p.tr_atten_tpl)
      layer_p.tr_atten_tpl = fused_tr_atten_tpl

    # Use Triton Layer Norm.
    if self.USE_TRITON_LAYER_NORM:
      assert layer_p.ln_tpl.cls == layers.LayerNorm
      fused_ln_tpl = pax_fiddle.Config(
          gpu_fast_attention.GpuTritonFusedLayerNorm,
      )
      fused_ln_tpl.copy_fields_from(layer_p.ln_tpl)
      layer_p.ln_tpl = fused_ln_tpl

    scale = self.SOFTMAX_INIT_STD
    if not scale:
      scale = 1.0 / math.sqrt(self.MODEL_DIMS)
    softmax_init = WeightInit.Gaussian(scale)

    lp = task_p.train.learner
    lp.loss_name = 'total_loss'

    if self.USE_ADAFACTOR:
      lp.optimizer = pax_fiddle.Config(
          optimizers.ShardedAdafactor,
          decay_method='adam',
          beta1=self.ADAM_BETA1,
          decay_adam=0.99,
          weight_decay=self.WEIGHT_DECAY,
          clip_gradient_norm_to_value=self.CLIP_GRADIENT_NORM_TO_VALUE,
      )
    else:
      lp.optimizer = pax_fiddle.Config(
          optimizers.Adam,
          beta1=self.ADAM_BETA1,
          beta2=self.ADAM_BETA2,
          weight_decay=self.WEIGHT_DECAY,
          epsilon=self.ADAM_EPSILON,
          epsilon_root=self.ADAM_EPSILON_ROOT,
          clip_gradient_norm_to_value=self.CLIP_GRADIENT_NORM_TO_VALUE,
          clip_threshold=self.CLIP_THRESHOLD,
      )
    lp.optimizer.learning_rate = self.LEARNING_RATE

    lp.optimizer.lr_schedule = pax_fiddle.Config(
        schedules.LinearRampupCosineDecay,
        warmup_steps=self.LR_COS_WARMUP,
        decay_start=self.LR_COS_DECAY_START,
        decay_end=self.LR_COS_DECAY_END,
        min_ratio=self.LR_COS_MIN_RATIO,
        max=self.LR_COS_MAX,
    )
    return task_p


@experiment_registry.register
class NVIDIA8_3B(NVIDIA1_3B):
  USE_FLASH_ATTENTION = False
  USE_TRITON_LAYER_NORM = False

  DCN_MESH_SHAPE = [1, 1, 1]
  ICI_MESH_SHAPE = [4, 1, 4]
  PERCORE_BATCH_SIZE = 4

  NUM_LAYERS = 40
  NUM_HEADS = 64
  DIMS_PER_HEAD = 64
  MODEL_DIMS = 4096
  HIDDEN_DIMS = 4 * 4096


@experiment_registry.register
class NVIDIA10B(NVIDIA1_3B):
  USE_FLASH_ATTENTION = False
  USE_TRITON_LAYER_NORM = False

  DCN_MESH_SHAPE = [1, 1, 1]
  ICI_MESH_SHAPE = [2, 1, 8]
  PERCORE_BATCH_SIZE = 0.25

  NUM_LAYERS = 32
  NUM_HEADS = 40
  DIMS_PER_HEAD = 128
  MODEL_DIMS = 5120
  HIDDEN_DIMS = 4 * 5120


@experiment_registry.register
class NVIDIA40BProxy(NVIDIA5B):
  USE_FLASH_ATTENTION = False
  USE_TRITON_LAYER_NORM = False

  DCN_MESH_SHAPE = [1, 1, 1, 1]
  ICI_MESH_SHAPE = [2, 2, 1, 4]
  NUM_STAGES = 2

  MICROBATCH_SIZE = 4
  PERCORE_BATCH_SIZE = 16

  NUM_LAYERS = 12
  NUM_HEADS = 64
  DIMS_PER_HEAD = 128
  MODEL_DIMS = 8192
  HIDDEN_DIMS = 4 * 8192


@experiment_registry.register
class NVIDIA70BProxy(NVIDIA5B):
  USE_FLASH_ATTENTION = False
  USE_TRITON_LAYER_NORM = False

  DCN_MESH_SHAPE = [1, 1, 1, 1]
  ICI_MESH_SHAPE = [2, 2, 1, 4]
  NUM_STAGES = 2

  MAX_SEQ_LEN = 2048

  MICROBATCH_SIZE = 4
  PERCORE_BATCH_SIZE = 8

  NUM_LAYERS = 20
  NUM_HEADS = 64
  DIMS_PER_HEAD = 128
  MODEL_DIMS = 8192
  HIDDEN_DIMS = 4 * 8192


@experiment_registry.register
class NVIDIA116BProxy(NVIDIA5B):
  USE_FLASH_ATTENTION = False
  USE_TRITON_LAYER_NORM = False

  DCN_MESH_SHAPE = [1, 1, 1, 1]
  ICI_MESH_SHAPE = [4, 1, 1, 4]
  NUM_STAGES = 4

  MAX_SEQ_LEN = 2048

  MICROBATCH_SIZE = 2
  PERCORE_BATCH_SIZE = 6

  NUM_LAYERS = 16
  NUM_HEADS = 96
  DIMS_PER_HEAD = 128
  MODEL_DIMS = 12288
  HIDDEN_DIMS = 4 * 12288


@experiment_registry.register
class NVIDIA175BProxy(NVIDIA5B):
  """175B config that works with 4x16 A100-40G."""

  USE_FLASH_ATTENTION = False
  USE_TRITON_LAYER_NORM = False

  DCN_MESH_SHAPE = [1, 1, 1, 1]
  ICI_MESH_SHAPE = [4, 1, 1, 4]
  NUM_STAGES = 4
  MICROBATCH_SIZE = 1
  PERCORE_BATCH_SIZE = 6

  NUM_LAYERS = 24
  NUM_HEADS = 96
  DIMS_PER_HEAD = 128
  MODEL_DIMS = 12288
  HIDDEN_DIMS = 4 * 12288


@experiment_registry.register
class NVIDIA175B(NVIDIA175BProxy):
  """175B config that works with 6x16 A100-40G."""

  DCN_MESH_SHAPE = [6, 1, 1, 1]
  ICI_MESH_SHAPE = [1, 1, 1, 16]

  NUM_LAYERS = 96
  NUM_STAGES = 6
  MICROBATCH_SIZE = 1
  PERCORE_BATCH_SIZE = 0.0625

  ENABLE_BFLOAT16 = True
  USE_ADAFACTOR = True


@experiment_registry.register
class TestSmallConfig(NVIDIA5B):
  """Test config that works with 16 A100-40G."""

  DCN_MESH_SHAPE = [1, 1, 1, 1]
  ICI_MESH_SHAPE = [2, 2, 1, 4]
  NUM_STAGES = 2

  MICROBATCH_SIZE = 2
  PERCORE_BATCH_SIZE = 2

  NUM_LAYERS = 16
  NUM_HEADS = 32
  DIMS_PER_HEAD = 128
  MODEL_DIMS = 4096
  HIDDEN_DIMS = 4 * 4096


@experiment_registry.register
class Llama33BProxy(NVIDIA1_3B):
  """Llama 33B config that works with 1x16 A100-40G."""

  USE_FLASH_ATTENTION = False
  USE_TRITON_LAYER_NORM = False

  ICI_MESH_SHAPE = [1, 16, 1]
  PERCORE_BATCH_SIZE = 1

  NUM_LAYERS = 60
  VOCAB_SIZE = 32000
  DIMS_PER_HEAD = 128
  NUM_HEADS = 52
  MODEL_DIMS = 6656
  HIDDEN_DIMS = 17920
  ENABLE_BFLOAT16 = True
  USE_ADAFACTOR = True


@experiment_registry.register
class NVIDIA175B_FSDP(NVIDIA1_3B):
  """175B with fully-sharded data-parallel that works with 8x16 A100-40G."""

  USE_FLASH_ATTENTION = False
  USE_TRITON_LAYER_NORM = False

  ICI_MESH_SHAPE = [1, 16, 1]
  DCN_MESH_SHAPE = [1, 8, 1]
  PERCORE_BATCH_SIZE = 1

  USE_ADAFACTOR = True
  NUM_LAYERS = 96
  NUM_HEADS = 96
  DIMS_PER_HEAD = 128
  MODEL_DIMS = 12288
  HIDDEN_DIMS = 4 * 12288
