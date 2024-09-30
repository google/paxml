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

"""Configurations for GPU models."""

from typing import Type, cast
import fiddle as fdl
import jax.numpy as jnp
import numpy as np
from paxml import experiment_registry
from paxml import tasks_lib
from paxml.contrib.gpu.scripts_gpu.checkpoint_utils import CheckpointRestoreMixin
from paxml.contrib.gpu.scripts_gpu.llama_utils import BaseLLaMA
from paxml.contrib.gpu.scripts_gpu.lora_utils import LoRAMixin
from paxml.contrib.gpu.scripts_gpu.tasks import BoolQDataset
from paxml.contrib.gpu.scripts_gpu.tasks import LambadaDataset
from paxml.contrib.gpu.scripts_gpu.tasks import PileUnsupervisedDataset
from paxml.tasks.lm.model_params import maybe_setup_moe_params
from paxml.contrib.gpu.scripts_gpu.te_helper import TransformerEngineHelper
from paxml.tasks.lm.params.c4 import TransformerLmSpmdAdam
from paxml.tasks.lm.params.lm_cloud import SyntheticDataset
from praxis import base_layer
from praxis import layers
from praxis import optimizers
from praxis import pax_fiddle
from praxis import schedules
from praxis.contrib.gpu.scripts_gpu.models import CustomMetricsLM
from praxis.layers import glam, transformers

WeightInit = base_layer.WeightInit

GPT_EOS_ID = 1


## from https://github.com/google/paxml/commit/9b5682019806dcb058b82ec2f122aa30ed51f255
def configure_gpt3_task(
    cls,
    task_p: pax_fiddle.Config[tasks_lib.SingleTask],
) -> pax_fiddle.Config[tasks_lib.SingleTask]:
  """Returns task with gpt3 related configs."""
  model_p = task_p.model  # pytype: disable=attribute-error  # enable-nested-classes

  model_p.decoder_tpl.eos_id = (
      GPT_EOS_ID  # pytype: disable=attribute-error  # enable-nested-classes
  )
  model_p.decoder_tpl.seqlen = cls.MAX_SEQ_LEN  # pytype: disable=attribute-error  # enable-nested-classes

  model_p.params_init = WeightInit.Gaussian(0.006)

  softmax_init = WeightInit.Gaussian(0.006)
  model_p.lm_tpl.softmax_tpl.params_init = softmax_init
  model_p.lm_tpl.softmax_tpl.feed_forward_tpl.has_bias = False
  model_p.lm_tpl.softmax_tpl.soft_cap_logits = None

  if cls.SEPARATE_EMBEDDING:
    model_p.lm_tpl.separate_embedding_tpl.scale_sqrt_depth = False
    model_p.lm_tpl.separate_embedding_tpl.lookup_style = (
        cls.EMBEDDING_LOOKUP_STYLE
    )
  else:
    model_p.lm_tpl.softmax_tpl.scale_sqrt_depth = False
    model_p.lm_tpl.softmax_tpl.lookup_style = cls.EMBEDDING_LOOKUP_STYLE
  if cls.TRAINABLE_POSITION_EMB:
    model_p.lm_tpl.position_emb_tpl.lookup_style = cls.EMBEDDING_LOOKUP_STYLE

  stacked_p = model_p.lm_tpl.stacked_transformer_tpl
  if fdl.get_callable(stacked_p) == transformers.PipelinedTransformer:
    stacked_p = stacked_p.pipeline_stage
  if issubclass(
      fdl.get_callable(stacked_p), transformers.StackedTransformerRepeated
  ):
    stacked_p = stacked_p.block
  transformer_layer_p = stacked_p.transformer_layer_params_tpl

  transformer_layer_p.ln_tpl.epsilon = cls.LAYERNORM_EPSILON
  transformer_layer_p.tr_fflayer_tpl.ln_tpl.epsilon = cls.LAYERNORM_EPSILON
  model_p.lm_tpl.final_ln_tpl.epsilon = cls.LAYERNORM_EPSILON
  transformer_layer_p.tr_atten_tpl.internal_enable_per_dim_scale = False
  transformer_layer_p.tr_atten_tpl.use_bias = True

  transformer_layer_p.tr_fflayer_tpl.activation_tpl.approximate = True

  for atten_p in (
      transformer_layer_p.tr_atten_tpl,
      transformer_layer_p.cross_atten_tpl,
  ):
    if atten_p is None:
      continue
    atten_wp = atten_p.weight_split_dims_mapping
    atten_wp.proj = ['data', 'mdl', None]

  return task_p


## 8 node
class GPT126MBase(TransformerLmSpmdAdam):

  MODEL_CLASS = CustomMetricsLM

  USE_REPEATED_LAYER = False
  ICI_MESH_SHAPE = [8, 1, 1]
  DCN_MESH_SHAPE = [8, 1, 1]
  FPROP_DTYPE = jnp.bfloat16
  MAX_STEPS = 600000

  MAX_SEQ_LEN = 2048
  VOCAB_SIZE = 50304
  PACKED_INPUT = False
  PERCORE_BATCH_SIZE = 4

  NUM_LAYERS = 12
  NUM_HEADS = 12
  MODEL_DIMS = 768
  HIDDEN_DIMS = 3072
  DIMS_PER_HEAD = 64

  TRAINABLE_POSITION_EMB = True
  USE_BIAS = True
  LAYERNORM_EPSILON = 1e-5
  ATTEN_LOGIT_CAP = -1.0
  INIT_STD = 0.023
  SOFTMAX_INIT_STD = 0.023
  ACTIVATION_CLS = layers.GELU

  ## optimizer-related
  ADAM_BETA1 = 0.9
  ADAM_BETA2 = 0.95
  LEARNING_RATE = 6e-4
  ADAM_EPSILON_ROOT = 0.0
  ADAM_EPSILON = 1e-8
  WEIGHT_DECAY = 0.1
  ADAM_CLIP_THRESHOLD = -1.0
  CLIP_GRADIENT_NORM_TO_VALUE = 1.0

  ## lr schedule
  LR_SCHEDULE = 'linear_rampup_cosine_decay'
  LR_COS_WARMUP = 636
  LR_COS_DECAY_START = LR_COS_WARMUP+1
  LR_COS_DECAY_END = 500000
  R_COS_MIN_RATIO = 0.1
  LR_COS_MAX = 1.0

  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    self.TRAINABLE_PE_MAX_SEQ_LEN = self.MAX_SEQ_LEN

    task_p = super().task()
    task_p = configure_gpt3_task(self, task_p)
    task_p.train.num_train_steps = self.MAX_STEPS
    task_p.train.compute_steps_per_sec_interval_steps = (
        self.SUMMARY_INTERVAL_STEPS
    )

    model_p = task_p.model

    ### compute layernorm reductions in fp32. Needed for stable training on GPUs
    stacked_p = model_p.lm_tpl.stacked_transformer_tpl
    if fdl.get_callable(stacked_p) == transformers.PipelinedTransformer:
      stacked_p = stacked_p.pipeline_stage
    if issubclass(
        fdl.get_callable(stacked_p), transformers.StackedTransformerRepeated
    ):
      stacked_p = stacked_p.block

    task_p.model.lm_tpl.final_ln_tpl.reductions_in_fp32 = True
    if not TransformerEngineHelper.is_enabled_te():
      transformer_layer_p = stacked_p.transformer_layer_params_tpl
      transformer_layer_p.ln_tpl.reductions_in_fp32 = True
      transformer_layer_p.tr_fflayer_tpl.ln_tpl.reductions_in_fp32 = True
    else:
      stacked_p = TransformerEngineHelper.get_stack_transformer(
        stacked_p, jnp.dtype(self.FPROP_DTYPE))
      if issubclass(fdl.get_callable(model_p.lm_tpl.stacked_transformer_tpl),
                    transformers.StackedTransformerRepeated):
        model_p.lm_tpl.stacked_transformer_tpl.block = stacked_p
      else:
        model_p.lm_tpl.stacked_transformer_tpl = stacked_p


    model_p.params_init = WeightInit.Gaussian(self.INIT_STD)
    softmax_init = WeightInit.Gaussian(self.SOFTMAX_INIT_STD)
    model_p.lm_tpl.softmax_tpl.params_init = softmax_init

    model_p.apply_eval_sample_weights = True

    return task_p


## 32 node
class GPT5BBase(GPT126MBase):

  USE_REPEATED_LAYER = True
  ICI_MESH_SHAPE = [1, 8, 1]
  DCN_MESH_SHAPE = [1, 32, 1]
  CHECKPOINT_POLICY = layers.AutodiffCheckpointType.SAVE_DOT_WITH_NO_BATCH_DIM
  MAX_STEPS = 75000

  PERCORE_BATCH_SIZE = 8

  NUM_LAYERS = 24
  NUM_HEADS = 32
  MODEL_DIMS = 4096
  HIDDEN_DIMS = 16384
  DIMS_PER_HEAD = 128

  INIT_STD = 0.01
  SOFTMAX_INIT_STD = 0.01

  ## optimizer-related
  LEARNING_RATE = 1.6e-4

  ## lr schedule
  LR_COS_WARMUP = 115
  LR_COS_DECAY_START = LR_COS_WARMUP+1
  LR_COS_DECAY_END = 62500

  CHECKPOINT_EVERY_N_STEPS = 250
  SUMMARY_INTERVAL_STEPS = 10

  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    task_p = super().task()

    model_p = task_p.model
    stacked_p = model_p.lm_tpl.stacked_transformer_tpl
    if issubclass(
        fdl.get_callable(stacked_p), transformers.StackedTransformerRepeated
    ):
      stacked_p = stacked_p.block

    stacked_p.input_dropout_prob = 0.1
    return task_p


## 96 node
class GPT175BBase(GPT126MBase):

  NUM_LAYERS = 96
  NUM_HEADS = 96
  MODEL_DIMS = 12288
  # Known as MLP_DIM in t5x
  HIDDEN_DIMS = MODEL_DIMS * 4
  # Defaults to MODEL_DIMS // NUM_HEADS.
  DIMS_PER_HEAD = 128
  # Known as NUM_EMBEDDINGS in t5x
  VOCAB_SIZE = 50257
  USE_REPEATED_LAYER = True
  MAX_STEPS = 75000

  # Model configs
  ACTIVATION_CLS = layers.GELU
  USE_GATED_ACTIVATION = False
  SEPARATE_EMBEDDING = False
  TRAINABLE_POSITION_EMB = True
  ATTEN_LOGIT_CAP = -1.0  # Disable logits cap in atten

  # HPs
  WEIGHT_DECAY = 0.1
  ADAM_BETA1 = 0.9
  ADAM_BETA2 = 0.95
  ADAM_EPSILON = 1e-8
  ADAM_CLIP_THRESHOLD = -1.0  # Disable Adam clip_threshold
  CLIP_GRADIENT_NORM_TO_VALUE = 1.0
  LAYERNORM_EPSILON = 1e-5

  # In units of steps for BS1.5k
  LEARNING_RATE = 2e-5
  LR_SCHEDULE = 'linear_rampup_cosine_decay'
  LR_COS_WARMUP = 265
  LR_COS_DECAY_START = LR_COS_WARMUP + 1
  LR_COS_DECAY_END = 108600
  LR_COS_MAX = 1.0
  LR_COS_MIN_RATIO = 0.1

  # Checkpoint
  EVAL_INTERVAL_STEPS = 100
  SUMMARY_INTERVAL_STEPS = 10
  CHECKPOINT_EVERY_N_STEPS = 100
  CHECKPOINT_MAX_TO_KEEP = 10

  ## GPU-specific settings
  ICI_MESH_SHAPE = [1, 8, 1]
  DCN_MESH_SHAPE = [1, 32, 1]
  PERCORE_BATCH_SIZE = 6

  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    task_p = super().task()
    return task_p


### synthetic configs
@experiment_registry.register
class Synthetic126M(GPT126MBase, SyntheticDataset):

  MAX_STEPS = 100

  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    task_p = super().task()
    return task_p


@experiment_registry.register
class Synthetic5B(GPT5BBase, SyntheticDataset):

  MAX_STEPS = 100

  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    task_p = super().task()
    return task_p


@experiment_registry.register
class Synthetic175B(GPT175BBase, SyntheticDataset):

  MAX_STEPS = 100

  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    task_p = super().task()
    return task_p


### configs with the Pile dataset
@experiment_registry.register
class Pile126M(GPT126MBase, PileUnsupervisedDataset):

  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    task_p = super().task()
    return task_p


@experiment_registry.register
class Pile5B(GPT5BBase, PileUnsupervisedDataset):

  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    task_p = super().task()
    return task_p


@experiment_registry.register
class Pile175B(GPT175BBase, PileUnsupervisedDataset):

  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    task_p = super().task()
    return task_p


### example of a config that runs evaluation on the lambada dataset
@experiment_registry.register
class Lambada126M(GPT126MBase, LambadaDataset):

  ICI_MESH_SHAPE = [8, 1, 1]

  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    task_p = super().task()
    task_p.train.always_use_train_for_model_init = False
    task_p.model.eval_task = 'lambada'
    return task_p


### legacy aliases
GPT5B = Pile5B
GPT175B = Pile175B


@experiment_registry.register
class LLaMA7B(BaseLLaMA, BoolQDataset, LoRAMixin, CheckpointRestoreMixin):
  """7B model on a A100-40GB.

  Checkpoint:
  gs://sax-data/pax-llama/7B/checkpoint_00000000/

  April 14, 2023
  Latency = 3.619s with 128 decoded tokens. 27ms per output token
  """

  NUM_LAYERS = 32
  VOCAB_SIZE = 32000
  DIMS_PER_HEAD = 128
  NUM_HEADS = 32
  MODEL_DIMS = 4096
  HIDDEN_DIMS = 11008

  PERCORE_BATCH_SIZE = 16

  ICI_MESH_SHAPE = [1, 8, 1]
  DCN_MESH_SHAPE = [1, 1, 1]

  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    task_p = super().task()
    task_p.train.always_use_train_for_model_init = False
    task_p.model.apply_eval_sample_weights = True
    task_p.model.eval_task = 'boolq'
    task_p.model.boolq_yn_tokens = jnp.array(
        [self.TRUE_TOKEN, self.FALSE_TOKEN]
    )

    task_p = self.configure_lora(task_p)
    task_p = self.configure_checkpoint_restore(task_p)

    return task_p


@experiment_registry.register
class LLaMA13B(LLaMA7B):
  """13B model on a A100-40GB.

  April 12, 2023
  Latency = 5.06s with 128 decoded tokens. 38ms per output token.
  """

  NUM_LAYERS = 40
  VOCAB_SIZE = 32000
  DIMS_PER_HEAD = 128
  NUM_HEADS = 40
  MODEL_DIMS = 5120
  HIDDEN_DIMS = 13824

  PERCORE_BATCH_SIZE = 8

  ICI_MESH_SHAPE = [1, 8, 1]
  DCN_MESH_SHAPE = [1, 1, 1]


@experiment_registry.register
class LLaMA70B(LLaMA7B):
  """LlaMA-2 70B model on TPUv5-16."""

  NUM_LAYERS = 80
  VOCAB_SIZE = 32000
  DIMS_PER_HEAD = 128
  NUM_HEADS = 64
  MODEL_DIMS = 8192
  HIDDEN_DIMS = 28672
  USE_MQA = True
  NUM_KV_HEADS = 8

  PERCORE_BATCH_SIZE = 4

  ICI_MESH_SHAPE = [1, 8, 1]
  DCN_MESH_SHAPE = [1, 2, 1]


class GLaM126M64EBase(GPT126MBase):

  NUM_GPUS = 64
  NUM_EXPERTS = 64

  ICI_MESH_SHAPE = [1, NUM_GPUS, 1]
  DCN_MESH_SHAPE = [1, 1, 1]
  USE_REPEATED_LAYER = False

  PERCORE_BATCH_SIZE = 4

  LOAD_BALANCING_WEIGHT = 0.01
  Z_LOSS_WEIGHT = 0

  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    task_p = super().task()
    task_p.model.lm_tpl = glam.GlamUniTransformerLmHParams(
        name='glam_lm',
        vocab_size=self.VOCAB_SIZE,
        num_transformer_layers=self.NUM_LAYERS,
        moe=True,
        model_dim=self.MODEL_DIMS,
        ff_dim=self.HIDDEN_DIMS,
        moe_hidden_dim=self.HIDDEN_DIMS,
        attention_num_heads=self.NUM_HEADS,
        attention_key_value_dim=self.MODEL_DIMS // self.NUM_HEADS,
        attention_extra_logit=0.0,
        use_tgt_labels_size_as_loss_denominator=True,
        moe_load_balance_loss_weight=self.LOAD_BALANCING_WEIGHT
        / (self.NUM_LAYERS / 2),
        z_loss_weight=self.Z_LOSS_WEIGHT,
        moe_gating_func='top2',
        moe_gating_embedding_level='token',
        c_dim=None,  ## determined automatically when capacity_factor is set
        capacity_factor=2.0,
        e_dim=self.NUM_EXPERTS,
        num_groups=np.prod(self.ICI_MESH_SHAPE[:-1])
        * np.prod(
            self.DCN_MESH_SHAPE[:-1]
        ),  ## product of all data-related axes
        use_gated_activation=True,
        repeat=self.USE_REPEATED_LAYER,
    )

    ## set sharding
    lm_cls = cast(
        Type[layers.TransformerLm], pax_fiddle.get_callable(task_p.model.lm_tpl)
    )

    replica_axis = 'replica'
    data_axis = 'data'
    data_expert_axis = 'data_expert'
    mdl_axis = 'mdl'

    task_p.model.lm_tpl = lm_cls.set_sharding_params_v1(
        task_p.model.lm_tpl,
        replica_axis='replica',
        data_axis='data',
        mdl_axis='mdl',
        ici_mesh_shape=task_p.model.ici_mesh_shape,
        dcn_mesh_shape=task_p.model.dcn_mesh_shape,
        mesh_axis_names=['replica', 'data', 'mdl'],
        training_optimized=self.TRAINING_OPTIMIZED_SHARDING,
    )

    task_p.train.always_use_train_for_model_init = False
    task_p.model.report_strict_acc = True

    return task_p


class GLaM64B64EBase(GLaM126M64EBase):

  MAX_SEQ_LEN = 1024

  NUM_GPUS = 512
  DCN_MESH_SHAPE = [1, int(NUM_GPUS / 8), 1]
  ICI_MESH_SHAPE = [1, 1, 8]

  NUM_LAYERS = 64
  NUM_HEADS = 64
  DIMS_PER_HEAD = 128
  MODEL_DIMS = 8192
  HIDDEN_DIMS = 32768

  NUM_EXPERTS = 64

  USE_REPEATED_LAYER = True

  ADAM_BETA1 = 0
  ADAM_BETA2 = 0.99
  CLIP_GRADIENT_NORM_TO_VALUE = 1.0

  PERCORE_BATCH_SIZE = 4

  # In units of steps for BS 1k
  LEARNING_RATE = 2e-5
  LR_SCHEDULE = 'linear_rampup_cosine_decay'
  LR_COS_WARMUP = 398
  LR_COS_DECAY_START = LR_COS_WARMUP + 1
  LR_COS_DECAY_END = 162900
  LR_COS_MAX = 1.0
  LR_COS_MIN_RATIO = 0.1

  # Checkpoint
  EVAL_INTERVAL_STEPS = 100
  SUMMARY_INTERVAL_STEPS = 10
  CHECKPOINT_EVERY_N_STEPS = 100
  CHECKPOINT_MAX_TO_KEEP = 2

  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:

    task_p = super().task()

    return task_p


class PileGLaM126M64E(GLaM126M64EBase, PileUnsupervisedDataset):

  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    task_p = super().task()
    return task_p


class PileGLaM64B64E(GLaM64B64EBase, PileUnsupervisedDataset):

  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    task_p = super().task()
    return task_p


class LambadaGLaM126M64E(GLaM126M64EBase, LambadaDataset):

  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    task_p = super().task()
    return task_p


class LambadaGLaM64B64E(GLaM64B64EBase, LambadaDataset):

  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    task_p = super().task()
    return task_p
