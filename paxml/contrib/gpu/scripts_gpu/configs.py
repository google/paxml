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

import jax.numpy as jnp
from paxml.tasks.lm.params.c4 import TransformerLmSpmdAdam, C4SpmdPipelineGpt3AdamOrgHP
from paxml.contrib.gpu.scripts_gpu.tasks import LambadaDataset, PileUnsupervisedDataset
from paxml import experiment_registry
from paxml import tasks_lib
from praxis import base_layer
from praxis import layers
from praxis import optimizers
from praxis import schedules
from praxis.layers import transformers

WeightInit = base_layer.WeightInit

GPT_EOS_ID = 1

## from https://github.com/google/paxml/commit/9b5682019806dcb058b82ec2f122aa30ed51f255
def configure_gpt3_task(
    cls,
    task_p: tasks_lib.SingleTask.HParams,
) -> tasks_lib.SingleTask.HParams:
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
  if stacked_p.cls == transformers.PipelinedTransformer:
    stacked_p = stacked_p.pipeline_stage
  if issubclass(stacked_p.cls, transformers.StackedTransformerRepeated):
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
@experiment_registry.register
class GPT126M(TransformerLmSpmdAdam):
    
  USE_REPEATED_LAYER = False
  ICI_MESH_SHAPE = [64,1,1]
  FPROP_DTYPE = jnp.bfloat16
  MAX_STEPS = 600000
  
  MAX_SEQ_LEN = 2048
  VOCAB_SIZE = 50304
  PACKED_INPUT = True
  PERCORE_BATCH_SIZE = 4
  
  NUM_LAYERS = 12
  NUM_HEADS = 12
  MODEL_DIMS = 768
  HIDDEN_DIMS = 3072
  DIMS_PER_HEAD = 64

  TRAINABLE_POSITION_EMB = True
  TRAINABLE_PE_MAX_SEQ_LEN = MAX_SEQ_LEN
  
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

  
  def task(self) -> tasks_lib.SingleTask.HParams:
    task_p = super().task()
    task_p = configure_gpt3_task(self, task_p)
    task_p.train.num_train_steps = self.MAX_STEPS
    
    model_p = task_p.model
    
    ### compute layernorm reductions in fp32. Needed for stable training on GPUs
    stacked_p = model_p.lm_tpl.stacked_transformer_tpl
    if stacked_p.cls == transformers.PipelinedTransformer:
      stacked_p = stacked_p.pipeline_stage
    if issubclass(stacked_p.cls, transformers.StackedTransformerRepeated):
      stacked_p = stacked_p.block
    transformer_layer_p = stacked_p.transformer_layer_params_tpl
    transformer_layer_p.ln_tpl.reductions_in_fp32 = True
    transformer_layer_p.tr_fflayer_tpl.ln_tpl.reductions_in_fp32 = True
    task_p.model.lm_tpl.final_ln_tpl.reductions_in_fp32 = True
    
    model_p.params_init = WeightInit.Gaussian(self.INIT_STD)
    softmax_init = WeightInit.Gaussian(self.SOFTMAX_INIT_STD)
    model_p.lm_tpl.softmax_tpl.params_init = softmax_init
    
    model_p.apply_eval_sample_weights = True
    
    return task_p

@experiment_registry.register
class Pile126M(GPT126M, PileUnsupervisedDataset):
  def task(self) -> tasks_lib.SingleTask.HParams:
    task_p = super().task()
    return task_p

@experiment_registry.register
class Lambada126M(GPT126M, LambadaDataset):
  
  ICI_MESH_SHAPE = [8,1,1]

  def task(self) -> tasks_lib.SingleTask.HParams:
    task_p = super().task()
    task_p.train.always_use_train_for_model_init=False
    task_p.model.report_strict_acc=True
    return task_p

## 32 node
@experiment_registry.register
class GPT5B(Pile126M):
  
  USE_REPEATED_LAYER = True
  ICI_MESH_SHAPE = [4,1,2]
  DCN_MESH_SHAPE = [32,1,1]
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

## 96 node
@experiment_registry.register
class GPT175B(C4SpmdPipelineGpt3AdamOrgHP):
    
  USE_REPEATED_LAYER = False
  ICI_MESH_SHAPE = [2,1,1,4]
  DCN_MESH_SHAPE = [8,12,1,1]
  NUM_STAGES = 16
  PERCORE_BATCH_SIZE = 2
  MICROBATCH_SIZE = 24
  MAX_STEPS = 75000

  LEARNING_RATE = 2e-5
  
  def task(self) -> tasks_lib.SingleTask.HParams:
    task_p = super().task()
    model_p = task_p.model
    
    ### compute layernorm reductions in fp32. Needed for stable training on GPUs
    stacked_p = model_p.lm_tpl.stacked_transformer_tpl
    if stacked_p.cls == transformers.PipelinedTransformer:
      stacked_p = stacked_p.pipeline_stage
    if issubclass(stacked_p.cls, transformers.StackedTransformerRepeated):
      stacked_p = stacked_p.block
    transformer_layer_p = stacked_p.transformer_layer_params_tpl
    
    transformer_layer_p.ln_tpl.reductions_in_fp32 = True
    transformer_layer_p.tr_fflayer_tpl.ln_tpl.reductions_in_fp32 = True
    task_p.model.lm_tpl.final_ln_tpl.reductions_in_fp32 = True
    
    return task_p

## single node training
@experiment_registry.register
class SmallPileTest(Pile126M):
  """Base config for an SPMD model."""
  ICI_MESH_SHAPE = [8,1,1]
    
