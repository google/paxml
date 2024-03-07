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

## adapted from https://github.com/google/saxml/blob/main/saxml/server/pax/lm/params/lm_cloud.py

from typing import Type, cast
import fiddle as fdl
import jax.numpy as jnp
from paxml import experiment_registry
from paxml import tasks_lib
from paxml.contrib.gpu.scripts_gpu import saxml_layers
from paxml.tasks.lm.params.c4 import TransformerLmSpmdAdam
from praxis import base_input
from praxis import layers
from praxis import optimizers
from praxis import pax_fiddle
from praxis import schedules
from praxis.contrib.gpu.scripts_gpu.models import CustomMetricsLM
from praxis.layers import activations
from praxis.layers import multi_query_attention

LLaMARotaryEmbedding = saxml_layers.LLaMARotaryEmbedding


@experiment_registry.register
class BaseLLaMA(TransformerLmSpmdAdam):
  """Base LLaMA Transformer LM configuration."""

  MODEL_CLASS = CustomMetricsLM

  BOS_ID = 1
  EOS_ID = 2

  ## only eval supported currently
  MAX_STEPS = 0

  # architecture related
  NUM_LAYERS = 32
  VOCAB_SIZE = 32000
  DIMS_PER_HEAD = 128
  NUM_HEADS = 32
  MODEL_DIMS = 4096
  HIDDEN_DIMS = MODEL_DIMS * 4
  FPROP_DTYPE = jnp.bfloat16
  MODEL_DTYPE = jnp.bfloat16
  USE_MQA = False

  ACTIVATION_CLS = activations.SiLU
  USE_GATED_ACTIVATION = True
  RMS_NORM_EPSILON = 1.0e-05

  # Sub-class has to specify a mesh.
  ICI_MESH_SHAPE = [1, 1, 1]
  DCN_MESH_SHAPE = None

  TRAINING_OPTIMIZED_SHARDING = True

  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    """Returns the task parameters."""
    task_p = super().task()

    task_p.train.num_train_steps = self.MAX_STEPS

    model_p = task_p.model

    model_p.ici_mesh_shape = self.ICI_MESH_SHAPE
    model_p.dcn_mesh_shape = self.DCN_MESH_SHAPE
    replica_axis = 'replica'
    data_axis = 'data'
    mdl_axis = 'mdl'
    model_p.mesh_axis_names = [replica_axis, data_axis, mdl_axis]

    model_p.lm_tpl.packed_input = False
    model_p.lm_tpl.model_dims = self.MODEL_DIMS
    model_p.lm_tpl.vocab_size = self.VOCAB_SIZE
    model_p.lm_tpl.position_emb_tpl = None
    model_p.lm_tpl.softmax_tpl = pax_fiddle.Config(
        layers.FullSoftmax,
        name='output',
        input_dims=self.MODEL_DIMS,
        num_classes=self.VOCAB_SIZE,
    )
    model_p.lm_tpl.softmax_tpl.feed_forward_tpl.has_bias = False
    model_p.lm_tpl.separate_embedding_tpl = pax_fiddle.Config(
        layers.Embedding,
        name='tok_embeddings',
        input_dims=self.MODEL_DIMS,
        num_classes=self.VOCAB_SIZE,
    )
    ln_tpl = pax_fiddle.Config(
        layers.RmsNorm,
        name='norm',
        direct_scale=True,
        epsilon=self.RMS_NORM_EPSILON,
    )
    model_p.lm_tpl.final_ln_tpl = ln_tpl.clone()

    stacked_transformer_tpl = pax_fiddle.Config(layers.StackedTransformer)
    stacked_transformer_tpl.model_dims = self.MODEL_DIMS
    stacked_transformer_tpl.hidden_dims = self.HIDDEN_DIMS
    stacked_transformer_tpl.num_layers = self.NUM_LAYERS
    stacked_transformer_tpl.num_heads = self.NUM_HEADS
    stacked_transformer_tpl.dim_per_head = self.DIMS_PER_HEAD

    transformer_layer_p = cast(
        pax_fiddle.Config[layers.Transformer],
        stacked_transformer_tpl.transformer_layer_params_tpl,
    )
    transformer_layer_p.norm_policy = 'pre'
    transformer_layer_p.ln_tpl = ln_tpl.clone()

    if self.USE_MQA:
      transformer_layer_p.tr_atten_tpl = pax_fiddle.Config(
          multi_query_attention.MultiQueryDotProductAttention,
          num_kv_heads=self.NUM_KV_HEADS,
      )
      transformer_layer_p.tr_atten_tpl.combine_qkv = False
      transformer_layer_p.tr_atten_tpl.proj_tpl.use_bias = False
    else:
      transformer_layer_p.tr_atten_tpl.internal_enable_per_dim_scale = False
      transformer_layer_p.tr_atten_tpl.combine_qkv = True
      transformer_layer_p.tr_atten_tpl.combined_qkv_proj_tpl.use_bias = False

    transformer_layer_p.tr_atten_tpl.internal_enable_query_scale = True
    transformer_layer_p.tr_atten_tpl.use_bias = False
    transformer_layer_p.tr_atten_tpl.rotary_position_emb_tpl = (
        pax_fiddle.Config(LLaMARotaryEmbedding)
    )
    transformer_layer_p.tr_atten_tpl.use_rotary_position_emb = True

    transformer_layer_p.tr_fflayer_tpl.has_bias = False
    transformer_layer_p.tr_fflayer_tpl.fflayer_tpl.has_bias = False
    transformer_layer_p.tr_fflayer_tpl.ln_tpl = ln_tpl.clone()
    transformer_layer_p.tr_fflayer_tpl.activation_tpl = pax_fiddle.Config(
        self.ACTIVATION_CLS
    )
    transformer_layer_p.tr_fflayer_tpl.fflayer_tpl.activation_tpl = (
        pax_fiddle.Config(self.ACTIVATION_CLS)
    )
    model_p.lm_tpl.softmax_tpl.feed_forward_tpl.activation_tpl = (
        pax_fiddle.Config(self.ACTIVATION_CLS)
    )

    transformer_layer_p.tr_fflayer_tpl.use_gated_activation = (
        self.USE_GATED_ACTIVATION
    )

    model_p.lm_tpl.stacked_transformer_tpl = stacked_transformer_tpl

    model_p.fprop_dtype = self.FPROP_DTYPE
    ## for training, we want model dtype to be fp32
    # model_p.dtype = self.MODEL_DTYPE

    ### intermediate dtypes in fp32 for stable training on GPU
    transformer_layer_p.ln_tpl.intermediate_dtype = jnp.float32
    transformer_layer_p.tr_fflayer_tpl.ln_tpl.intermediate_dtype = jnp.float32
    task_p.model.lm_tpl.final_ln_tpl.intermediate_dtype = jnp.float32

    # Set sharding
    lm_cls = cast(
        Type[layers.TransformerLm], pax_fiddle.get_callable(task_p.model.lm_tpl)
    )
    task_p.model.lm_tpl = lm_cls.set_sharding_params_v1(
        task_p.model.lm_tpl,
        replica_axis=replica_axis,
        data_axis=data_axis,
        mdl_axis=mdl_axis,
        ici_mesh_shape=model_p.ici_mesh_shape,
        dcn_mesh_shape=model_p.dcn_mesh_shape,
        mesh_axis_names=model_p.mesh_axis_names,
        training_optimized=self.TRAINING_OPTIMIZED_SHARDING,
    )

    return task_p
