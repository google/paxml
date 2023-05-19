"""DenseBuilder-based LM with TF record inputs."""

from lingvo import compat as tf
from lingvo import model_registry
from lingvo.core import base_input_generator
from lingvo.core import base_model_params
from lingvo.core import gshard_builder
from lingvo.core import gshard_utils
from lingvo.core import optimizer
from lingvo.core import program
from lingvo.core import py_utils
from lingvo.core import schedule
from lingvo.core import tokenizers
from lingvo.core import generic_input
from lingvo.tasks.lm import input_generator as lm_inp
import numpy as np
import os, sys, math, random, copy
from google.cloud import storage
from paxml.tasks.lm import model_params

"""Base language model configurations."""

import math
import typing
from typing import Optional, Sequence, Type, cast

import fiddle as fdl
from jax import numpy as jnp
from paxml import base_experiment
from paxml import tasks_lib
from praxis import asserts
from praxis import base_layer
from praxis import base_model
from praxis import layers
from praxis import optimizers
from praxis import pax_fiddle
from praxis import py_utils
from praxis import schedules
from praxis.layers import activations
from praxis.layers import embedding_softmax
from praxis.layers import models
from praxis.layers import transformer_models


NestedMap = py_utils.NestedMap
WeightInit = base_layer.WeightInit

#class DataBuild(base_input_generator.BaseSequenceInputGenerator):
class DataBuild(base_input_generator.BaseInputGeneratorFromFiles):

  @classmethod
  def Params(cls):
    """Defaults params for `LMInput`."""
    params = super().Params()
    params.Define('seq_len', 0, 'input sequence length')
    params.Define('last_global_step', 0, 'for dataset resume steps')  # TODO: Need to using the pyutils.global_steps()
    return params

  def __init__(self, params):
    super().__init__(params)

  def _DataSourceFromFilePattern(self, file_pattern):
    def Proc(record):
      seq_len = self.params.seq_len
      outputs = [('text', tf.io.VarLenFeature(tf.int64))]
      features = tf.io.parse_single_example(record, dict(outputs))
      for k, v in features.items():
          features[k] = v.values
      bucket_key = tf.size(features['text'])

      return [features[k] for k, v in features.items()], bucket_key

    args = self.CommonInputOpArgs()

    features, bucket_key = generic_input.GenericInput(
        file_pattern=file_pattern,
        processor=Proc,
        **args)

    return self.BuildInputBatch(
        batch_size=self.InfeedBatchSize(),
        features_list=features)

  def BuildInputBatch(self, batch_size, features_list, bucket_keys=None):

    p = self.params

    ret = py_utils.NestedMap()
    bs = batch_size

    ret.tgt = py_utils.NestedMap()

    def SetShape(x):
      x.set_shape([bs, p.seq_len +1])

    ids = features_list[0]
    SetShape(ids)
    label = tf.roll(ids, -1, axis=-1)
    ids = tf.strided_slice(ids, [0,0], [bs, p.seq_len])
    label = tf.strided_slice(label, [0,0], [bs, p.seq_len])

    ret.ids = tf.cast(ids, dtype=tf.int32)
    #Anisha: adding dummy paddings and weights
    ret.paddings = tf.zeros_like(ids)
    ret.weights = tf.ones_like(ids)
    
    ret.labels = tf.cast(label, dtype=tf.int32)
    ret.segment_ids = tf.minimum(ret.ids, 1)
    seg_pos = tf.range(p.seq_len, dtype=tf.int32)
    seg_pos = tf.expand_dims(seg_pos, axis=0)
    ret.segment_pos = tf.tile(seg_pos, [bs, 1])
    #ret.tgt.segment_pos = tf.cast(label, dtype=tf.int32)

    if (p.fprop_dtype is None or
       p.dtype==p.fprop_dtype):
      return ret

    def _Cast(v):
      if not v.dtype.is_floating:
        return v
      return tf.cast(v, p.fprop_dtype)

    ret = ret.Transform(
            lambda t: tf.ensure_shape(t, (bs, p.seq_len)))
    ret = ret.Transform(_Cast)
    return ret
  
  # def BuildInputBatch(self, batch_size, features_list, bucket_keys=None):

  #   p = self.params

  #   ret = py_utils.NestedMap()
  #   bs = batch_size

  #   ret.tgt = py_utils.NestedMap()

  #   def SetShape(x):
  #     x.set_shape([bs, p.seq_len +1])

  #   ids = features_list[0]
  #   SetShape(ids)
  #   label = tf.roll(ids, -1, axis=-1)
  #   ids = tf.strided_slice(ids, [0,0], [bs, p.seq_len])
  #   label = tf.strided_slice(label, [0,0], [bs, p.seq_len])

  #   ret.tgt.ids = tf.cast(ids, dtype=tf.int32)
  #   ret.tgt.labels = tf.cast(label, dtype=tf.int32)
  #   ret.tgt.segment_ids = tf.minimum(ret.tgt.ids, 1)
  #   seg_pos = tf.range(p.seq_len, dtype=tf.int32)
  #   seg_pos = tf.expand_dims(seg_pos, axis=0)
  #   ret.tgt.segment_pos = tf.tile(seg_pos, [bs, 1])
  #   #ret.tgt.segment_pos = tf.cast(label, dtype=tf.int32)

  #   if (p.fprop_dtype is None or
  #      p.dtype==p.fprop_dtype):
  #     return ret

  #   def _Cast(v):
  #     if not v.dtype.is_floating:
  #       return v
  #     return tf.cast(v, p.fprop_dtype)

  #   ret = ret.Transform(
  #           lambda t: tf.ensure_shape(t, (bs, p.seq_len)))
  #   ret = ret.Transform(_Cast)
  #   return ret


class DenseLMTemplateLG(base_experiment.BaseExperiment):
  """Base SPMD Transformer LM configuration using Adafactor."""
  # architecture related
  NUM_LAYERS = 10 #?
  VOCAB_SIZE = 50272
  DIMS_PER_HEAD = 64 #?
  NUM_HEADS = 64
  MODEL_DIMS = 4 * 1024
  HIDDEN_DIMS = MODEL_DIMS * 4 # originally was 32 * 1024 
  FPROP_DTYPE = jnp.bfloat16 #?
  PACKED_INPUT = True #?

  USE_REPEATED_LAYER = True
  SEPARATE_EMBEDDING = False #?
  TRAINABLE_POSITION_EMB = True  
  TRAINABLE_PE_MAX_SEQ_LEN = 16 * 1024 #?
  RELATIVE_BIAS = False #?
  USE_ROTARY_POSITION_EMB = False #?
  NORM_POLICY = 'pre' #?
  ENABLE_DCONV = False #?
  COMBINE_QKV = True #?
  ACTIVATION_CLS = activations.GELU #.ReLU
  USE_GATED_ACTIVATION = True
  DECAY_END = 100000 #?

  # optimizer related
  DROPOUT_PROB = 0.0 #?
  LEARNING_RATE =  0.0001 #2.5e-4
  CLIP_GRADIENT_NORM_TO_VALUE = 1.0
  WEIGHT_DECAY = 1e-3
  SOFTMAX_CAP_LOGITS = 30.0
  ATTEN_LOGIT_CAP = 50.0
  # Autodiff remat.
  CHECKPOINT_POLICY = layers.AutodiffCheckpointType.SAVE_NOTHING

  # checkpoint
  CHECKPOINT_EVERY_N_STEPS = 5000
  SUMMARY_INTERVAL_STEPS = 100
  CHECKPOINT_MAX_TO_KEEP = 10000
  EVAL_INTERVAL_STEPS = 100

  # Profiler related
  PROFILER_NUM_STEPS = 2
  PROFILER_MIN_DURATION_SEC = 1
  PROFILER_CAPTURE_STEP = None

  # Sub-class has to specify a mesh.
  ICI_MESH_SHAPE = [1, 4, 1]
  # Default to a single slice
  DCN_MESH_SHAPE = [1, 1, 1]
  TRAINING_OPTIMIZED_SHARDING = True
  MAX_SEQ_LEN = 2048

  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    """Returns the task parameters."""
    if self.DIMS_PER_HEAD is not None:
      if self.NUM_HEADS is None:
        assert self.MODEL_DIMS % self.DIMS_PER_HEAD == 0
        num_heads = int(self.MODEL_DIMS / self.DIMS_PER_HEAD)
      else:
        assert self.MODEL_DIMS == self.NUM_HEADS * self.DIMS_PER_HEAD
        num_heads = self.NUM_HEADS
    else:
      assert self.NUM_HEADS is not None
      num_heads = self.NUM_HEADS

    task_p = pax_fiddle.Config(tasks_lib.SingleTask, name='xformer_task')
    task_p.model = pax_fiddle.Config(models.LanguageModel, name='xformer_lm')
    model_p = task_p.model
    model_p.lm_tpl.packed_input = self.PACKED_INPUT
    model_p.lm_tpl.model_dims = self.MODEL_DIMS
    model_p.lm_tpl.vocab_size = self.VOCAB_SIZE

    if self.SEPARATE_EMBEDDING:
      model_p.lm_tpl.separate_embedding_tpl = pax_fiddle.Config(
          layers.Embedding
      )
      model_p.lm_tpl.softmax_tpl = pax_fiddle.Config(layers.FullSoftmax)

    softmax_init = WeightInit.Gaussian(1.0 / math.sqrt(self.MODEL_DIMS))
    # pytype: disable=attribute-error  # enable-nested-classes
    model_p.lm_tpl.softmax_tpl.params_init = softmax_init
    if self.SEPARATE_EMBEDDING:
      model_p.lm_tpl.separate_embedding_tpl.scale_sqrt_depth = True
    else:
      model_p.lm_tpl.softmax_tpl.scale_sqrt_depth = True
    model_p.lm_tpl.softmax_tpl.soft_cap_logits = self.SOFTMAX_CAP_LOGITS

    if self.TRAINABLE_POSITION_EMB:
      model_p.lm_tpl.position_emb_tpl = pax_fiddle.Config(
          layers.TrainablePositionalEmbedding,
          max_seq_length=self.TRAINABLE_PE_MAX_SEQ_LEN,
      )

    stacked_transformer_tpl = pax_fiddle.Config(layers.StackedTransformer)
    stacked_transformer_tpl.model_dims = self.MODEL_DIMS
    stacked_transformer_tpl.hidden_dims = self.HIDDEN_DIMS
    stacked_transformer_tpl.num_layers = self.NUM_LAYERS
    stacked_transformer_tpl.num_heads = num_heads
    stacked_transformer_tpl.dim_per_head = self.DIMS_PER_HEAD

    stacked_transformer_tpl.dropout_prob = self.DROPOUT_PROB
    transformer_layer_p = typing.cast(
        pax_fiddle.Config[layers.Transformer],
        stacked_transformer_tpl.transformer_layer_params_tpl,
    )
    transformer_layer_p.tr_atten_tpl.atten_logit_cap = self.ATTEN_LOGIT_CAP
    transformer_layer_p.norm_policy = self.NORM_POLICY
    transformer_layer_p.tr_atten_tpl.use_bias = False
    transformer_layer_p.tr_atten_tpl.combine_qkv = self.COMBINE_QKV
    transformer_layer_p.tr_fflayer_tpl.activation_tpl = pax_fiddle.Config(
        self.ACTIVATION_CLS
    )
    transformer_layer_p.tr_fflayer_tpl.use_gated_activation = (
        self.USE_GATED_ACTIVATION)
    transformer_layer_p.tr_atten_tpl.dconv_qkv = self.ENABLE_DCONV
    # pytype: enable=attribute-error  # enable-nested-classes

    # Only one of RELATIVE_BIAS or USE_ROTARY_POSITION_EMB can be True.
    assert (not self.RELATIVE_BIAS) or (not self.USE_ROTARY_POSITION_EMB)
    if self.RELATIVE_BIAS:
      transformer_layer_p.tr_atten_tpl.relative_bias_tpl = pax_fiddle.Config(
          layers.RelativeBias
      )
    if self.USE_ROTARY_POSITION_EMB:
      transformer_layer_p.tr_atten_tpl.use_rotary_position_emb = True

    if self.USE_REPEATED_LAYER:
      model_p.lm_tpl.stacked_transformer_tpl = pax_fiddle.Config(
          layers.StackedTransformerRepeated
      )
      stacked_transformer_tpl.num_layers = 1
      model_p.lm_tpl.stacked_transformer_tpl.block = stacked_transformer_tpl
      model_p.lm_tpl.stacked_transformer_tpl.x_times = self.NUM_LAYERS
      model_p.lm_tpl.stacked_transformer_tpl.checkpoint_policy = (
          self.CHECKPOINT_POLICY)
    else:
      model_p.lm_tpl.stacked_transformer_tpl = stacked_transformer_tpl

    # Enable bf16.
    model_p.fprop_dtype = self.FPROP_DTYPE

    model_params.set_default_adafactor(
        task_p,
        self.LEARNING_RATE,
        self.WEIGHT_DECAY,
        decay_end=self.DECAY_END,
        clip_gradient_norm_to_value=self.CLIP_GRADIENT_NORM_TO_VALUE)

    task_p.train.save_interval_steps = self.CHECKPOINT_EVERY_N_STEPS
    task_p.train.summary_interval_steps = self.SUMMARY_INTERVAL_STEPS
    task_p.train.save_max_to_keep = self.CHECKPOINT_MAX_TO_KEEP
    task_p.train.eval_interval_steps = self.EVAL_INTERVAL_STEPS
    task_p.train.profiler_num_steps = self.PROFILER_NUM_STEPS
    task_p.train.profiler_min_duration_sec = self.PROFILER_MIN_DURATION_SEC
    task_p.train.profiler_capture_step = self.PROFILER_CAPTURE_STEP

    if self.ICI_MESH_SHAPE is not None:
      model_params.set_sharding_annotations_v1(task_p, self.TRAINING_OPTIMIZED_SHARDING,
                                  self.ICI_MESH_SHAPE, self.DCN_MESH_SHAPE)
    model_params.maybe_setup_moe_params(model_p.lm_tpl.stacked_transformer_tpl)

    return task_p
  