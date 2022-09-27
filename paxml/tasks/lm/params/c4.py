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

"""Language Model configurations on the T5/C4 dataset."""

import functools
import time
from typing import List, Optional

from absl import logging
import jax
from jax import numpy as jnp
from paxml import base_experiment
from paxml import experiment_registry
from paxml import seqio_input
from paxml import tasks_lib
from paxml.tasks.lm import model_params
from paxml.tasks.lm.params import lm_cloud
from praxis import base_input
from praxis import base_layer
from praxis import layers
from praxis import optimizers
from praxis import schedules
import seqio
import t5.data
from t5.data import preprocessors as t5_preprocessors


WeightInit = base_layer.WeightInit

GPT_SPM_PATH = 'gs://mlperf-llm-public2/vocab/c4_en_301_5Mexp2_spm.model'
GPT_EOS_ID = 1
GPT_VOCABULARY = t5.data.SentencePieceVocabulary(GPT_SPM_PATH)
C4_GPT_OUTPUT_FEATURES_LM = {
    'targets': t5.data.Feature(vocabulary=GPT_VOCABULARY, add_eos=True)
}
C4_TFDS_DATADIR = 'gs://mlperf-llm-public2'
C4_EVAL_DATADIR = 'gs://mlperf-llm-public2'


class TaskRegistry(t5.data.TaskRegistry):
  """Task registry with extra tracking."""

  TASK_NAMES = []

  @classmethod
  def add_versioned_tfds_task(cls,
                              name: str,
                              *,
                              versions: List[str],
                              pinned_version: Optional[str] = None,
                              tfds_name: str,
                              tfds_data_dir: Optional[str] = None,
                              **kwargs) -> List[seqio.Task]:
    tasks = []
    for version in versions:
      tasks.append(
          cls.add(
              f'{name}_{version}',
              seqio.Task,
              source=seqio.TfdsDataSource(
                  tfds_name=f'{tfds_name}:{version}',
                  tfds_data_dir=tfds_data_dir,
              ),
              **kwargs,
          ))
    if pinned_version is not None:
      tasks.append(
          cls.add(
              name,
              seqio.Task,
              source=seqio.TfdsDataSource(
                  tfds_name=f'{tfds_name}:{pinned_version}',
                  tfds_data_dir=tfds_data_dir,
              ),
              **kwargs,
          ))
    return tasks


# C4 corpus for language model pretraining
TaskRegistry.add_versioned_tfds_task(
    'c4_lm_v301_gpt',
    versions=['3.0.1'],
    pinned_version='3.0.1',
    tfds_name='c4/en',
    tfds_data_dir=C4_TFDS_DATADIR,
    preprocessors=[
        functools.partial(
            t5_preprocessors.rekey,
            key_map={
                'inputs': None,
                'targets': 'text',
            }),
        seqio.preprocessors.tokenize,
        t5_preprocessors.reduce_concat_tokens,
        t5_preprocessors.split_tokens_to_targets_length,
        seqio.preprocessors.append_eos_after_trim,
    ],
    output_features=C4_GPT_OUTPUT_FEATURES_LM,
    metric_fns=[],
    shuffle_buffer_size=100000,
)

TaskRegistry.add_versioned_tfds_task(
    'c4_lm_v301_gpt_eval',
    versions=['3.0.1'],
    pinned_version='3.0.1',
    tfds_name='c4/en',
    tfds_data_dir=C4_EVAL_DATADIR,
    preprocessors=[
        functools.partial(
            t5_preprocessors.rekey,
            key_map={
                'inputs': None,
                'targets': 'text',
            }),
        seqio.preprocessors.tokenize,
        t5_preprocessors.reduce_concat_tokens,
        t5_preprocessors.split_tokens_to_targets_length,
        seqio.preprocessors.append_eos_after_trim,
    ],
    output_features=C4_GPT_OUTPUT_FEATURES_LM,
    metric_fns=[],
    shuffle_buffer_size=100000,
)


class C4UnsupervisedDataset(base_experiment.BaseExperiment):
  """Used for training Baseline ULM."""
  PERCORE_BATCH_SIZE = 1
  MAX_SEQ_LEN = 1024

  def _dataset_common(self, is_training) -> base_input.BaseInput.HParams:
    num_local_devices = jax.local_device_count()
    if self.PERCORE_BATCH_SIZE >= 1:
      batch_size_per_process = int(self.PERCORE_BATCH_SIZE * num_local_devices)
      num_infeed_hosts = jax.process_count()
    else:
      global_batch_size = int(self.PERCORE_BATCH_SIZE * num_local_devices *
                              jax.process_count())
      if jax.process_count() > 1:
        assert global_batch_size % num_local_devices == 0
        batch_size_per_process = num_local_devices
        num_infeed_hosts = global_batch_size // batch_size_per_process
      else:
        batch_size_per_process = int(self.PERCORE_BATCH_SIZE *
                                     num_local_devices)
        num_infeed_hosts = 1
    seed = None
    if is_training:
      seed = jnp.int32(time.time())
      # TODO(sgpyc): enable sync of seeds across hosts, currently the
      # following failed because of "sync_global_devices name mismatch"
      # seed = jnp.int32(multihost_utils.broadcast_one_to_all(seed))
      logging.info('Train input seed: %d', seed)
    p = seqio_input.SeqIOInput.HParams(
        name='C4Train' if is_training else 'C4Validation',
        mixture_name='c4_lm_v301_gpt' if is_training else 'c4_lm_v301_gpt_eval',
        split_name='train' if is_training else 'validation_24567exp',
        task_feature_lengths={'targets': self.MAX_SEQ_LEN},
        use_cached=True,
        repeat=True if is_training else False,
        feature_converter=seqio_input.LanguageModelFeatures(
            pack=True if is_training else False,
            use_custom_packing_ops=False),
        is_training=is_training,
        input_random_seed=(seed if is_training else 4321),
        batch_size=batch_size_per_process,
        num_infeed_hosts=num_infeed_hosts,
        reset_for_eval=False if is_training else True)
    return p

  def datasets(self) -> List[base_input.BaseInput.HParams]:
    """Returns a list of dataset parameters."""
    return [
        self._dataset_common(is_training=True),
        self._dataset_common(is_training=False)
    ]


class TransformerLmSpmdAdam(model_params.TransformerLmSpmdAdafactor):
  """Base SPMD Transformer LM configuration using Adam.

  Only things different from TransformerLmSpmdAdafactor are listed.
  """
  # architecture related
  NUM_LAYERS = 32
  NUM_HEADS = 16
  MODEL_DIMS = 1024
  HIDDEN_DIMS = MODEL_DIMS * 4
  FPROP_DTYPE = jnp.float32
  PACKED_INPUT = True
  USE_BIAS = False

  # optimizer related
  LEARNING_RATE = 1e-3
  ADAM_BETA1 = 0.9
  ADAM_BETA2 = 0.99
  ADAM_CLIP_THRESHOLD = 1.0
  ADAM_EPSILON = 1e-6
  ADAM_EPSILON_ROOT = 0.0

  # Learning rate schedule
  LR_SCHEDULE = 'linear_rampup_exponential_decay'
  LR_LRED_WARMUP = 4000
  LR_LRED_DECAY_START = 4001
  LR_LRED_DECAY_END = 300000
  LR_LRED_MIN_RATIO = 0.1
  LR_LRED_MAX = 1.0

  LR_COS_MIN_RATIO = 0.1
  LR_COS_MAX = 1.0
  LR_COS_WARMUP = 4000
  LR_COS_DECAY_START = 4001
  LR_COS_DECAY_END = 300000

  def task(self) -> tasks_lib.SingleTask.HParams:
    """Returns the task parameters."""
    task_p = super().task()
    model_p = task_p.model
    model_p.lm_tpl.packed_input = self.PACKED_INPUT  # pytype: disable=attribute-error  # enable-nested-classes

    if self.USE_REPEATED_LAYER:
      stacked_transformer_tpl = model_p.lm_tpl.stacked_transformer_tpl.block  # pytype: disable=attribute-error
    else:
      stacked_transformer_tpl = model_p.lm_tpl.stacked_transformer_tpl  # pytype: disable=attribute-error
    transformer_layer_p = stacked_transformer_tpl.transformer_layer_params_tpl
    transformer_layer_p.tr_atten_tpl.use_bias = self.USE_BIAS

    lp = task_p.train.learner
    lp.loss_name = 'total_loss'
    lp.optimizer = optimizers.Adam.HParams(
        beta1=self.ADAM_BETA1,
        beta2=self.ADAM_BETA2,
        weight_decay=self.WEIGHT_DECAY,
        epsilon=self.ADAM_EPSILON,
        epsilon_root=self.ADAM_EPSILON_ROOT,
        clip_gradient_norm_to_value=self.CLIP_GRADIENT_NORM_TO_VALUE,
        clip_threshold=self.ADAM_CLIP_THRESHOLD)
    lp.optimizer.learning_rate = self.LEARNING_RATE

    if self.LR_SCHEDULE == 'linear_rampup_exponential_decay':
      lp.optimizer.lr_schedule = (
          schedules.LinearRampupExponentialDecay.HParams(
              warmup_steps=self.LR_LRED_WARMUP,
              decay_start=self.LR_LRED_DECAY_START,
              decay_end=self.LR_LRED_DECAY_END,
              min_ratio=self.LR_LRED_MIN_RATIO,
              max=self.LR_LRED_MAX))
    elif self.LR_SCHEDULE == 'linear_rampup_cosine_decay':
      lp.optimizer.lr_schedule = (
          schedules.LinearRampupCosineDecay.HParams(
              warmup_steps=self.LR_COS_WARMUP,
              decay_start=self.LR_COS_DECAY_START,
              decay_end=self.LR_COS_DECAY_END,
              min_ratio=self.LR_COS_MIN_RATIO,
              max=self.LR_COS_MAX))
    else:
      raise NotImplementedError(f'Learning rate schedule {self.LR_SCHEDULE} is '
                                'not supported.')

    return task_p


@experiment_registry.register
class LmCloudSpmdAdam(TransformerLmSpmdAdam, lm_cloud.SyntheticDataset):
  """Base config for an SPMD model."""

  NUM_LAYERS = 2
  MODEL_DIMS = 2048
  HIDDEN_DIMS = MODEL_DIMS * 4
  ACTIVATION_CLS = layers.GELU
  USE_GATED_ACTIVATION = False

  # Autodiff remat.
  CHECKPOINT_POLICY = layers.AutodiffCheckpointType.SAVE_NOTHING

  # Sub-class has to specify a mesh.
  ICI_MESH_SHAPE = [1, 4, 2]


@experiment_registry.register
class C4SpmdAdam(TransformerLmSpmdAdam,
                 C4UnsupervisedDataset):
  r"""Base config for a decoder only transformer."""

  NUM_LAYERS = 24
  NUM_HEADS = 32
  MODEL_DIMS = 2048
  # Known as MLP_DIM in t5x
  HIDDEN_DIMS = MODEL_DIMS * 4
  # Defaults to MODEL_DIMS // NUM_HEADS.
  DIMS_PER_HEAD = None
  # Known as NUM_EMBEDDINGS in t5x
  VOCAB_SIZE = 32128
  ACTIVATION_CLS = layers.GELU
  USE_GATED_ACTIVATION = False

  CHECKPOINT_POLICY = layers.AutodiffCheckpointType.SAVE_DOT_FOR_MLPERF_200B
  CHECKPOINT_EVERY_N_STEPS = 1000

  # Sub-class has to specify a mesh.
  ICI_MESH_SHAPE = [1, 4, 2]

  def task(self) -> tasks_lib.SingleTask.HParams:
    """Returns the task parameters."""
    task_p = super().task()
    model_p = task_p.model  # pytype: disable=attribute-error  # enable-nested-classes
    model_p.decoder_tpl.eos_id = GPT_EOS_ID  # pytype: disable=attribute-error  # enable-nested-classes
    model_p.decoder_tpl.seqlen = self.MAX_SEQ_LEN  # pytype: disable=attribute-error  # enable-nested-classes

    return task_p


@experiment_registry.register
class C4SpmdGpt3Adam(C4SpmdAdam):
  r"""GPT-3 config for a decoder only transformer."""

  MAX_SEQ_LEN = 2048

  NUM_LAYERS = 96
  NUM_HEADS = 96
  MODEL_DIMS = 12288
  # Known as MLP_DIM in t5x
  HIDDEN_DIMS = MODEL_DIMS * 4
  # Defaults to MODEL_DIMS // NUM_HEADS.
  DIMS_PER_HEAD = None
  # Known as NUM_EMBEDDINGS in t5x
  VOCAB_SIZE = 50257

  # Autodiff remat.
  CHECKPOINT_POLICY = layers.AutodiffCheckpointType.SAVE_NOTHING
  CHECKPOINT_EVERY_N_STEPS = 200
  CHECKPOINT_MAX_TO_KEEP = 2

  # 768 replicas with 1.5k global batch size
  PERCORE_BATCH_SIZE = 2
  ICI_MESH_SHAPE = [1, 64, 12]


@experiment_registry.register
class C4SpmdGpt3L16Adam(C4SpmdGpt3Adam):
  r"""a few layers of GPT-3 config for a decoder only transformer."""
  NUM_LAYERS = 16
  USE_REPEATED_LAYER = True
  # pad vocab to TPU-friendly size
  VOCAB_SIZE = 50304

  # 128 replicas with 128 global batch size using fp32
  PERCORE_BATCH_SIZE = 1
  ICI_MESH_SHAPE = [1, 32, 4]


@experiment_registry.register
class C4SpmdGpt3AdamHP(C4SpmdGpt3Adam):
  r"""GPT-3 config for a decoder only transformer."""
  NUM_LAYERS = 96
  USE_REPEATED_LAYER = True
  # pad vocab to TPU-friendly size
  VOCAB_SIZE = 50304

  # 1536 replicas with 1536 global batch size
  PERCORE_BATCH_SIZE = 1
  ICI_MESH_SHAPE = [1, 192, 8]
  FPROP_DTYPE = jnp.bfloat16

  # HPs
  ACTIVATION_CLS = layers.GELU
  USE_GATED_ACTIVATION = False
  LR_SCHEDULE = 'linear_rampup_exponential_decay'
  CLIP_GRADIENT_NORM_TO_VALUE = 1.0

  CHECKPOINT_MAX_TO_KEEP = 10


class C4SpmdGpt3AdamOrgHP(C4SpmdGpt3Adam):
  r"""GPT-3 config with original HPs.

  From the paper & after convergence matching with
  NVIDIA's Megatron-LM framework.
  """
  NUM_LAYERS = 96
  USE_REPEATED_LAYER = True

  # HPs
  ACTIVATION_CLS = layers.GELU
  USE_GATED_ACTIVATION = False
  SEPARATE_EMBEDDING = False
  TRAINABLE_POSITION_EMB = True
  TRAINABLE_PE_MAX_SEQ_LEN = 16384
  ATTEN_LOGIT_CAP = -1.0  # Disable logits cap in atten

  LEARNING_RATE = 6e-5
  WEIGHT_DECAY = 0.1
  ADAM_BETA1 = 0.9
  ADAM_BETA2 = 0.95
  ADAM_EPSILON = 1e-8
  ADAM_CLIP_THRESHOLD = -1.0  # Disable Adam clip_threshold
  CLIP_GRADIENT_NORM_TO_VALUE = 1.0
  LAYERNORM_EPSILON = 1e-5

  LR_SCHEDULE = 'linear_rampup_cosine_decay'
  LR_COS_WARMUP = 265
  LR_COS_DECAY_START = 266
  LR_COS_DECAY_END = 108600
  LR_COS_MAX = 1.0
  LR_COS_MIN_RATIO = 0.1

  # Checkpoint
  EVAL_INTERVAL_STEPS = 100
  SUMMARY_INTERVAL_STEPS = 10
  CHECKPOINT_EVERY_N_STEPS = 100
  CHECKPOINT_MAX_TO_KEEP = 10

  def task(self) -> tasks_lib.SingleTask.HParams:
    """Returns the task parameters."""
    task_p = super().task()

    task_p.model.params_init = WeightInit.Gaussian(0.006)

    softmax_init = WeightInit.Gaussian(0.006)
    task_p.model.lm_tpl.softmax_tpl.params_init = softmax_init
    task_p.model.lm_tpl.softmax_tpl.feed_forward_tpl.has_bias = False

    if self.SEPARATE_EMBEDDING:
      task_p.model.lm_tpl.separate_embedding_tpl.scale_sqrt_depth = False
      task_p.model.lm_tpl.separate_embedding_tpl.lookup_style = 'index'
    else:
      task_p.model.lm_tpl.softmax_tpl.scale_sqrt_depth = False
      task_p.model.lm_tpl.softmax_tpl.lookup_style = 'index'
    if self.TRAINABLE_POSITION_EMB:
      task_p.model.lm_tpl.position_emb_tpl.lookup_style = 'index'

    if self.USE_REPEATED_LAYER:
      stacked_transformer_tpl = task_p.model.lm_tpl.stacked_transformer_tpl.block
    else:
      stacked_transformer_tpl = task_p.model.lm_tpl.stacked_transformer_tpl
    transformer_layer_p = stacked_transformer_tpl.transformer_layer_params_tpl

    transformer_layer_p.ln_tpl.epsilon = self.LAYERNORM_EPSILON
    transformer_layer_p.tr_fflayer_tpl.ln_tpl.epsilon = self.LAYERNORM_EPSILON
    task_p.model.lm_tpl.final_ln_tpl.epsilon = self.LAYERNORM_EPSILON
    transformer_layer_p.tr_atten_tpl.internal_enable_per_dim_scale = False
    transformer_layer_p.tr_atten_tpl.use_bias = True

    transformer_layer_p.tr_fflayer_tpl.activation_tpl.approximate = True

    return task_p


@experiment_registry.register
class C4SpmdGpt3AdamOrgHP1536Replicas(C4SpmdGpt3AdamOrgHP):
  r"""GPT-3 config in fp32 for 1536 replicas with 1536 global batch size."""
  PERCORE_BATCH_SIZE = 1
  ICI_MESH_SHAPE = [1, 64, 24]
  FPROP_DTYPE = jnp.float32
  CHECKPOINT_MAX_TO_KEEP = 100
  EVAL_INTERVAL_STEPS = 25


@experiment_registry.register
class C4SpmdGpt3AdamOrgHP512Replicas(C4SpmdGpt3AdamOrgHP):
  r"""GPT-3 config in bf16 for 512 replicas with 1536 global batch size."""
  PERCORE_BATCH_SIZE = 3
  ICI_MESH_SHAPE = [1, 64, 8]
  FPROP_DTYPE = jnp.bfloat16


@experiment_registry.register
class C4SpmdGpt3AdamOrgHP768Replicas(C4SpmdGpt3AdamOrgHP):
  r"""GPT-3 config in bf16 for 768 replicas with 1536 global batch size."""
  PERCORE_BATCH_SIZE = 2
  ICI_MESH_SHAPE = [1, 64, 12]
  FPROP_DTYPE = jnp.bfloat16
  CHECKPOINT_MAX_TO_KEEP = 100
  EVAL_INTERVAL_STEPS = 25


@experiment_registry.register
class C4SpmdGpt3AdamOrgHP512Replicas2(C4SpmdGpt3AdamOrgHP):
  r"""GPT-3 config in bf16 for 512 replicas with 1536 global batch size."""
  PERCORE_BATCH_SIZE = 3
  ICI_MESH_SHAPE = [1, 32, 16]
  FPROP_DTYPE = jnp.bfloat16


@experiment_registry.register
class C4SpmdGpt3AdamOrgHPBS4k2048Replicas(C4SpmdGpt3AdamOrgHP):
  r"""GPT-3 config in bf16 for 1024 replicas with 4k global batch size."""
  PERCORE_BATCH_SIZE = 2
  ICI_MESH_SHAPE = [1, 128, 16]
  FPROP_DTYPE = jnp.bfloat16


@experiment_registry.register
class C4SpmdGpt3L16AdamOrgHP(C4SpmdGpt3AdamOrgHP):
  r"""Small GPT-3 config in bf16 for 64 replicas with 192 global batch size."""

  NUM_LAYERS = 16
  FPROP_DTYPE = jnp.bfloat16
  PERCORE_BATCH_SIZE = 3
  ICI_MESH_SHAPE = [1, 16, 4]


@experiment_registry.register
class C4SpmdGpt3L16AdamOrgHP128Replicas(C4SpmdGpt3AdamOrgHP):
  r"""Small GPT-3 config in bf16 for 128 replicas with 256 global batch size."""
  NUM_LAYERS = 16
  FPROP_DTYPE = jnp.bfloat16
  PERCORE_BATCH_SIZE = 2
  ICI_MESH_SHAPE = [1, 8, 16]


@experiment_registry.register
class C4SpmdGpt3L16AdamOrgHP96Replicas(C4SpmdGpt3AdamOrgHP):
  r"""Small GPT-3 config in bf16 for 96 replicas with 192 global batch size."""

  NUM_LAYERS = 16
  FPROP_DTYPE = jnp.bfloat16
  PERCORE_BATCH_SIZE = 2
  ICI_MESH_SHAPE = [1, 6, 16]


@experiment_registry.register
class C4SpmdGpt3L16AdamOrgHP64Replicas(C4SpmdGpt3AdamOrgHP):
  r"""Small GPT-3 config in bf16 for 64 replicas with 256 global batch size."""
  NUM_LAYERS = 16
  FPROP_DTYPE = jnp.bfloat16
  PERCORE_BATCH_SIZE = 4
  ICI_MESH_SHAPE = [1, 4, 16]


@experiment_registry.register
class C4SpmdGpt3L16AdamOrgHP32Replicas(C4SpmdGpt3AdamOrgHP):
  r"""Small GPT-3 config in bf16 for 32 replicas with 128 global batch size."""
  NUM_LAYERS = 16
  FPROP_DTYPE = jnp.bfloat16
  PERCORE_BATCH_SIZE = 8
  ICI_MESH_SHAPE = [1, 2, 16]
