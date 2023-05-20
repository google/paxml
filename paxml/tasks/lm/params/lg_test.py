"""Decoder-only language model configurations."""

from typing import List
import jax
from jax import numpy as jnp
from paxml import base_experiment
from paxml import experiment_registry
from paxml import tasks_lib
from paxml.tasks.lm import input_generator
from paxml.tasks.lm import lg_gpt3_pax
from paxml.tasks.lm import model_params
from praxis import base_input
from praxis import layers
from praxis import pax_fiddle
from absl import logging

class LGSyntheticDataset(base_experiment.BaseExperiment):
  """Synthetic LM dataset."""
  PERCORE_BATCH_SIZE = 16
  MAX_SEQ_LEN = 1024

  def _dataset_train(
      self, is_training
  ) -> pax_fiddle.Config[base_input.LingvoInputAdaptor]:
    num_local_devices = jax.local_device_count()
    batch_size = round(self.PERCORE_BATCH_SIZE * num_local_devices)
    input_p = lg_gpt3_pax.DataBuild.Params()
    if is_training:
      input_p.batch_size = batch_size
    else:
      # TODO(zhangqiaorjc): Is this batch size too big for test?
      input_p.batch_size = batch_size
    input_p.seq_len = self.MAX_SEQ_LEN
    input_p.file_pattern="tfrecord:gs://yejingxin-us-central2/external/lg/dummy-data/train/*.tfrecords"
    input_p.file_parallelism = 16
    input_p.file_buffer_size = 16  # janghoon.han
    input_p.num_batcher_threads = 16
    input_p.file_random_seed =0
    p = pax_fiddle.Config(
        base_input.LingvoInputAdaptor, name='train_dataset', input=input_p, is_training=is_training
    )
    return p
  
  def _dataset_eval(
    self, is_training
  )  -> pax_fiddle.Config[base_input.LingvoEvalAdaptor]:
    num_local_devices = jax.local_device_count()
    batch_size = round(self.PERCORE_BATCH_SIZE * num_local_devices)
    input_p = lg_gpt3_pax.DataBuild.Params()
    if is_training:
      input_p.batch_size = batch_size
    else:
      # TODO(zhangqiaorjc): Is this batch size too big for test?
      input_p.batch_size = batch_size
    input_p.seq_len = self.MAX_SEQ_LEN
    input_p.file_pattern="tfrecord:gs://yejingxin-us-central2/external/lg/dummy-data/eval/*.tfrecords"
    input_p.file_parallelism = 1
    input_p.file_buffer_size = 1  # janghoon.han
    input_p.file_random_seed =0
    p = pax_fiddle.Config(
        base_input.LingvoEvalAdaptor, input=input_p, name='eval_dataset', 
        is_training=is_training, reset_for_eval=True, batch_size=input_p.batch_size
    )
    logging.info(f"LingvoEvalAdaptor.get_batch_size={base_input.LingvoEvalAdaptor.get_batch_size(p)}")
    logging.info(f"LingvoEvalAdaptor.num_samples={base_input.LingvoEvalAdaptor.num_samples}")
    logging.info(f"LingvoEvalAdaptor.num_infeed_hosts={base_input.LingvoEvalAdaptor.num_infeed_hosts}")
    return p

  def datasets(self) -> List[pax_fiddle.Config[base_input.BaseInput]]:
    """Returns a list of dataset parameters."""
    return [
        self._dataset_train(is_training=True),
        self._dataset_eval(is_training=False)
    ]

@experiment_registry.register
class LGLmLayers10(lg_gpt3_pax.DenseLMTemplateLG, LGSyntheticDataset):
  """Base config for an SPMD model."""

  NUM_LAYERS = 10
#   MODEL_DIMS = 2048
#   HIDDEN_DIMS = MODEL_DIMS * 4
#   ACTIVATION_CLS = layers.GELU
#   USE_GATED_ACTIVATION = False

  USE_REPEATED_LAYER = True

  # Autodiff remat.
  CHECKPOINT_POLICY = layers.AutodiffCheckpointType.SAVE_NOTHING

  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    """Returns the task parameters."""
    task_p = super().task()
    model_params.set_default_adam(task_p, self.LEARNING_RATE, self.WEIGHT_DECAY)
    task_p.train.learner.repeat_prefix_sep = '_'
    task_p.train.num_train_steps = 2
    return task_p
  
@experiment_registry.register
class LGDenseLmTiny(lg_gpt3_pax.DenseLMTemplateLG, LGSyntheticDataset):
  """Base config for an SPMD model."""

  NUM_LAYERS = 1
  MODEL_DIMS = 512
  HIDDEN_DIMS = MODEL_DIMS * 4
#   ACTIVATION_CLS = layers.GELU
#   USE_GATED_ACTIVATION = False
  NUM_HEADS = 4
  DIMS_PER_HEAD = MODEL_DIMS // NUM_HEADS

  USE_REPEATED_LAYER = True

  # Autodiff remat.
  CHECKPOINT_POLICY = layers.AutodiffCheckpointType.SAVE_NOTHING

  CHECKPOINT_EVERY_N_STEPS = 1
  SUMMARY_INTERVAL_STEPS = 1

  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    """Returns the task parameters."""
    task_p = super().task()
    model_params.set_default_adam(task_p, self.LEARNING_RATE, self.WEIGHT_DECAY)
    task_p.train.learner.repeat_prefix_sep = '_'
    task_p.train.num_train_steps = 200
    return task_p