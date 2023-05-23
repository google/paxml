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

"""ResNets classifiers on the Imagenet dataset."""
from typing import List

from absl import flags
import jax
import jax.numpy as jnp
from paxml import base_experiment
from paxml import experiment_registry
from paxml import learners
from paxml import tasks_lib
from paxml.tasks.vision import input_generator
from praxis import base_input
from praxis import base_layer
from praxis import layers
from praxis import optimizers
from praxis import pax_fiddle
from praxis import py_utils
from praxis import pytypes
from praxis import schedules


FLAGS = flags.FLAGS
WeightInit = base_layer.WeightInit
NestedMap = py_utils.NestedMap
NestedShapeDtypeStruct = pytypes.NestedShapeDtypeStruct
LayerTpl = pax_fiddle.Config[base_layer.BaseLayer]


class ImageClassificationInputSpecsProvider(base_input.BaseInputSpecsProvider):
  """Encapsulates input specs for image classification tasks."""
  height: int = 224
  width: int = 224
  num_color_channels: int = 3
  num_classes: int = 1000
  batch_size: int = 1

  def get_input_specs(self) -> NestedShapeDtypeStruct:
    """Returns specs from the input pipeline for model init."""
    if self.height is None or self.width is None:
      raise ValueError(
          f'Both `height` (`{self.height}`) and `width` (`{self.width}`) params'
          ' must be set.'
      )
    if self.num_classes is None:
      raise ValueError('Parameter `num_classes` must be set.')
    image_shape = (
        self.batch_size,
        self.height,
        self.width,
        self.num_color_channels,
    )
    return NestedMap(
        eval_sample_weights=jax.ShapeDtypeStruct(
            shape=(self.batch_size,), dtype=jnp.float32
        ),
        image=jax.ShapeDtypeStruct(shape=image_shape, dtype=jnp.float32),
        label_probs=jax.ShapeDtypeStruct(
            shape=(self.batch_size, self.num_classes), dtype=jnp.float32
        ),
        weight=jax.ShapeDtypeStruct(
            shape=(self.batch_size,), dtype=jnp.float32
        ),
    )


@experiment_registry.register(tags=['smoke_test_abstract_init'])
class ResNet50Pjit(base_experiment.BaseExperiment):
  """ResNet-50 baseline.

  The experiment settings follow the paper
  "ImageNet in 1 Hour" (https://arxiv.org/abs/1706.02677).
  Most notable difference is:
  1. We use SyncBN. This hurts accuracy by ~0.3% according to
     Sec.4 of https://arxiv.org/abs/2105.07576. It hurts more
     for larger batch sizes.

  Other differences that likely do not matter:
  1. We use bfloat16. This appears to have no significant effect.
  2. Our LR warmup starts from 0, instead of 0.1.

  """
  # Set input hparams.
  IMAGE_SIZE = 224
  TRAIN_BATCH_SIZE = 512
  EVAL_BATCH_SIZE = 512

  # Set optimization hparams.
  LOSS_NAME = 'avg_xent'
  NUM_EPOCHS = 90
  LEARNING_RATE = (TRAIN_BATCH_SIZE / 256) * 0.1
  L2_REGULARIZER_WEIGHT = 1e-4
  BPROP_VARIABLE_EXCLUSION = []
  CLIP_GRADIENT_NORM_TO_VALUE = 0.0
  EMA_DECAY = 0.0

  # Set checkpointing hparams.
  SAVE_MAX_TO_KEEP = 1_000  # Save more checkpoints for post-analysis.
  SAVE_INTERVAL_STEPS = 5_000
  SUMMARY_INTERVAL_STEPS = 1_000
  EVAL_INTERVAL_STEPS = 1_000
  DECODE_INTERVAL_STEPS = 5_000

  # SPMD related hparams.
  MESH_AXIS_NAMES = ('replica', 'data', 'mdl')
  BATCH_SPLIT_AXES = ('replica', 'data')
  # full data parallelism
  MESH_SHAPE = [1, 8, 1]

  def get_input_specs_provider_params(
      self,
  ) -> pax_fiddle.Config[base_input.BaseInputSpecsProvider]:
    """Returns the hparams of the input specs provider.

    Returns:
      An InputSpecsProvider instance.
    """
    return pax_fiddle.Config(ImageClassificationInputSpecsProvider).set(
        batch_size=self.TRAIN_BATCH_SIZE
    )

  def _dataset_train(self) -> pax_fiddle.Config[base_input.LingvoInputAdaptor]:
    """Returns training data input configs."""
    input_obj = input_generator.ImageNetTrain
    input_p = input_obj.Params()
    input_p.batch_size = self.TRAIN_BATCH_SIZE
    return pax_fiddle.Config(
        base_input.LingvoInputAdaptor, input=input_p, is_training=True
    )

  def _dataset_test(self) -> pax_fiddle.Config[base_input.LingvoInputAdaptor]:
    """Returns test / validation data input configs."""
    input_obj = input_generator.ImageNetValidation
    input_p = input_obj.Params()
    input_p.batch_size = self.EVAL_BATCH_SIZE
    return pax_fiddle.Config(
        base_input.LingvoInputAdaptor, input=input_p, is_training=False
    )

  def datasets(self) -> List[pax_fiddle.Config[base_input.BaseInput]]:
    """Returns a list of dataset configs."""
    return [self._dataset_train(), self._dataset_test()]

  def _network(self) -> LayerTpl:
    net = layers.ResNet.HParamsResNet50()
    # Zero-init BN-gamma in the residual branch, needed to support larger
    # batch size training. See Table 2(b) in "ImageNet in 1 hour".
    # Note that this also requires optimizer.skip_lp_1d_vectors=True due to
    # reparameterization.
    net.block_params = pax_fiddle.Config(
        layers.ResNetBlock, zero_init_residual=True
    )
    return net

  def _optimizer(self) -> pax_fiddle.Config[optimizers.BaseOptimizer]:
    return pax_fiddle.Config(optimizers.ShardedSgd, momentum=0.9, nesterov=True)

  def _lr_schedule(self) -> pax_fiddle.Config[schedules.BaseSchedule]:
    lrs = [1, 0.1, 0.01, 0.001, 0.0]
    epoch_boundaries = [5, 30, 60, 80, 90]
    iters = input_generator.TRAIN_EXAMPLES // self.TRAIN_BATCH_SIZE
    boundaries = [iters * e for e in epoch_boundaries]
    return pax_fiddle.Config(
        schedules.LinearRampupPiecewiseConstant,
        boundaries=boundaries,
        values=lrs,
    )

  def _learner(self) -> pax_fiddle.Config[learners.Learner]:
    lp = pax_fiddle.Config(learners.Learner)
    lp.loss_name = self.LOSS_NAME
    lp.bprop_variable_exclusion = self.BPROP_VARIABLE_EXCLUSION

    lp.optimizer = self._optimizer()
    op = lp.optimizer
    op.lr_schedule = self._lr_schedule()
    op.learning_rate = self.LEARNING_RATE
    lp.optimizer.skip_lp_1d_vectors = True
    # For following hparams, only override if not already set.
    if op.l2_regularizer_weight is None:
      op.l2_regularizer_weight = self.L2_REGULARIZER_WEIGHT
    if op.clip_gradient_norm_to_value == 0.0:
      op.clip_gradient_norm_to_value = self.CLIP_GRADIENT_NORM_TO_VALUE
    if op.ema_decay == 0.0:
      op.ema_decay = self.EMA_DECAY
    return lp

  def _configure_task(
      self, task_p: pax_fiddle.Config[tasks_lib.SingleTask]
  ) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    """Configures commonly used task_p settings."""

    train_p = task_p.train

    # Set learner and optimizer.
    train_p.learner = self._learner()

    # Set sharding annotations.
    if self.MESH_SHAPE is not None:
      model = task_p.model
      model.ici_mesh_shape = self.MESH_SHAPE
      model.dcn_mesh_shape = None
      model.mesh_axis_names = self.MESH_AXIS_NAMES

      batch_split = self.BATCH_SPLIT_AXES
      train_p.inputs_split_mapping = py_utils.NestedMap(
          map_5d=(batch_split, None, None, None, None),
          map_4d=(batch_split, None, None, None),
          map_3d=(batch_split, None, None),
          map_2d=(batch_split, None),
          map_1d=(batch_split,))

    # Set summary, checkpointing and evaluation.
    steps_per_epoch = input_generator.TRAIN_EXAMPLES // self.TRAIN_BATCH_SIZE
    train_p.num_train_steps = steps_per_epoch * 90  # Train for 90 epochs.
    train_p.eval_interval_steps = steps_per_epoch  # Eval every epoch.
    train_p.summary_interval_steps = self.SUMMARY_INTERVAL_STEPS
    train_p.save_interval_steps = self.SAVE_INTERVAL_STEPS
    train_p.decode_interval_steps = self.DECODE_INTERVAL_STEPS
    train_p.save_max_to_keep = self.SAVE_MAX_TO_KEEP
    train_p.summary_accumulate_interval_steps = 1
    train_p.variable_norm_summary = True
    train_p.eval_skip_train = True  # Disable eval of train input data.
    return task_p

  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    """Returns the task configs."""
    resnet = self._network()
    task_p = pax_fiddle.Config(tasks_lib.SingleTask, name='classifier_task')
    task_p.model = pax_fiddle.Config(
        layers.ClassificationModel,
        name='classifier',
        network_tpl=resnet,
        softmax_tpl=pax_fiddle.Config(
            layers.FullSoftmax,
            params_init=WeightInit.Gaussian(scale=0.01),
            input_dims=resnet.channels[-1],  # pytype: disable=attribute-error
            num_classes=1000,
        ),
    )
    task_p = self._configure_task(task_p)
    return task_p
