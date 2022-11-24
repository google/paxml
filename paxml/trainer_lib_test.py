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

"""Tests for trainer_lib."""
import os

from typing import Any, Dict, Tuple, Union

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
from jax.experimental import maps
import numpy as np

from paxml import tasks_lib
from paxml import trainer_lib
from praxis import base_input
from praxis import base_layer
from praxis import base_model
from praxis import layers
from praxis import optimizers
from praxis import py_utils
from praxis import pytypes
from praxis import schedules

BaseModel = base_model.BaseModel
BaseLayer = base_layer.BaseLayer
BaseInput = base_input.BaseInput
NestedMap = py_utils.NestedMap
Predictions = base_model.Predictions

JTensor = pytypes.JTensor
WeightedScalars = pytypes.WeightedScalars

instantiate = base_layer.instantiate


class TestInput(base_input.BaseInput):
  """Input for testing purpose."""

  class HParams(base_input.BaseInput.HParams):
    seq_length: int = 2

  def get_next(self):
    p = self.hparams
    return py_utils.NestedMap(
        image=jnp.zeros((p.batch_size, p.seq_length), dtype=jnp.float32))


class TestModel(base_model.BaseModel):

  class HParams(BaseLayer.HParams):
    layer: BaseLayer.HParams = layers.FeedForward.HParams(
        input_dims=2, output_dims=1)

  def setup(self):
    self.create_child('layer_a', self.hparams.layer)

  def compute_predictions(self, input_batch: NestedMap) -> Predictions:
    return self.layer_a(input_batch['image'])

  def compute_loss(
      self, predictions: Union[JTensor, NestedMap],
      input_batch: NestedMap) -> Tuple[WeightedScalars, Dict[str, Any]]:
    return {'loss': (jnp.sum(predictions), 1)}, NestedMap()

  def decode(self, input_batch: base_model.NestedMap):
    return {'a': (1, 1)}, {}, {}


class TrainLibTest(parameterized.TestCase):
  """Trainer_lib tests under 2 CPU devices."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    # Construct a 1d mesh with 2 devices on x.
    devices = np.array(jax.local_devices()[:2]).reshape((2,))
    cls._mesh = maps.Mesh(devices, ('x'))
    # Set up input data
    train_input_p = TestInput.HParams(batch_size=2)
    train_input_p = trainer_lib.adjust_input_params_for_small_batch(
        train_input_p, cls._mesh)
    cls.train_input = instantiate(train_input_p)
    # Set up the task.
    task_p = tasks_lib.SingleTask.HParams(name='test_task')
    task_p.model = TestModel.HParams(name='test_ffn')
    task_p.model.ici_mesh_shape = cls._mesh.shape
    task_p.model.mesh_axis_names = cls._mesh.axis_names
    lp = task_p.train.learner
    lp.loss_name = 'loss'
    lp.optimizer = optimizers.Adam.HParams()
    lp.optimizer.learning_rate = 0.0
    lp.optimizer.lr_schedule = schedules.Constant.HParams()
    cls.task = instantiate(task_p)

  @parameterized.product(use_auto_shard=[True, False])
  def test_spmd_partitioner_output_spec(self, use_auto_shard):
    prng_key = jax.random.PRNGKey(0)
    train_sample_inputs = self.train_input.get_next()
    # Single-host test, per-host shape/dtype is global.
    inputs_shape_dtype = jax.tree_map(
        lambda x: jax.ShapeDtypeStruct(shape=x.shape, dtype=x.dtype),
        train_sample_inputs)

    train_state_metadata = trainer_lib.create_train_state_metadata(
        self.task, inputs_shape_dtype)

    sharding_info = None
    if not use_auto_shard:
      sharding_info = trainer_lib._SpmdModelPartitioner.ShardingInfo(
          train_state_metadata.partitioned_specs, 2)

    partitioner = trainer_lib._SpmdModelPartitioner(
        self.task,
        self._mesh,
        prng_key,
        inputs_shape_dtype,
        train_inputs_shape_dtype=None,
        sharding_info=sharding_info)
    _, _, train_state_partition_spec = partitioner.partition(
        trainer_lib.train_step_single_learner, is_eval=False)
    if use_auto_shard:
      self.assertIsNotNone(train_state_partition_spec)
      self.assertNotEqual(train_state_metadata.partitioned_specs,
                          train_state_partition_spec)
    else:
      self.assertEqual(train_state_metadata.partitioned_specs,
                       train_state_partition_spec)

  def test_trainer_partitioned_step(self):
    trainer = trainer_lib.SingleTaskPjitTrainer(
        self.task,
        self.train_input,
        mesh=self._mesh,
        enable_auto_sharding=False)
    prng_key = jax.random.PRNGKey(0)

    step_fn, _, train_state_spec = trainer.compile_step(prng_key)

    self.assertIsNotNone(trainer.task)
    self.assertEqual(step_fn, trainer._step_fn)
    self.assertEqual(train_state_spec,
                     trainer.train_state_metadata.partitioned_specs)


if __name__ == '__main__':
  os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=2'
  absltest.main()
