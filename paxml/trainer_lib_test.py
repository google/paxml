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
import numpy as np
from paxml import tasks_lib
from paxml import trainer_lib
from praxis import base_input
from praxis import base_layer
from praxis import base_model
from praxis import layers
from praxis import optimizers
from praxis import pax_fiddle
from praxis import py_utils
from praxis import pytypes
from praxis import schedules

BaseModel = base_model.BaseModel
BaseLayer = base_layer.BaseLayer
BaseInput = base_input.BaseInput
NestedMap = py_utils.NestedMap
Predictions = base_model.Predictions
RunningMode = trainer_lib.RunningMode

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


@pax_fiddle.auto_config
def _test_model_layer_default():
  return pax_fiddle.Config(layers.FeedForward, input_dims=2, output_dims=1)


class TestModel(base_model.BaseModel):
  layer: pax_fiddle.Config[BaseLayer] = pax_fiddle.fdl_field(
      default_factory=_test_model_layer_default, tags=pax_fiddle.DoNotBuild)

  def setup(self):
    self.create_child('layer_a', self.layer)

  def compute_predictions(self, input_batch: NestedMap) -> Predictions:
    return self.layer_a(input_batch['image'])

  def compute_loss(
      self, predictions: Union[JTensor, NestedMap],
      input_batch: NestedMap) -> Tuple[WeightedScalars, Dict[str, Any]]:
    return {'loss': (jnp.sum(predictions), 1)}, NestedMap()

  def decode(self, input_batch: base_model.NestedMap):
    return {'a': (1, 1)}, {}, {}


class TrainLibTestBase(parameterized.TestCase):
  """Trainer_lib tests under 2 CPU devices."""

  mesh = None
  train_input = None
  task = None

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    # Construct a 1d mesh with 2 devices on x.
    devices = np.array(jax.local_devices()[:2]).reshape((2,))
    cls.mesh = jax.sharding.Mesh(devices, 'x')

    # Set up input data
    train_input_p = TestInput.HParams(batch_size=2)
    train_input_p = trainer_lib.adjust_input_params_for_small_batch(
        train_input_p, cls.mesh
    )
    cls.train_input = instantiate(train_input_p)

    # Set up the task.
    task_p = tasks_lib.SingleTask.HParams(name='test_task')
    task_p.model = pax_fiddle.Config(TestModel, name='test_ffn')
    task_p.model.ici_mesh_shape = [2]
    task_p.model.mesh_axis_names = cls.mesh.axis_names
    lp = task_p.train.learner
    lp.loss_name = 'loss'
    lp.optimizer = optimizers.Adam.HParams()
    lp.optimizer.learning_rate = 0.0
    lp.optimizer.lr_schedule = schedules.Constant.HParams()
    cls.task = instantiate(task_p)


class PjitPartitionerTest(TrainLibTestBase):

  def setUp(self):
    train_sample_inputs = self.train_input.get_next()
    # Single-host test, per-host shape/dtype is global.
    self._inputs_shape_dtype = jax.tree_map(
        lambda x: jax.ShapeDtypeStruct(shape=x.shape, dtype=x.dtype),
        train_sample_inputs,
    )

  def _create_partitioner(
      self, auto_sharding_step_fn=None, auto_sharding_is_eval=False
  ):
    auto_sharding_info = None
    if auto_sharding_step_fn:
      auto_sharding_info = trainer_lib._PjitPartitioner.AutoShardingInfo(
          auto_sharding_step_fn,
          is_eval=auto_sharding_is_eval,
          replicate_output=False,
      )
    return trainer_lib._PjitPartitioner(
        self.task,
        jax.random.PRNGKey(0),
        train_inputs_shape_dtype=self._inputs_shape_dtype,
        init_is_eval=False,
        auto_sharding_info=auto_sharding_info,
    )

  @parameterized.parameters([True, False])
  def test_output_spec(self, use_auto_sharding):
    prng_key = jax.random.PRNGKey(0)
    metadata = trainer_lib.create_train_state_metadata(
        self.task, self._inputs_shape_dtype
    )
    partitioner = self._create_partitioner(
        trainer_lib.train_step_single_learner if use_auto_sharding else None
    )
    train_state_partition_spec = partitioner.get_train_state_metadata(
    ).partition_specs

    if use_auto_sharding:
      self.assertIsNotNone(train_state_partition_spec)
      self.assertNotEqual(metadata.partition_specs, train_state_partition_spec)
    else:
      self.assertEqual(metadata.partition_specs, train_state_partition_spec)

  @parameterized.parameters(
      [RunningMode.TRAIN, RunningMode.EVAL, RunningMode.DECODE]
  )
  def test_cache_auto_sharding_result(self, mode):
    prng_key = jax.random.PRNGKey(0)
    step_fn, is_eval = trainer_lib.get_step_fn(mode)
    partitioner = self._create_partitioner(step_fn, is_eval)
    # TODO(laigd): split the get_train_state_metadata logic to separate test
    # cases.
    metadata_1 = partitioner.get_train_state_metadata(
        discard_opt_states=is_eval
    )
    metadata_2 = partitioner.get_train_state_metadata(
        discard_opt_states=is_eval
    )
    self.assertEqual(metadata_1.partition_specs, metadata_2.partition_specs)

    partitioned_step_fn_1, input_partition_specs_1 = partitioner.partition(
        step_fn, self._inputs_shape_dtype, is_eval
    )
    partitioned_step_fn_2, input_partition_specs_2 = partitioner.partition(
        step_fn, self._inputs_shape_dtype, is_eval
    )
    self.assertIs(partitioned_step_fn_1, partitioned_step_fn_2)
    self.assertIs(input_partition_specs_1, input_partition_specs_2)


class SingleTaskPjitTrainerTest(TrainLibTestBase):

  def test_trainer_partitioned_step(self):
    trainer = trainer_lib.SingleTaskPjitTrainer(
        self.task, self.train_input, mesh=self.mesh, enable_auto_sharding=False
    )
    prng_key = jax.random.PRNGKey(0)

    step_fn, _, train_state_spec = trainer.compile_step(prng_key)

    self.assertIsNotNone(trainer.task)
    self.assertEqual(step_fn, trainer._step_fn)
    self.assertEqual(
        train_state_spec, trainer.train_state_metadata.partition_specs
    )


if __name__ == '__main__':
  os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=2'
  absltest.main()
