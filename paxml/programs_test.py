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

"""Tests for programs."""
import os
from typing import Any, Dict, Tuple, Union

from absl.testing import absltest
from etils import epath
import jax
from jax import numpy as jnp
import numpy as np
from paxml import partitioning
from paxml import programs
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
from praxis import test_utils

BaseModel = base_model.BaseModel
BaseLayer = base_layer.BaseLayer
BaseInput = base_input.BaseInput
NestedMap = py_utils.NestedMap
PartitionSpec = jax.sharding.PartitionSpec
Predictions = base_model.Predictions
RunningMode = trainer_lib.RunningMode

JTensor = pytypes.JTensor
WeightedScalars = pytypes.WeightedScalars

instantiate = base_layer.instantiate


class TestInput(base_input.BaseInput):
  """Input for testing purpose."""
  seq_length: int = 2

  def get_next(self):
    return NestedMap(
        image=jnp.zeros((self.batch_size, self.seq_length), dtype=jnp.float32)
    )


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

  def compute_loss(  # pytype: disable=signature-mismatch  # jax-ndarray
      self, predictions: Union[JTensor, NestedMap],
      input_batch: NestedMap) -> Tuple[WeightedScalars, Dict[str, Any]]:
    return {'loss': (jnp.sum(predictions), 1)}, NestedMap()  # pytype: disable=bad-return-type  # jax-ndarray

  def decode(self, input_batch: base_model.NestedMap):
    return {'a': (1, 1)}, {}, {}


class ProgramTestBase(test_utils.TestCase):
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
    train_input_p = pax_fiddle.Config(TestInput, batch_size=2)
    cls.train_input = instantiate(train_input_p)

    # Set up the task.
    task_p = pax_fiddle.Config(tasks_lib.SingleTask, name='test_task')
    task_p.model = pax_fiddle.Config(TestModel, name='test_ffn')
    task_p.model.ici_mesh_shape = [2]
    task_p.model.mesh_axis_names = cls.mesh.axis_names
    lp = task_p.train.learner
    lp.loss_name = 'loss'
    lp.optimizer = pax_fiddle.Config(optimizers.Adam)
    lp.optimizer.learning_rate = 0.0
    lp.optimizer.lr_schedule = pax_fiddle.Config(schedules.Constant)
    cls.task = instantiate(task_p)


class SingleTaskPjitTrainProgramTest(ProgramTestBase):

  def test_train_program(self):
    inputs_shape_dtype = jax.tree_map(
        lambda x: jax.ShapeDtypeStruct(shape=x.shape, dtype=x.dtype),
        self.train_input.get_next(),
    )
    partitioner = partitioning.PjitPartitioner(
        init_is_eval=False, reshard_inputs=True, task=self.task
    )
    prng_key = jax.random.PRNGKey(0)
    partitioner.setup(self.task, prng_key, inputs_shape_dtype)
    train_pg = programs.SingleTaskTrainProgram()
    train_pg.setup(
        self.task,
        self.train_input,
        partitioner,
        epath.Path('/tmp'),
        prng_key,
        prng_key,
        0,
    )
    self.assertEqual(2, train_pg.train_unpadded_global_batch_size)


if __name__ == '__main__':
  os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=2'
  absltest.main()
