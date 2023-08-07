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

"""Unit tests for tasks_lib."""

from __future__ import annotations

import re
from typing import Tuple

from absl.testing import absltest
from absl.testing import parameterized
import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from paxml import checkpoints
from paxml import learners
from paxml import partitioning
from paxml import tasks_lib
from paxml import trainer_lib
from praxis import base_hyperparams
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


CheckpointType = checkpoints.CheckpointType
NestedMap = py_utils.NestedMap
WeightHParams = base_layer.WeightHParams
JTensor = pytypes.JTensor
WeightInit = base_layer.WeightInit

PMAP_PARALLEL_AXIS_NAME = base_layer.PMAP_PARALLEL_AXIS_NAME

RANDOM = base_layer.RANDOM
PARAMS = base_layer.PARAMS

instantiate = base_hyperparams.instantiate


def get_model_inputs():
  inputs = NestedMap(
      ids=np.zeros([16, 100], dtype=np.int32),
      labels=np.zeros([16, 100], dtype=np.int32),
      paddings=np.zeros([16, 100], dtype=np.float32),
      weights=np.zeros([16, 100], dtype=np.float32))
  return inputs


class CustomInputSpecsProvider(base_input.BaseInputSpecsProvider):
  """Class to provide input specs for model initialization."""
  input_dims: int = 0

  def get_input_specs(self) -> pytypes.NestedShapeDtypeStruct:
    batch_size = 1

    return NestedMap(
        inputs=jax.ShapeDtypeStruct(
            (batch_size, self.input_dims), dtype=jnp.float32
        )
    )


class LMInputSpecsProvider(base_input.BaseInputSpecsProvider):
  """Class to provide input specs for model initialization."""

  def get_input_specs(self):
    return jax.tree_map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype),
                        get_model_inputs())


class TestModel01(base_model.BaseModel):
  """Simple model for testing.

  Attributes:
    input_dims: Depth of the input.
    output_dims: Depth of the output.
  """
  input_dims: int = 0
  output_dims: int = 0

  def setup(self) -> None:
    self.create_variable(
        'var01',
        base_layer.WeightHParams(shape=[self.input_dims, self.output_dims]),
    )

  def compute_predictions(self, input_batch: NestedMap) -> JTensor:
    ret = jnp.einsum('bi,io->bo', input_batch.inputs, self.theta.var01)
    self.add_summary('debug', ret, verbosity=4)
    self.add_summary('info', ret, verbosity=3)
    return ret

  def compute_loss(self, predictions: JTensor,
                   input_batch: NestedMap) -> Tuple[NestedMap, NestedMap]:
    del input_batch
    loss = jnp.sum(predictions)
    loss02 = jnp.max(jnp.abs(self.theta.var01))
    # Here loss is the main loss to back-prop into, and loss02 is an eval
    # metric.
    per_example_out = NestedMap()
    return NestedMap(
        loss=(loss, jnp.array(1.0, loss.dtype)),
        loss02=(loss02, jnp.array(1.0, loss02.dtype))), per_example_out


class TestModel02(base_model.BaseModel):
  """Similar as above but with two variables.

  Attributes:
    input_dims: Depth of the input.
    output_dims: Depth of the output.
  """

  input_dims: int = 0
  output_dims: int = 0

  def setup(self) -> None:
    sub_p = pax_fiddle.Config(layers.FeedForward, input_dims=2, output_dims=2)
    p = pax_fiddle.Config(
        layers.Repeat, name='repeated_ffn', sub_tpl=sub_p, x_times=3
    )
    self.create_child('repeated_ffn', p)
    self.create_variable(
        'var01',
        base_layer.WeightHParams(shape=[self.input_dims, self.output_dims]),
    )

  def compute_predictions(self, input_batch: NestedMap) -> JTensor:
    return self.repeated_ffn(input_batch.inputs)

  def compute_loss(
      self, predictions: JTensor, input_batch: NestedMap
  ) -> Tuple[NestedMap, NestedMap]:
    del input_batch
    loss = jnp.sum(predictions)
    per_example_out = NestedMap()
    return NestedMap(loss=(loss, jnp.array(1.0, loss.dtype))), per_example_out


class TestModel03(base_model.BaseModel):
  """Simple model for testing.

  Attributes:
    input_dims: Depth of the input.
    output_dims: Depth of the output.
  """
  input_dims: int = 0
  output_dims: int = 0

  def setup(self) -> None:
    self.create_variable(
        'var01',
        base_layer.WeightHParams(shape=[self.input_dims, self.output_dims]),
    )
    self.create_variable(
        'var02',
        base_layer.WeightHParams(shape=[self.output_dims, self.input_dims]),
    )

  def compute_predictions(self, input_batch: NestedMap) -> JTensor:
    x = jnp.einsum('bi,io->bo', input_batch.inputs, self.theta.var01)
    return jnp.einsum('bo,oi->bi', x, self.theta.var02)

  def compute_loss(
      self, predictions: JTensor, input_batch: NestedMap
  ) -> Tuple[NestedMap, NestedMap]:
    del input_batch
    loss = jnp.sum(predictions)
    loss02 = jnp.max(jnp.abs(self.theta.var01))
    # Here loss is the main loss to back-prop into, and loss02 is an eval
    # metric.
    per_example_out = NestedMap()
    return (
        NestedMap(
            loss=(loss, jnp.array(1.0, loss.dtype)),
            loss02=(loss02, jnp.array(1.0, loss02.dtype)),
        ),
        per_example_out,
    )


class BaseTaskTest(test_utils.TestCase):

  def test_dataclass_params(self):
    dataclass_params = pax_fiddle.Config(tasks_lib.SingleTask, name='foo_task')
    self.assertIsInstance(dataclass_params, pax_fiddle.Config)
    self.assertIsInstance(dataclass_params.train,
                          tasks_lib.SingleTask.TrainHParams)
    self.assertIsInstance(dataclass_params.vn,
                          tasks_lib.SingleTask.VariationalNoiseHParams)

  def test_mutate_nested_dataclass_params(self):
    """Tests that nested dataclass settings are copied to Lingvo params."""
    task_p = pax_fiddle.Config(tasks_lib.SingleTask, name='foo_task')
    task_p.vn.vn_scale = 0.075
    task_p.vn.vn_regex = 'dec_embedding/emb_var|decoder/cell'
    self.assertEqual(task_p.vn.vn_scale, 0.075)
    self.assertEqual(task_p.vn.vn_regex, 'dec_embedding/emb_var|decoder/cell')

  @parameterized.named_parameters(
      ('summary_verbosity_default', 3),
      ('summary_verbosity_more_verbose', 4),
  )
  def test_model_linear_regression_vn(self, summary_verbosity):
    # Set up the model.
    input_dims = 52
    output_dims = 32
    vn_start_step = 5
    task_p = pax_fiddle.Config(tasks_lib.SingleTask, name='task')
    task_p.model = pax_fiddle.Config(
        TestModel01, name='mdl', input_dims=input_dims, output_dims=output_dims
    )
    task_p.model.params_init = WeightInit.Constant(0.0)

    # Use VN on all params
    task_p.vn.vn_scale = 1.0
    task_p.vn.vn_regex = '.*'
    task_p.vn.vn_start_step = vn_start_step

    lp = task_p.train.learner
    lp.loss_name = 'loss'
    lp.optimizer = pax_fiddle.Config(optimizers.Adam)
    # we set learning rate to 0.0 to check if it is changed by VN
    lp.optimizer.learning_rate = 0.0
    lp.optimizer.lr_schedule = pax_fiddle.Config(schedules.Constant)

    # Create the mdl.
    jax_task = instantiate(task_p)
    jax_task.summary_verbosity = summary_verbosity
    prng_key = jax.random.PRNGKey(12345)
    prng_key, init_key = jax.random.split(prng_key)
    sample_inputs = NestedMap(
        inputs=jnp.ones((1, input_dims), dtype=jnp.float32))
    replicated_mdl_states = trainer_lib.initialize_replicate_model_state(
        jax_task, init_key, sample_inputs)

    def train_step(states, prng_key, inputs):
      return trainer_lib.train_step_single_learner(jax_task, states, prng_key,
                                                   inputs)

    def eval_step(states, prng_key, inputs):
      states = states.to_eval_state()
      return trainer_lib.eval_step_single_learner(jax_task, states, prng_key,
                                                  inputs)

    num_devices = jax.local_device_count()
    batch_size = 4
    mdl_inputs = NestedMap(inputs=np.random.normal(
        size=[num_devices, batch_size, input_dims]).astype(np.float32))
    prng_key, train_key, eval_key = jax.random.split(prng_key, 3)
    train_prng_key = jax.random.split(train_key, num=num_devices)
    eval_prng_key = jax.random.split(eval_key, num=num_devices)

    p_train_step = jax.pmap(
        train_step, donate_argnums=(0,), axis_name=PMAP_PARALLEL_AXIS_NAME)
    p_eval_step = jax.pmap(eval_step, axis_name=PMAP_PARALLEL_AXIS_NAME)

    # Train the model for one single step.
    for step in range(10):
      replicated_mdl_states, train_outputs = p_train_step(
          replicated_mdl_states, train_prng_key, mdl_inputs
      )
      metrics = train_outputs.weighted_scalars
      summary_tensors = train_outputs.summary_tensors

      if step >= vn_start_step:
        # The VN is applied for training
        self.assertGreater(np.array(metrics['loss02'])[0], 0.0)
      else:
        # The VN is not applied for training
        self.assertEqual(np.array(metrics['loss02'])[0], 0.0)

      # The parameter is not changed by VN when lr = 0.0
      param = replicated_mdl_states.mdl_vars[PARAMS]['var01'][0]
      self.assertAllClose(param, np.zeros((input_dims, output_dims)), atol=1e-5)

      # Assert summaries logged at summary_verbosity levels are present:
      self.assertEqual(bool(summary_tensors), True)
      self.assertIn('info_scalar', summary_tensors)
      if summary_verbosity == 4:
        self.assertIn('debug_scalar', summary_tensors)

    _, eval_outputs = p_eval_step(
        replicated_mdl_states, eval_prng_key, mdl_inputs
    )

    # The VN is not applied for eval
    self.assertEqual(np.array(eval_outputs.weighted_scalars['loss02'])[0], 0.0)

  def test_model_linear_regression_ema(self):
    # Set up the model.
    input_dims = 52
    output_dims = 32
    decay = 0.9999

    task_p = pax_fiddle.Config(tasks_lib.SingleTask, name='task')
    task_p.model = pax_fiddle.Config(
        TestModel01, name='mdl', input_dims=input_dims, output_dims=output_dims
    )

    lp = task_p.train.learner
    lp.loss_name = 'loss'
    lp.optimizer = pax_fiddle.Config(optimizers.Adam)
    lp.optimizer.learning_rate = 5.0
    lp.optimizer.lr_schedule = pax_fiddle.Config(schedules.Constant)

    lp.optimizer.ema_decay = decay

    # Create the mdl.
    jax_task = instantiate(task_p)
    prng_key = jax.random.PRNGKey(12345)
    prng_key, init_key = jax.random.split(prng_key)
    sample_inputs = NestedMap(
        inputs=jnp.ones((1, input_dims), dtype=jnp.float32))
    replicated_mdl_states = trainer_lib.initialize_replicate_model_state(
        jax_task, init_key, sample_inputs)

    def find_ema(model_states):
      for i in range(len(model_states.opt_states[0])):
        if 'ema' in model_states.opt_states[0][i]:
          return model_states.opt_states[0][i].ema
      raise Exception('Coundn\'t find EMA from train state.')

    def train_step(states, prng_key, inputs):
      return trainer_lib.train_step_single_learner(jax_task, states, prng_key,
                                                   inputs)

    num_devices = jax.local_device_count()
    batch_size = 4
    mdl_inputs = NestedMap(inputs=np.random.normal(
        size=[num_devices, batch_size, input_dims]).astype(np.float32))
    prng_key, train_key = jax.random.split(prng_key, 2)
    train_prng_key = jax.random.split(train_key, num=num_devices)

    p_train_step = jax.pmap(
        train_step, donate_argnums=(0,), axis_name=PMAP_PARALLEL_AXIS_NAME)

    shallow_state = replicated_mdl_states.mdl_vars[PARAMS]['var01']

    # Train the model for one single step.
    for step in range(10):
      self.assertAllClose(shallow_state,
                          find_ema(replicated_mdl_states)[PARAMS]['var01'])

      param = replicated_mdl_states.mdl_vars[PARAMS]['var01']

      decay_rate = min(decay, (1 + step) / (10 + step))
      shallow_state = decay_rate * shallow_state + (1. - decay_rate) * param

      replicated_mdl_states, _ = p_train_step(
          replicated_mdl_states, train_prng_key, mdl_inputs
      )


class ExternalCheckpointLoaderTest(test_utils.TestCase):

  @parameterized.named_parameters(
      ('partial_load_opt_states', True, False, False),
      ('load_no_opt_states', False, False, False),
      ('load_all_opt_states', False, True, False),
      ('partial_load_opt_states_opt_native', True, False, True),
      ('load_no_opt_states_opt_native', False, False, True),
      ('load_all_opt_states_opt_native', False, True, True),
  )
  def test_load(
      self, partial_load_opt_states, load_opt_states, use_native_optax_opt
  ):
    input_dims = 3
    output_dims = 5

    # Initialize external task and save checkpoint
    ext_task_p = pax_fiddle.Config(tasks_lib.SingleTask, name='task')
    ext_task_p.model = pax_fiddle.Config(
        TestModel01,
        name='mdl_ext',
        input_dims=input_dims,
        output_dims=output_dims,
    )
    lp = ext_task_p.train.learner
    lp.loss_name = 'loss'

    if use_native_optax_opt:
      lp.optimizer = pax_fiddle.Config(optimizers.OptaxOptimizer)
      lp.optimizer.lr_schedule = pax_fiddle.Config(schedules.Constant)
      lr_scdule_inst = instantiate(lp.optimizer.lr_schedule)

      def lr_schedule(step):
        return lr_scdule_inst.value_at(step) * lp.optimizer.learning_rate

      lp.optimizer.grad_tx = optax.adam(
          learning_rate=lr_schedule, b1=0.9, b2=0.99, eps=0.1
      )
      # TODO(b/277132394): Add parameterized test for prefix vectorization.
      lp.vectorize_on_repeat_prefix = False
    else:
      lp.optimizer = pax_fiddle.Config(optimizers.Adam)
      lp.optimizer.lr_schedule = pax_fiddle.Config(schedules.Constant)

    # Enable ema
    lp.optimizer.ema_decay = 0.9999

    sample_inputs = NestedMap(
        inputs=jnp.ones((1, input_dims), dtype=jnp.float32)
    )
    ext_task = instantiate(ext_task_p)
    ext_train_state = trainer_lib.initialize_replicate_model_state(
        ext_task, jax.random.PRNGKey(0), sample_inputs
    )
    ext_train_state_metadata = trainer_lib.create_train_state_metadata(
        ext_task, sample_inputs
    )
    if use_native_optax_opt:
      opt_state_shape = (
          ext_train_state.opt_states[0][2][0].mu['params']['var01'].shape
      )
      ext_train_state.opt_states[0][2][0].mu['params']['var01'] = jnp.array(
          np.random.normal(size=opt_state_shape)
      )
      ext_train_state.opt_states[0][2][0].nu['params']['var01'] = jnp.array(
          np.random.normal(size=opt_state_shape)
      )
    else:
      opt_state_shape = ext_train_state.opt_states[0][2]['m']['params'][
          'var01'
      ].shape
      ext_train_state.opt_states[0][2]['m']['params']['var01'] = jnp.array(
          np.random.normal(size=opt_state_shape)
      )
      ext_train_state.opt_states[0][2]['v']['params']['var01'] = jnp.array(
          np.random.normal(size=opt_state_shape)
      )

    # Modify var01 to be random
    var_shape = ext_train_state.mdl_vars['params']['var01'].shape
    random_var = jnp.array(np.random.normal(size=var_shape))
    ext_train_state.mdl_vars['params']['var01'] = random_var

    tempdir = self.create_tempdir()
    checkpoints.save_checkpoint(
        ext_train_state,
        tempdir.full_path,
        train_state_unpadded_shape_dtype_struct=(
            ext_train_state_metadata.unpadded_global_shapes
        ),
    )

    # Create task with warm-start
    task_p = pax_fiddle.Config(tasks_lib.SingleTask, name='task')
    task_p.model = pax_fiddle.Config(
        TestModel01, name='mdl', input_dims=input_dims, output_dims=output_dims
    )
    task_p.train.learner = lp.clone()
    load_rules = [(r'params/(.*)', 'params/{}')]
    if partial_load_opt_states:
      if use_native_optax_opt:
        load_rules.extend([
            (r'(.*)mu/params/(.*)', '{}mu/params/{}'),
            (r'(.*)nu/params/(.*)', '{}nu/params/{}'),
        ])
      else:
        load_rules.extend([
            (r'(.*)m/params/(.*)', '{}m/params/{}'),
            (r'(.*)v/params/(.*)', '{}v/params/{}'),
        ])
    task_p.train.init_from_checkpoint_rules = {
        tempdir.full_path: tasks_lib.CheckpointLoadingRules(
            task_p=ext_task_p,
            load_rules=load_rules,
            input_specs_provider_p=pax_fiddle.Config(
                CustomInputSpecsProvider, input_dims=input_dims
            ),
            load_opt_states=load_opt_states,
            partial_load_opt_states=partial_load_opt_states,
        ),
    }
    task = instantiate(task_p)

    # Now initialize also includes warm start (loading from ckpt)
    train_state = trainer_lib.initialize_replicate_model_state(
        task, jax.random.PRNGKey(1), sample_inputs
    )

    self.assertAllClose(
        ext_train_state.mdl_vars['params']['var01'],
        train_state.mdl_vars['params']['var01'][0],
    )

    if partial_load_opt_states:
      if use_native_optax_opt:
        self.assertAllClose(
            ext_train_state.opt_states[0][2][0].mu['params']['var01'],
            train_state.opt_states[0][2][0].mu['params']['var01'][0],
        )
        self.assertAllClose(
            ext_train_state.opt_states[0][2][0].nu['params']['var01'],
            train_state.opt_states[0][2][0].nu['params']['var01'][0],
        )
      else:
        self.assertAllClose(
            ext_train_state.opt_states[0][2]['m']['params']['var01'],
            train_state.opt_states[0][2]['m']['params']['var01'][0],
        )
        self.assertAllClose(
            ext_train_state.opt_states[0][2]['v']['params']['var01'],
            train_state.opt_states[0][2]['v']['params']['var01'][0],
        )

    # When loading opt states from a checkpoint, ema is not auto-initialized to
    # mdl_vars by default.
    if not partial_load_opt_states and not load_opt_states:
      for v in train_state.opt_states[0]:
        if 'ema' in v:
          self.assertAllClose(
              ext_train_state.mdl_vars['params']['var01'],
              v.ema['params']['var01'][0],
          )

  @parameterized.named_parameters(
      ('partial_load_opt_states', True, False),
      ('load_all_opt_states', False, True),
  )
  def test_load_checkpoint_from_base_optmizer_to_optax_optimizer(
      self, partial_load_opt_states, load_opt_states
  ):
    input_dims = 3
    output_dims = 5

    # Initialize external task and save checkpoint
    ext_task_p = pax_fiddle.Config(tasks_lib.SingleTask, name='task')
    ext_task_p.model = pax_fiddle.Config(
        TestModel01,
        name='mdl_ext',
        input_dims=input_dims,
        output_dims=output_dims,
    )
    lp = ext_task_p.train.learner
    lp.loss_name = 'loss'

    lp.optimizer = pax_fiddle.Config(optimizers.Adam)
    lp.optimizer.lr_schedule = pax_fiddle.Config(schedules.Constant)

    # Enable ema
    lp.optimizer.ema_decay = 0.9999

    sample_inputs = NestedMap(
        inputs=jnp.ones((1, input_dims), dtype=jnp.float32))
    ext_task = instantiate(ext_task_p)
    ext_train_state = trainer_lib.initialize_replicate_model_state(
        ext_task, jax.random.PRNGKey(0), sample_inputs
    )
    ext_train_state_metadata = trainer_lib.create_train_state_metadata(
        ext_task, sample_inputs
    )

    opt_state_shape = ext_train_state.opt_states[0][2]['m']['params'][
        'var01'].shape
    ext_train_state.opt_states[0][2]['m']['params']['var01'] = jnp.array(
        np.random.normal(size=opt_state_shape))
    ext_train_state.opt_states[0][2]['v']['params']['var01'] = jnp.array(
        np.random.normal(size=opt_state_shape))

    # Modify var01 to be random
    var_shape = ext_train_state.mdl_vars['params']['var01'].shape
    random_var = jnp.array(np.random.normal(size=var_shape))
    ext_train_state.mdl_vars['params']['var01'] = random_var

    tempdir = self.create_tempdir()
    checkpoints.save_checkpoint(
        ext_train_state,
        tempdir.full_path,
        train_state_unpadded_shape_dtype_struct=(
            ext_train_state_metadata.unpadded_global_shapes
        ),
    )

    # Create task with warm-start
    task_p = pax_fiddle.Config(tasks_lib.SingleTask, name='task')
    task_p.model = pax_fiddle.Config(
        TestModel01, name='mdl', input_dims=input_dims, output_dims=output_dims
    )
    # Create learner with new optax optimizer
    lp = task_p.train.learner
    lp.loss_name = 'loss'

    lp.optimizer = pax_fiddle.Config(optimizers.OptaxOptimizer)
    lp.optimizer.lr_schedule = pax_fiddle.Config(schedules.Constant)
    lr_scdule_inst = instantiate(lp.optimizer.lr_schedule)

    def lr_schedule(step):
      return lr_scdule_inst.value_at(step) * lp.optimizer.learning_rate

    lp.optimizer.grad_tx = optax.adam(
        learning_rate=lr_schedule, b1=0.9, b2=0.99, eps=0.1
    )
    # TODO(b/277132394): Add parameterized test for prefix vectorization.
    lp.vectorize_on_repeat_prefix = False

    # Enable ema
    lp.optimizer.ema_decay = 0.9999
    load_rules = [(r'params/(.*)', 'params/{}')]

    load_rules.extend([
        (r'(.*)/0/mu/params/(.*)', '{}/m/params/{}'),
        (r'(.*)/0/nu/params/(.*)', '{}/v/params/{}'),
    ])
    task_p.train.init_from_checkpoint_rules = {
        tempdir.full_path: tasks_lib.CheckpointLoadingRules(
            task_p=ext_task_p,
            load_rules=load_rules,
            input_specs_provider_p=pax_fiddle.Config(
                CustomInputSpecsProvider, input_dims=input_dims
            ),
            load_opt_states=load_opt_states,
            partial_load_opt_states=partial_load_opt_states,
        ),
    }
    task = instantiate(task_p)

    # Now initialize also includes warm start (loading from ckpt)
    train_state = trainer_lib.initialize_replicate_model_state(
        task, jax.random.PRNGKey(1), sample_inputs)

    self.assertAllClose(ext_train_state.mdl_vars['params']['var01'],
                        train_state.mdl_vars['params']['var01'][0])

    self.assertAllClose(
        ext_train_state.opt_states[0][2]['m']['params']['var01'],
        train_state.opt_states[0][2][0].mu['params']['var01'][0],
    )
    self.assertAllClose(
        ext_train_state.opt_states[0][2]['v']['params']['var01'],
        train_state.opt_states[0][2][0].nu['params']['var01'][0],
    )

  @parameterized.named_parameters(
      ('partial_load_opt_states', True, False, False),
      ('load_no_opt_states', False, False, False),
      ('load_all_opt_states', False, True, False),
      ('restore_from_local_checkpoint', False, False, True),
  )
  def test_load_provenance(
      self,
      partial_load_opt_states,
      load_opt_states,
      load_from_cp_train_state,
  ):
    input_dims = 3
    output_dims = 5

    # Initialize external task and save checkpoint
    ext_task_p = pax_fiddle.Config(tasks_lib.SingleTask, name='task')
    ext_task_p.model = pax_fiddle.Config(
        TestModel01,
        name='mdl_ext',
        input_dims=input_dims,
        output_dims=output_dims,
    )
    lp = ext_task_p.train.learner
    lp.loss_name = 'loss'
    lp.optimizer = pax_fiddle.Config(optimizers.Adam)
    lp.optimizer.lr_schedule = pax_fiddle.Config(schedules.Constant)

    # Enable ema
    lp.optimizer.ema_decay = 0.9999

    sample_inputs = NestedMap(
        inputs=jnp.ones((1, input_dims), dtype=jnp.float32)
    )
    ext_task = instantiate(ext_task_p)
    ext_train_state = trainer_lib.initialize_replicate_model_state(
        ext_task, jax.random.PRNGKey(0), sample_inputs
    )
    ext_train_state_metadata = trainer_lib.create_train_state_metadata(
        ext_task, sample_inputs
    )

    tempdir = self.create_tempdir()
    checkpoints.save_checkpoint(
        ext_train_state,
        tempdir.full_path,
        train_state_unpadded_shape_dtype_struct=(
            ext_train_state_metadata.unpadded_global_shapes
        ),
    )

    # Create task with warm-start
    task_p = pax_fiddle.Config(tasks_lib.SingleTask, name='task')
    task_p.model = pax_fiddle.Config(
        TestModel01, name='mdl', input_dims=input_dims, output_dims=output_dims
    )
    task_p.train.learner = lp.clone()
    load_rules = [(r'params/(.*)', 'params/{}')]
    if partial_load_opt_states:
      load_rules.extend([(r'(.*)m/params/(.*)', '{}m/params/{}')])
    task_p.train.init_from_checkpoint_rules = {
        tempdir.full_path: tasks_lib.CheckpointLoadingRules(
            task_p=ext_task_p,
            load_rules=load_rules,
            input_specs_provider_p=pax_fiddle.Config(
                CustomInputSpecsProvider, input_dims=input_dims
            ),
            load_opt_states=load_opt_states,
            partial_load_opt_states=partial_load_opt_states,
        ),
    }
    task = instantiate(task_p)

    partitioner = partitioning.create_partitioner(
        task,
        reshard_inputs=False,
        auto_sharding_mode=True,
    )
    prng_key = jax.random.PRNGKey(1)
    partitioner.setup(task, prng_key, sample_inputs)
    if load_from_cp_train_state:
      train_state = trainer_lib.initialize_replicate_model_state(
          task, prng_key, sample_inputs
      )
    else:
      train_state = None
    _, _, train_state_provenance = (
        partitioner.initialize_prng_key_and_train_state(
            prng_key,
            train_state,
            CheckpointType.FLAX,
        )
    )

    # train_state_provenance should be None if restoring train_state
    # from an existing checkpoint i.e. in the case of preemption
    if load_from_cp_train_state:
      self.assertIsNone(train_state_provenance)
      return

    var_provenance_serialized = flax.serialization.to_state_dict(
        train_state_provenance.mdl_vars
    )
    opt_provenance_serialized = flax.serialization.to_state_dict(
        train_state_provenance.opt_states
    )
    mdl_var01 = (
        repr(var_provenance_serialized['params']['var01'])
        .replace('"(', '')
        .replace(')"', '')
    )
    self.assertEqual(
        mdl_var01,
        tempdir.full_path + ':latest',
    )
    opt_state_v_params_var01 = (
        repr(opt_provenance_serialized['0']['2']['v']['params']['var01'])
        .replace('"(', '')
        .replace(')"', '')
    )
    opt_state_m_params_var01 = (
        repr(opt_provenance_serialized['0']['2']['m']['params']['var01'])
        .replace('"(', '')
        .replace(')"', '')
    )

    # loads opt_states in its entirety
    if load_opt_states and not partial_load_opt_states:
      self.assertEqual(
          opt_state_m_params_var01,
          tempdir.full_path + ':latest',
      )
      self.assertEqual(
          opt_state_v_params_var01,
          tempdir.full_path + ':latest',
      )

    # only load r'(.*)m/params/(.*)' opt_states
    if partial_load_opt_states:
      self.assertEqual(
          opt_state_m_params_var01,
          tempdir.full_path + ':latest',
      )
      self.assertEqual(
          opt_state_v_params_var01,
          'random_init',
      )

    # don't load any opt_states
    if not partial_load_opt_states and not load_opt_states:
      self.assertEqual(
          opt_state_m_params_var01,
          'random_init',
      )
      self.assertEqual(
          opt_state_v_params_var01,
          'random_init',
      )

  def test_load_with_incorrect_loading_rules(self):
    input_dims = 3
    output_dims = 5

    # Initialize external task and save checkpoint
    ext_task_p = pax_fiddle.Config(tasks_lib.SingleTask, name='task')
    ext_task_p.model = pax_fiddle.Config(
        TestModel01,
        name='mdl_ext',
        input_dims=input_dims,
        output_dims=output_dims,
    )
    lp = ext_task_p.train.learner
    lp.loss_name = 'loss'
    lp.optimizer = pax_fiddle.Config(optimizers.Adam)
    lp.optimizer.lr_schedule = pax_fiddle.Config(schedules.Constant)

    # Enable ema
    lp.optimizer.ema_decay = 0.9999

    sample_inputs = NestedMap(
        inputs=jnp.ones((1, input_dims), dtype=jnp.float32))
    ext_task = instantiate(ext_task_p)
    ext_train_state = trainer_lib.initialize_replicate_model_state(
        ext_task, jax.random.PRNGKey(0), sample_inputs
    )
    ext_train_state_metadata = trainer_lib.create_train_state_metadata(
        ext_task, sample_inputs
    )

    # Modify var01 to be random
    var_shape = ext_train_state.mdl_vars['params']['var01'].shape
    random_var = jnp.array(np.random.normal(size=var_shape))
    ext_train_state.mdl_vars['params']['var01'] = random_var

    tempdir = self.create_tempdir()
    checkpoints.save_checkpoint(
        ext_train_state,
        tempdir.full_path,
        train_state_unpadded_shape_dtype_struct=(
            ext_train_state_metadata.unpadded_global_shapes
        ),
    )

    # Create task with incorrect load_rules and safe_load=True.
    task_p = pax_fiddle.Config(tasks_lib.SingleTask, name='task')
    task_p.model = pax_fiddle.Config(
        TestModel01, name='mdl', input_dims=input_dims, output_dims=output_dims
    )
    task_p.train.learner = lp.clone()
    task_p.train.init_from_checkpoint_rules = {
        tempdir.full_path: tasks_lib.CheckpointLoadingRules(
            task_p=ext_task_p,
            load_rules=[(r'params/_(.*)', 'params/{}')],
            safe_load=True,
            input_specs_provider_p=pax_fiddle.Config(
                CustomInputSpecsProvider, input_dims=input_dims
            ),
        ),
    }
    task = instantiate(task_p)

    error_message = (
        'do not serve the intended purpose; some model variables that were '
        'meant to be loaded from checkpoint are left to their initial (random) '
        'values due to wrong pattern(s): {\'params/_(.*)\'}.')
    predicate = lambda err: error_message in str(err)
    with self.assertRaisesWithPredicateMatch(ValueError, predicate):
      _ = trainer_lib.initialize_replicate_model_state(task,
                                                       jax.random.PRNGKey(1),
                                                       sample_inputs)

  def test_lm_partial_load(self):
    model_dims = 128

    # Initialize external task and save checkpoint
    ext_task_p = pax_fiddle.Config(tasks_lib.SingleTask, name='task')
    ext_task_p.model = pax_fiddle.Config(layers.LanguageModel, name='lm')
    model_p = ext_task_p.model
    model_p.lm_tpl.model_dims = model_dims
    stacked_transformer_tpl = model_p.lm_tpl.stacked_transformer_tpl.clone()
    stacked_transformer_tpl.hidden_dims = model_dims * 4
    stacked_transformer_tpl.num_layers = 1
    stacked_transformer_tpl.num_heads = 4
    model_p.lm_tpl.stacked_transformer_tpl = pax_fiddle.Config(
        layers.StackedTransformerRepeated
    )
    model_p.lm_tpl.stacked_transformer_tpl.block = stacked_transformer_tpl
    model_p.lm_tpl.stacked_transformer_tpl.x_times = 2
    model_p.lm_tpl.vocab_size = 64

    lp = ext_task_p.train.learner
    lp.loss_name = 'loss'
    lp.optimizer = pax_fiddle.Config(optimizers.Adam)
    lp.optimizer.lr_schedule = pax_fiddle.Config(schedules.Constant)

    sample_inputs = get_model_inputs()
    ext_task = instantiate(ext_task_p)
    ext_train_state = trainer_lib.initialize_replicate_model_state(
        ext_task, jax.random.PRNGKey(0), sample_inputs
    )
    ext_train_state_metadata = trainer_lib.create_train_state_metadata(
        ext_task, sample_inputs
    )

    ext_opt_states_flat, treedef = jax.tree_util.tree_flatten(
        flax.serialization.to_state_dict(ext_train_state.opt_states))
    randomized_opt_states_flat = [
        np.random.normal(size=v.shape) for v in ext_opt_states_flat
    ]
    randomized_opt_states = jax.tree_util.tree_unflatten(
        treedef, randomized_opt_states_flat)
    ext_train_state = ext_train_state.replace(opt_states=randomized_opt_states)

    tempdir = self.create_tempdir()
    checkpoints.save_checkpoint(
        ext_train_state,
        tempdir.full_path,
        train_state_unpadded_shape_dtype_struct=(
            ext_train_state_metadata.unpadded_global_shapes
        ),
    )

    # Create task with warm-start
    task_p = ext_task_p.clone()
    task_p.train.learner = lp.clone()
    load_rules = [(r'(.*)', '{}')]
    ignore_rules = ['.*softmax.*', 'params/lm/final_ln/bias.*',
                    'params/lm/transformer/repeat/.*/query/b']
    task_p.train.init_from_checkpoint_rules = {
        tempdir.full_path: tasks_lib.CheckpointLoadingRules(
            task_p=ext_task_p,
            load_rules=load_rules,
            ignore_rules=ignore_rules,
            input_specs_provider_p=pax_fiddle.Config(LMInputSpecsProvider),
            partial_load_opt_states=True,
        ),
    }
    task = instantiate(task_p)

    # Now initialize also includes warm start (loading from ckpt)
    train_state = trainer_lib.initialize_replicate_model_state(
        task, jax.random.PRNGKey(1), sample_inputs)

    ext_mdl_vars = tasks_lib._flatten_dict(
        flax.serialization.to_state_dict(ext_train_state.mdl_vars))
    mdl_vars = tasks_lib._flatten_dict(
        flax.serialization.to_state_dict(train_state.mdl_vars))
    self.assertEqual(len(ext_mdl_vars), len(mdl_vars))
    for x, y in zip(ext_mdl_vars, mdl_vars):
      x_tensor = x[1]
      y_tensor = y[1]
      if len(x_tensor.shape) < len(y_tensor.shape):
        y_tensor = y_tensor[0]
      # Ensure that the softmax weights are not intialized. Softmax biases on
      # the other hand are always initialized to all zeros.
      if 'softmax' in x[0] and '.w' in x[0]:
        self.assertFalse(np.allclose(x_tensor, y_tensor))
      else:
        self.assertAllClose(x_tensor, y_tensor)

    ext_opt_states = tasks_lib._flatten_dict(
        flax.serialization.to_state_dict(ext_train_state.opt_states))
    opt_states = tasks_lib._flatten_dict(
        flax.serialization.to_state_dict(train_state.opt_states))
    self.assertEqual(len(ext_opt_states), len(opt_states))
    for x, y in zip(ext_opt_states, opt_states):
      x_tensor = x[1]
      y_tensor = y[1]
      if len(x_tensor.shape) < len(y_tensor.shape):
        y_tensor = y_tensor[0]
      if 'softmax' in x[0]:
        self.assertFalse(np.allclose(x_tensor, y_tensor))
      else:
        self.assertAllClose(x_tensor, y_tensor)

  def test_load_ema_prefixed(self):
    """Checks that EMA vars can be loaded from ckpts with repeated layers."""

    model_dims = 128

    # Initialize external task and save checkpoint
    ext_task_p = pax_fiddle.Config(tasks_lib.SingleTask, name='task')
    ext_task_p.model = pax_fiddle.Config(layers.LanguageModel, name='lm')
    model_p = ext_task_p.model
    model_p.lm_tpl.model_dims = model_dims
    stacked_transformer_tpl = model_p.lm_tpl.stacked_transformer_tpl.clone()
    stacked_transformer_tpl.hidden_dims = model_dims * 4
    stacked_transformer_tpl.num_layers = 1
    stacked_transformer_tpl.num_heads = 4
    model_p.lm_tpl.stacked_transformer_tpl = pax_fiddle.Config(
        layers.StackedTransformerRepeated
    )
    model_p.lm_tpl.stacked_transformer_tpl.block = stacked_transformer_tpl
    model_p.lm_tpl.stacked_transformer_tpl.x_times = 2
    model_p.lm_tpl.vocab_size = 64

    lp = ext_task_p.train.learner
    lp.loss_name = 'loss'
    lp.optimizer = pax_fiddle.Config(optimizers.Adam)
    lp.optimizer.lr_schedule = pax_fiddle.Config(schedules.Constant)
    # Enable ema
    lp.optimizer.ema_decay = 0.9999

    sample_inputs = get_model_inputs()
    ext_task = instantiate(ext_task_p)
    ext_train_state = trainer_lib.initialize_replicate_model_state(
        ext_task, jax.random.PRNGKey(0), sample_inputs
    )
    ext_train_state_metadata = trainer_lib.create_train_state_metadata(
        ext_task, sample_inputs
    )

    ext_opt_states_flat, treedef = jax.tree_util.tree_flatten(
        flax.serialization.to_state_dict(ext_train_state.opt_states)
    )
    randomized_opt_states_flat = [
        v
        if py_utils.is_optax_masked_node(v)
        else np.random.normal(size=v.shape)
        for v in ext_opt_states_flat
    ]
    randomized_opt_states = jax.tree_util.tree_unflatten(
        treedef, randomized_opt_states_flat
    )
    ext_train_state = ext_train_state.replace(opt_states=randomized_opt_states)

    tempdir = self.create_tempdir()
    checkpoints.save_checkpoint(
        ext_train_state,
        tempdir.full_path,
        train_state_unpadded_shape_dtype_struct=(
            ext_train_state_metadata.unpadded_global_shapes
        ),
    )

    # Create task with warm-start
    task_p = ext_task_p.clone()
    task_p.train.learner = lp.clone()
    load_rules = [(r'params/(.*)', 'ema/params/{}')]
    ignore_rules = ['.*softmax.*']
    task_p.train.init_from_checkpoint_rules = {
        tempdir.full_path: tasks_lib.CheckpointLoadingRules(
            task_p=ext_task_p,
            load_rules=load_rules,
            ignore_rules=ignore_rules,
            input_specs_provider_p=pax_fiddle.Config(LMInputSpecsProvider),
            load_ema_states=True,
        ),
    }
    task = instantiate(task_p)

    # Now initialize also includes warm start (loading from ckpt)
    train_state, _ = trainer_lib.initialize_model_state(
        task, jax.random.PRNGKey(1), sample_inputs
    )

    mdl_vars = tasks_lib._flatten_dict(
        flax.serialization.to_state_dict(train_state.mdl_vars)
    )
    ext_opt_states = tasks_lib._flatten_dict(
        flax.serialization.to_state_dict(ext_train_state.opt_states)
    )

    param_name_to_ema_value = {}
    for k, v in ext_opt_states:
      if 'ema.params' in k:
        param_name_to_ema_value[re.sub(r'.*?ema\.params', 'params', k)] = v

    self.assertEqual(len(mdl_vars), len(param_name_to_ema_value))
    for k, v in mdl_vars:
      ema_v = param_name_to_ema_value[k]
      if 'softmax' in k:
        self.assertFalse(np.allclose(v, ema_v))
      else:
        self.assertAllClose(v, ema_v, err_msg=k)

  @parameterized.named_parameters(('partial_load_opt_states_true', True),
                                  ('partial_load_opt_states_false', False))
  def test_load_ema_gda(self, partial_load_opt_states):
    input_dims = 2
    output_dims = 2

    # Initialize external task and save checkpoint
    ext_task_p = pax_fiddle.Config(tasks_lib.SingleTask, name='task')
    ext_task_p.model = pax_fiddle.Config(
        TestModel02,
        name='mdl_ext',
        input_dims=input_dims,
        output_dims=output_dims,
    )
    ext_task_p.train.learner = pax_fiddle.Config(
        learners.MultiOptimizerLearner
    ).set(
        loss_name='loss',
        optimizer=pax_fiddle.Config(optimizers.ShardedSgd).set(
            learning_rate=0.0,
            lr_schedule=pax_fiddle.Config(schedules.Constant, value=0.0),
            ema_decay=0.9999,
        ),
        auxiliary_optimizers=[
            optimizers.ShardedAdafactor.HParamsAdamB().set(
                lr_schedule=pax_fiddle.Config(schedules.Constant)
            )
        ],
        auxiliary_regex=(['.*var01']),
        auxiliary_names=(['var01']),
    )

    sample_inputs = NestedMap(
        inputs=jnp.ones((1, input_dims), dtype=jnp.float32))
    ext_task = instantiate(ext_task_p)
    ext_train_state, _ = trainer_lib.initialize_model_state(
        ext_task, jax.random.PRNGKey(1245), sample_inputs
    )
    ext_train_state_metadata = trainer_lib.create_train_state_metadata(
        ext_task, sample_inputs
    )

    tempdir = self.create_tempdir()
    checkpoints.save_checkpoint(
        ext_train_state,
        tempdir.full_path,
        checkpoint_type=checkpoints.CheckpointType.GDA,
        train_state_unpadded_shape_dtype_struct=(
            ext_train_state_metadata.unpadded_global_shapes
        ),
    )

    task_p = pax_fiddle.Config(tasks_lib.SingleTask, name='task')
    task_p.model = pax_fiddle.Config(
        TestModel02, name='mdl', input_dims=input_dims, output_dims=output_dims
    )
    task_p.train.learner = ext_task_p.train.learner.clone()
    task_p.train.init_from_checkpoint_rules = {
        tempdir.full_path: tasks_lib.CheckpointLoadingRules(
            task_p=ext_task_p,
            load_rules=[
                (
                    r'params/repeated_ffn/sub/bias/b',
                    'ema/params/repeated_ffn/sub/bias/b',
                ),
                (r'params/var01', 'ema/params/var01'),
            ],
            input_specs_provider_p=pax_fiddle.Config(
                CustomInputSpecsProvider, input_dims=input_dims
            ),
            partial_load_opt_states=partial_load_opt_states,
        ),
    }
    task = instantiate(task_p)

    for v in ext_train_state.opt_states[0]:
      if 'ema' in v:
        pass

    # Now initialize also includes warm start (loading from ckpt)
    sample_inputs = NestedMap(
        inputs=jnp.ones((1, input_dims), dtype=jnp.float32))
    train_state, _ = trainer_lib.initialize_model_state(
        task,
        jax.random.PRNGKey(5678),
        sample_inputs,
        checkpoint_type=checkpoints.CheckpointType.GDA,
    )

    self.assertAllClose(
        ext_train_state.mdl_vars['params']['var01'],
        train_state.mdl_vars['params']['var01'],
    )

    self.assertAllClose(
        ext_train_state.mdl_vars['params']['repeated_ffn']['sub']['bias']['b'],
        train_state.mdl_vars['params']['repeated_ffn']['sub']['bias']['b'],
    )

    for v in train_state.opt_states[0]:
      if 'ema' in v:
        self.assertAllClose(
            ext_train_state.mdl_vars['params']['var01'],
            v.ema['params']['var01'],
        )

  @parameterized.named_parameters(
      ('load_excluded_true', True), ('load_excluded_false', False)
  )
  def test_load_ema_gda_with_bprop_exclusion(self, load_excluded):
    # Test an EMA var excluded by bprop_exclusion rule, and it should sitll be
    # loaded via the original var.
    input_dims = 2
    output_dims = 2

    # Initialize external task and save checkpoint
    ext_task_p = pax_fiddle.Config(tasks_lib.SingleTask, name='task')
    ext_task_p.model = pax_fiddle.Config(
        TestModel03,
        name='mdl_ext',
        input_dims=input_dims,
        output_dims=output_dims,
    )
    ext_task_p.train.learner = pax_fiddle.Config(learners.Learner).set(
        loss_name='loss',
        optimizer=pax_fiddle.Config(optimizers.ShardedSgd).set(
            learning_rate=0.0,
            lr_schedule=pax_fiddle.Config(schedules.Constant, value=0.0),
            ema_decay=0.9999,
        ),
        bprop_variable_exclusion='.*var01',
    )

    sample_inputs = NestedMap(
        inputs=jnp.ones((1, input_dims), dtype=jnp.float32)
    )
    ext_task = instantiate(ext_task_p)
    ext_train_state, _ = trainer_lib.initialize_model_state(
        ext_task, jax.random.PRNGKey(1245), sample_inputs
    )
    ext_train_state_metadata = trainer_lib.create_train_state_metadata(
        ext_task, sample_inputs
    )

    tempdir = self.create_tempdir()
    checkpoints.save_checkpoint(
        ext_train_state,
        tempdir.full_path,
        checkpoint_type=checkpoints.CheckpointType.GDA,
        train_state_unpadded_shape_dtype_struct=(
            ext_train_state_metadata.unpadded_global_shapes
        ),
    )

    task_p = pax_fiddle.Config(tasks_lib.SingleTask, name='task')
    task_p.model = pax_fiddle.Config(
        TestModel03, name='mdl', input_dims=input_dims, output_dims=output_dims
    )
    task_p.train.learner = pax_fiddle.Config(learners.Learner).set(
        loss_name='loss',
        optimizer=pax_fiddle.Config(optimizers.ShardedSgd).set(
            learning_rate=0.0,
            lr_schedule=pax_fiddle.Config(schedules.Constant, value=0.0),
        ),
    )
    load_rules = [
        (r'params/var02', 'ema/params/var02'),
    ]
    if load_excluded:
      load_rules.append((r'params/var01', 'ema/params/var01'))
    task_p.train.init_from_checkpoint_rules = {
        tempdir.full_path: tasks_lib.CheckpointLoadingRules(
            task_p=ext_task_p,
            load_rules=load_rules,
            input_specs_provider_p=pax_fiddle.Config(
                CustomInputSpecsProvider, input_dims=input_dims
            ),
        ),
    }
    task = instantiate(task_p)

    # Now initialize also includes warm start (loading from ckpt)
    sample_inputs = NestedMap(
        inputs=jnp.ones((1, input_dims), dtype=jnp.float32)
    )
    train_state, _ = trainer_lib.initialize_model_state(
        task,
        jax.random.PRNGKey(5678),
        sample_inputs,
        checkpoint_type=checkpoints.CheckpointType.GDA,
    )

    if load_excluded:
      self.assertAllClose(
          ext_train_state.mdl_vars['params']['var01'],
          train_state.mdl_vars['params']['var01'],
      )
    self.assertAllClose(
        ext_train_state.mdl_vars['params']['var02'],
        train_state.mdl_vars['params']['var02'],
    )

  @parameterized.named_parameters(('load_excluded_true', True),
                                  ('load_excluded_false', False))
  def test_load_ema_gda_repeat_with_bprop_exclusion(self, load_excluded):
    # Test an EMA var excluded by bprop_exclusion rule, and it should sitll be
    # loaded via the original var.
    input_dims = 2
    output_dims = 2

    # Initialize external task and save checkpoint
    ext_task_p = pax_fiddle.Config(tasks_lib.SingleTask, name='task')
    ext_task_p.model = pax_fiddle.Config(
        TestModel02,
        name='mdl_ext',
        input_dims=input_dims,
        output_dims=output_dims,
    )
    ext_task_p.train.learner = pax_fiddle.Config(learners.Learner).set(
        loss_name='loss',
        optimizer=pax_fiddle.Config(optimizers.ShardedSgd).set(
            learning_rate=0.0,
            lr_schedule=pax_fiddle.Config(schedules.Constant, value=0.0),
            ema_decay=0.9999,
        ),
        bprop_variable_exclusion='.*var01',
    )

    sample_inputs = NestedMap(
        inputs=jnp.ones((1, input_dims), dtype=jnp.float32))
    ext_task = instantiate(ext_task_p)
    ext_train_state, _ = trainer_lib.initialize_model_state(
        ext_task, jax.random.PRNGKey(1245), sample_inputs
    )
    ext_train_state_metadata = trainer_lib.create_train_state_metadata(
        ext_task, sample_inputs
    )

    tempdir = self.create_tempdir()
    checkpoints.save_checkpoint(
        ext_train_state,
        tempdir.full_path,
        checkpoint_type=checkpoints.CheckpointType.GDA,
        train_state_unpadded_shape_dtype_struct=(
            ext_train_state_metadata.unpadded_global_shapes
        ),
    )

    task_p = pax_fiddle.Config(tasks_lib.SingleTask, name='task')
    task_p.model = pax_fiddle.Config(
        TestModel02, name='mdl', input_dims=input_dims, output_dims=output_dims
    )
    task_p.train.learner = pax_fiddle.Config(learners.Learner).set(
        loss_name='loss',
        optimizer=pax_fiddle.Config(optimizers.ShardedSgd).set(
            learning_rate=0.0,
            lr_schedule=pax_fiddle.Config(schedules.Constant, value=0.0),
        ),
    )
    load_rules = [(
        r'params/repeated_ffn/sub/bias/b',
        'ema/params/repeated_ffn/sub/bias/b',
    )]
    if load_excluded:
      load_rules.append((r'params/var01', 'ema/params/var01'))
    task_p.train.init_from_checkpoint_rules = {
        tempdir.full_path: tasks_lib.CheckpointLoadingRules(
            task_p=ext_task_p,
            load_rules=load_rules,
            input_specs_provider_p=pax_fiddle.Config(
                CustomInputSpecsProvider, input_dims=input_dims
            ),
        ),
    }
    task = instantiate(task_p)

    # Now initialize also includes warm start (loading from ckpt)
    sample_inputs = NestedMap(
        inputs=jnp.ones((1, input_dims), dtype=jnp.float32))
    train_state, _ = trainer_lib.initialize_model_state(
        task,
        jax.random.PRNGKey(5678),
        sample_inputs,
        checkpoint_type=checkpoints.CheckpointType.GDA,
    )

    if load_excluded:
      self.assertAllClose(
          ext_train_state.mdl_vars['params']['var01'],
          train_state.mdl_vars['params']['var01'],
      )

    self.assertAllClose(
        ext_train_state.mdl_vars['params']['repeated_ffn']['sub']['bias']['b'],
        train_state.mdl_vars['params']['repeated_ffn']['sub']['bias']['b'],
    )

  def test_var_exclusion(self):
    # Set up the model.
    input_dims = 2
    output_dims = 2

    task_p = pax_fiddle.Config(tasks_lib.SingleTask, name='task')
    task_p.model = pax_fiddle.Config(
        TestModel02, name='mdl', input_dims=input_dims, output_dims=output_dims
    )

    lp = task_p.train.learner
    lp.loss_name = 'loss'
    lp.optimizer = pax_fiddle.Config(optimizers.Adam)
    lp.optimizer.learning_rate = 5.0
    lp.optimizer.lr_schedule = pax_fiddle.Config(schedules.Constant)

    task_p_exclusion = task_p.clone()
    task_p_exclusion.train.learner.bprop_variable_exclusion = '.*var01'
    task_p_inclusion = task_p.clone()
    task_p_inclusion.train.learner.bprop_variable_inclusion = '.*repeated_ffn.*'

    sample_inputs = NestedMap(
        inputs=jnp.ones((1, input_dims), dtype=jnp.float32)
    )
    train_state, _ = trainer_lib.initialize_model_state(
        instantiate(task_p), jax.random.PRNGKey(1245), sample_inputs
    )
    train_state_exclusion, _ = trainer_lib.initialize_model_state(
        instantiate(task_p_exclusion), jax.random.PRNGKey(1245), sample_inputs
    )
    train_state_inclusion, _ = trainer_lib.initialize_model_state(
        instantiate(task_p_inclusion), jax.random.PRNGKey(1245), sample_inputs
    )

    self.assertFalse(
        py_utils.is_bprop_masked_node(
            train_state.opt_states[0]['no_prefix'][2]['m']['params']['var01']
        )
    )
    self.assertTrue(
        py_utils.is_bprop_masked_node(
            train_state_exclusion.opt_states[0]['no_prefix'][2]['m']['params'][
                'var01'
            ]
        )
    )
    self.assertTrue(
        py_utils.is_bprop_masked_node(
            train_state_inclusion.opt_states[0]['no_prefix'][2]['m']['params'][
                'var01'
            ]
        )
    )

  def test_extract_ema_with_var_exclusion(self):
    input_dims = 2
    output_dims = 2

    task_p = pax_fiddle.Config(tasks_lib.SingleTask, name='task')
    task_p.model = pax_fiddle.Config(
        TestModel02,
        name='mdl',
        input_dims=input_dims,
        output_dims=output_dims,
    )
    lp = task_p.train.learner
    lp.loss_name = 'loss'
    lp.optimizer = pax_fiddle.Config(optimizers.Adam)
    lp.optimizer.learning_rate = 5.0
    lp.optimizer.lr_schedule = pax_fiddle.Config(schedules.Constant)
    lp.optimizer.ema_decay = 0.9999
    lp.bprop_variable_exclusion = '.*var01'

    sample_inputs = NestedMap(
        inputs=jnp.ones((1, input_dims), dtype=jnp.float32))
    train_state, _ = trainer_lib.initialize_model_state(
        instantiate(task_p), jax.random.PRNGKey(1245), sample_inputs
    )

    extracted = tasks_lib.extract_ema(train_state)

    self.assertAllClose(
        train_state.mdl_vars['params']['var01'],
        extracted.mdl_vars['params']['var01'],
    )
    self.assertIsInstance(extracted.opt_states, list)

if __name__ == '__main__':
  absltest.main()
