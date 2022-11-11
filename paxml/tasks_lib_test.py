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

"""Unit tests for tasks_lib."""

from __future__ import annotations

from typing import Tuple

from absl.testing import absltest
from absl.testing import parameterized
import flax
import jax
import jax.numpy as jnp
import numpy as np
from paxml import checkpoints
from paxml import tasks_lib
from paxml import trainer_lib
from praxis import base_hyperparams
from praxis import base_input
from praxis import base_layer
from praxis import base_model
from praxis import layers
from praxis import optimizers
from praxis import py_utils
from praxis import pytypes
from praxis import schedules
from praxis import test_utils

NestedMap = py_utils.NestedMap
WeightHParams = base_layer.WeightHParams
JTensor = pytypes.JTensor
WeightInit = base_layer.WeightInit

PMAP_PARALLEL_AXIS_NAME = base_layer.PMAP_PARALLEL_AXIS_NAME

BaseHParams = base_layer.BaseLayer.HParams

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

  class HParams(base_input.BaseInputSpecsProvider.HParams):
    input_dims: int = 0

  def get_input_specs(self) -> pytypes.NestedShapeDtypeStruct:
    p = self.hparams
    batch_size = 1

    return NestedMap(
        inputs=jax.ShapeDtypeStruct(
            (batch_size, p.input_dims), dtype=jnp.float32)
        )


class LMInputSpecsProvider(base_input.BaseInputSpecsProvider):
  """Class to provide input specs for model initialization."""

  def get_input_specs(self):
    return jax.tree_map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype),
                        get_model_inputs())


class TestModel01(base_model.BaseModel):
  """Simple model for testing."""

  class HParams(BaseHParams):
    """Associated hyper-params for this layer class.

    Attributes:
      input_dims: Depth of the input.
      output_dims: Depth of the output.
    """
    input_dims: int = 0
    output_dims: int = 0

  def setup(self) -> None:
    p = self.hparams
    self.create_variable(
        'var01', base_layer.WeightHParams(shape=[p.input_dims, p.output_dims]))

  def compute_predictions(self, input_batch: NestedMap) -> JTensor:
    return jnp.einsum('bi,io->bo', input_batch.inputs, self.theta.var01)

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


class BaseTaskTest(test_utils.TestCase):

  def test_dataclass_params(self):
    dataclass_params = tasks_lib.SingleTask.HParams(name='foo_task')
    self.assertIsInstance(dataclass_params, tasks_lib.SingleTask.HParams)
    self.assertIsInstance(dataclass_params.train,
                          tasks_lib.SingleTask.TrainHParams)
    self.assertIsInstance(dataclass_params.vn,
                          tasks_lib.SingleTask.VariationalNoiseHParams)

  def test_mutate_nested_dataclass_params(self):
    """Tests that nested dataclass settings are copied to Lingvo params."""
    task_p = tasks_lib.SingleTask.HParams(name='foo_task')
    task_p.vn.vn_scale = 0.075
    task_p.vn.vn_regex = 'dec_embedding/emb_var|decoder/cell'
    self.assertEqual(task_p.vn.vn_scale, 0.075)
    self.assertEqual(task_p.vn.vn_regex, 'dec_embedding/emb_var|decoder/cell')

  def test_model_linear_regression_vn(self):
    # Set up the model.
    input_dims = 52
    output_dims = 32
    vn_start_step = 5
    task_p = tasks_lib.SingleTask.HParams(name='task')
    task_p.model = TestModel01.HParams(
        name='mdl', input_dims=input_dims, output_dims=output_dims)
    task_p.model.params_init = WeightInit.Constant(0.0)

    # Use VN on all params
    task_p.vn.vn_scale = 1.0
    task_p.vn.vn_regex = '.*'
    task_p.vn.vn_start_step = vn_start_step

    lp = task_p.train.learner
    lp.loss_name = 'loss'
    lp.optimizer = optimizers.Adam.HParams()
    # we set learning rate to 0.0 to check if it is changed by VN
    lp.optimizer.learning_rate = 0.0
    lp.optimizer.lr_schedule = schedules.Constant.HParams()

    # Create the mdl.
    jax_task = instantiate(task_p)
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
      (replicated_mdl_states, _, metrics, _,
       _) = p_train_step(replicated_mdl_states, train_prng_key, mdl_inputs)
      if step >= vn_start_step:
        # The VN is applied for training
        self.assertGreater(np.array(metrics['loss02'])[0], 0.0)
      else:
        # The VN is not applied for training
        self.assertEqual(np.array(metrics['loss02'])[0], 0.0)

      # The parameter is not changed by VN when lr = 0.0
      param = replicated_mdl_states.mdl_vars[PARAMS]['var01'][0]
      self.assertAllClose(param, np.zeros((input_dims, output_dims)), atol=1e-5)

    _, mean_metrics, _, _ = p_eval_step(replicated_mdl_states, eval_prng_key,
                                        mdl_inputs)

    # The VN is not applied for eval
    self.assertEqual(np.array(mean_metrics['loss02'])[0], 0.0)

  def test_model_linear_regression_ema(self):
    # Set up the model.
    input_dims = 52
    output_dims = 32
    decay = 0.9999

    task_p = tasks_lib.SingleTask.HParams(name='task')
    task_p.model = TestModel01.HParams(
        name='mdl', input_dims=input_dims, output_dims=output_dims)

    lp = task_p.train.learner
    lp.loss_name = 'loss'
    lp.optimizer = optimizers.Adam.HParams()
    lp.optimizer.learning_rate = 5.0
    lp.optimizer.lr_schedule = schedules.Constant.HParams()

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

      (replicated_mdl_states, _, _, _,
       _) = p_train_step(replicated_mdl_states, train_prng_key, mdl_inputs)


class ExternalCheckpointLoaderTest(test_utils.TestCase):

  @parameterized.named_parameters(('partial_load_opt_states_true', True),
                                  ('partial_load_opt_states_false', False))
  def test_load(self, partial_load_opt_states):
    input_dims = 3
    output_dims = 5

    # Initialize external task and save checkpoint
    ext_task_p = tasks_lib.SingleTask.HParams(name='task')
    ext_task_p.model = TestModel01.HParams(
        name='mdl_ext', input_dims=input_dims, output_dims=output_dims)
    lp = ext_task_p.train.learner
    lp.loss_name = 'loss'
    lp.optimizer = optimizers.Adam.HParams()
    lp.optimizer.lr_schedule = schedules.Constant.HParams()

    # Enable ema
    lp.optimizer.ema_decay = 0.9999

    sample_inputs = NestedMap(
        inputs=jnp.ones((1, input_dims), dtype=jnp.float32))
    ext_train_state = trainer_lib.initialize_replicate_model_state(
        instantiate(ext_task_p), jax.random.PRNGKey(0), sample_inputs)

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
    checkpoints.save_checkpoint(ext_train_state, tempdir.full_path)

    # Create task with warm-start
    task_p = tasks_lib.SingleTask.HParams(name='task')
    task_p.model = TestModel01.HParams(
        name='mdl', input_dims=input_dims, output_dims=output_dims)
    task_p.train.learner = lp.clone()
    load_rules = [(r'params/(.*)', 'params/{}')]
    if partial_load_opt_states:
      load_rules.extend([(r'(.*)m/params/(.*)', '{}m/params/{}'),
                         (r'(.*)v/params/(.*)', '{}v/params/{}')])
    task_p.train.init_from_checkpoint_rules = {
        tempdir.full_path:
            tasks_lib.CheckpointLoadingRules(
                task_p=ext_task_p,
                load_rules=load_rules,
                input_specs_provider_p=CustomInputSpecsProvider.HParams(
                    input_dims=input_dims),
                partial_load_opt_states=partial_load_opt_states),
    }
    task = instantiate(task_p)

    # Now initialize also includes warm start (loading from ckpt)
    train_state = trainer_lib.initialize_replicate_model_state(
        task, jax.random.PRNGKey(1), sample_inputs)

    self.assertAllClose(ext_train_state.mdl_vars['params']['var01'],
                        train_state.mdl_vars['params']['var01'][0])

    if partial_load_opt_states:
      self.assertAllClose(
          ext_train_state.opt_states[0][2]['m']['params']['var01'],
          train_state.opt_states[0][2]['m']['params']['var01'][0])
      self.assertAllClose(
          ext_train_state.opt_states[0][2]['v']['params']['var01'],
          train_state.opt_states[0][2]['v']['params']['var01'][0])

    # When loading opt states from a checkpoint, ema is not auto-initialized to
    # mdl_vars by default.
    if not partial_load_opt_states:
      for v in train_state.opt_states[0]:
        if 'ema' in v:
          self.assertAllClose(ext_train_state.mdl_vars['params']['var01'],
                              v.ema['params']['var01'][0])

  def test_load_with_incorrect_loading_rules(self):
    input_dims = 3
    output_dims = 5

    # Initialize external task and save checkpoint
    ext_task_p = tasks_lib.SingleTask.HParams(name='task')
    ext_task_p.model = TestModel01.HParams(
        name='mdl_ext', input_dims=input_dims, output_dims=output_dims)
    lp = ext_task_p.train.learner
    lp.loss_name = 'loss'
    lp.optimizer = optimizers.Adam.HParams()
    lp.optimizer.lr_schedule = schedules.Constant.HParams()

    # Enable ema
    lp.optimizer.ema_decay = 0.9999

    sample_inputs = NestedMap(
        inputs=jnp.ones((1, input_dims), dtype=jnp.float32))
    ext_train_state = trainer_lib.initialize_replicate_model_state(
        instantiate(ext_task_p), jax.random.PRNGKey(0), sample_inputs)

    # Modify var01 to be random
    var_shape = ext_train_state.mdl_vars['params']['var01'].shape
    random_var = jnp.array(np.random.normal(size=var_shape))
    ext_train_state.mdl_vars['params']['var01'] = random_var

    tempdir = self.create_tempdir()
    checkpoints.save_checkpoint(ext_train_state, tempdir.full_path)

    # Create task with incorrect load_rules and safe_load=True.
    task_p = tasks_lib.SingleTask.HParams(name='task')
    task_p.model = TestModel01.HParams(
        name='mdl', input_dims=input_dims, output_dims=output_dims)
    task_p.train.learner = lp.clone()
    task_p.train.init_from_checkpoint_rules = {
        tempdir.full_path:
            tasks_lib.CheckpointLoadingRules(
                task_p=ext_task_p,
                load_rules=[(r'params/_(.*)', 'params/{}')],
                safe_load=True,
                input_specs_provider_p=CustomInputSpecsProvider.HParams(
                    input_dims=input_dims)),
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
    ext_task_p = tasks_lib.SingleTask.HParams(name='task')
    ext_task_p.model = layers.LanguageModel.HParams(name='lm')
    model_p = ext_task_p.model
    model_p.lm_tpl.model_dims = model_dims
    stacked_transformer_tpl = model_p.lm_tpl.stacked_transformer_tpl.clone()
    stacked_transformer_tpl.hidden_dims = model_dims * 4
    stacked_transformer_tpl.num_layers = 1
    stacked_transformer_tpl.num_heads = 4
    model_p.lm_tpl.stacked_transformer_tpl = (
        layers.StackedTransformerRepeated.HParams())
    model_p.lm_tpl.stacked_transformer_tpl.block = stacked_transformer_tpl
    model_p.lm_tpl.stacked_transformer_tpl.x_times = 2
    model_p.lm_tpl.vocab_size = 64

    lp = ext_task_p.train.learner
    lp.loss_name = 'loss'
    lp.optimizer = optimizers.Adam.HParams()
    lp.optimizer.lr_schedule = schedules.Constant.HParams()

    sample_inputs = get_model_inputs()
    ext_train_state = trainer_lib.initialize_replicate_model_state(
        instantiate(ext_task_p), jax.random.PRNGKey(0), sample_inputs)

    ext_opt_states_flat, treedef = jax.tree_util.tree_flatten(
        flax.serialization.to_state_dict(ext_train_state.opt_states))
    randomized_opt_states_flat = [
        np.random.normal(size=v.shape) for v in ext_opt_states_flat
    ]
    randomized_opt_states = jax.tree_util.tree_unflatten(
        treedef, randomized_opt_states_flat)
    ext_train_state = ext_train_state.replace(opt_states=randomized_opt_states)

    tempdir = self.create_tempdir()
    checkpoints.save_checkpoint(ext_train_state, tempdir.full_path)

    # Create task with warm-start
    task_p = ext_task_p.clone()
    task_p.train.learner = lp.clone()
    load_rules = [(r'(.*)', '{}')]
    ignore_rules = ['.*softmax.*']
    task_p.train.init_from_checkpoint_rules = {
        tempdir.full_path:
            tasks_lib.CheckpointLoadingRules(
                task_p=ext_task_p,
                load_rules=load_rules,
                ignore_rules=ignore_rules,
                input_specs_provider_p=LMInputSpecsProvider.HParams(),
                partial_load_opt_states=True),
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

  @parameterized.named_parameters(('partial_load_opt_states_true', True),
                                  ('partial_load_opt_states_false', False))
  def test_load_ema(self, partial_load_opt_states):
    input_dims = 3
    output_dims = 5

    # Initialize external task and save checkpoint
    ext_task_p = tasks_lib.SingleTask.HParams(name='task')
    ext_task_p.model = TestModel01.HParams(
        name='mdl_ext', input_dims=input_dims, output_dims=output_dims)
    lp = ext_task_p.train.learner
    lp.loss_name = 'loss'
    lp.optimizer = optimizers.Adam.HParams()
    lp.optimizer.lr_schedule = schedules.Constant.HParams()

    # Enable ema
    lp.optimizer.ema_decay = 0.9999

    sample_inputs = NestedMap(
        inputs=jnp.ones((1, input_dims), dtype=jnp.float32))
    ext_train_state = trainer_lib.initialize_replicate_model_state(
        instantiate(ext_task_p), jax.random.PRNGKey(0), sample_inputs)

    # Modify var01 to be random
    var_shape = ext_train_state.mdl_vars['params']['var01'].shape
    random_var = jnp.array(np.random.normal(size=var_shape))
    ext_train_state.mdl_vars['params']['var01'] = random_var

    tempdir = self.create_tempdir()
    checkpoints.save_checkpoint(ext_train_state, tempdir.full_path)

    task_p = tasks_lib.SingleTask.HParams(name='task')
    task_p.model = TestModel01.HParams(
        name='mdl', input_dims=input_dims, output_dims=output_dims)
    task_p.train.learner = lp.clone()
    task_p.train.init_from_checkpoint_rules = {
        tempdir.full_path:
            tasks_lib.CheckpointLoadingRules(
                task_p=ext_task_p,
                load_rules=[(r'params/(.*)', 'ema/params/{}')],
                input_specs_provider_p=CustomInputSpecsProvider.HParams(
                    input_dims=input_dims),
                partial_load_opt_states=partial_load_opt_states),
    }
    task = instantiate(task_p)

    for v in ext_train_state.opt_states[0]:
      if 'ema' in v:
        ext_ema = v.ema

    # Now initialize also includes warm start (loading from ckpt)
    sample_inputs = NestedMap(
        inputs=jnp.ones((1, input_dims), dtype=jnp.float32))
    train_state = trainer_lib.initialize_replicate_model_state(
        task, jax.random.PRNGKey(1), sample_inputs)

    self.assertAllClose(ext_ema['params']['var01'],
                        train_state.mdl_vars['params']['var01'][0])

    for v in train_state.opt_states[0]:
      if 'ema' in v:
        self.assertAllClose(ext_ema['params']['var01'],
                            v.ema['params']['var01'][0])


if __name__ == '__main__':
  absltest.main()
