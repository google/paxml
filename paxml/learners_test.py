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

"""Tests for Learners."""
from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
import numpy as np
import optax
from paxml import learners
from praxis import asserts
from praxis import base_hyperparams
from praxis import base_layer
from praxis import optimizer_prefix_vectorization as opt_vec
from praxis import optimizers
from praxis import pax_fiddle
from praxis import py_utils
from praxis import schedules
from praxis import test_utils

NestedMap = py_utils.NestedMap
instantiate = base_hyperparams.instantiate


class LearnersTest(test_utils.TestCase):

  @parameterized.parameters(
      (0.5, 0.5, 1.5, 1., 0.),
      (0., 0., 1.5, 1., 0.),
      (0.5, 0.5, 1.5, 0., 1.),
      (0., 0., 1.5, 0., 1.),
  )
  def test_learner_clip_gradients(self, g1a, g1b, g2, global_clip_norm,
                                  single_clip_norm):
    learner_p = pax_fiddle.Config(learners.Learner)
    learner_p.name = 'learner'
    learner_p.loss_name = 'loss'
    learner_p.grad_norm_individual_vars = True
    learner_p.optimizer = pax_fiddle.Config(optimizers.Sgd)
    learner_p.optimizer.learning_rate = 1.
    learner_p.optimizer.lr_schedule = pax_fiddle.Config(schedules.Constant)
    if global_clip_norm:
      learner_p.optimizer.clip_gradient_norm_to_value = global_clip_norm
    elif single_clip_norm:
      learner_p.optimizer.clip_gradient_single_norm_to_value = single_clip_norm

    learner_instance = instantiate(learner_p)

    grads = NestedMap(
        grad1=jnp.array([g1a, g1b], dtype=jnp.float32),
        grad2=jnp.array([g2], dtype=jnp.float32))

    with base_layer.JaxContext.new_context():
      transformed_grads, _ = learner_instance.scale_gradients(grads)

    global_norm = np.linalg.norm([g1a, g1b, g2])
    local_norm1 = np.linalg.norm([g1a, g1b])
    local_norm2 = np.linalg.norm([g2])
    if global_clip_norm:
      gn1a = g1a * global_clip_norm / max(global_norm, global_clip_norm)
      gn1b = g1b * global_clip_norm / max(global_norm, global_clip_norm)
      gn2 = g2 * global_clip_norm / max(global_norm, global_clip_norm)
    elif single_clip_norm:
      gn1a = g1a * single_clip_norm / max(local_norm1, single_clip_norm)
      gn1b = g1b * single_clip_norm / max(local_norm1, single_clip_norm)
      gn2 = g2 * single_clip_norm / max(local_norm2, single_clip_norm)
    expected_grad1 = jnp.array([gn1a, gn1b], dtype=jnp.float32)
    expected_grad2 = jnp.array([gn2], dtype=jnp.float32)

    self.assertAllClose(expected_grad1, transformed_grads.grad1)
    self.assertAllClose(expected_grad2, transformed_grads.grad2)

  @parameterized.parameters(
      (None, False),
      (0.9, False),
      (0.9, True),
  )
  def test_sharded_sgd(self, momentum, nesterov):
    learner_p = pax_fiddle.Config(learners.Learner)
    learner_p.name = 'learner'
    learner_p.loss_name = 'loss'
    learner_p.optimizer = pax_fiddle.Config(optimizers.ShardedSgd)
    lr = 0.1
    learner_p.optimizer.learning_rate = lr
    learner_p.optimizer.lr_schedule = pax_fiddle.Config(schedules.Constant)
    learner_p.optimizer.momentum = momentum
    learner_p.optimizer.nesterov = nesterov

    learner_instance = instantiate(learner_p)

    grads = NestedMap()
    grads.lm = NestedMap(
        w=jnp.asarray(np.random.normal(1.4, 2.0, [1, 2]).astype('float32')))
    grads.ffn = jnp.asarray(
        np.random.normal(1.6, 2.0, [1, 2]).astype('float32'))
    old_vars = grads.DeepCopy()
    old_vars.lm.w = jnp.asarray(
        np.random.normal(1.2, 4.0, [1, 2]).astype('float32'))
    old_vars.ffn = jnp.asarray(
        np.random.normal(1.6, 2.0, [1, 2]).astype('float32'))
    var_weight_hparams = jax.tree_map(
        lambda v: base_layer.WeightHParams(v.shape), old_vars)

    grad_tx = learner_instance.get_grad_tx(var_weight_hparams)

    opt_states_pspec = grad_tx.init_partition_spec(var_weight_hparams)
    # Due to a new optax update, chained pytrees are masked.
    opt_states_pspec = opt_states_pspec.inner_state
    logging.info('opt_states_pspec=%s', opt_states_pspec)

    # opt_states = (
    # {"count": DeviceArray(0, dtype=int32)},
    # {"count": DeviceArray(0, dtype=int32)},
    # (
    #     TraceState(
    #       trace={
    #         "ffn": DeviceArray([[0.0, 0.0], [0.0, 0.0]], dtype=float32),
    #         "lm": {"w": DeviceArray([[0.0, 0.0], [0.0, 0.0]], dtype=float32)},
    #       }
    #     ),
    #     ScaleByScheduleState(count=DeviceArray(0, dtype=int32)),
    # ),
    # {"count": DeviceArray(0, dtype=int32)},
    # )
    opt_states = grad_tx.init(old_vars)
    logging.info('opt_states: %s', opt_states)

    # Similar to tf.nest.assert_same_structure(opt_states_pspec, opt_states),
    # but takes is_leaf arg to treat WeightHParams as a leaf.
    _ = jax.tree_map(
        lambda x, y: True,
        opt_states_pspec,
        opt_states,
        is_leaf=lambda x: isinstance(x, base_layer.WeightHParams))

    with base_layer.JaxContext.new_context():
      transformed_grads, updated_opt_states = learner_instance.update_states(
          grads, opt_states, old_vars, var_weight_hparams)

    logging.info('updated_opt_states: %s', updated_opt_states)

    sgd = optax.sgd(learning_rate=lr, momentum=momentum, nesterov=nesterov)
    # updated_state = (
    #     TraceState(
    #       trace={
    #         "ffn": DeviceArray([[0.2296397, -1.6318845]], dtype=float32),
    #         "lm": {"w": DeviceArray([[4.0486345, 0.8042676]], dtype=float32)},
    #       }
    #     ),
    #     ScaleByScheduleState(count=DeviceArray(0, dtype=int32)),
    # )
    updated_grads, updated_state = sgd.update(grads, opt_states[2])
    logging.info('updated_state: %s', updated_state)
    self.assertAllClose(transformed_grads.lm.w, updated_grads['lm']['w'])
    self.assertAllClose(transformed_grads.ffn, updated_grads['ffn'])

  @parameterized.parameters(
      (True),
      (False),
  )
  def test_adam_with_sharding(self, use_prefix_vectorization):
    learner_params = pax_fiddle.Config(learners.Learner)
    learner_params.name = 'learner'
    learner_params.loss_name = 'loss'
    learner_params.optimizer = pax_fiddle.Config(optimizers.OptaxOptimizer)

    learner_params.optimizer.learning_rate = 0.1
    learner_params.optimizer.lr_schedule = pax_fiddle.Config(
        schedules.LinearRampupExponentialDecay,
        warmup_steps=10,
        decay_start=11,
        decay_end=4000,
        min_ratio=0.1,
        max=1.0,
    )
    lr_scdule_inst = instantiate(learner_params.optimizer.lr_schedule)

    def lr_schedule(step):
      return (
          lr_scdule_inst.value_at(step) * learner_params.optimizer.learning_rate
      )

    learner_params.optimizer.grad_tx = optax.adam(
        learning_rate=lr_schedule, b1=0.9, b2=0.99, eps=0.1
    )

    learner_params.vectorize_on_repeat_prefix = use_prefix_vectorization

    learner_instance = instantiate(learner_params)

    # Define device mesh.
    mesh_shape = [1, 2, 1]
    num_devices = np.prod(mesh_shape)
    logging.info('num_local_devices: %s', num_devices)
    mesh_shape = np.arange(num_devices).reshape(mesh_shape)

    grads = NestedMap()
    grads.lm = NestedMap(
        w=jnp.asarray(np.random.normal(1.4, 2.0, [1, 2]).astype('float32'))
    )
    grads.ffn = jnp.asarray(
        np.random.normal(1.6, 2.0, [1, 2]).astype('float32')
    )
    old_vars = grads.DeepCopy()
    old_vars.lm.w = jnp.asarray(
        np.random.normal(1.2, 4.0, [1, 2]).astype('float32')
    )
    old_vars.ffn = jnp.asarray(
        np.random.normal(1.6, 2.0, [1, 2]).astype('float32')
    )
    var_weight_hparams = jax.tree_map(
        lambda v: base_layer.WeightHParams(
            v.shape, mesh_shape=mesh_shape, tensor_split_dims_mapping=[-1, 1]
        ),
        old_vars,
    )

    grad_tx = learner_instance.get_grad_tx(var_weight_hparams)
    # Initialize optimizer
    opt_states = grad_tx.init(old_vars)
    logging.info('grad_tx: %s', grad_tx)
    logging.info('Init state: %s', opt_states)
    logging.info('var params %s', var_weight_hparams)
    # Apply sharding with tree_map_params instead of calling init_partition_spec
    opt_states_pspec = opt_vec.partition_params(
        grad_tx, var_weight_hparams, opt_states
    )
    logging.info('opt_states_pspec=%s', opt_states_pspec)
    # Similar to tf.nest.assert_same_structure(opt_states_pspec, opt_states),
    # but takes is_leaf arg to treat WeightHParams as a leaf.
    jax.tree_map(
        lambda x, y: True,
        opt_states_pspec,
        opt_states,
        is_leaf=lambda x: isinstance(x, base_layer.WeightHParams),
    )

    with base_layer.JaxContext.new_context():
      transformed_grads, updated_opt_states = learner_instance.update_states(
          grads, opt_states, old_vars, var_weight_hparams
      )

    logging.info('updated_opt_states: %s', updated_opt_states)
    adam_opt = optax.adam(learning_rate=lr_schedule, b1=0.9, b2=0.99, eps=0.1)
    updated_grads, updated_state = adam_opt.update(grads, opt_states[2])
    logging.info('updated_state: %s', updated_state)
    self.assertAllClose(transformed_grads.lm.w, updated_grads['lm']['w'])
    self.assertAllClose(transformed_grads.ffn, updated_grads['ffn'])

  @parameterized.parameters(
      (0.5, 2.0, True),
      (1.5, 3.0, False),
      (10., 0.1, True),
      (100., 2.0, False),
  )
  def test_multioptimizer_learner(self, lr_multiplier1, lr_multiplier2,
                                  use_vq_ngrams):
    learner_p = pax_fiddle.Config(learners.MultiOptimizerLearner)
    learner_p.name = 'learner'
    learner_p.loss_name = 'loss'
    learner_p.optimizer = pax_fiddle.Config(optimizers.Sgd)
    learner_p.optimizer.learning_rate = 1.
    learner_p.optimizer.lr_schedule = pax_fiddle.Config(schedules.Constant)
    aux_p1 = pax_fiddle.Config(optimizers.Sgd)
    aux_p1.lr_schedule = pax_fiddle.Config(schedules.Constant)
    aux_p1.learning_rate = lr_multiplier1
    aux_p2 = pax_fiddle.Config(optimizers.Sgd)
    aux_p2.lr_schedule = pax_fiddle.Config(schedules.Constant)
    aux_p2.learning_rate = lr_multiplier2

    learner_p.auxiliary_optimizers = [aux_p1, aux_p2]
    learner_p.auxiliary_regex = ['.*ngram', '.*transformer']
    learner_p.auxiliary_names = ['ngram', 'transformer']
    learner_instance = instantiate(learner_p)

    grads = NestedMap()
    grads.lm = NestedMap()
    grads.lm.ngrammer = NestedMap()
    old_vars = grads.DeepCopy()
    emb_var1 = jnp.asarray(np.random.normal(1.0, 0.5, [4, 8]).astype('float32'))
    emb_var2 = jnp.asarray(np.random.normal(1.0, 0.5, [4, 8]).astype('float32'))
    grad1 = jnp.asarray(np.random.normal(1.0, 0.5, [4, 8]).astype('float32'))
    grad2 = jnp.asarray(np.random.normal(1.0, 0.5, [4, 8]).astype('float32'))
    if use_vq_ngrams:
      grads.lm.ngrammer.ngram_layer = NestedMap()
      grads.lm.ngrammer.ngram_layer.ngram_table = [
          NestedMap(emb_var=grad1),
          NestedMap(emb_var=grad2)
      ]
      old_vars.lm.ngrammer.ngram_layer = NestedMap()
      old_vars.lm.ngrammer.ngram_layer.ngram_table = [
          NestedMap(emb_var=emb_var1),
          NestedMap(emb_var=emb_var2)
      ]
    else:
      grads.lm.ngrammer.ngram_table = [
          NestedMap(emb_var=grad1),
          NestedMap(emb_var=grad2)
      ]
      old_vars.lm.ngrammer.ngram_table = [
          NestedMap(emb_var=emb_var1),
          NestedMap(emb_var=emb_var2)
      ]
    # Create some other keys.
    grads.lm.transformer = NestedMap(
        w=jnp.asarray(np.random.normal(1.4, 2.0, [4, 4]).astype('float32')))
    old_vars.lm.transformer = NestedMap(
        w=jnp.asarray(np.random.normal(1.2, 4.0, [4, 4]).astype('float32')))
    grads.lm.ffn = NestedMap(
        k=jnp.asarray(np.random.normal(1.6, 2.0, [4, 4]).astype('float32')))
    old_vars.lm.ffn = NestedMap(
        k=jnp.asarray(np.random.normal(1.3, 2.0, [4, 4]).astype('float32')))
    var_weight_hparams = jax.tree_map(
        lambda v: base_layer.WeightHParams(v.shape), old_vars)
    grad_tx = learner_instance.get_grad_tx(var_weight_hparams)
    opt_states = grad_tx.init(old_vars)
    with base_layer.JaxContext.new_context():
      transformed_grads, _ = learner_instance.update_states(
          grads, opt_states, old_vars, var_weight_hparams)

    expected_grad1 = -lr_multiplier1 * grad1
    expected_grad2 = -lr_multiplier1 * grad2
    if use_vq_ngrams:
      new_grad1 = (
          transformed_grads.lm.ngrammer.ngram_layer.ngram_table[0].emb_var)
      new_grad2 = (
          transformed_grads.lm.ngrammer.ngram_layer.ngram_table[1].emb_var)
    else:
      new_grad1 = transformed_grads.lm.ngrammer.ngram_table[0].emb_var
      new_grad2 = transformed_grads.lm.ngrammer.ngram_table[1].emb_var
    self.assertAllClose(new_grad1, expected_grad1)
    self.assertAllClose(new_grad2, expected_grad2)
    expected_grad_transformer = -lr_multiplier2 * grads.lm.transformer.w
    new_grad_transformer = transformed_grads.lm.transformer.w
    expected_grad_ffn = -grads.lm.ffn.k
    new_grad_ffn = transformed_grads.lm.ffn.k
    self.assertAllClose(new_grad_transformer, expected_grad_transformer)
    self.assertAllClose(new_grad_ffn, expected_grad_ffn)

  @parameterized.parameters(
      (0.5, 2.0, True, True),
      (1.5, 3.0, False, True),
      (10.0, 0.1, True, True),
      (100.0, 2.0, False, True),
      (0.5, 2.0, True, False),
      (1.5, 3.0, False, False),
      (10.0, 0.1, True, False),
      (100.0, 2.0, False, False),
  )
  def test_multioptimizer_learner_with_optax_optimizers(
      self,
      lr_multiplier1,
      lr_multiplier2,
      use_vq_ngrams,
      vectorize_on_repeat_prefix,
  ):
    learner_p = pax_fiddle.Config(learners.MultiOptimizerLearner)
    learner_p.name = 'learner'
    learner_p.loss_name = 'loss'
    learner_p.optimizer = pax_fiddle.Config(optimizers.OptaxOptimizer)
    learning_rate = 1.0
    learner_p.optimizer.grad_tx = optax.sgd(
        learning_rate=learning_rate, momentum=0.9
    )
    learner_p.vectorize_on_repeat_prefix = vectorize_on_repeat_prefix
    learner_p.optimizer.learning_rate = learning_rate
    learner_p.optimizer.lr_schedule = pax_fiddle.Config(schedules.Constant)
    aux_p1 = pax_fiddle.Config(optimizers.OptaxOptimizer)
    aux_p1.grad_tx = optax.sgd(learning_rate=lr_multiplier1, momentum=0.9)
    aux_p1.lr_schedule = pax_fiddle.Config(schedules.Constant)
    aux_p1.learning_rate = lr_multiplier1
    aux_p2 = pax_fiddle.Config(optimizers.OptaxOptimizer)
    aux_p2.grad_tx = optax.sgd(learning_rate=lr_multiplier2, momentum=0.9)
    aux_p2.lr_schedule = pax_fiddle.Config(schedules.Constant)
    aux_p2.learning_rate = lr_multiplier2

    learner_p.auxiliary_optimizers = [aux_p1, aux_p2]
    learner_p.auxiliary_regex = ['.*ngram', '.*transformer']
    learner_p.auxiliary_names = ['ngram', 'transformer']
    learner_instance = instantiate(learner_p)

    # Add a single instance optimizer.
    learner_p = pax_fiddle.Config(learners.Learner)
    learner_p.name = 'learner'
    learner_p.loss_name = 'loss'
    learner_p.optimizer = pax_fiddle.Config(optimizers.OptaxOptimizer)
    learning_rate = 1.0
    learner_p.optimizer.grad_tx = optax.sgd(
        learning_rate=learning_rate, momentum=0.9
    )
    learner_p.optimizer.learning_rate = learning_rate
    learner_p.optimizer.lr_schedule = pax_fiddle.Config(schedules.Constant)

    learner_p.vectorize_on_repeat_prefix = vectorize_on_repeat_prefix
    learner_instance_single = instantiate(learner_p)

    # Define device mesh.
    mesh_shape = [1, 2, 1]
    num_devices = np.prod(mesh_shape)
    mesh_shape = np.arange(num_devices).reshape(mesh_shape)

    grads = NestedMap()
    grads.lm = NestedMap()
    grads.lm.ngrammer = NestedMap()
    old_vars = grads.DeepCopy()
    grad_v1 = jnp.asarray(np.random.normal(1.0, 0.5, [4, 8]).astype('float32'))
    grad_v2 = jnp.asarray(np.random.normal(1.0, 0.5, [4, 8]).astype('float32'))
    emb_var1 = jnp.asarray(np.random.normal(1.0, 0.5, [4, 8]).astype('float32'))
    emb_var2 = jnp.asarray(np.random.normal(1.0, 0.5, [4, 8]).astype('float32'))
    if use_vq_ngrams:
      grads.lm.ngrammer.ngram_layer = NestedMap()
      grads.lm.ngrammer.ngram_layer.ngram_table = [
          NestedMap(emb_var=grad_v1),
          NestedMap(emb_var=grad_v2),
      ]
      old_vars.lm.ngrammer.ngram_layer = NestedMap()
      old_vars.lm.ngrammer.ngram_layer.ngram_table = [
          NestedMap(emb_var=emb_var1),
          NestedMap(emb_var=emb_var2),
      ]
    else:
      grads.lm.ngrammer.ngram_table = [
          NestedMap(emb_var=grad_v1),
          NestedMap(emb_var=grad_v2),
      ]
      old_vars.lm.ngrammer.ngram_table = [
          NestedMap(emb_var=emb_var1),
          NestedMap(emb_var=emb_var2),
      ]

    # Create some other keys.
    grads.lm.transformer = NestedMap(
        w=jnp.asarray(np.random.normal(1.4, 2.0, [4, 4]).astype('float32'))
    )
    old_vars.lm.transformer = NestedMap(
        w=jnp.asarray(np.random.normal(1.2, 4.0, [4, 4]).astype('float32'))
    )
    grads.lm.ffn = NestedMap(
        k=jnp.asarray(np.random.normal(1.6, 2.0, [4, 4]).astype('float32'))
    )
    old_vars.lm.ffn = NestedMap(
        k=jnp.asarray(np.random.normal(1.3, 2.0, [4, 4]).astype('float32'))
    )

    var_weight_hparams = jax.tree_map(
        lambda v: base_layer.WeightHParams(
            v.shape, mesh_shape=mesh_shape, tensor_split_dims_mapping=[-1, 1]
        ),
        old_vars,
    )
    var_weight_hparams.lm.ffn = NestedMap(
        k=base_layer.WeightHParams(
            shape=[4, 8],
            mesh_shape=mesh_shape,
            tensor_split_dims_mapping=[0, 1],
        )
    )

    grad_tx = learner_instance.get_grad_tx(var_weight_hparams)
    grad_tx_single = learner_instance_single.get_grad_tx(var_weight_hparams)
    opt_state = grad_tx.init(old_vars)
    logging.info('opt_state: %s', opt_state)
    opt_state_single = grad_tx_single.init(old_vars)
    logging.info('opt_state_single: %s', opt_state_single)
    partition_spec = opt_vec.partition_params(
        grad_tx, var_weight_hparams=old_vars, opt_states=opt_state
    )

    partition_spec_single = opt_vec.partition_params(
        grad_tx_single, var_weight_hparams=old_vars, opt_states=opt_state_single
    )
    # Assert that the length of partition spec is the same as the total
    # auxiliary optimizers plus 1 (for the primary optimizer).
    self.assertLen(
        partition_spec, len(learner_instance._auxiliary_optimizer_insts) + 1
    )
    # MaskedState has inner_state representing the single optimizer state
    # and the masked states are chained for optimizer and auziliary optimizers.
    for p in partition_spec:
      jax.tree_map(
          asserts.assert_same_structure,
          partition_spec_single,
          p.inner_state,
      )
    with base_layer.JaxContext.new_context():
      transformed_grads, _ = learner_instance.update_states(
          grads, opt_state, old_vars, var_weight_hparams
      )

    expected_grad1 = -lr_multiplier1 * grad_v1
    expected_grad2 = -lr_multiplier1 * grad_v2
    if use_vq_ngrams:
      new_grad1 = transformed_grads.lm.ngrammer.ngram_layer.ngram_table[
          0
      ].emb_var
      new_grad2 = transformed_grads.lm.ngrammer.ngram_layer.ngram_table[
          1
      ].emb_var
    else:
      new_grad1 = transformed_grads.lm.ngrammer.ngram_table[0].emb_var
      new_grad2 = transformed_grads.lm.ngrammer.ngram_table[1].emb_var
    self.assertAllClose(new_grad1, expected_grad1)
    self.assertAllClose(new_grad2, expected_grad2)
    expected_grad_transformer = -lr_multiplier2 * grads.lm.transformer.w
    new_grad_transformer = transformed_grads.lm.transformer.w
    expected_grad_ffn = -grads.lm.ffn.k
    new_grad_ffn = transformed_grads.lm.ffn.k
    self.assertAllClose(new_grad_transformer, expected_grad_transformer)
    self.assertAllClose(new_grad_ffn, expected_grad_ffn)

  def test_multioptimizer_learner_adam_adagrad(self):
    learner_p = pax_fiddle.Config(learners.MultiOptimizerLearner)
    learner_p.name = 'learner'
    learner_p.loss_name = 'loss'
    learner_p.optimizer = optimizers.Adam.HParamsA()
    learner_p.optimizer.learning_rate = 1.
    learner_p.optimizer.lr_schedule = pax_fiddle.Config(schedules.Constant)
    aux_p1 = optimizers.Adam.HParamsA()
    aux_p1.lr_schedule = pax_fiddle.Config(schedules.Constant)
    aux_p1.learning_rate = 2.0
    aux_p2 = pax_fiddle.Config(optimizers.Adagrad)
    aux_p2.lr_schedule = pax_fiddle.Config(schedules.Constant)
    aux_p2.learning_rate = 3.0

    learner_p.auxiliary_optimizers = [aux_p1, aux_p2]
    learner_p.auxiliary_regex = ['.*ngram', '.*transformer']
    learner_p.auxiliary_names = ['ngram', 'transformer']
    learner_instance = instantiate(learner_p)

    grads = NestedMap()
    grads.lm = NestedMap()
    grads.lm.ngrammer = NestedMap()
    old_vars = grads.DeepCopy()
    emb_var1 = jnp.asarray(np.random.normal(1.0, 0.5, [4, 8]).astype('float32'))
    emb_var2 = jnp.asarray(np.random.normal(1.0, 0.5, [4, 8]).astype('float32'))
    grad1 = jnp.asarray(np.random.normal(1.0, 0.5, [4, 8]).astype('float32'))
    grad2 = jnp.asarray(np.random.normal(1.0, 0.5, [4, 8]).astype('float32'))
    grads.lm.ngrammer.ngram_table = [
        NestedMap(emb_var=grad1),
        NestedMap(emb_var=grad2)
    ]
    old_vars.lm.ngrammer.ngram_table = [
        NestedMap(emb_var=emb_var1),
        NestedMap(emb_var=emb_var2)
    ]
    # Create some other keys.
    grads.lm.transformer = NestedMap(
        w=jnp.asarray(np.random.normal(1.4, 2.0, [4, 4]).astype('float32')))
    old_vars.lm.transformer = NestedMap(
        w=jnp.asarray(np.random.normal(1.2, 4.0, [4, 4]).astype('float32')))
    grads.lm.ffn = NestedMap(
        k=jnp.asarray(np.random.normal(1.6, 2.0, [4, 4]).astype('float32')))
    old_vars.lm.ffn = NestedMap(
        k=jnp.asarray(np.random.normal(1.3, 2.0, [4, 4]).astype('float32')))
    var_weight_hparams = jax.tree_map(
        lambda v: base_layer.WeightHParams(v.shape), old_vars)
    grad_tx = learner_instance.get_grad_tx(var_weight_hparams)
    opt_states = grad_tx.init(old_vars)
    logging.info('opt_states: %s', opt_states)

  def test_multioptimizer_learner_value_error(self):
    learner_p = pax_fiddle.Config(learners.MultiOptimizerLearner)
    learner_p.name = 'learner'
    learner_p.loss_name = 'loss'
    learner_p.optimizer = optimizers.Adam.HParamsA()
    learner_p.optimizer.learning_rate = 1.
    learner_p.optimizer.lr_schedule = pax_fiddle.Config(schedules.Constant)
    aux_p1 = optimizers.Adam.HParamsA()
    aux_p1.lr_schedule = pax_fiddle.Config(schedules.Constant)
    aux_p1.learning_rate = 2.0
    aux_p2 = pax_fiddle.Config(optimizers.Adagrad)
    aux_p2.lr_schedule = pax_fiddle.Config(schedules.Constant)
    aux_p2.learning_rate = 3.0

    learner_p.auxiliary_optimizers = [aux_p1, aux_p2]
    learner_p.auxiliary_regex = ['.*ngrammer', '.*ngram']
    learner_p.auxiliary_names = ['ngram', 'transformer']
    learner_instance = instantiate(learner_p)

    grads = NestedMap()
    grads.lm = NestedMap()
    grads.lm.ngrammer = NestedMap()
    old_vars = grads.DeepCopy()
    emb_var1 = jnp.asarray(np.random.normal(1.0, 0.5, [4, 8]).astype('float32'))
    emb_var2 = jnp.asarray(np.random.normal(1.0, 0.5, [4, 8]).astype('float32'))
    grad1 = jnp.asarray(np.random.normal(1.0, 0.5, [4, 8]).astype('float32'))
    grad2 = jnp.asarray(np.random.normal(1.0, 0.5, [4, 8]).astype('float32'))
    grads.lm.ngrammer.ngram_table = [
        NestedMap(emb_var=grad1),
        NestedMap(emb_var=grad2)
    ]
    old_vars.lm.ngrammer.ngram_table = [
        NestedMap(emb_var=emb_var1),
        NestedMap(emb_var=emb_var2)
    ]
    # Create some other keys.
    grads.lm.transformer = NestedMap(
        w=jnp.asarray(np.random.normal(1.4, 2.0, [4, 4]).astype('float32')))
    old_vars.lm.transformer = NestedMap(
        w=jnp.asarray(np.random.normal(1.2, 4.0, [4, 4]).astype('float32')))
    grads.lm.ffn = NestedMap(
        k=jnp.asarray(np.random.normal(1.6, 2.0, [4, 4]).astype('float32')))
    old_vars.lm.ffn = NestedMap(
        k=jnp.asarray(np.random.normal(1.3, 2.0, [4, 4]).astype('float32')))
    var_weight_hparams = jax.tree_map(
        lambda v: base_layer.WeightHParams(v.shape), old_vars)
    with self.assertRaises(ValueError):
      learner_instance.get_grad_tx(var_weight_hparams)

  def test_multioptimizer_learner_sharding(self):
    learner_p = pax_fiddle.Config(learners.MultiOptimizerLearner)
    learner_p.name = 'learner'
    learner_p.loss_name = 'loss'
    learner_p.optimizer = pax_fiddle.Config(optimizers.ShardedAdafactor)
    learner_p.optimizer.learning_rate = 1.
    learner_p.optimizer.decay_method = 'pow'
    learner_p.optimizer.lr_schedule = pax_fiddle.Config(schedules.Constant)
    aux_p1 = pax_fiddle.Config(optimizers.ShardedAdafactor)
    aux_p1.lr_schedule = pax_fiddle.Config(schedules.Constant)
    aux_p1.learning_rate = 2.0
    aux_p1.decay_method = 'pow'
    aux_p2 = pax_fiddle.Config(optimizers.ShardedAdafactor)
    aux_p2.lr_schedule = pax_fiddle.Config(schedules.Constant)
    aux_p2.decay_method = 'adam'
    aux_p2.learning_rate = 3.0

    # Add auxiliary optimizers.
    learner_p.auxiliary_optimizers = [aux_p1, aux_p2]
    learner_p.auxiliary_regex = ['.*ngrammer', '.*transformer']
    learner_p.auxiliary_names = ['ngram', 'transformer']
    learner_instance = instantiate(learner_p)

    # Add a single instance optimizer.
    learner_p = pax_fiddle.Config(learners.Learner)
    learner_p.name = 'learner'
    learner_p.loss_name = 'loss'
    learner_p.optimizer = pax_fiddle.Config(optimizers.ShardedAdafactor)
    learner_p.optimizer.learning_rate = 1.
    learner_p.optimizer.decay_method = 'pow'
    learner_p.optimizer.lr_schedule = pax_fiddle.Config(schedules.Constant)
    learner_instance_single = instantiate(learner_p)

    # Define device mesh.
    mesh_shape = [1, 2, 1]
    num_devices = np.prod(mesh_shape)
    logging.info('num_local_devices: %s', num_devices)
    mesh_shape = np.arange(num_devices).reshape(mesh_shape)

    grads = NestedMap()
    grads.lm = NestedMap()
    grads.lm.ngrammer = NestedMap()
    old_vars = grads.DeepCopy()
    emb_var1 = base_layer.WeightHParams(
        shape=[4, 8], mesh_shape=mesh_shape, tensor_split_dims_mapping=[-1, 1])
    emb_var2 = base_layer.WeightHParams(
        shape=[4, 8], mesh_shape=mesh_shape, tensor_split_dims_mapping=[-1, 1])
    grad1 = base_layer.WeightHParams(
        shape=[4, 8], mesh_shape=mesh_shape, tensor_split_dims_mapping=[-1, 1])
    grad2 = base_layer.WeightHParams(
        shape=[4, 8], mesh_shape=mesh_shape, tensor_split_dims_mapping=[-1, 1])
    grads.lm.ngrammer.ngram_table = [
        NestedMap(emb_var=grad1),
        NestedMap(emb_var=grad2)
    ]
    old_vars.lm.ngrammer.ngram_table = [
        NestedMap(emb_var=emb_var1),
        NestedMap(emb_var=emb_var2)
    ]
    # Create some other keys.
    grads.lm.transformer = NestedMap(
        w=base_layer.WeightHParams(
            shape=[4, 8],
            mesh_shape=mesh_shape,
            tensor_split_dims_mapping=[-1, 1]))
    old_vars.lm.transformer = NestedMap(
        w=base_layer.WeightHParams(
            shape=[4, 8],
            mesh_shape=mesh_shape,
            tensor_split_dims_mapping=[-1, 1]))
    grads.lm.ffn = NestedMap(
        k=base_layer.WeightHParams(
            shape=[4, 8],
            mesh_shape=mesh_shape,
            tensor_split_dims_mapping=[-1, 1]))
    old_vars.lm.ffn = NestedMap(
        k=base_layer.WeightHParams(
            shape=[4, 8],
            mesh_shape=mesh_shape,
            tensor_split_dims_mapping=[0, 1]))

    grad_tx = learner_instance.get_grad_tx(var_weight_hparams=old_vars)
    grad_tx_single = learner_instance_single.get_grad_tx(
        var_weight_hparams=old_vars)
    partition_spec = grad_tx.init_partition_spec(old_vars)
    # Due to a new optax update, chained pytrees are masked.
    partition_spec = partition_spec.inner_state
    partition_spec_single = grad_tx_single.init_partition_spec(old_vars)
    # Assert that the length of partition spec is the same as the total
    # auxiliary optimizers plus 1 (for the primary optimizer).
    self.assertLen(
        partition_spec, len(learner_instance._auxiliary_optimizer_insts) + 1
    )
    # Optimizers are chained as l1 - l2 - optimizer update - weight_decay.
    for k in partition_spec_single[0][2]._fields:
      for p in partition_spec:
        asserts.assert_same_structure(
            getattr(p[0][2], k), getattr(partition_spec_single[0][2], k))

  def test_vectorized_prefix(self):

    def _opt_init(params):
      # Reduction over each variable. Behavior will depend on vectorization.
      logging.info(f'Init called with params {params}')
      return jax.tree_map(jnp.sum, params)

    def _opt_update(updates, state, params):
      del params
      return jax.tree_map(lambda u, s: u + s, updates, state), state

    def _init_partition_spec(var_hparams):

      def _init_one(p):
        assert not p.repeat_prefix
        return p

      return jax.tree_map(_init_one, var_hparams)

    class TestOptimizer(optimizers.BaseOptimizer):

      def _get_raw_grad_transformation(self, lr):
        return optimizers.ShardedGradientTransformation(
            init=_opt_init,
            update=_opt_update,
            init_partition_spec=_init_partition_spec)

    learner_p = pax_fiddle.Config(
        learners.Learner,
        name='learner',
        loss_name='loss',
        grad_norm_individual_vars=True,
    )
    learner_p.optimizer = pax_fiddle.Config(
        TestOptimizer,
        learning_rate=1.0,
        lr_schedule=pax_fiddle.Config(schedules.Constant),
    )

    learner_instance = instantiate(learner_p)

    grads = NestedMap(
        a=jnp.array([1, 2], dtype=jnp.float32),
        b=jnp.array([1, 2], dtype=jnp.float32),
        c=jnp.array([[1, 2], [3, 4]], dtype=jnp.float32))
    variables = grads.copy()
    a_var_param = base_layer.WeightHParams(())
    a_var_param.repeat_prefix = [2]
    a_var_param.repeat_prefix_split_dims_mapping = [-1]
    b_var_param = base_layer.WeightHParams((2,))
    c_var_param = base_layer.WeightHParams(())
    c_var_param.repeat_prefix = [2, 2]
    c_var_param.repeat_prefix_split_dims_mapping = [('data', 'mdl'), None]
    var_hparams = NestedMap(a=a_var_param, b=b_var_param, c=c_var_param)

    grad_tx = learner_instance.get_grad_tx(var_weight_hparams=var_hparams)
    partition_spec = grad_tx.init_partition_spec(var_hparams)
    # Optimizers are chained as l1 - l2 - optimizer update - weight_decay.
    # Due to a new optax update, chained pytrees are masked.
    opt_idx = 2
    pspec_1 = partition_spec['p#2#i-1'].inner_state
    pspec_2 = partition_spec[opt_vec.NO_PREFIX_KEY].inner_state
    pspec_3 = partition_spec['p#2.2#tsdata,smdl.'].inner_state
    self.assertEqual(pspec_1[opt_idx].a.shape, ())
    self.assertEqual(pspec_1[opt_idx].a.repeat_prefix, [2])
    self.assertEqual(
        pspec_1[opt_idx].a.repeat_prefix_split_dims_mapping,
        [-1])
    self.assertEqual(pspec_2[opt_idx].b.shape,
                     (2,))
    self.assertEmpty(
        pspec_2[opt_idx].b.repeat_prefix or [])
    self.assertEqual(pspec_3[opt_idx].c.shape, ())
    self.assertEqual(pspec_3[opt_idx].c.repeat_prefix, [2, 2])
    self.assertEqual(pspec_3[opt_idx].c.repeat_prefix_split_dims_mapping,
                     [('data', 'mdl'), None])
    logging.info(f'Prefix vectorization partition spec .. {partition_spec} ')
    state = grad_tx.init(variables)
    logging.info('Prefix vectorization state after init .. ')
    logging.info(state)
    # Computed update is 0 + state, and state is sum of each variable.
    update, state = grad_tx.update(
        jax.tree_map(jnp.zeros_like, variables), state, variables)
    # Variables a and c are scalars excluding the prefix, so the update must be
    # equal to the initial variable values.
    self.assertAllClose(update.a, variables.a)
    self.assertAllClose(update.c, variables.c)
    # b is not vectorized, so the update equals the sum reduction of the initial
    # variable value.
    logging.info(f'Prefix vectorization a after update .. {update.a}')
    logging.info(f'Prefix vectorization b after update .. {update.b}')
    logging.info(f'Prefix vectorization c after update .. {update.c}')
    self.assertAllClose(update.b,
                        jnp.zeros_like(variables.b) + jnp.sum(variables.b))

  def test_vectorized_prefix_with_tree_map_params(self):
    def _opt_init(params):
      # Reduction over each variable. Behavior will depend on vectorization.
      logging.info(f'Init called with params {params}')
      return jax.tree_map(jnp.sum, params)

    def _opt_update(updates, state, params):
      del params
      return jax.tree_map(lambda u, s: u + s, updates, state), state

    learner_p = pax_fiddle.Config(
        learners.Learner,
        name='learner',
        loss_name='loss',
        grad_norm_individual_vars=True,
    )
    learner_p.optimizer = pax_fiddle.Config(optimizers.OptaxOptimizer)
    learner_p.optimizer.learning_rate = 1.0
    learner_p.optimizer.lr_schedule = pax_fiddle.Config(schedules.Constant)
    learner_p.optimizer.grad_tx = optax.GradientTransformationExtraArgs(
        init=_opt_init, update=_opt_update
    )

    learner_instance = instantiate(learner_p)

    grads = NestedMap(
        a=jnp.array([1, 2], dtype=jnp.float32),
        b=jnp.array([1, 2], dtype=jnp.float32),
        c=jnp.array([[1, 2], [3, 4]], dtype=jnp.float32),
    )
    variables = grads.copy()
    a_var_param = base_layer.WeightHParams(())
    a_var_param.repeat_prefix = [2]
    a_var_param.repeat_prefix_split_dims_mapping = [-1]
    b_var_param = base_layer.WeightHParams((2,))
    c_var_param = base_layer.WeightHParams(())
    c_var_param.repeat_prefix = [2, 2]
    c_var_param.repeat_prefix_split_dims_mapping = [('data', 'mdl'), None]
    var_hparams = NestedMap(a=a_var_param, b=b_var_param, c=c_var_param)

    grad_tx = learner_instance.get_grad_tx(var_weight_hparams=var_hparams)

    state = grad_tx.init(variables)
    logging.info(state)
    opt_states_pspec = opt_vec.partition_params(grad_tx, var_hparams, state)
    logging.info('opt_states_pspec=%s', opt_states_pspec)
    logging.info('Prefix vectorization state after init .. ')
    # Computed update is 0 + state, and state is sum of each variable.
    update, state = grad_tx.update(
        jax.tree_map(jnp.zeros_like, variables), state, variables
    )
    # Variables a and c are scalars excluding the prefix, so the update must be
    # equal to the initial variable values.
    self.assertAllClose(update.a, variables.a)
    self.assertAllClose(update.c, variables.c)
    # b is not vectorized, so the update equals the sum reduction of the initial
    # variable value.
    logging.info(f'Prefix vectorization a after update .. {update.a}')
    logging.info(f'Prefix vectorization b after update .. {update.b}')
    logging.info(f'Prefix vectorization c after update .. {update.c}')
    self.assertAllClose(
        update.b, jnp.zeros_like(variables.b) + jnp.sum(variables.b)
    )

  def test_scale_update_by_var_norm(self):
    def _opt_init(params):
      # Reduction over each variable. Behavior will depend on vectorization.
      return jax.tree_map(jnp.sum, params)

    def _opt_update(updates, state, params):
      del params
      return jax.tree_map(lambda u, s: u + s, updates, state), state

    def _init_partition_spec(var_hparams):
      def _init_one(p):
        assert not p.repeat_prefix
        return p

      return jax.tree_map(_init_one, var_hparams)

    class TestOptimizer(optimizers.BaseOptimizer):

      def _get_raw_grad_transformation(self, lr):
        return optimizers.ShardedGradientTransformation(
            init=_opt_init,
            update=_opt_update,
            init_partition_spec=_init_partition_spec,
        )

    learner_p = pax_fiddle.Config(
        learners.Learner,
        name='learner',
        loss_name='loss',
        grad_norm_individual_vars=True,
        scale_update_by_var_norm=True,
    )
    learner_p.optimizer = pax_fiddle.Config(
        TestOptimizer,
        learning_rate=1.0,
        lr_schedule=pax_fiddle.Config(schedules.Constant),
    )

    learner_instance = instantiate(learner_p)

    grads = NestedMap(
        a=jnp.array([[1, 2], [3, 4]], dtype=jnp.float32),
        b=jnp.array([1, 2], dtype=jnp.float32),
        c=jnp.array([[1, 2], [3, 4]], dtype=jnp.float32),
    )
    variables = grads.copy()
    a_var_param = base_layer.WeightHParams((2, 2))
    a_var_param.init = base_layer.WeightInit('gaussian', 0.003)
    b_var_param = base_layer.WeightHParams((2,))
    c_var_param = base_layer.WeightHParams(())
    c_var_param.repeat_prefix = [2, 2]
    c_var_param.repeat_prefix_split_dims_mapping = [('data', 'mdl'), None]
    var_hparams = NestedMap(a=a_var_param, b=b_var_param, c=c_var_param)

    with base_layer.JaxContext.new_context():
      learner_instance.apply_gradient(variables, grads, var_hparams)

  def test_vectorized_prefix_with_global_summary(self):

    def _opt_init(params):
      # Reduction over each variable. Behavior will depend on vectorization.
      return jax.tree_map(jnp.sum, params)

    def _opt_update(updates, state, params):
      del params

      def _opt_update_with_global_summary(u, s):
        base_layer.add_global_summary('u', u.sum())
        base_layer.add_global_summary('s', s.sum())
        return u + s

      return jax.tree_map(_opt_update_with_global_summary, updates,
                          state), state

    def _init_partition_spec(var_hparams):

      def _init_one(p):
        assert not p.repeat_prefix
        return p

      return jax.tree_map(_init_one, var_hparams)

    class TestOptimizer(optimizers.BaseOptimizer):

      def _get_raw_grad_transformation(self, lr):
        return optimizers.ShardedGradientTransformation(
            init=_opt_init,
            update=_opt_update,
            init_partition_spec=_init_partition_spec)

    learner_p = pax_fiddle.Config(
        learners.Learner,
        name='learner',
        loss_name='loss',
        grad_norm_individual_vars=True,
    )
    learner_p.optimizer = pax_fiddle.Config(
        TestOptimizer,
        learning_rate=1.0,
        lr_schedule=pax_fiddle.Config(schedules.Constant),
    )

    learner_instance = instantiate(learner_p)

    grads = NestedMap(
        a=jnp.array([1, 2], dtype=jnp.float32),
        b=jnp.array([1, 2], dtype=jnp.float32),
        c=jnp.array([[1, 2], [3, 4]], dtype=jnp.float32))
    variables = grads.copy()
    a_var_param = base_layer.WeightHParams(())
    a_var_param.repeat_prefix = [2]
    a_var_param.repeat_prefix_split_dims_mapping = [-1]
    b_var_param = base_layer.WeightHParams((2,))
    c_var_param = base_layer.WeightHParams(())
    c_var_param.repeat_prefix = [2, 2]
    c_var_param.repeat_prefix_split_dims_mapping = [('data', 'mdl'), None]
    var_hparams = NestedMap(a=a_var_param, b=b_var_param, c=c_var_param)

    with base_layer.JaxContext.new_context():
      grad_tx = learner_instance.get_grad_tx(var_weight_hparams=var_hparams)
      partition_spec = grad_tx.init_partition_spec(var_hparams)
      # Optimizers are chained as l1 - l2 - optimizer update - weight_decay.
      # Due to a new optax update, chained pytrees are masked.
      opt_idx = 2
      pspec_1 = partition_spec['p#2#i-1'].inner_state
      pspec_2 = partition_spec[opt_vec.NO_PREFIX_KEY].inner_state
      pspec_3 = partition_spec['p#2.2#tsdata,smdl.'].inner_state
      self.assertEqual(pspec_1[opt_idx].a.shape, ())
      self.assertEqual(pspec_1[opt_idx].a.repeat_prefix, [2])
      self.assertEqual(pspec_1[opt_idx].a.repeat_prefix_split_dims_mapping,
                       [-1])
      self.assertEqual(pspec_2[opt_idx].b.shape, (2,))
      self.assertEmpty(pspec_2[opt_idx].b.repeat_prefix or [])
      self.assertEqual(pspec_3[opt_idx].c.shape, ())
      self.assertEqual(pspec_3[opt_idx].c.repeat_prefix, [2, 2])
      self.assertEqual(pspec_3[opt_idx].c.repeat_prefix_split_dims_mapping,
                       [('data', 'mdl'), None])

      state = grad_tx.init(variables)
      # Computed update is 0 + state, and state is sum of each variable.
      update, state = grad_tx.update(
          jax.tree_map(jnp.zeros_like, variables), state, variables)
      # Variables a and c are scalars excluding the prefix, so the update must
      # be equal to the initial variable values.
      self.assertAllClose(update.a, variables.a)
      self.assertAllClose(update.c, variables.c)
      # b is not vectorized, so the update equals the sum reduction of the
      # initial variable value.
      self.assertAllClose(update.b,
                          jnp.zeros_like(variables.b) + jnp.sum(variables.b))

      expected_summary_keys = [
          's.[2, 2]_scalar', 's.[2]_scalar', 's.[]_scalar', 'u.[2, 2]_scalar',
          'u.[2]_scalar', 'u.[]_scalar'
      ]
      summaries = base_layer.all_global_summaries()
      with self.subTest('test_keys'):
        self.assertCountEqual(expected_summary_keys, sorted(summaries))
      self.assertEqual(summaries['s.[2, 2]_scalar'].shape, (2, 2))
      self.assertEqual(summaries['s.[2, 2]_scalar'].dtype, np.float32)
      self.assertEqual(summaries['s.[2, 2]_scalar']._value.tolist(),
                       [[1.0, 2.0], [3.0, 4.0]])
      self.assertEqual(summaries['s.[2]_scalar'].shape, (2,))
      self.assertEqual(summaries['s.[2]_scalar'].dtype, np.float32)
      self.assertEqual(summaries['s.[2]_scalar']._value.tolist(), [1.0, 2.0])
      self.assertEqual(summaries['s.[]_scalar'].shape, ())
      self.assertEqual(summaries['s.[]_scalar'].dtype, np.float32)
      self.assertEqual(summaries['s.[]_scalar']._value.tolist(), 3.0)
      self.assertEqual(summaries['u.[2, 2]_scalar'].shape, (2, 2))
      self.assertEqual(summaries['u.[2, 2]_scalar'].dtype, np.float32)
      self.assertEqual(summaries['u.[2, 2]_scalar']._value.tolist(),
                       [[0.0, 0.0], [0.0, 0.0]])
      self.assertEqual(summaries['u.[2]_scalar'].shape, (2,))
      self.assertEqual(summaries['u.[2]_scalar'].dtype, np.float32)
      self.assertEqual(summaries['u.[2]_scalar']._value.tolist(), [0.0, 0.0])
      self.assertEqual(summaries['u.[]_scalar'].shape, ())
      self.assertEqual(summaries['u.[]_scalar'].dtype, np.float32)
      self.assertEqual(summaries['u.[]_scalar']._value.tolist(), 0.0)


if __name__ == '__main__':
  absltest.main()
