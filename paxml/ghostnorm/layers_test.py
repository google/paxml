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

"""Parameteric tests for layers that implemented the ghost norm protocol."""

from absl.testing import absltest
from absl.testing import parameterized

import jax
import jax.numpy as jnp
import numpy as np
import optax
from paxml.ghostnorm import base
from paxml.ghostnorm import linears
from praxis import base_layer
from praxis import pax_fiddle
from praxis.layers import linears as praxis_linears
import tensorflow as tf

instantiate = base_layer.instantiate
PARAMS = base_layer.PARAMS
RANDOM = base_layer.RANDOM


class LayersTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(123456)
    tf.random.set_seed(123)

  @parameterized.named_parameters(
      {
          'testcase_name': 'Linear',
          'reference_layer_cls': praxis_linears.Linear,
          'ghostnorm_layer_cls': linears.LinearGhostNorm,
          'configs': dict(input_dims=10, output_dims=3),
          'inputs_shape': (32, 10),
      },
      {
          'testcase_name': 'LinearMultiDimInputs',
          'reference_layer_cls': praxis_linears.Linear,
          'ghostnorm_layer_cls': linears.LinearGhostNorm,
          'configs': dict(input_dims=10, output_dims=3),
          'inputs_shape': (32, 24, 10),
      },
      {
          'testcase_name': 'Bias',
          'reference_layer_cls': praxis_linears.Bias,
          'ghostnorm_layer_cls': linears.BiasGhostNorm,
          'configs': {'dims': 3},
          'inputs_shape': (32, 24, 3),
      },
  )
  def test_calculate_grad_norms(self, reference_layer_cls, ghostnorm_layer_cls,
                                configs, inputs_shape):
    ref_layer_tpl = pax_fiddle.Config(reference_layer_cls, **configs)
    ghostnorm_layer_tpl = pax_fiddle.Config(ghostnorm_layer_cls, **configs)
    ref_layer = instantiate(ref_layer_tpl)
    ghostnorm_layer = instantiate(ghostnorm_layer_tpl)
    inputs = jnp.asarray(np.random.normal(0, 0.1, size=inputs_shape))

    prng_key = jax.random.PRNGKey(seed=1234)
    prng_key, init_key, random_key, noise_key = jax.random.split(prng_key, 4)
    initial_vars = ghostnorm_layer.init({PARAMS: init_key, RANDOM: random_key},
                                        inputs)

    # make sure the layer behaves the same as the reference layer in forward
    ghostnorm_outputs = ghostnorm_layer.apply(
        initial_vars, inputs, rngs={RANDOM: noise_key})
    ref_outputs = ref_layer.apply(
        initial_vars, inputs, rngs={RANDOM: noise_key})
    np.testing.assert_allclose(ghostnorm_outputs, ref_outputs,
                               rtol=1e-5, atol=1e-5)

    def simple_loss(outputs):
      # simple loss for testing purpose
      # note the ghost clipping library assumes mean loss over the batch
      outputs = outputs.reshape((outputs.shape[0], -1))
      target = 1  # an arbitrary value for testing purpose
      return jnp.mean(jnp.sum(jnp.square(outputs - target), axis=1))

    def loss_fn(mdl_vars, inputs):
      outputs = ghostnorm_layer.apply(mdl_vars, inputs,
                                      rngs={RANDOM: noise_key})
      return simple_loss(outputs)

    def loss_fn_ref(mdl_vars, inputs):
      outputs = ref_layer.apply(mdl_vars, inputs, rngs={RANDOM: noise_key})
      return simple_loss(outputs)

    # get expected per-example gradient norms by explicitly materialize
    # per-example gradients with jax.vmap
    per_eg_grad_fn = jax.vmap(jax.grad(loss_fn), in_axes=(None, 0))
    vmap_inputs = jax.tree_map(lambda x: jnp.expand_dims(x, axis=1), inputs)
    per_eg_grad = per_eg_grad_fn(initial_vars, vmap_inputs)
    per_eg_grad_norms = jax.vmap(optax.global_norm)(per_eg_grad)

    # first pass to compute gradient norm
    grad_fn = jax.grad(loss_fn)
    scales = jnp.ones(inputs_shape[0])
    params_with_sq_norms = jax.tree_map(
        lambda x: base.ParamWithAux(x, scales), initial_vars[PARAMS])
    grad_with_sq_norms = grad_fn(
        {**initial_vars, PARAMS: params_with_sq_norms}, inputs)[PARAMS]

    is_leaf = lambda x: isinstance(x, base.ParamWithAux)
    fast_per_eg_grad_norms = jnp.sqrt(sum(
        x.aux for x in
        jax.tree_util.tree_flatten(grad_with_sq_norms, is_leaf=is_leaf)[0]))

    # test if the computed per-example gradient norms match expected values
    np.testing.assert_allclose(per_eg_grad_norms, fast_per_eg_grad_norms,
                               rtol=1e-5, atol=1e-5)

    # test if the computed gradient matches the grad of the reference layer
    grad_fn_ref = jax.grad(loss_fn_ref)
    grad_ref = grad_fn_ref(initial_vars, inputs)[PARAMS]
    grad_diffs = jax.tree_map(lambda x, y: np.mean(np.abs(x-y.param)),
                              grad_ref, grad_with_sq_norms)
    np.testing.assert_allclose(jax.tree_util.tree_flatten(grad_diffs)[0], 0,
                               rtol=1e-5, atol=1e-5)

    # second pass to compute norm-clipped gradients
    l2_clip = 0.2
    scales = jnp.minimum(1.0, l2_clip / fast_per_eg_grad_norms)
    params_with_sq_norms = jax.tree_map(
        lambda x: base.ParamWithAux(x, scales), initial_vars[PARAMS])
    grad_with_sq_norms = grad_fn(
        {**initial_vars, PARAMS: params_with_sq_norms}, inputs)[PARAMS]

    fast_per_eg_grad_norms = jnp.sqrt(sum(
        x.aux for x in
        jax.tree_util.tree_flatten(grad_with_sq_norms, is_leaf=is_leaf)[0]))

    # test if the norm clipping conditions are satisfied
    self.assertTrue(np.all(fast_per_eg_grad_norms <= l2_clip + 1e-5))

    # compute the expected average gradients from the clipped per-example grads
    grads_flat, grads_treedef = jax.tree_flatten(per_eg_grad[PARAMS])
    sum_clipped, _ = optax.per_example_global_norm_clip(
        grads=grads_flat, l2_norm_clip=l2_clip)
    sum_grads = jax.tree_unflatten(grads_treedef, sum_clipped)
    expected_grads = jax.tree_map(lambda x: x / inputs_shape[0], sum_grads)

    obtained_grads = jax.tree_map(
        lambda x: x.param, grad_with_sq_norms, is_leaf=is_leaf)

    # test if the ghost norm clipping outputs expected average gradients
    diffs = jax.tree_map(lambda x, y: np.mean(np.abs(x-y)),
                         expected_grads, obtained_grads)
    np.testing.assert_allclose(jax.tree_util.tree_flatten(diffs)[0], 0,
                               rtol=1e-5, atol=1e-5)


if __name__ == '__main__':
  absltest.main()
