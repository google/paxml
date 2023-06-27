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
from paxml.ghostnorm import embedding
from paxml.ghostnorm import generic_wrapper
from paxml.ghostnorm import linears
from praxis import base_layer
from praxis import pax_fiddle
from praxis.layers import embedding_softmax as praxis_embedding
from praxis.layers import linears as praxis_linears
import tensorflow as tf

instantiate = base_layer.instantiate
PARAMS = base_layer.PARAMS
RANDOM = base_layer.RANDOM
LAYER = generic_wrapper.LAYER


def mean_loss_for_testing(outputs):
  # simple loss for testing purpose
  # note the ghost clipping library assumes mean loss over the batch
  outputs = outputs.reshape((outputs.shape[0], -1))
  arbitrary_target_for_test = 1
  return jnp.mean(
      jnp.sum(jnp.square(outputs - arbitrary_target_for_test), axis=1)
  )


class LayersTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(123456)
    tf.random.set_seed(123)

  def _get_loss_fns(self, ghostnorm_layer, ref_layer):
    noise_key = jax.random.PRNGKey(seed=12345)

    def loss_fn(mdl_vars, inputs):
      outputs = ghostnorm_layer.apply(
          mdl_vars, inputs, rngs={RANDOM: noise_key}
      )
      return mean_loss_for_testing(outputs)

    def loss_fn_ref(mdl_vars, inputs):
      outputs = ref_layer.apply(mdl_vars, inputs, rngs={RANDOM: noise_key})
      return mean_loss_for_testing(outputs)

    return loss_fn, loss_fn_ref

  def _get_per_eg_grad(self, inputs, initial_vars, loss_fn):
    # per-example gradients with jax.vmap
    per_eg_grad_fn = jax.vmap(jax.grad(loss_fn), in_axes=(None, 0))
    vmap_inputs = jax.tree_map(lambda x: jnp.expand_dims(x, axis=1), inputs)
    per_eg_grad = per_eg_grad_fn(initial_vars, vmap_inputs)
    return per_eg_grad

  def _get_grad_and_norms(self, inputs, initial_vars, loss_fn, scales):
    grad_fn = jax.grad(loss_fn)
    params_with_sq_norms = jax.tree_map(
        lambda x: base.ParamWithAux(x, scales), initial_vars[PARAMS]
    )
    params_with_sq_norms = {**initial_vars, PARAMS: params_with_sq_norms}
    grad_with_sq_norms = grad_fn(params_with_sq_norms, inputs)[PARAMS]

    is_leaf = lambda x: isinstance(x, base.ParamWithAux)
    fast_per_eg_grad_norms = jnp.sqrt(
        sum(
            x.aux
            for x in jax.tree_util.tree_flatten(
                grad_with_sq_norms, is_leaf=is_leaf
            )[0]
        )
    )
    return grad_with_sq_norms, fast_per_eg_grad_norms

  def assertOutputsAreSame(
      self,
      inputs,
      ghostnorm_initial_vars,
      initial_vars,
      ghostnorm_layer,
      ref_layer,
  ):
    noise_key = jax.random.PRNGKey(seed=12345)

    # make sure the layer behaves the same as the reference layer in forward
    ghostnorm_outputs = ghostnorm_layer.apply(
        ghostnorm_initial_vars, inputs, rngs={RANDOM: noise_key}
    )
    ref_outputs = ref_layer.apply(
        initial_vars, inputs, rngs={RANDOM: noise_key}
    )
    np.testing.assert_allclose(
        ghostnorm_outputs,
        ref_outputs,
        rtol=1e-5,
        atol=1e-5,
        err_msg='outputs are different.',
    )

  def assertGradNormsAreSame(self, inputs, initial_vars, loss_fn):
    # get expected per-example gradient norms by explicitly materialize
    per_eg_grad = self._get_per_eg_grad(inputs, initial_vars, loss_fn)
    per_eg_grad_norms = jax.vmap(optax.global_norm)(per_eg_grad)

    # first pass to compute gradient norm
    _, fast_per_eg_grad_norms = self._get_grad_and_norms(
        inputs, initial_vars, loss_fn, jnp.ones(inputs.shape[0])
    )

    # test if the computed per-example gradient norms match expected values
    np.testing.assert_allclose(
        per_eg_grad_norms,
        fast_per_eg_grad_norms,
        rtol=1e-5,
        atol=1e-5,
        err_msg=(
            "computed per-example gradient norms don't match expected values."
        ),
    )

  def assertGradIsSame(
      self,
      inputs,
      ghost_initial_vars,
      initial_vars,
      loss_fn,
      loss_fn_ref,
      unwrap_grad=False,
  ):
    grad_with_sq_norms, _ = self._get_grad_and_norms(
        inputs,
        ghost_initial_vars,
        loss_fn,
        jnp.ones(inputs.shape[0]),
    )
    if unwrap_grad:
      grad_with_sq_norms = grad_with_sq_norms[LAYER]

    # test if the computed gradient matches the grad of the reference layer
    grad_fn_ref = jax.grad(loss_fn_ref)
    grad_ref = grad_fn_ref(initial_vars, inputs)[PARAMS]
    grad_diffs = jax.tree_map(
        lambda x, y: np.mean(np.abs(x - y.param)), grad_ref, grad_with_sq_norms
    )
    np.testing.assert_allclose(
        jax.tree_util.tree_flatten(grad_diffs)[0],
        0,
        rtol=1e-5,
        atol=1e-5,
        err_msg='average gradients are different.',
    )

  def assertGradIsClippedToScale(self, inputs, initial_vars, loss_fn):
    # first pass to compute gradient norm
    _, fast_per_eg_grad_norms = self._get_grad_and_norms(
        inputs, initial_vars, loss_fn, jnp.ones(inputs.shape[0])
    )

    # second pass to compute norm-clipped gradients
    l2_clip = 0.2
    scales = jnp.minimum(1.0, l2_clip / fast_per_eg_grad_norms)
    _, scaled_fast_per_eg_grad_norms = self._get_grad_and_norms(
        inputs, initial_vars, loss_fn, scales
    )

    self.assertTrue(
        np.all(scaled_fast_per_eg_grad_norms <= l2_clip + 1e-5),
        msg='norm clipping conditions are not satisfied.',
    )

  def assertClippedGradEqualsScaledGrad(self, inputs, initial_vars, loss_fn):
    l2_clip = 0.2
    # get expected per-example gradient norms by explicitly materialize
    per_eg_grad = self._get_per_eg_grad(inputs, initial_vars, loss_fn)

    # compute the expected average gradients from the clipped per-example grads
    grads_flat, grads_treedef = jax.tree_flatten(per_eg_grad[PARAMS])
    sum_clipped, _ = optax.per_example_global_norm_clip(
        grads=grads_flat, l2_norm_clip=l2_clip
    )
    sum_grads = jax.tree_unflatten(grads_treedef, sum_clipped)
    expected_grads = jax.tree_map(lambda x: x / inputs.shape[0], sum_grads)

    _, fast_per_eg_grad_norms = self._get_grad_and_norms(
        inputs, initial_vars, loss_fn, jnp.ones(inputs.shape[0])
    )
    scales = jnp.minimum(1.0, l2_clip / fast_per_eg_grad_norms)
    grad_with_sq_norms, _ = self._get_grad_and_norms(
        inputs,
        initial_vars,
        loss_fn,
        scales,
    )

    is_leaf = lambda x: isinstance(x, base.ParamWithAux)
    obtained_grads = jax.tree_map(
        lambda x: x.param, grad_with_sq_norms, is_leaf=is_leaf
    )

    diffs = jax.tree_map(
        lambda x, y: np.mean(np.abs(x - y)), expected_grads, obtained_grads
    )
    np.testing.assert_allclose(
        jax.tree_util.tree_flatten(diffs)[0],
        0,
        rtol=1e-5,
        atol=1e-5,
        err_msg='ghost norm clipping outputs incorrect average gradients.',
    )

  @parameterized.named_parameters(
      {
          'testcase_name': 'Linear',
          'reference_layer_cls': praxis_linears.Linear,
          'ghostnorm_layer_cls': linears.LinearGhostNorm,
          'configs': dict(input_dims=10, output_dims=3),
          'inputs_fn': lambda: np.random.normal(0, 0.1, size=(32, 10)),
      },
      {
          'testcase_name': 'LinearMultiDimInputs',
          'reference_layer_cls': praxis_linears.Linear,
          'ghostnorm_layer_cls': linears.LinearGhostNorm,
          'configs': dict(input_dims=10, output_dims=3),
          'inputs_fn': lambda: np.random.normal(0, 0.1, size=(32, 24, 10)),
      },
      {
          'testcase_name': 'Bias',
          'reference_layer_cls': praxis_linears.Bias,
          'ghostnorm_layer_cls': linears.BiasGhostNorm,
          'configs': {'dims': 3},
          'inputs_fn': lambda: np.random.normal(0, 0.1, size=(32, 24, 3)),
      },
      {
          'testcase_name': 'Embedding',
          'reference_layer_cls': praxis_embedding.Embedding,
          'ghostnorm_layer_cls': embedding.EmbeddingGhostNorm,
          'configs': {'input_dims': 3, 'num_classes': 10},
          'inputs_fn': lambda: np.random.randint(0, 10, size=(32,)),
      },
      {
          'testcase_name': 'EmbeddingOOB',
          'reference_layer_cls': praxis_embedding.Embedding,
          'ghostnorm_layer_cls': embedding.EmbeddingGhostNorm,
          'configs': {'input_dims': 3, 'num_classes': 2},
          'inputs_fn': lambda: np.random.randint(0, 10, size=(32,)),
      },
      {
          'testcase_name': 'EmbeddingScaleSqrt',
          'reference_layer_cls': praxis_embedding.Embedding,
          'ghostnorm_layer_cls': embedding.EmbeddingGhostNorm,
          'configs': {
              'input_dims': 3,
              'num_classes': 10,
              'scale_sqrt_depth': True,
          },
          'inputs_fn': lambda: np.random.randint(0, 10, size=(32,)),
      },
      {
          'testcase_name': 'EmbeddingMultiDimInputs',
          'reference_layer_cls': praxis_embedding.Embedding,
          'ghostnorm_layer_cls': embedding.EmbeddingGhostNorm,
          'configs': {'input_dims': 3, 'num_classes': 10},
          'inputs_fn': lambda: np.random.randint(0, 10, size=(32, 24, 20)),
      },
      {
          'testcase_name': 'EmbeddingMultiDimInputsOOB',
          'reference_layer_cls': praxis_embedding.Embedding,
          'ghostnorm_layer_cls': embedding.EmbeddingGhostNorm,
          'configs': {'input_dims': 3, 'num_classes': 2},
          'inputs_fn': lambda: np.random.randint(0, 10, size=(32, 24, 3)),
      },
      {
          'testcase_name': 'EmbeddingMatMul',
          'reference_layer_cls': praxis_embedding.Embedding,
          'ghostnorm_layer_cls': embedding.EmbeddingGhostNorm,
          'configs': {
              'input_dims': 3,
              'num_classes': 10,
              'lookup_style': 'matmul',
          },
          'inputs_fn': lambda: np.random.randint(0, 10, size=(32,)),
      },
      {
          'testcase_name': 'EmbeddingOOBSetNan',
          'reference_layer_cls': praxis_embedding.Embedding,
          'ghostnorm_layer_cls': embedding.EmbeddingGhostNorm,
          'configs': {
              'input_dims': 3,
              'num_classes': 2,
              'set_nan_for_oob_id': True,
          },
          'inputs_fn': lambda: np.random.randint(0, 10, size=(32,)),
      },
  )
  def test_calculate_grad_norms(
      self, reference_layer_cls, ghostnorm_layer_cls, configs, inputs_fn
  ):
    ref_layer_tpl = pax_fiddle.Config(reference_layer_cls, **configs)
    ghostnorm_layer_tpl = pax_fiddle.Config(ghostnorm_layer_cls, **configs)
    ref_layer = instantiate(ref_layer_tpl)
    ghostnorm_layer = instantiate(ghostnorm_layer_tpl)
    inputs = jnp.asarray(inputs_fn())

    prng_key = jax.random.PRNGKey(seed=1234)
    init_key, random_key = jax.random.split(prng_key)
    initial_vars = ghostnorm_layer.init(
        {PARAMS: init_key, RANDOM: random_key}, inputs
    )

    self.assertOutputsAreSame(
        inputs, initial_vars, initial_vars, ghostnorm_layer, ref_layer
    )

    loss_fn, loss_fn_ref = self._get_loss_fns(ghostnorm_layer, ref_layer)

    self.assertGradNormsAreSame(inputs, initial_vars, loss_fn)
    self.assertGradIsSame(
        inputs, initial_vars, initial_vars, loss_fn, loss_fn_ref
    )
    self.assertGradIsClippedToScale(inputs, initial_vars, loss_fn)
    self.assertClippedGradEqualsScaledGrad(inputs, initial_vars, loss_fn)

  @parameterized.named_parameters(
      {
          'testcase_name': 'Embedding',
          'reference_layer_cls': praxis_embedding.Embedding,
          'configs': {'input_dims': 3, 'num_classes': 10},
          'inputs_fn': lambda: np.random.randint(0, 10, size=(2, 1)),
      },
      {
          'testcase_name': 'Linear',
          'reference_layer_cls': praxis_linears.Linear,
          'configs': {'input_dims': 10, 'output_dims': 10},
          'inputs_fn': lambda: np.random.normal(0, 0.1, size=(32, 10)),
      },
      {
          'testcase_name': 'FeedForward',
          'reference_layer_cls': praxis_linears.FeedForward,
          'configs': {'input_dims': 3, 'output_dims': 4},
          'inputs_fn': lambda: np.random.normal(0, 0.1, size=(2, 3)),
      },
  )
  def test_calculate_grad_norms_generic(
      self, reference_layer_cls, configs, inputs_fn
  ):
    """Expected values calculated using reference layer that is wrapped."""
    ref_layer_tpl = pax_fiddle.Config(reference_layer_cls, **configs)
    ghostnorm_layer_tpl = pax_fiddle.Config(generic_wrapper.WrappedGhostNorm)
    ghostnorm_layer_tpl.layer_tpl = ref_layer_tpl.clone()

    ref_layer = instantiate(ref_layer_tpl)
    ghostnorm_layer = instantiate(ghostnorm_layer_tpl)
    inputs = jnp.asarray(inputs_fn())

    prng_key = jax.random.PRNGKey(seed=1234)
    init_key, random_key = jax.random.split(prng_key)
    ghost_initial_vars = ghostnorm_layer.init(
        {PARAMS: init_key, RANDOM: random_key}, inputs
    )
    initial_vars = {PARAMS: ghost_initial_vars[PARAMS][LAYER]}

    self.assertOutputsAreSame(
        inputs, ghost_initial_vars, initial_vars, ghostnorm_layer, ref_layer
    )

    loss_fn, loss_fn_ref = self._get_loss_fns(ghostnorm_layer, ref_layer)

    self.assertGradNormsAreSame(inputs, ghost_initial_vars, loss_fn)
    self.assertGradIsSame(
        inputs, ghost_initial_vars, initial_vars, loss_fn, loss_fn_ref, True
    )
    self.assertGradIsClippedToScale(inputs, ghost_initial_vars, loss_fn)
    self.assertClippedGradEqualsScaledGrad(inputs, ghost_initial_vars, loss_fn)


if __name__ == '__main__':
  absltest.main()
