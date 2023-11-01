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
import copy
import functools
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
from praxis import py_utils
from praxis.layers import embedding_softmax as praxis_embedding
from praxis.layers import linears as praxis_linears
from praxis.layers import transformer_models as praxis_transformer_models
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


def _remove_generic_layer_structure(pytree):
  if isinstance(pytree, dict):
    if LAYER in pytree:
      if len(pytree.keys()) != 1:
        raise ValueError('Unsupported pytree structure.')
      return pytree[LAYER]
    for key in pytree:
      pytree[key] = _remove_generic_layer_structure(pytree[key])
  return pytree


class LayersTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(123456)
    tf.random.set_seed(123)

  def _get_loss_fns(self, ghostnorm_layer, ref_layer, extract_outputs_fn=None):
    noise_key = jax.random.PRNGKey(seed=12345)

    def loss_fn(mdl_vars, *inputs_args, **inputs_kwargs):
      outputs = ghostnorm_layer.apply(
          mdl_vars, *inputs_args, **inputs_kwargs, rngs={RANDOM: noise_key}
      )
      if extract_outputs_fn:
        outputs = extract_outputs_fn(outputs)
      return mean_loss_for_testing(outputs)

    def loss_fn_ref(mdl_vars, *inputs_args, **inputs_kwargs):
      outputs = ref_layer.apply(
          mdl_vars, *inputs_args, **inputs_kwargs, rngs={RANDOM: noise_key}
      )
      if extract_outputs_fn:
        outputs = extract_outputs_fn(outputs)
      return mean_loss_for_testing(outputs)

    return loss_fn, loss_fn_ref

  def _get_per_eg_grad(
      self, initial_vars, loss_fn, *inputs_args, **inputs_kwargs
  ):
    # per-example gradients with jax.vmap
    grad_fn = jax.grad(loss_fn)
    grad_fn_with_vars = functools.partial(grad_fn, initial_vars)
    per_eg_grad_fn = jax.vmap(grad_fn_with_vars)
    vmap_inputs_args = jax.tree_map(
        lambda x: jnp.expand_dims(x, axis=1), inputs_args
    )
    vmap_inputs_kwargs = jax.tree_map(
        lambda x: jnp.expand_dims(x, axis=1), inputs_kwargs
    )
    per_eg_grad = per_eg_grad_fn(*vmap_inputs_args, **vmap_inputs_kwargs)
    return per_eg_grad

  def _get_grad_and_norms(
      self, initial_vars, loss_fn, scales, *inputs_args, **inputs_kwargs
  ):
    grad_fn = jax.grad(loss_fn)
    params_with_sq_norms = jax.tree_map(
        lambda x: base.ParamWithAux(x, scales), initial_vars[PARAMS]
    )
    params_with_sq_norms = {**initial_vars, PARAMS: params_with_sq_norms}
    grad_with_sq_norms = grad_fn(
        params_with_sq_norms, *inputs_args, **inputs_kwargs
    )[PARAMS]

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
      ghostnorm_initial_vars,
      initial_vars,
      ghostnorm_layer,
      ref_layer,
      *input_args,
      **input_kwargs,
  ):
    extract_outputs_fn = input_kwargs.pop('extract_outputs_fn', None)
    noise_key = jax.random.PRNGKey(seed=12345)

    # make sure the layer behaves the same as the reference layer in forward
    ghostnorm_outputs = ghostnorm_layer.apply(
        ghostnorm_initial_vars,
        *input_args,
        **input_kwargs,
        rngs={RANDOM: noise_key},
    )
    ref_outputs = ref_layer.apply(
        initial_vars, *input_args, **input_kwargs, rngs={RANDOM: noise_key}
    )
    if extract_outputs_fn is not None:
      ghostnorm_outputs = extract_outputs_fn(ghostnorm_outputs)
      ref_outputs = extract_outputs_fn(ref_outputs)
    np.testing.assert_allclose(
        ghostnorm_outputs,
        ref_outputs,
        rtol=1e-5,
        atol=1e-5,
        err_msg='outputs are different.',
    )

  def assertGradNormsAreSame(
      self, initial_vars, loss_fn, *inputs_args, **inputs_kwargs
  ):
    # get expected per-example gradient norms by explicitly materialize
    per_eg_grad = self._get_per_eg_grad(
        initial_vars, loss_fn, *inputs_args, **inputs_kwargs
    )
    per_eg_grad_norms = jax.vmap(optax.global_norm)(per_eg_grad)

    # first pass to compute gradient norm
    batch_size = inputs_args[0].shape[0]
    _, fast_per_eg_grad_norms = self._get_grad_and_norms(
        initial_vars,
        loss_fn,
        jnp.ones(batch_size),
        *inputs_args,
        **inputs_kwargs,
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
      ghost_initial_vars,
      initial_vars,
      loss_fn,
      loss_fn_ref,
      *inputs_args,
      **inputs_kwargs,
  ):
    unwrap_grad = inputs_kwargs.pop('unwrap_grad', False)
    batch_size = inputs_args[0].shape[0]
    grad_with_sq_norms, _ = self._get_grad_and_norms(
        ghost_initial_vars,
        loss_fn,
        jnp.ones(batch_size),
        *inputs_args,
        **inputs_kwargs,
    )
    if unwrap_grad:
      grad_with_sq_norms = _remove_generic_layer_structure(grad_with_sq_norms)

    # test if the computed gradient matches the grad of the reference layer
    grad_fn_ref = jax.grad(loss_fn_ref)
    grad_ref = grad_fn_ref(initial_vars, *inputs_args, **inputs_kwargs)[PARAMS]
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

  def assertGradIsClippedToScale(
      self, initial_vars, loss_fn, *inputs_args, **inputs_kwargs
  ):
    # first pass to compute gradient norm
    batch_size = inputs_args[0].shape[0]
    _, fast_per_eg_grad_norms = self._get_grad_and_norms(
        initial_vars,
        loss_fn,
        jnp.ones(batch_size),
        *inputs_args,
        **inputs_kwargs,
    )

    # second pass to compute norm-clipped gradients
    l2_clip = 0.2
    scales = jnp.minimum(1.0, l2_clip / fast_per_eg_grad_norms)
    _, scaled_fast_per_eg_grad_norms = self._get_grad_and_norms(
        initial_vars, loss_fn, scales, *inputs_args, **inputs_kwargs
    )

    self.assertTrue(
        np.all(scaled_fast_per_eg_grad_norms <= l2_clip + 1e-5),
        msg='norm clipping conditions are not satisfied.',
    )

  def assertClippedGradEqualsScaledGrad(
      self, initial_vars, loss_fn, *inputs_args, **inputs_kwargs
  ):
    l2_clip = 0.2
    batch_size = inputs_args[0].shape[0]
    # get expected per-example gradient norms by explicitly materialize
    per_eg_grad = self._get_per_eg_grad(
        initial_vars, loss_fn, *inputs_args, **inputs_kwargs
    )

    # compute the expected average gradients from the clipped per-example grads
    grads_flat, grads_treedef = jax.tree_flatten(per_eg_grad[PARAMS])
    sum_clipped, _ = optax.per_example_global_norm_clip(
        grads=grads_flat, l2_norm_clip=l2_clip
    )
    sum_grads = jax.tree_unflatten(grads_treedef, sum_clipped)
    expected_grads = jax.tree_map(lambda x: x / batch_size, sum_grads)

    _, fast_per_eg_grad_norms = self._get_grad_and_norms(
        initial_vars,
        loss_fn,
        jnp.ones(batch_size),
        *inputs_args,
        **inputs_kwargs,
    )
    scales = jnp.minimum(1.0, l2_clip / fast_per_eg_grad_norms)
    grad_with_sq_norms, _ = self._get_grad_and_norms(
        initial_vars, loss_fn, scales, *inputs_args, **inputs_kwargs
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
        atol=1e-2,
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
          'testcase_name': 'LinearRank3InputsFast',
          'reference_layer_cls': praxis_linears.Linear,
          'ghostnorm_layer_cls': linears.LinearGhostNorm,
          'configs': dict(input_dims=10, output_dims=10),
          'inputs_fn': lambda: np.random.normal(0, 0.1, size=(32, 2, 10)),
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
        initial_vars, initial_vars, ghostnorm_layer, ref_layer, inputs
    )

    loss_fn, loss_fn_ref = self._get_loss_fns(ghostnorm_layer, ref_layer)

    self.assertGradNormsAreSame(initial_vars, loss_fn, inputs)
    self.assertGradIsSame(
        initial_vars, initial_vars, loss_fn, loss_fn_ref, inputs
    )
    self.assertGradIsClippedToScale(initial_vars, loss_fn, inputs)
    self.assertClippedGradEqualsScaledGrad(initial_vars, loss_fn, inputs)

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
        ghost_initial_vars, initial_vars, ghostnorm_layer, ref_layer, inputs
    )

    loss_fn, loss_fn_ref = self._get_loss_fns(ghostnorm_layer, ref_layer)

    self.assertGradNormsAreSame(ghost_initial_vars, loss_fn, inputs)
    self.assertGradIsSame(
        ghost_initial_vars,
        initial_vars,
        loss_fn,
        loss_fn_ref,
        inputs,
        unwrap_grad=True,
    )
    self.assertGradIsClippedToScale(ghost_initial_vars, loss_fn, inputs)
    self.assertClippedGradEqualsScaledGrad(ghost_initial_vars, loss_fn, inputs)

  @parameterized.named_parameters(
      {
          'testcase_name': 'Causal',
          'model_type': praxis_transformer_models.LanguageModelType.CAUSAL,
      },
      {
          'testcase_name': 'Prefix',
          'model_type': praxis_transformer_models.LanguageModelType.PREFIX,
      },
  )
  def test_calculate_grad_norms_transformer(self, model_type):
    batch_size = 2
    seq_len = 10
    model_dim = 32
    vocab_size = 52
    num_heads = 4
    ref_layer_tpl = pax_fiddle.Config(
        praxis_transformer_models.TransformerLm,
        name='bert_lm',
        model_type=model_type,
        model_dims=model_dim,
        vocab_size=vocab_size,
        skip_compute_loss=False,
    )

    stacked_transformer_tpl = ref_layer_tpl.stacked_transformer_tpl
    stacked_transformer_tpl.model_dims = model_dim
    stacked_transformer_tpl.hidden_dims = num_heads * model_dim
    stacked_transformer_tpl.num_heads = num_heads
    stacked_transformer_tpl.num_layers = 1

    tr_atten_tpl = (
        stacked_transformer_tpl.transformer_layer_params_tpl.tr_atten_tpl
    )
    tr_atten_tpl.combine_qkv = True

    ref_layer_tpl.softmax_tpl.scale_sqrt_depth = True

    # Use a separate embedding as shared weights are not compatible with
    # ghost clipping.
    ref_layer_tpl.separate_embedding_tpl = pax_fiddle.Config(
        praxis_embedding.Embedding
    )

    inputs = jax.random.randint(
        jax.random.PRNGKey(1234), [batch_size, seq_len], 0, vocab_size
    )
    input_paddings = jnp.zeros([batch_size, seq_len])
    input_weights = jnp.ones([batch_size, seq_len])
    input_segment_ids = jnp.ones([batch_size, seq_len])
    input_segment_pos = jnp.tile(
        jnp.arange(0, seq_len)[jnp.newaxis, :], [batch_size, 1]
    )
    labels = py_utils.NestedMap()
    labels.class_ids = inputs
    labels.class_weights = input_weights

    ghostnorm_layer_tpl = generic_wrapper.generate_wrapped_template(
        ref_layer_tpl.clone()
    )

    ref_layer = instantiate(ref_layer_tpl)
    ghostnorm_layer = instantiate(ghostnorm_layer_tpl)

    ghost_initial_vars = ghostnorm_layer.init(
        jax.random.PRNGKey(1234),
        inputs,
        paddings=input_paddings,
        labels=labels,
        segment_ids=input_segment_ids,
        segment_pos=input_segment_pos,
    )
    initial_vars = _remove_generic_layer_structure(
        copy.deepcopy(ghost_initial_vars)
    )

    extract_outputs_fn = lambda out: out.logits

    self.assertOutputsAreSame(
        ghost_initial_vars,
        initial_vars,
        ghostnorm_layer,
        ref_layer,
        inputs,
        paddings=input_paddings,
        labels=labels,
        segment_ids=input_segment_ids,
        segment_pos=input_segment_pos,
        extract_outputs_fn=extract_outputs_fn,
    )

    loss_fn, loss_fn_ref = self._get_loss_fns(
        ghostnorm_layer, ref_layer, extract_outputs_fn=extract_outputs_fn
    )

    self.assertGradNormsAreSame(
        ghost_initial_vars,
        loss_fn,
        inputs,
        paddings=input_paddings,
        labels=labels,
        segment_ids=input_segment_ids,
        segment_pos=input_segment_pos,
    )
    self.assertGradIsSame(
        ghost_initial_vars,
        initial_vars,
        loss_fn,
        loss_fn_ref,
        inputs,
        paddings=input_paddings,
        labels=labels,
        segment_ids=input_segment_ids,
        segment_pos=input_segment_pos,
        unwrap_grad=True,
    )
    self.assertGradIsClippedToScale(
        ghost_initial_vars,
        loss_fn,
        inputs,
        paddings=input_paddings,
        labels=labels,
        segment_ids=input_segment_ids,
        segment_pos=input_segment_pos,
    )
    self.assertClippedGradEqualsScaledGrad(
        ghost_initial_vars,
        loss_fn,
        inputs,
        paddings=input_paddings,
        labels=labels,
        segment_ids=input_segment_ids,
        segment_pos=input_segment_pos,
    )


if __name__ == '__main__':
  absltest.main()
