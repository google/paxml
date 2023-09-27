import os
from contextlib import contextmanager
from typing import Optional, Sequence

import jax
import jax.numpy as jnp
from jax.ad_checkpoint import checkpoint_name
from praxis import base_layer
from praxis import pax_fiddle
from praxis import pytypes
from praxis.layers import transformers
from praxis.layers import stochastics

try:
  import transformer_engine.jax as te
  import transformer_engine.jax.flax as te_flax
  import transformer_engine.jax.praxis as te_praxis
  from transformer_engine.common import recipe
  _IS_TRANSFORMER_ENGINE_INSTALLED = True
  DEFAULT_INIT_MUTABLE_LIST = base_layer.DEFAULT_INIT_MUTABLE_LIST + [te.fp8.FP8Helper.FP8_COLLECTION_NAME]
  import praxis.layers.repeats as praxis_repeat
  # This is to make Repeat module correctly generate collections we need.
  praxis_repeat.SCAN_VARIABLE_AXES.update({base_layer.NON_PAX_VAR_COLLECTION[1]: 0, # 1-idx = params_axes
                                           te.fp8.FP8Helper.FP8_COLLECTION_NAME:0})

except ModuleNotFoundError as e:
  _IS_TRANSFORMER_ENGINE_INSTALLED = False
  DEFAULT_INIT_MUTABLE_LIST = base_layer.DEFAULT_INIT_MUTABLE_LIST


LayerTpl = pax_fiddle.Config[base_layer.BaseLayer]
JTensor = pytypes.JTensor

class StackedTransformer(transformers.StackedTransformer):
  """A mirror of StackedTransformer layers in Praxis."""

  def setup(self) -> None:

    assert self.num_layers > 0
    assert self.model_dims > 0
    assert self.hidden_dims > 0
    assert self.num_heads > 0
    assert 0.0 <= self.dropout_prob < 1.0
    assert 0.0 <= self.input_dropout_prob < 1.0

    def _layer_params(i):
        """Construct i-th layer params."""
        if isinstance(self.transformer_layer_params_tpl, Sequence):
            factor = self.num_layers // len(self.transformer_layer_params_tpl)
            ii = i // factor
            p_i = self._clone_layer_params(self.transformer_layer_params_tpl[ii])
        else:
            p_i = self._clone_layer_params(self.transformer_layer_params_tpl)
        p_i.name = f'layer_{i}'

        p_i.logical_axes_rules = te_flax.extend_logical_axis_rules(tuple())
        p_i.layer_type = te_praxis.TransformerLayerType.DECODER if self.use_cross_attention \
                        else te_praxis.TransformerLayerType.ENCODER
        p_i.num_attention_heads = self.num_heads
        p_i.hidden_size = self.model_dims
        p_i.mlp_hidden_size = self.hidden_dims

        p_i.dropout_rng_name = base_layer.RANDOM
        p_i.attention_dropout = self.atten_dropout_prob or self.dropout_prob
        p_i.hidden_dropout = self.residual_dropout_prob or self.dropout_prob
        p_i.intermediate_dropout = self.relu_dropout_prob or self.dropout_prob
        if self.residual_droppath_prob > 0.0:
            p_i.drop_path = (
                self.residual_droppath_prob * i / max(1, self.num_layers)
            )

        assert self.dim_per_head == self.model_dims // self.num_heads
        assert self.packed_input == False
        assert len(self.moe_layers) == 0
        assert self.ngrammer_tpls is None

        if self.ngrammer_tpls is not None:
            if self.ngrammer_tpls[i] is not None:
                p_i.ngrammer_tpl = self.ngrammer_tpls[i]
        return p_i

    if isinstance(self.transformer_layer_params_tpl, (list, tuple)):
        if self.num_layers % len(self.transformer_layer_params_tpl):
            raise ValueError('num_layers should be divisible by '
                                'transformer_layer_params_tpl')

    layer_params = [_layer_params(i) for i in range(self.num_layers)]
    self.create_children('x_layers', layer_params)

    if self.input_dropout_prob > 0.0:
        self.create_child(
            'input_dropout',
            pax_fiddle.Config(
                  stochastics.Dropout, keep_prob=1.0 - self.input_dropout_prob
            ),
        )

  def __call__(self,
               inputs: JTensor,
               paddings: JTensor,
               segment_mask: Optional[JTensor] = None,
               cross_inputs: Optional[JTensor] = None,
               cross_paddings: Optional[JTensor] = None,
               cross_segment_mask: Optional[JTensor] = None,
               segment_pos: Optional[JTensor] = None) -> JTensor:

    if self.packed_input:
        assert segment_mask is not None

    if self.use_cross_attention:
        assert cross_inputs is not None
        assert cross_paddings is not None
        if self.packed_input:
            assert cross_segment_mask is not None

    attention_mask, cross_attention_mask = transformers.compute_attention_masks_for_fprop(
        inputs,
        paddings,
        self.mask_self_attention,
        segment_mask,
        cross_inputs,
        cross_paddings,
        cross_segment_mask,
        fold_padding_with_segment_mask=self.fold_padding_with_segment_mask,
    )

    x_out = inputs
    if self.input_dropout_prob > 0.0:
        x_out = self.input_dropout(x_out)

    attention_mask = 1 - (attention_mask == 0)
    attention_mask = attention_mask.astype(jnp.uint8)

    if cross_attention_mask is not None:
        cross_attention_mask = 1 - (cross_attention_mask == 0)
        cross_attention_mask = cross_attention_mask.astype(jnp.uint8)

    for i in range(self.num_layers):
        x_in = x_out
        x_out = self.x_layers[i](
            inputs=x_in,
            attention_mask=attention_mask,
            encoded=cross_inputs,
            encoder_decoder_mask=cross_attention_mask)
        x_out = checkpoint_name(x_out, 'transformer_layer_out')
    return x_out


class TransformerEngineHelperBase:

    @staticmethod
    def get_stack_transformer(stacked_transformer_p, dtype):
        raise NotImplementedError

    @staticmethod
    def update_fp8_metas_if_needed(mdl_vars, grads):
        raise NotImplementedError

    @staticmethod
    def include_fp8_for_grads_if_needed(variables):
        raise NotImplementedError

    @staticmethod
    def mask_out_fp8_meta_grads_if_needed(grads, vars_with_opt):
        raise NotImplementedError

    @staticmethod
    @contextmanager
    def fp8_autocast(dp_mesh_axis="replica", tp_mesh_axis="mdl", fsdp_mesh_axis="data"):
        raise NotImplementedError


class TENotInstalledHelper(TransformerEngineHelperBase):

    @staticmethod
    def get_stack_transformer(stacked_transformer_p, dtype):
        return stacked_transformer_p

    @staticmethod
    def update_fp8_metas_if_needed(mdl_vars, grads):
        return mdl_vars

    @staticmethod
    def include_fp8_for_grads_if_needed(variables):
        return variables

    @staticmethod
    def mask_out_fp8_meta_grads_if_needed(grads, vars_with_opt):
        return grads

    @staticmethod
    @contextmanager
    def fp8_autocast(dp_mesh_axis="replica", tp_mesh_axis="mdl", fsdp_mesh_axis="data"):
        try:
            yield
        finally:
            pass


class TEInstalledHelper(TransformerEngineHelperBase):

    @staticmethod
    def get_stack_transformer(stacked_transformer_p, dtype):

        assert stacked_transformer_p.cls == transformers.StackedTransformer

        te_stacked_transformer_p = pax_fiddle.Config(StackedTransformer,
            use_cross_attention=stacked_transformer_p.use_cross_attention,
            mask_self_attention=stacked_transformer_p.mask_self_attention,
            num_layers=stacked_transformer_p.num_layers,
            model_dims=stacked_transformer_p.model_dims,
            hidden_dims=stacked_transformer_p.hidden_dims,
            num_heads=stacked_transformer_p.num_heads,
            dim_per_head=stacked_transformer_p.dim_per_head,
            dropout_prob=stacked_transformer_p.dropout_prob,
            atten_dropout_prob=stacked_transformer_p.atten_dropout_prob,
            residual_dropout_prob=stacked_transformer_p.residual_dropout_prob,
            relu_dropout_prob=stacked_transformer_p.relu_dropout_prob,
            residual_droppath_prob=stacked_transformer_p.residual_droppath_prob,
            input_dropout_prob=stacked_transformer_p.input_dropout_prob,
            gating_func=stacked_transformer_p.gating_func,
            unadjusted_expert_capacity_factor=stacked_transformer_p.unadjusted_expert_capacity_factor,
            packed_input=stacked_transformer_p.packed_input,
            fold_padding_with_segment_mask=stacked_transformer_p.fold_padding_with_segment_mask,
            moe_layer_tpl=stacked_transformer_p.moe_layer_tpl,
            num_experts=stacked_transformer_p.num_experts,
            num_groups=stacked_transformer_p.num_groups,
            min_group_size=stacked_transformer_p.min_group_size,
            moe_layers=stacked_transformer_p.moe_layers,
            ngrammer_tpls=stacked_transformer_p.ngrammer_tpls
        )

        ori_transformer_engine_p = stacked_transformer_p.transformer_layer_params_tpl

        te_stacked_transformer_p.transformer_layer_params_tpl = pax_fiddle.Config(te_praxis.TransformerLayer,
            name='transformer_layer',
            params_init=stacked_transformer_p.params_init,
            dtype=dtype,
            hidden_size=stacked_transformer_p.model_dims,
            mlp_hidden_size=stacked_transformer_p.hidden_dims,
            num_attention_heads=stacked_transformer_p.num_heads,
            layernorm_type='layernorm',
            layernorm_epsilon=ori_transformer_engine_p.ln_tpl.epsilon,
            zero_centered_gamma = True,
            hidden_dropout=ori_transformer_engine_p.residual_dropout_prob,
            attention_dropout=ori_transformer_engine_p.atten_dropout_prob,
            mlp_activations=('gelu',),
            use_bias=True,
            layer_type=te_praxis.TransformerLayerType.ENCODER,
            self_attn_mask_type='causal',
            enable_relative_embedding=False,
            drop_path=ori_transformer_engine_p.residual_droppath_prob,
            scaled_query_init=False,
            scale_attn_logits=True,
            transpose_batch_sequence=False
        )

        return te_stacked_transformer_p

    @staticmethod
    def update_fp8_metas_if_needed(mdl_vars, grads):
        FP8_COLLECTION_NAME = te.fp8.FP8Helper.FP8_COLLECTION_NAME
        if FP8_COLLECTION_NAME in grads:
            mdl_vars[FP8_COLLECTION_NAME] = te.update_fp8_metas(grads)[FP8_COLLECTION_NAME]
        return mdl_vars

    @staticmethod
    def include_fp8_for_grads_if_needed(variables):
        FP8_COLLECTION_NAME = te.fp8.FP8Helper.FP8_COLLECTION_NAME
        if FP8_COLLECTION_NAME in variables:
            variables[FP8_COLLECTION_NAME] = \
                jax.tree_util.tree_map(lambda x: False, variables[FP8_COLLECTION_NAME])
        return variables

    @staticmethod
    def mask_out_fp8_meta_grads_if_needed(grads, vars_with_opt):
        FP8_COLLECTION_NAME = te.fp8.FP8Helper.FP8_COLLECTION_NAME
        if FP8_COLLECTION_NAME in grads:
            grads[FP8_COLLECTION_NAME] = vars_with_opt[FP8_COLLECTION_NAME].copy()
        return grads

    @staticmethod
    @contextmanager
    def fp8_autocast(dp_mesh_axis="replica", tp_mesh_axis="mdl", fsdp_mesh_axis="data"):
        fp8_recipe = recipe.DelayedScaling(margin=0, interval=1, fp8_format=recipe.Format.HYBRID,
                                           amax_history_len=1024, amax_compute_algo='max')

        enable_fp8 = bool(int((os.environ.get("ENABLE_FP8", False))))
        try:
            with te.fp8_autocast(enabled=enable_fp8,
                                 fp8_recipe=fp8_recipe,
                                 sharding_resource=te.ShardingResource(dp_mesh_axis, tp_mesh_axis, fsdp_mesh_axis)):
                yield
        finally:
            pass


class TransformerEngineHelper(TransformerEngineHelperBase):

    @staticmethod
    def is_enabled_te():
        enable_te = bool(int((os.environ.get("ENABLE_TE", False))))
        return (_IS_TRANSFORMER_ENGINE_INSTALLED and enable_te)

    @staticmethod
    def get_helper():
        if TransformerEngineHelper.is_enabled_te():
            return TEInstalledHelper
        return TENotInstalledHelper

    @staticmethod
    def get_stack_transformer(stacked_transformer_p, dtype):
        return TransformerEngineHelper.get_helper().get_stack_transformer(stacked_transformer_p, dtype)

    @staticmethod
    def update_fp8_metas_if_needed(mdl_vars, grads):
        return TransformerEngineHelper.get_helper().update_fp8_metas_if_needed(mdl_vars, grads)

    @staticmethod
    def include_fp8_for_grads_if_needed(variables):
        return TransformerEngineHelper.get_helper().include_fp8_for_grads_if_needed(variables)

    @staticmethod
    def mask_out_fp8_meta_grads_if_needed(grads, vars_with_opt):
        return TransformerEngineHelper.get_helper().mask_out_fp8_meta_grads_if_needed(grads, vars_with_opt)

    @staticmethod
    @contextmanager
    def fp8_autocast(dp_mesh_axis="replica", tp_mesh_axis="mdl", fsdp_mesh_axis="data"):
        try:
            with TransformerEngineHelper.get_helper().fp8_autocast(dp_mesh_axis, tp_mesh_axis, fsdp_mesh_axis):
                yield
        finally:
            pass
