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

"""Module with the Learner class."""

from __future__ import annotations

import dataclasses
import re
from typing import Optional, Sequence, Tuple, Union

import jax
from jax import numpy as jnp
import optax
from paxml import sgf
from praxis import asserts
from praxis import base_hyperparams
from praxis import base_layer
from praxis import optimizer_prefix_vectorization as opt_vec
from praxis import optimizers
from praxis import py_utils

JTensor = jnp.ndarray
NestedMap = py_utils.NestedMap
NestedJTensor = base_layer.NestedJTensor
NestedBool = base_layer.NestedBool
NestedWeightHParams = base_layer.NestedWeightHParams
SummaryType = base_layer.SummaryType
InstantiableHyperParams = base_hyperparams.InstantiableHyperParams
instantiate = base_hyperparams.instantiate


def _compute_grad_norm(grads: NestedMap) -> JTensor:
  """Computes total grad norm."""
  grad_norms_squared = jax.tree_map(lambda x: jnp.sum(x * x), grads)
  grad_norms_squared, _ = jax.tree_util.tree_flatten(grad_norms_squared)
  return jnp.sqrt(jnp.sum(jnp.stack(grad_norms_squared)))


class Learner(base_hyperparams.BaseParameterizable):
  """A learner.

  Example client code:

  p = Learner.HParams().set(...)
  learner = base_hyperparams.instantiate(p)

  mdl_vars = ...
  var_weight_hparams = jax.tree_map(
    lambda v: base_layer.WeightHParams(v.shape), mdl_vars)

  grad_tx = learner.get_grad_tx(var_weight_hparams)
  opt_states0 = grad_tx.init(mdl_vars)

  grads0 = ...
  grads1, opt_states1 = learner.update_states(
    grads0, opt_states0, mdl_vars, var_weight_hparams)
  updated_mdl_vars = learner.apply_gradient(
    mdl_vars, grads1, var_weight_hparams)
  """

  class HParams(base_hyperparams.BaseParameterizable.HParams):
    """Returns the Learner params.

    Attributes:
      loss_name: Name of the loss this learner optimizes. If the task has a
        loss_aggregator this param will be ignored and is expected to be None,
        otherwise it must be set. This loss_name must be in the metrics dict
        (the first return of compute_loss).
      stochastic_gradient: Params for the stochastic gradient function.
      optimizer: Params for the optimizer.
      skip_zero_gradients: If set, skips aggregating zero gradients while
        computing gradients.This helps in case where some weights may not be
        used in forward computation, e.g., sparsely activated networks or
        switchable layers in neural architectural search. Possible values are:
        None: do not skip zero gradients; "variable": skip if the entire
          variable gradients are almost zero.
      grad_norm_summary: Whether or not to export accumulated grad_norm
        summaries. Disable to save some compute.
      grad_norm_individual_vars: Whether or not to export grad_norm for each
        individual variable as summaries.
      var_norm_summary: Whether or not to export accumulated var_norm summaries.
        Disable to save some compute.
      check_valid_step: Whether or not to run sanity check to ensure that the
        training step is valid.
      vectorize_on_repeat_prefix: Whether to vectorize optimizers on the
        repeat_prefix dims of the variables. This allows stacking variables of
        different layers while not affecting the behavior of optimizers like
        Adafactor.
      skip_step_gradient_norm_value: If non-zero, we skip a step entirely if
        gradient_norm exceeds this value.
      enable_skip_step_on_gradient_anomalies: Skips the step if gradient anomaly
        (NaN/Inf) is detected.
      bprop_variable_exclusion: Regular expression or a list of regular
        expressions. If a variable name matches one of the regular expressions,
        the variable should be fixed during model training.
      repeat_prefix_sep: Repeat prefix separator character, for use in filename
        separator during checkpointing.
    """

    # TODO(pax): loss_name is not used anywhere anymore other than to
    # create a LossAggregator on the task. Consider moving loss_name to the
    # task or having everyone set it through the loss_aggregator.
    loss_name: Optional[str] = None
    stochastic_gradient: Optional[sgf.BaseStochasticGradient.HParams] = None
    optimizer: Optional[optimizers.BaseOptimizer.HParams] = None
    skip_zero_gradients: Optional[bool] = None
    grad_norm_summary: bool = True
    grad_norm_individual_vars: bool = False
    var_norm_summary: bool = True
    check_valid_step: bool = True
    vectorize_on_repeat_prefix: bool = True
    skip_step_gradient_norm_value: float = 0.0
    enable_skip_step_on_gradient_anomalies: bool = True
    bprop_variable_exclusion: Union[str, Sequence[str]] = dataclasses.field(
        default_factory=list
    )
    repeat_prefix_sep: str = '#'

  def __init__(self, hparams: Learner.HParams) -> None:
    """Constructor for the learner."""
    assert hparams.name, (
        'Learner params for %s must have a "name"' % self.__class__.__name__
    )
    super().__init__(hparams)
    module_name = hparams.name
    NestedMap.CheckKey(module_name)

    p = self._hparams
    asserts.not_none(p.optimizer)
    self._optimizer_inst = instantiate(p.optimizer)
    self._stochastic_gradient_inst = (
        None
        if p.stochastic_gradient is None
        else instantiate(p.stochastic_gradient)
    )
    self._get_grad_tx = self.optimizer.get_grad_transformation

  @property
  def optimizer(self) -> optimizers.BaseOptimizer:
    """Returns the Optimizer object of this learner."""
    return self._optimizer_inst

  @property
  def stochastic_gradient(self) -> Optional[sgf.BaseStochasticGradient]:
    """Returns the stochastic gradient function object of this learner."""
    return self._stochastic_gradient_inst

  def plot_learning_rate(self, step: int) -> None:
    learning_rate = self.optimizer.get_learning_rate(step)
    base_layer.add_global_summary(
        'lr', learning_rate, SummaryType.AGGREGATE_SCALAR
    )
    base_layer.add_global_summary(
        'learning/lr', learning_rate, SummaryType.AGGREGATE_SCALAR
    )

  def get_grad_tx(
      self, var_weight_hparams: NestedWeightHParams
  ) -> optimizers.GeneralGradientTransformation:
    # Apply vectorization on prefix dims.
    if not self._hparams.vectorize_on_repeat_prefix:
      return self._get_grad_tx(var_weight_hparams)
    return opt_vec.get_transformations_with_vectorized_repeat_prefix(
        self._get_grad_tx(var_weight_hparams),
        var_weight_hparams,
        self._hparams.repeat_prefix_sep,
    )

  def scale_gradients(
      self,
      raw_grads: NestedMap,
      optimizer_name: Optional[str] = None,
      clip_gradient_norm_to_value: Optional[float] = None,
      clip_gradient_single_norm_to_value: Optional[float] = None,
  ) -> Tuple[NestedMap, JTensor]:
    """Scales the gradient.

    Args:
      raw_grads: A nested structure of gradient values.
      optimizer_name: Optional. If None no prefix is used. otherwise it starts
        with the name of optimizers. Doesn't include '/'.
      clip_gradient_norm_to_value: Optional. If None,
        p.optimizer.clip_gradient_norm_to_value will be used.
      clip_gradient_single_norm_to_value: Optional. If None,
        p.optimizer.clip_gradient_single_norm_to_value will be used.

    Returns:
     A nested structure with the rescaled gradient values.
     A predicate tensor indicating whether the step is valid, i.e., it does not
       have anomaly detected (e.g. Nan or Inf, or excessively big gradient norm)
       and should not be skipped.
    """
    p = self._hparams
    if optimizer_name is None:
      optimizer_name = ''
    else:
      optimizer_name = optimizer_name + '/'
    if clip_gradient_norm_to_value is None:
      clip_gradient_norm_to_value = p.optimizer.clip_gradient_norm_to_value
    if clip_gradient_single_norm_to_value is None:
      clip_gradient_single_norm_to_value = (
          p.optimizer.clip_gradient_single_norm_to_value
      )
    # Compute gradient norm.

    if p.grad_norm_individual_vars:
      grad_norms = jax.tree_map(lambda x: jnp.sqrt(jnp.sum(x * x)), raw_grads)
      var_keys = py_utils.extract_prefixed_keys_from_nested_map(grad_norms)

      def add_grad_norm_summary(key, value):
        base_layer.add_global_summary(
            f'per_var_grad_norm/{optimizer_name}{key}',
            value,
            SummaryType.AGGREGATE_SCALAR,
        )

      jax.tree_map(add_grad_norm_summary, var_keys, grad_norms)

    if (
        p.grad_norm_summary
        or p.check_valid_step
        or clip_gradient_norm_to_value
        or clip_gradient_single_norm_to_value
    ):
      raw_grad_norm = _compute_grad_norm(raw_grads)
      if p.grad_norm_summary:
        base_layer.add_global_summary(
            'learning/' + optimizer_name + 'raw_grad_norm',
            raw_grad_norm,
            SummaryType.AGGREGATE_SCALAR,
        )
    else:
      raw_grad_norm = None

    def keep_step(grad_norm):
      keep_threshold = p.skip_step_gradient_norm_value
      if keep_threshold:
        return jnp.logical_and(
            jnp.all(jnp.isfinite(grad_norm)),
            jnp.all(jnp.less(grad_norm, keep_threshold)),
        )
      else:
        return jnp.all(jnp.isfinite(grad_norm))

    def clip_grads(grads, grad_norm):
      if clip_gradient_norm_to_value:
        assert clip_gradient_single_norm_to_value == 0.0
        grad_scale = jnp.minimum(
            jnp.array(1, grad_norm.dtype),
            jnp.array(clip_gradient_norm_to_value, grad_norm.dtype) / grad_norm,
        )
        grads = jax.tree_map(lambda g: g * grad_scale, grads)
      elif clip_gradient_single_norm_to_value:
        assert clip_gradient_norm_to_value == 0.0
        grad_single_norm = jax.tree_map(
            lambda x: jnp.sqrt(jnp.sum(x * x)), grads
        )

        def scale_gradient(grad, norm):
          return grad * jnp.minimum(
              jnp.array(1, norm.dtype),
              jnp.array(clip_gradient_single_norm_to_value, norm.dtype) / norm,
          )

        grads = jax.tree_map(scale_gradient, grads, grad_single_norm)
        grad_scale = jnp.array(1.0)
      else:
        # no clipping is needed.
        grad_scale = jnp.array(1.0)
      return grads, grad_scale

    if p.check_valid_step:
      # Mark the step as invalid if any gradient anomaly is detected (e.g. Nan
      # or Inf, or excessively big gradient norm).
      valid_step = keep_step(raw_grad_norm)
      base_layer.add_global_summary(
          'learning/' + optimizer_name + 'is_valid_step',
          valid_step.astype(jnp.float32),
          SummaryType.AGGREGATE_SCALAR,
      )
    else:
      valid_step = True
    grads, grad_scale = clip_grads(raw_grads, raw_grad_norm)
    base_layer.add_global_summary(
        'learning/' + optimizer_name + 'grad_scale',
        grad_scale,
        SummaryType.AGGREGATE_SCALAR,
    )

    if p.grad_norm_summary:
      clipped_grad_norm = _compute_grad_norm(grads)
      base_layer.add_global_summary(
          'learning/' + optimizer_name + 'clipped_grad_norm',
          clipped_grad_norm,
          SummaryType.AGGREGATE_SCALAR,
      )
    return grads, valid_step

  def update_states(
      self,
      grads: NestedMap,
      states: optax.OptState,
      old_vars: NestedJTensor,
      var_weight_hparams: NestedWeightHParams,
  ) -> Tuple[NestedMap, optax.OptState]:
    """Applies gradient transformation, updates optimizer states.

    Args:
      grads: A nested structure of gradient values.
      states: Optimizer states.
      old_vars: Current model weights.
      var_weight_hparams: Weight params of the vars.

    Returns:
      transformed_grad, new_states pair.
    """
    p = self._hparams

    grads, valid_step = self.scale_gradients(grads)
    transformed_grad, new_states = self.get_grad_tx(var_weight_hparams).update(
        grads, states, old_vars
    )

    if p.enable_skip_step_on_gradient_anomalies:
      # Set grads to 0 if the step is invalid.
      transformed_grad = jax.tree_map(
          lambda x: jnp.where(valid_step, x, jnp.zeros_like(x)),
          transformed_grad,
      )

      # Keep the old state if the step is invalid.
      def _update(updated, original):
        if any([py_utils.is_optax_masked_node(x) for x in (updated, original)]):
          return updated
        else:
          return jnp.where(valid_step, updated, original)

      new_states = jax.tree_map(
          _update, new_states, states, is_leaf=py_utils.is_optax_masked_node
      )
    # Final applied grad norm.
    if p.grad_norm_summary:
      applied_grad_norm = _compute_grad_norm(transformed_grad)
      base_layer.add_global_summary(
          'learning/applied_grad_norm',
          applied_grad_norm,
          SummaryType.AGGREGATE_SCALAR,
      )
    return transformed_grad, new_states

  def apply_gradient(
      self,
      old_vars: NestedJTensor,
      transformed_grads: NestedJTensor,
      var_weight_hparams: NestedWeightHParams,
  ) -> NestedJTensor:
    """Applies grads to model_variables.

    Note, in a flax model learnable variables are often referred to as 'params'.
    But since 'params' in Lingvo often refers to a hyperparams.HParams, we
    refer to learnable weights of a network as 'variables'.

    Args:
      old_vars: a nested structure of model variables.
      transformed_grads: grads of loss wrt to the old_vars. Must be of the same
        structure as old_var. 'transformed_grads' have already gone through
        various gradient transformations.
      var_weight_hparams: a nested structure of variable weight params. Must be
        of the same structure as old_vars. A variable weight param contains all
        the meta information about a variable.

    Returns:
      updated variables. Only learnable variables are updated.
    """
    p = self._hparams
    asserts.assert_same_structure(old_vars, transformed_grads)
    asserts.assert_same_structure(old_vars, var_weight_hparams)

    assert p.skip_zero_gradients is None

    if p.var_norm_summary:
      # Add a summary of total var norm.
      var_squared = jax.tree_map(lambda x: jnp.sum(x * x), old_vars)
      var_squared, _ = jax.tree_util.tree_flatten(var_squared)
      var_squared = jnp.concatenate([x[jnp.newaxis] for x in var_squared])
      var_norm = jnp.sqrt(jnp.sum(var_squared))
      base_layer.add_global_summary(
          'learning/var_norm', var_norm, SummaryType.AGGREGATE_SCALAR
      )

    # TODO(yonghui): implement skip_zero_gradients.
    # TODO(yonghui): implement numerical checks.

    def _adjust_var(old_var, transformed_grad, is_learnable):
      if is_learnable:
        return old_var + transformed_grad
      else:
        return old_var

    var_is_learnable = jax.tree_util.tree_map(
        lambda x: not base_layer.var_not_trainable(x), var_weight_hparams
    )

    return jax.tree_util.tree_map(
        _adjust_var, old_vars, transformed_grads, var_is_learnable
    )
    # TODO(yonghui): export gradient / variable summaries.

  @property
  def loss_name(self) -> str:
    return self._hparams.loss_name


class MultiOptimizerLearner(Learner):
  """Multi-optimizer learner which supports multiple optimizers.

  This class assumes a primary optimizer as in the Learner class. To add more
  optimizers, call the `MultiOptimizerLearner` class with a list of
  `auxiliary_optimizers` which are applied on variables matched using the list
  `auxiliary_regex`, which will walk the PyTree of variables, and set all
  descendants to use the corresponding auxiliary optimizer. Note that the length
  of `auxiliary_optimizers` and `auxiliary_regex` must be the same.
  """

  class HParams(Learner.HParams):
    """HParams for MultiOptimizerLearner.

    Attributes:
      auxiliary_optimizers: Additional auxiliary optimizers for optimizing a
        subset of model variables.
      auxiliary_regex: A regular expression which if matches the variable name,
        will activate the corresponding auxiliary optimizer. The length of this
        list must be the same as auxiliary optimiers.
      auxiliary_names: Names of all auxiliary optimizers. This is mainly used
        for tensorboard.
      apply_separate_scaling: Whether to apply gradient scaling separately for
        each auxiliary optimizer. By default, all gradients are scaled together
        so all configurations under auxiliary optimizers are ignored.
    """

    auxiliary_optimizers: Sequence[optimizers.BaseOptimizer.HParams] = ()
    auxiliary_regex: Sequence[str] = ()
    auxiliary_names: Sequence[str] = ()
    apply_separate_scaling: bool = False

  def __init__(self, hparams: MultiOptimizerLearner.HParams) -> None:
    """Constructor for the MultiOptimizer learner."""
    super().__init__(hparams)
    p = self._hparams
    asserts.not_none(p.optimizer)
    if len(p.auxiliary_optimizers) != len(p.auxiliary_regex) or len(
        p.auxiliary_regex
    ) != len(p.auxiliary_names):
      raise ValueError(
          f'The length of the {p.auxiliary_regex} must match the length'
          f' of the {p.auxiliary_optimizers} and length of the '
          f'{p.auxiliary_names}.'
      )
    self._optimizer_inst = instantiate(p.optimizer)
    self._auxiliary_optimizer_insts = [
        instantiate(opt) for opt in p.auxiliary_optimizers
    ]
    self._grad_tx_fn = self._optimizer_inst.get_grad_transformation
    self._auxiliary_grad_tx_fn = [
        opt.get_grad_transformation for opt in self._auxiliary_optimizer_insts
    ]

  def plot_learning_rate(self, step: int) -> None:
    p = self._hparams
    learning_rate = self.optimizer.get_learning_rate(step)
    base_layer.add_global_summary(
        'learning/lr_main', learning_rate, SummaryType.AGGREGATE_SCALAR
    )

    for name, optimizer in zip(
        p.auxiliary_names, self._auxiliary_optimizer_insts
    ):
      learning_rate = optimizer.get_learning_rate(step)
      base_layer.add_global_summary(
          f'learning/lr_{name}', learning_rate, SummaryType.AGGREGATE_SCALAR
      )

  def get_masks(
      self, var_weight_hparams: NestedWeightHParams
  ) -> Tuple[Sequence[NestedMap], NestedMap]:
    p = self._hparams
    optimizer_mask = []

    # Aggregate all the auxiliary optimizer masks.
    for regex, grad_tx_fn in zip(p.auxiliary_regex, self._auxiliary_grad_tx_fn):
      regexp = re.compile(regex)
      prefix = py_utils.extract_prefixed_keys_from_nested_map(
          var_weight_hparams
      )
      mask = jax.tree_map(
          lambda x, regexp=regexp: regexp.match(x) is not None, prefix
      )
      optimizer_mask.append(mask)

    # Create the default optimizer mask.
    def check_var_in_auxiliary_regex(*args):
      """Check if a variable is already activated by an auxiliary optimizer."""
      r = False
      for x in args:
        if x:
          # Check if it was already activated by some other optimizer.
          if r:
            raise ValueError(
                'The regex pattern of auxiliary optimizers should'
                'be non-overlapping.'
            )
          r = True
      return r

    default_mask = jax.tree_map(check_var_in_auxiliary_regex, *optimizer_mask)
    default_mask = jax.tree_map(lambda mask: not mask, default_mask)

    return optimizer_mask, default_mask

  def get_grad_tx(
      self, var_weight_hparams: NestedWeightHParams
  ) -> optimizers.GeneralGradientTransformation:
    """The gradient transformation the MultiOptimizer lerner.

    Args:
      var_weight_hparams: The model vars' params which will be used to filter
        using the regex to determine which optimizer will be applied to which
        variable.

    Returns:
      Optax sharded gradient transformation.
    """
    p = self._hparams
    optimizer_chain = []
    optimizer_mask, default_mask = self.get_masks(var_weight_hparams)

    for mask, grad_tx_fn in zip(optimizer_mask, self._auxiliary_grad_tx_fn):
      optimizer_chain.append(
          optimizers.sharded_masked(
              grad_tx_fn(var_weight_hparams, include_ema=False), mask
          )
      )

    optimizer_chain.insert(
        0,
        optimizers.sharded_masked(
            self._grad_tx_fn(var_weight_hparams, include_ema=False),
            default_mask,
        ),
    )

    # Include ema in the beginning before masking
    op = self._optimizer_inst.hparams
    if op.ema_decay > 0.0:
      optimizer_chain.insert(
          0, optimizers.apply_ema_weights(decay=op.ema_decay)
      )

    grad_tx = optimizers.sharded_chain(*optimizer_chain)
    # Finally, apply vectorization on prefix dims.
    if p.vectorize_on_repeat_prefix:
      grad_tx = opt_vec.get_transformations_with_vectorized_repeat_prefix(
          grad_tx, var_weight_hparams
      )
    return grad_tx

  def scale_gradients_by_optimizer(
      self, raw_grads: NestedMap, var_weight_hparams: NestedWeightHParams
  ) -> Tuple[NestedMap, JTensor]:
    optimizer_mask, default_mask = self.get_masks(var_weight_hparams)

    all_grads, all_valid_step = self.scale_gradients(
        jax.tree_map(lambda x, y: x * y, raw_grads, default_mask),
        optimizer_name='main',
    )

    for name, mask, optimizer in zip(
        self._hparams.auxiliary_names,
        optimizer_mask,
        self._hparams.auxiliary_optimizers,
    ):
      assert optimizer.clip_gradient_norm_to_value is not None
      assert optimizer.clip_gradient_single_norm_to_value is not None
      grads, valid_step = self.scale_gradients(
          jax.tree_map(lambda x, y: x * y, raw_grads, mask),
          optimizer_name=name,
          clip_gradient_norm_to_value=optimizer.clip_gradient_norm_to_value,
          clip_gradient_single_norm_to_value=optimizer.clip_gradient_single_norm_to_value,
      )
      all_grads = jax.tree_map(lambda x, y: x + y, all_grads, grads)
      all_valid_step = jnp.logical_and(all_valid_step, valid_step)
    return all_grads, all_valid_step

  def update_states(
      self,
      grads: NestedMap,
      states: optax.OptState,
      old_vars: NestedJTensor,
      var_weight_hparams: NestedWeightHParams,
  ) -> Tuple[NestedMap, optax.OptState]:
    """Applies gradient transformation, updates optimizer states.

    Args:
      grads: A nested structure of gradient values.
      states: Optimizer states.
      old_vars: Current model weights.
      var_weight_hparams: Weight params of the vars.

    Returns:
      transformed_grad, new_states pair.
    """
    if self._hparams.apply_separate_scaling:
      grads, valid_step = self.scale_gradients_by_optimizer(
          grads, var_weight_hparams
      )
    else:
      grads, valid_step = self.scale_gradients(grads)
    grad_tx = self.get_grad_tx(var_weight_hparams)
    transformed_grad, new_states = grad_tx.update(grads, states, old_vars)
    if self._hparams.enable_skip_step_on_gradient_anomalies:
      # Set grads to 0 if the step is invalid.
      transformed_grad = jax.tree_map(
          lambda x: jnp.where(valid_step, x, jnp.zeros_like(x)),
          transformed_grad,
      )

      # Keep the old state if the step is invalid.
      def _update(x, y):
        if not py_utils.is_optax_masked_node(
            x
        ) and not py_utils.is_optax_masked_node(y):
          return jnp.where(valid_step, x, y)
        return x

      new_states = jax.tree_map(
          _update, new_states, states, is_leaf=py_utils.is_optax_masked_node
      )
    return transformed_grad, new_states
