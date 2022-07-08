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

"""Shared trainer lib utilities."""

import functools
import os
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

from absl import logging
import jax
from jax import numpy as jnp
from jax.experimental import maps
from jax.experimental import pjit
from paxml import base_metrics
from paxml import summary_utils
from paxml import tasks_lib
from praxis import base_hyperparams
from praxis import base_input
from praxis import base_layer
from praxis import base_model
from praxis import py_utils
from praxis import pytypes
from praxis import train_states
import tensorflow.compat.v2 as tf

from paxml import checkpoints  # mapped to internal

JTensor = pytypes.JTensor
NestedJTensor = pytypes.NestedJTensor
NestedMap = py_utils.NestedMap
NestedShape = NestedMap
PRNGKey = pytypes.PRNGKey
ParamsT = pytypes.HParamsT
NestedShapeDtypeStruct = pytypes.NestedShapeDtypeStruct
TrainState = train_states.TrainState
SummaryDict = pytypes.SummaryDict
MetricDict = pytypes.Metrics
TrainStepFn = Callable[[TrainState, JTensor, NestedJTensor], Tuple[TrainState,
                                                                   ...]]
EvalStepFn = Callable[[NestedJTensor, JTensor, JTensor, NestedJTensor], Tuple]
DecodeFn = Callable[[NestedJTensor, JTensor, JTensor, NestedJTensor],
                    NestedJTensor]
EarlyStoppingFn = Callable[[Optional[Dict[str, MetricDict]], int, bool], bool]
CheckpointType = checkpoints.CheckpointType

PARAMS = base_layer.PARAMS
AUX_LOSS = base_layer.AUX_LOSS
SUMMARIES = base_layer.SUMMARIES
NON_TRAINABLE = base_layer.NON_TRAINABLE
NON_PAX_RNG_KEY = base_layer.NON_PAX_RNG_KEY
RANDOM = base_layer.RANDOM
DECODE_CACHE = base_layer.DECODE_CACHE
NON_PAX_VAR_COLLECTION = base_layer.NON_PAX_VAR_COLLECTION
PMAP_PARALLEL_AXIS_NAME = base_layer.PMAP_PARALLEL_AXIS_NAME

instantiate = base_hyperparams.instantiate


def write_post_init_model_hparams_file(model, vars_weight_params,
                                       job_log_dir: str) -> None:
  """Writes a post-init params file into the root `job_log_dir`.

  This file is the source of truth of how model is constructed. It contains two
  parts:
  1) how each layer is configured during layer construction time.
  2) variable WeightHParams for each of the model weight.

  Args:
    model: A BaseModel
    vars_weight_params: A pytree of WeightHParams
    job_log_dir: The root dir for the training job.
  """
  if jax.process_index() == 0:
    params_fpath = os.path.join(job_log_dir, 'post_init_model_params.txt')
    logging.info('post_init_model_params: %s', params_fpath)
    if not tf.io.gfile.exists(job_log_dir):
      tf.io.gfile.makedirs(job_log_dir)
    with tf.io.gfile.GFile(params_fpath, 'w') as params_file:
      prng_key = jax.random.PRNGKey(seed=123)

      def gen_post_init_hparams(prng_key):
        return model.apply({},
                           rngs={base_layer.PARAMS: prng_key},
                           method=model.post_init_hparams,
                           mutable=True)[1]

      variables_abstract = jax.eval_shape(gen_post_init_hparams, prng_key)
      assert base_layer.HYPER_PARAMS in variables_abstract

      hyper_params = jax.tree_map(
          lambda x: x.meta,
          variables_abstract[base_layer.HYPER_PARAMS],
          is_leaf=lambda x: isinstance(x, base_layer.WrappedHParams))

      hyper_params_dump = base_hyperparams.nested_struct_to_text(hyper_params)
      params_file.write(hyper_params_dump)
      params_file.write('\n\n')

      if vars_weight_params:
        params_inits_text = base_hyperparams.nested_struct_to_text(
            vars_weight_params)
        params_file.write(params_inits_text)


def initialize_model_state(jax_task: tasks_lib.SingleTask,
                           prng_key: PRNGKey,
                           discard_opt_states: bool = False,
                           do_init_checkpoint_rules: bool = True) -> TrainState:
  """Initializes the model states.

  Weights are random initialized first.
  Then we restores weights based on the init_checkpoint_rules.

  Args:
    jax_task: An instance of tasks.SingleTask.
    prng_key: A PRNGKey, of shape [2], of type np.uint32.
    discard_opt_states: whetehr to discard optimizer states.
    do_init_checkpoint_rules: whether to apply init checkpoint rules or not.

  Returns:
    TrainStates - training states.
  """
  model = jax_task.model
  vars_weight_params = model.abstract_init_with_metadata(prng_key)
  logging.info('init_var prng_seed: %s', prng_key)
  logging.info('vars_weight_params: %s', vars_weight_params)
  initial_vars = model.init(prng_key)

  # In case jax_task.model wraps a t5x model, let's remove the params_axes
  # variable collection.
  if 'params_axes' in initial_vars:
    del initial_vars['params_axes']
  train_state = jax_task.create_train_state(initial_vars, vars_weight_params,
                                            discard_opt_states)
  if do_init_checkpoint_rules:
    # Overwrite some parts if init_checkpoint_rules are set (warm-start)
    # Note that this assumes a pmap model with Flax checkpoint(s).
    train_state, update_opt_states = jax_task.apply_init_checkpoint_rules(
        train_state)
    if update_opt_states:
      # Re-compute opt_states after the model variables are updated.
      opt_states = jax_task.create_opt_states(train_state.mdl_vars,
                                              vars_weight_params)
      train_state = train_state.replace(opt_states=opt_states)
  return train_state


def replicate_model_state(model_states: TrainState) -> TrainState:
  """Replicates the model states."""
  return jax.device_put_replicated(model_states, jax.local_devices())


def initialize_replicate_model_state(
    jax_task: tasks_lib.SingleTask,
    prng_key: PRNGKey,
    discard_opt_states: bool = False) -> TrainState:
  """Initializes and replicates the model states."""
  model_states = initialize_model_state(jax_task, prng_key, discard_opt_states)
  return replicate_model_state(model_states)


def _maybe_to_bfloat16(x: JTensor) -> JTensor:
  if x.dtype == jnp.float32:
    return x.astype(jnp.bfloat16)
  return x


def _maybe_to_float32(x: JTensor) -> JTensor:
  if x.dtype == jnp.bfloat16:
    return x.astype(jnp.float32)
  return x


# TODO(pax): maybe move to metric_utils.py.
def _maybe_aggregate_metrics_summaries(
    loss_aggregator: base_metrics.LossAggregator,
    metric_dict: MetricDict,
    summary_dict: SummaryDict,
    per_example_out: NestedMap,
) -> Tuple[JTensor, JTensor, MetricDict, SummaryDict, NestedMap]:
  """If in pmap, aggregate metrics and summaries across model replicas.

  Args:
    loss_aggregator: An instance of a LossAggregator class to aggregate the
      loss. Defaults to the a single Loss weighted loss calculation.
    metric_dict: a MetricDict.
    summary_dict: a SummaryDict.
    per_example_out: a NestedMap.

  Returns:
    (weighted_loss, mean_loss, aggregated_metrics, aggregated_summaries)
    weighted_loss - the per-replica loss to back-propagate from. Useful for
      computing gradients only.
    mean_loss - the avg per-replica loss. This often is the psum of
      weighted_loss.
    aggregated_metrics - the aggregated metrics.
    aggregated_summaries - the aggregated summaries.
    per_example_out - the aggregated per_example_out.
  """
  # compute weighted loss and mean across shards
  weighted_loss, mean_loss = loss_aggregator.aggregate(metric_dict)

  if base_layer.is_running_under_pmap():
    # aggregate metrics.
    mean_metrics = type(metric_dict)()
    for key in metric_dict:
      value, weight = metric_dict[key]
      sum_value = jax.lax.psum(
          value * weight, axis_name=PMAP_PARALLEL_AXIS_NAME)
      sum_weight = jax.lax.psum(weight, axis_name=PMAP_PARALLEL_AXIS_NAME)
      mean_metrics[key] = (sum_value / (sum_weight + 1e-8), sum_weight)
    aggregated_summaries = summary_utils.aggregate_per_replica_summaries(
        summary_dict)
    per_example_out = jax.tree_map(
        lambda x: jax.lax.all_gather(x, axis_name=PMAP_PARALLEL_AXIS_NAME),
        per_example_out)
  else:
    # No aggregation of metrics is needed.
    mean_metrics = metric_dict
    # No aggregation of summaries is needed.
    aggregated_summaries = summary_dict

  return (weighted_loss, mean_loss, mean_metrics, aggregated_summaries,
          per_example_out)


def _zero_gradient_for_non_learnable_vars(grads, var_weight_hparams):
  """A helper function to zero out grads for non-learnable vars.

  Args:
    grads: a nested structure of var gradients.
    var_weight_hparams: a nested structure of the variable weight params.
      var_weight_hparams must have the same structure as grads.

  Returns:
    grads with gradient for non-learnable vars zero-ed out.
  """
  tf.nest.assert_same_structure(grads, var_weight_hparams)
  var_is_learnable = tf.nest.map_structure(
      lambda x: not base_layer.var_not_trainable(x), var_weight_hparams)

  def _maybe_zero_out_grad_fn(var_grad, var_learnable):
    if var_learnable:
      return var_grad
    elif var_grad.dtype is jax.dtypes.float0:
      # Gradient of an integer-valued input cannot be consumed by jnp operation.
      # Zerso dtype should be int32 same as the original input that produced
      # float0.
      return jnp.zeros((), dtype=jnp.int32)
    else:
      return jnp.zeros_like(var_grad)

  # Zero-out gradient for non-learnable vars.
  return tf.nest.map_structure(_maybe_zero_out_grad_fn, grads, var_is_learnable)


def _maybe_synchronize_non_learnable_vars(old_vars, new_vars,
                                          var_weight_hparams):
  """A helper function to synchronize non-learnable vars for pmap training.

  Each non-learnable variable declares how it should be synchronized across
  model replicas during training. Currently, we only support mean aggregation.

  Args:
    old_vars: a nested structure containing value of all variables before a
      training step. old_vars is expected to only contain non-learnable vars.
    new_vars: a nested structure containing value of all variables after a
      training step. new_vars must be of the same structure as old_vars.
    var_weight_hparams: a nested structure of the variable weight params. Must
      be of the same structure as old_vars & new_vars

  Returns:
    synchronized new_vars.
  """

  tf.nest.assert_same_structure(old_vars, new_vars)
  tf.nest.assert_same_structure(old_vars, var_weight_hparams)

  def _synchronize_vars_using_mean(old_var: JTensor,
                                   new_var: JTensor) -> JTensor:
    """Synchronize a variable across replicas by averaging."""
    delta = new_var - old_var
    delta_mean = jax.lax.pmean(delta, axis_name=PMAP_PARALLEL_AXIS_NAME)
    updated_var = old_var + delta_mean
    return updated_var

  def _synchronize_vars_using_sum(old_var: JTensor,
                                  new_var: JTensor) -> JTensor:
    """Synchronize a variable across replicas by summing."""
    delta = new_var - old_var
    delta_total = jax.lax.psum(delta, axis_name=PMAP_PARALLEL_AXIS_NAME)
    updated_var = old_var + delta_total
    return updated_var

  def _synchronize_non_learnable_var(old_var: JTensor, new_var: JTensor,
                                     var_param: ParamsT) -> JTensor:
    """Update a non-trainable variable, using cross-replica synchronization.

    Args:
      old_var: The original variable before a training step.
      new_var: The new variable after a training step.
      var_param: Variable param which contains attributes such as whether a
        variable is trainable or requires synchornization across replicas.

    Returns:
      synchronized variable.

    Raises:
      ValueError if no synchronization method is provided for non-trainable
      variables.
    """
    assert base_layer.var_not_trainable(var_param)
    assert base_layer.is_running_under_pmap()

    if base_layer.var_requires_mean_sync(var_param):
      return _synchronize_vars_using_mean(old_var, new_var)
    elif base_layer.var_requires_sum_sync(var_param):
      return _synchronize_vars_using_sum(old_var, new_var)
    else:
      raise ValueError('Non-trainable variables must have a cross-replica '
                       'synchronization method specified.')

  if base_layer.is_running_under_pmap():

    def _sync_var(old_var, new_var, var_param):
      return _synchronize_non_learnable_var(old_var, new_var, var_param)

    return tf.nest.map_structure(_sync_var, old_vars, new_vars,
                                 var_weight_hparams)
  else:
    # no synchronization is needed.
    return new_vars


# TODO(yonghui): refactor to pass in learner separately.
def _train_step_single_learner_with_model(
    jax_task: tasks_lib.SingleTask,
    model: base_model.BaseModel,
    states: TrainState,
    prng_key: JTensor,
    inputs: Union[JTensor, NestedMap],
    fprop_dtype: jnp.dtype = jnp.float32
) -> Tuple[TrainState, Any, Any, Any, SummaryDict]:
  """Trains a model for a single step.

  This function works for both pmap-ed model and pjit-ed model.

  TODO(yonghui): Maybe refactor pmap and pjit into two functions.

  This utility is specialized for the singler learner case.

  Args:
    jax_task: An instance of tasks.SingleTask.
    model: An instance of base_model that is used for forward/back prop
    states: An instance of model.TrainState.
    prng_key: A PRNGKey, of shape [2], of type np.uint32.
    inputs: Inputs to the model() function.
    fprop_dtype: fprop datatype, can be either jnp.float32 or jnp.bfloat16.

  Returns:
    A tuple of the following elements.
    updated_states - updated states.
    loss - loss as computed by model.fprop.
    mean_metrics - a dict of metrics. Each element of the dict is a pair
    (metric, weight).
    per_example_out - auxilillary per-example output as computed in model.fprop.
    summary_tensors - A dict or nested map of summary tensors computed in
      forward as well as backward.
  """
  assert len(jax_task.learners) == 1
  learner = jax_task.learners[0]

  context_p = base_layer.JaxContext.HParams(do_eval=False)
  # Fold in global_step as part of the random seed key, so that random
  # numbers depends on global step.
  #
  # TODO(yonghui): also fold in the replica id.
  prng_key = jax.random.fold_in(prng_key, states.step)
  prng_key, k1, k2 = jax.random.split(prng_key, 3)
  init_key = {PARAMS: k1, NON_PAX_RNG_KEY: k2}
  parent_model = jax_task.model

  var_weight_hparams = parent_model.abstract_init_with_metadata(init_key)

  updated_mdl_vars = jax_task.maybe_adjust_train_state(states.step,
                                                       states.mdl_vars,
                                                       var_weight_hparams,
                                                       prng_key)

  def _loss_fn(
      mdl_vars: NestedJTensor, inputs: NestedMap, prng_key
  ) -> Tuple[JTensor, Tuple[JTensor, MetricDict, Dict[str, Any], SummaryDict,
                            SummaryDict]]:
    """Computes loss as well as other auxiliary outputs."""
    if fprop_dtype == jnp.float32:
      pass
    elif fprop_dtype == jnp.bfloat16:
      mdl_vars = jax.tree_map(_maybe_to_bfloat16, mdl_vars)
      inputs = jax.tree_map(_maybe_to_bfloat16, inputs)
    else:
      assert NotImplementedError(f'fprop_dtype {fprop_dtype} not supported.')

    with base_layer.JaxContext.new_context(hparams=context_p):
      prng_key, k1, k2, k3 = jax.random.split(prng_key, 4)
      apply_rng_keys = {PARAMS: k1, RANDOM: k2, NON_PAX_RNG_KEY: k3}
      (metrics, per_example_output), updated_vars = model.apply(
          mdl_vars,
          inputs,
          mutable=[AUX_LOSS, SUMMARIES, NON_TRAINABLE] + NON_PAX_VAR_COLLECTION,
          method=model.__call__,
          rngs=apply_rng_keys)

      # Fetch all the summary tensors.
      summary_tensors = updated_vars.get(SUMMARIES, {})
      # TODO(yonghui): Fetch aux losses and add them to summaries.
      summary_tensors = summary_utils.flatten_flax_summaries(summary_tensors)

      (weighted_loss, mean_loss, mean_metrics, aggregated_summaries,
       per_example_output) = _maybe_aggregate_metrics_summaries(
           jax_task.loss_aggregator, metrics, summary_tensors,
           per_example_output)
      # metrics and summary_tensors no longer needed.
      del metrics
      del summary_tensors

      forward_updated_vars = {}
      for collection in [NON_TRAINABLE] + NON_PAX_VAR_COLLECTION:
        if collection in updated_vars:
          forward_updated_vars[collection] = updated_vars[collection]
    if fprop_dtype == jnp.bfloat16 and weighted_loss.dtype == fprop_dtype:
      weighted_loss = weighted_loss.astype(jnp.float32)
    return weighted_loss, (mean_loss, mean_metrics, forward_updated_vars,
                           aggregated_summaries, per_example_output)

  # Layers may have integer-valued non-trainable vars. `allow_int=True` is
  # needed to allow jax.grad to differentiate wrt integer-values.
  # However, the gradient of an integer input will have a trivial vector-space
  # dtype (float0). They cannot be consumed by jnp operations.
  # _zero_gradient_for_non_learnable_vars needs to handle jax.dtypes.float0
  # specially.
  grad_fn = jax.value_and_grad(_loss_fn, has_aux=True, allow_int=True)

  prng_key, subkey = jax.random.split(prng_key)
  (weighted_loss,
   (mean_loss, mean_metrics, fwd_updated_vars, fwd_summary_tensors,
    per_example_out)), grads = grad_fn(updated_mdl_vars, inputs, subkey)

  # weighted_loss is only needed for computing gradients, but otherwise, not
  # needed.
  del weighted_loss

  # Carry out backward computation under a JaxContext.
  with base_layer.JaxContext.new_context(hparams=context_p):
    grads = _zero_gradient_for_non_learnable_vars(grads, var_weight_hparams)

    if base_layer.is_running_under_pmap():
      # Aggregate grads across different model replicas.
      grads = jax.lax.psum(grads, axis_name=PMAP_PARALLEL_AXIS_NAME)
    else:
      # No gradient aggregation is needed.
      pass

    # Add a summary for learning rate
    learner.plot_learning_rate(states.step)

    # Apply gradient transformations.
    mdl_vars = states.mdl_vars.copy()
    # Make updated nontrainable variable peekable from GradientTransformers.
    # Some optimizers, e.g. `optimizers.DynamicAccumulator`, assume special
    # non-trainable variables being set during fprop for controlling their
    # behavior.
    if NON_TRAINABLE in fwd_updated_vars:
      mdl_vars[NON_TRAINABLE] = fwd_updated_vars[NON_TRAINABLE]
    transformed_grads, new_opt_states = learner.update_states(
        grads, states.opt_states[0], mdl_vars, var_weight_hparams)
    mdl_vars = learner.apply_gradient(mdl_vars, transformed_grads,
                                      var_weight_hparams)

    for collection in [NON_TRAINABLE] + NON_PAX_VAR_COLLECTION:
      if collection in states.mdl_vars:
        # We need to update the non-trainable vars.
        tf.nest.assert_same_structure(states.mdl_vars[collection],
                                      fwd_updated_vars[collection])
        mdl_vars[collection] = _maybe_synchronize_non_learnable_vars(
            states.mdl_vars[collection], fwd_updated_vars[collection],
            var_weight_hparams[collection])

    # lastly, we avoid updating variables in learner.bprop_variable_exclusion.
    mdl_vars = py_utils.update_matched_variables(
        states.mdl_vars,
        mdl_vars,
        learner.hparams.bprop_variable_exclusion,
        invert=True)
    new_states = states.new_state(
        mdl_vars=mdl_vars, opt_states=[new_opt_states])
    # Finally fetch all backward summary tensors. We do not aggregate the scalar
    # summaries with pmean because the grads are already psum-ed.
    if jax_task.hparams.train.variable_norm_summary:
      var_summary_tensors = summary_utils.l2_mean(
          new_states.mdl_vars, prefix='vars', max_level=20)
      for name, norm in var_summary_tensors.items():
        base_layer.add_global_summary(name, norm)
    bwd_summary_tensors = base_layer.all_global_summaries()

  summary_tensors = NestedMap(
      fwd_summary_tensors=fwd_summary_tensors,
      bwd_summary_tensors=bwd_summary_tensors)

  return (new_states, mean_loss, mean_metrics, per_example_out, summary_tensors)


def train_step_single_learner(
    jax_task: tasks_lib.SingleTask,
    states: TrainState,
    prng_key: JTensor,
    inputs: Union[JTensor, NestedMap],
    fprop_dtype: jnp.dtype = jnp.float32,
    model_name: Optional[str] = None
) -> Tuple[TrainState, Any, Any, Any, SummaryDict]:
  """Trains a model (or a submodel) for a single step."""
  del model_name
  # default model is the base model in jax_task
  model = jax_task.model

  return _train_step_single_learner_with_model(jax_task, model, states,
                                               prng_key, inputs, fprop_dtype)


def _eval_step_single_learner_with_model(
    jax_task: tasks_lib.SingleTask,
    model: base_model.BaseModel,
    states: TrainState,
    prng_key: JTensor,
    inputs: Union[JTensor, NestedMap],
    fprop_dtype: jnp.dtype = jnp.float32
) -> Tuple[TrainState, Any, Any, Any, SummaryDict]:
  """Evaluates a model for a single step.

  This utility is specialized for the single learner case.

  Args:
    jax_task: An instance of tasks.SingleTask.
    model: An instance of base_model.BaseModel, that is used for training
    states: An instance of model.TrainState.
    prng_key: A prng seed, of shape [2], of type np.uint32.
    inputs: Inputs to the model() function.
    fprop_dtype: fprop datatype, can be either jnp.float32 or jnp.bfloat16.

  Returns:
    A tuple of the following elements.
    loss - loss as computed by model.fprop.
    mean_metrics - a dict of metrics. Each element of the dict is a pair
    (metric, weight).
    per_example_out - auxilillary per-example output as computed in model.fprop.
    summary_tensors - A nested map or dict of summary tensors computed in
      forward as well as backward pass.
  """
  context_p = base_layer.JaxContext.HParams(do_eval=True)
  # Fold in global_step as part of the random seed key, so that random
  # numbers depends on global step.
  prng_key = jax.random.fold_in(prng_key, states.step)
  mdl_vars = states.mdl_vars
  # assert not states.opt_states

  if fprop_dtype == jnp.float32:
    pass
  elif fprop_dtype == jnp.bfloat16:
    mdl_vars = jax.tree_map(_maybe_to_bfloat16, mdl_vars)
    inputs = jax.tree_map(_maybe_to_bfloat16, inputs)
  else:
    assert NotImplementedError(f'fprop_dtype {fprop_dtype} not supported.')

  with base_layer.JaxContext.new_context(hparams=context_p):
    prng_key, k1, k2, k3 = jax.random.split(prng_key, 4)
    apply_rng_keys = {PARAMS: k1, RANDOM: k2, NON_PAX_RNG_KEY: k3}
    (metrics, per_example_out), updated_vars = model.apply(
        mdl_vars,
        inputs,
        mutable=[AUX_LOSS, SUMMARIES, NON_TRAINABLE],
        method=model.__call__,
        rngs=apply_rng_keys)

    summary_tensors = updated_vars.get(SUMMARIES, {})
    # TODO(yonghui): Add aux-loss to summaries.
    summary_tensors = summary_utils.flatten_flax_summaries(summary_tensors)

    (_, mean_loss, mean_metrics, aggregated_summaries,
     per_example_out) = _maybe_aggregate_metrics_summaries(
         jax_task.loss_aggregator, metrics, summary_tensors, per_example_out)

    # metrics and summary_tensors no longer needed.
    del metrics
    del summary_tensors

  if fprop_dtype == jnp.bfloat16:
    (mean_loss, mean_metrics,
     per_example_out, aggregated_summaries) = jax.tree_map(
         _maybe_to_float32,
         (mean_loss, mean_metrics, per_example_out, aggregated_summaries))

  # Adding the unchanged state to the return list so that both
  # eval_step_single_learner and train_step_single_learner have the same api to
  # facilitate some down-stream code.
  return states, mean_loss, mean_metrics, per_example_out, aggregated_summaries


def eval_step_single_learner(
    jax_task: tasks_lib.SingleTask,
    states: TrainState,
    prng_key: JTensor,
    inputs: Union[JTensor, NestedMap],
    fprop_dtype: jnp.dtype = jnp.float32,
    model_name: Optional[str] = None,
) -> Tuple[TrainState, Any, Any, Any, SummaryDict]:
  """Evaluates a model (or submodel)."""
  del model_name  # TODO(johans) place holder for multi task info propagation

  # default model is the base model in jax_task
  model = jax_task.model
  return _eval_step_single_learner_with_model(jax_task, model, states, prng_key,
                                              inputs, fprop_dtype)


def decode_step(
    model: base_model.BaseModel,
    states: TrainState,
    prng_key: JTensor,
    inputs: Union[JTensor, NestedMap],
    fprop_dtype: jnp.dtype = jnp.float32,
) -> Tuple[NestedMap, NestedMap]:
  """Decodes a model for a single step.

  Args:
    model: An instance of models.BaseModel.
    states: An instance of TrainState..
    prng_key: A prng seed, of shape [2], of type np.uint32.
    inputs: A batch of inputs to model.decode().
    fprop_dtype: fprop datatype, can be either jnp.float32 or jnp.bfloat16.

  Returns:
    A tuple of (metrics, results) as computed by model.decode().
  """
  context_p = base_layer.JaxContext.HParams(do_eval=True)
  # Fold in global_step as part of the random seed key, so that random
  # numbers depends on global step.
  prng_key = jax.random.fold_in(prng_key, states.step)
  mdl_vars = states.mdl_vars
  assert not states.opt_states

  if fprop_dtype == jnp.bfloat16:
    mdl_vars = jax.tree_map(_maybe_to_bfloat16, mdl_vars)
    inputs = jax.tree_map(_maybe_to_bfloat16, inputs)
  elif fprop_dtype != jnp.float32:
    assert NotImplementedError(f'fprop_dtype {fprop_dtype} not supported.')

  with base_layer.JaxContext.new_context(hparams=context_p):
    k1, k2, k3 = jax.random.split(prng_key, 3)
    apply_rng_keys = {PARAMS: k1, RANDOM: k2, NON_PAX_RNG_KEY: k3}
    outputs, updated_vars = model.apply(
        mdl_vars,
        inputs,
        method=model.decode,
        rngs=apply_rng_keys,
        mutable=[
            DECODE_CACHE,
            SUMMARIES,
        ])
    # DECODE_CACHE are not read by caller. But they can be large. Tell XLA DCE
    # to remove it from output. Note MLP decoder don't have DECODE_CACHE.
    if DECODE_CACHE in updated_vars:
      del updated_vars[DECODE_CACHE]

    summary_tensors = updated_vars.get(base_layer.SUMMARIES, {})
    if summary_tensors:
      summary_tensors = jax.tree_map(_maybe_to_float32, summary_tensors)
      updated_vars[base_layer.SUMMARIES] = summary_tensors

    return outputs, updated_vars


def initialize_partitioned_model_states(
    jax_task: tasks_lib.SingleTask,
    prng_key: PRNGKey,
    discard_opt_states: bool = False,
    global_mesh: Optional[maps.Mesh] = None,
    checkpoint_type: CheckpointType = CheckpointType.CHECKPOINT_FLAX,
    state_specs: Optional[TrainState] = None,
) -> Tuple[TrainState, TrainState]:
  """Initializes model vars that are partitioned over TPU devices.

  Weights are random initialized first.
  Then we restores weights based on the init_checkpoint_rules.

  This function is equivalent to calling a pjit-ted version of
  InitializesModelStates().

  Args:
    jax_task: The task which is an instance of tasks.SingleTask.
    prng_key: A PRNGKey.
    discard_opt_states: bool, When true, optimizer slot variables are skipped.
    global_mesh: The global mesh to use when restoring weights based on the
      init_checkpoint_rules. Required for GDA-based checkpoints.
    checkpoint_type: The checkpoint type to use when restoring weights based on
      the init_checkpoint_rules.
    state_specs: The TrainState specs when restoring weights based on the
      init_checkpoint_rules. Required for GDA-based checkpoints.

  Returns:
    The partitioned specs and the partitioned vars themselves.
  """
  model = jax_task.model
  # At this point, variable specs are already known.
  vars_weight_params = model.abstract_init_with_metadata(prng_key)

  if state_specs is None:
    train_state_partition_specs = jax_task.create_train_state_partition_specs(
        vars_weight_params, discard_opt_states)
  else:
    train_state_partition_specs = state_specs

  train_state_unpadded_shapes = jax.tree_map(
      lambda x: x.shape,
      jax_task.create_train_state_unpadded_shapes(vars_weight_params,
                                                  discard_opt_states))
  assert train_state_partition_specs is not None

  def _maybe_pad(x, pspec, shape):
    if py_utils.is_optax_masked_node(x):
      return x
    return py_utils.maybe_pad_uneven_sharding(x, pspec, shape,
                                              model.hparams.mesh_shape,
                                              model.hparams.mesh_axis_names)

  def init_model_from_seed(prng_key):
    outs = initialize_model_state(
        jax_task, prng_key, discard_opt_states, do_init_checkpoint_rules=False)
    return jax.tree_map(_maybe_pad, outs, train_state_partition_specs,
                        train_state_unpadded_shapes,
                        is_leaf=py_utils.is_optax_masked_node)

  logging.info('unpadded_out_shape: %s', train_state_unpadded_shapes)
  logging.info('train_state_partition_specs: %s', train_state_partition_specs)
  tf.nest.assert_same_structure(train_state_unpadded_shapes,
                                train_state_partition_specs)

  init_fn = pjit.pjit(
      init_model_from_seed,
      in_axis_resources=(None,),
      out_axis_resources=train_state_partition_specs)

  assert py_utils.global_mesh_defined(), 'must be inside maps.mesh scope'
  partitioned_vars = init_fn(prng_key)
  # Overwrite some parts if init_checkpoint_rules are set (warm-start)
  if jax_task.hparams.train.init_from_checkpoint_rules:
    # TODO(b/230132535): Note that this application after constructing the
    # partitioned vars is currently inconsistent with what is being performed
    # for pmap models.
    partitioned_vars, _ = jax_task.apply_init_checkpoint_rules(
        partitioned_vars,
        global_mesh=global_mesh,
        checkpoint_type=checkpoint_type)

  return (train_state_partition_specs, partitioned_vars)


def shard_on_batch_dim_partition_spec(
    mesh_names: Sequence[str], x: jax.ShapeDtypeStruct) -> pjit.PartitionSpec:
  """Fully shards x on the batch dimension."""
  x_dim = len(x.shape)
  assert x_dim >= 1
  sharding = [-1] * x_dim
  # Assume the first dim is batch, and fully shard the batch dim over the entire
  # mesh.
  sharding[0] = tuple(mesh_names)
  return base_layer.to_partition_spec(sharding, mesh_names)


def reshard_input_based_on_rank_fn(
    mapping_dict: Dict[str, base_layer.SplitDimsMapping],
    mesh_names: Sequence[str],
    x: JTensor,
) -> JTensor:
  """Reshards input based on its rank.

  Args:
    mapping_dict: Dictionary which contains the split mapping for different
      shapes. For n-d shape, it must have an entry f'map_{n}d' which tells us
      how to partition tensors of this dimension.
    mesh_names: List of mesh axis names.
    x: JTensor which to shard.

  Returns:
    Resharded tensor.
  """
  key = f'map_{len(x.shape)}d'
  if key not in mapping_dict:
    raise ValueError(f'Split mapping must be provided for {len(x.shape)}-d '
                     f'in the form of key map_{len(x.shape)} in '
                     f'{mapping_dict}.')
  if mapping_dict[key] is not None:
    return base_layer.maybe_shard(x, mapping_dict[key], mesh_names)
  else:
    return x


def infer_partition_spec_based_on_rank_fn(
    mapping_dict: Dict[str, base_layer.SplitDimsMapping],
    mesh_names: Sequence[str],
    x: JTensor,
) -> Optional[pjit.PartitionSpec]:
  """Infers PartitionSpec of input from the rank of corresponding JTensors.

  Args:
    mapping_dict: Dictionary which contains the split mapping for different
      shapes. For n-d shape, it must have an entry f'map_{n}d' which tells us
      how to partition tensors of this dimension.
    mesh_names: List of mesh axis names.
    x: JTensor which to shard.

  Returns:
    PartitionSpec or None (if everything is replicated).
  """
  key = f'map_{len(x.shape)}d'
  if key not in mapping_dict:
    raise ValueError(f'Split mapping must be provided for {len(x.shape)}-d'
                     f'in the form of key map_{len(x.shape)} in'
                     f'{mapping_dict}.')
  if mapping_dict[key] is not None:
    return base_layer.to_partition_spec(mapping_dict[key], mesh_names)


def get_input_partition_specs(mesh_axis_names, inputs_shape):
  # Compute inputs PartitionSpec from inputs_shape
  inputs_partition_spec_fn = functools.partial(
      shard_on_batch_dim_partition_spec, mesh_axis_names)
  return tf.nest.map_structure(inputs_partition_spec_fn, inputs_shape)


def train_state_for_eval_step(state_with_opt_states):
  return TrainState(
      step=state_with_opt_states.step,
      mdl_vars=state_with_opt_states.mdl_vars,
      opt_states={})


# TODO(pax): merge with get_partitioned_spmd_model_decode_fn
def get_partitioned_spmd_model_step_fn(
    jax_task: tasks_lib.SingleTask,
    init_key: PRNGKey,
    model_state_partition_specs: TrainState,
    inputs_shape: NestedShapeDtypeStruct,
    is_eval: bool,
    unpadded_global_batch_size: Optional[int] = None):
  """Return sharded train or eval step function of the SPMD Model.

  Args:
    jax_task: The task which is an instance of tasks.SingleTask.
    init_key: PRNGKey for initializing the model variables.
    model_state_partition_specs: A TrainState contains PartitionSpecs for all
      the variables.
    inputs_shape: Shape of the inputs for use in pjit sharding.
    is_eval: bool, indicating if it's a eval/decode task or not.
    unpadded_global_batch_size: If not None, this is the unpadded size of global
      batch, and the padding is on the right side of inputs_shape.

  Returns:
    (step_fn, inputs_partition_spec):
    The step function and the partition spec for the inputs.
  """
  task_p = jax_task.hparams
  model_p = task_p.model
  mesh_names = model_p.mesh_axis_names

  reshard_inputs_fn = functools.partial(reshard_input_based_on_rank_fn,
                                        task_p.train.inputs_split_mapping,
                                        mesh_names)
  inputs_partition_spec = get_input_partition_specs(mesh_names, inputs_shape)

  vars_weight_params = jax_task.model.abstract_init_with_metadata(init_key)
  state_unpadded_shapes = jax.tree_map(
      lambda x: x.shape,
      jax_task.create_train_state_unpadded_shapes(
          vars_weight_params, discard_opt_states=is_eval))

  # TODO(bf-jax): prng_key is replicated. Would this be a problem?
  prng_key_partition_spec = base_layer.to_partition_spec((None,), mesh_names)

  def _maybe_pad(x, pspec, shape):
    if py_utils.is_optax_masked_node(x):
      return x
    return py_utils.maybe_pad_uneven_sharding(x, pspec, shape,
                                              model_p.mesh_shape,
                                              model_p.mesh_axis_names)

  def _step_fn(state, prng_key, inputs):
    # Vars/inputs are padded at program entry/exit to avoid uneven sharding. We
    # slice the vars to remove padding before the step computation, and pad them
    # after the step computation to make user code independent of paddings.
    # Internal uneven sharding in the step computation is supported by XLA.
    state = jax.tree_map(py_utils.maybe_slice_uneven_sharding, state,
                         model_state_partition_specs, state_unpadded_shapes,
                         is_leaf=py_utils.is_optax_masked_node)
    if unpadded_global_batch_size is not None:
      # At the beginning input is fully sharded on the batch dim which has
      # paddings. If we just slice out the padding, there won't be any overhead;
      # however, if reshard_inputs_fn (partially) replicates the batch dim,
      # there can be many halo exchange ops (collective permute) generated by
      # GSPMD since we are changing the amount of padding on an originally fully
      # sharded dim.
      #
      # To reduce the amount of halo exchange ops, we first reshape the padded
      # batch to [outer, inner, ...], where outer * inner == padded_batch_size,
      # and inner >= unpadded_global_batch_size. This way, halo exchange will be
      # limited inside the inner dim, and replication on the outer-dim is a
      # single all-reduce.
      padded_global_batch_size = jax.tree_leaves(inputs)[0].shape[0]
      inner_reshape_dim = padded_global_batch_size
      outer_reshape_dim = 1
      # Find a size for inner_reshape_dim that can evenly divide
      # padded_global_batch_size, and is also >= unpadded_global_batch_size.
      # Also make sure the two dims are both divisible by 2 as some TPU
      # topologies have limitations on communication group size being even.
      for maybe_outer_size in range(
          2, padded_global_batch_size // unpadded_global_batch_size + 1, 2):
        if padded_global_batch_size % maybe_outer_size == 0:
          maybe_inner_size = padded_global_batch_size // maybe_outer_size
          if maybe_inner_size % 2 == 0:
            inner_reshape_dim = maybe_inner_size
            outer_reshape_dim = maybe_outer_size

      def _remove_padding(x):
        x = jnp.reshape(x, (outer_reshape_dim, inner_reshape_dim) + x.shape[1:])
        # Add a no-op sharding annotation (all dims unspecified) to block any
        # optimization before SPMD partitioning.
        x = base_layer.maybe_shard(
            x, (None,) * x.ndim,
            mesh_names,
            unconstrained_dims=range(0, x.ndim))
        x = jax.lax.slice_in_dim(x, 0, 1, axis=0)
        # Partially annotate the sliced dim to be replicated.
        x = base_layer.maybe_shard(
            x, (None,) * x.ndim,
            mesh_names,
            unconstrained_dims=range(1, x.ndim))
        x = jnp.squeeze(x, axis=0)
        return x[:unpadded_global_batch_size]

      inputs = jax.tree_map(_remove_padding, inputs)
    # Reshard inputs.
    inputs = jax.tree_map(reshard_inputs_fn, inputs)

    fn = eval_step_single_learner if is_eval else train_step_single_learner
    fn_out = fn(
        jax_task, state, prng_key, inputs, fprop_dtype=model_p.fprop_dtype)

    assert len(fn_out) > 1

    new_states = jax.tree_map(_maybe_pad, fn_out[0],
                              model_state_partition_specs,
                              state_unpadded_shapes,
                              is_leaf=py_utils.is_optax_masked_node)
    return (new_states,) + fn_out[1:]

  def init_model_from_seed(init_key):
    outs = initialize_model_state(
        jax_task,
        init_key,
        discard_opt_states=is_eval,
        do_init_checkpoint_rules=False)
    return jax.tree_map(_maybe_pad, outs, model_state_partition_specs,
                        state_unpadded_shapes,
                        is_leaf=py_utils.is_optax_masked_node)

  var_padded_shapes = jax.eval_shape(init_model_from_seed, init_key)

  out_padded_shapes = jax.eval_shape(_step_fn, var_padded_shapes, init_key,
                                     inputs_shape)

  fn_in_partition_specs = (model_state_partition_specs, prng_key_partition_spec,
                           inputs_partition_spec)
  # Currently, all the outputs are fully replicated.
  # TODO(yonghui): Somehow fetch the output sharding spec from _eval_step fn.
  fn_out_partition_specs = tf.nest.map_structure(lambda _: None,
                                                 out_padded_shapes)

  fn_out_partition_specs = tuple([model_state_partition_specs] +
                                 list(fn_out_partition_specs[1:]))

  tf.nest.assert_same_structure(fn_out_partition_specs, out_padded_shapes)

  # pjit-ed step function.
  step_fn = pjit.pjit(
      _step_fn,
      in_axis_resources=fn_in_partition_specs,
      out_axis_resources=fn_out_partition_specs,
      donate_argnums=() if is_eval else (0,))

  return step_fn, inputs_partition_spec


# TODO(pax): merge with get_partitioned_spmd_model_decode_fn
# Q(pax-dev): how does the function below interact with padding/unpadding of
# variables along certain variable/mesh axis?
def get_partitioned_spmd_model_step_fn_auto_shard(
    jax_task: tasks_lib.SingleTask, init_key: Optional[PRNGKey],
    model_state_partition_specs: Optional[TrainState],
    inputs_shape: NestedShapeDtypeStruct, is_eval: bool):
  """Return sharded train or eval step function of the SPMD Model.

  Args:
    jax_task: The task which is an instance of tasks.SingleTask.
    init_key: (Unused) PRNGKey for initializing the model variables.
    model_state_partition_specs: (Unused) A TrainState contains PartitionSpecs
      for all the variables. This argument is ignored when auto sharding is
      enabled.
    inputs_shape: Shape of the inputs for use in pjit sharding.
    is_eval: bool, indicating if it's a eval/decode task or not.

  Returns:
    (step_fn, inputs_partition_spec):
    The step function and the partition spec for the inputs.
  """
  # TODO(pax-dev): Add support for padding and unpadding inputs.
  del init_key, model_state_partition_specs
  task_p = jax_task.hparams
  model_p = task_p.model
  mesh_names = model_p.mesh_axis_names

  reshard_inputs_fn = functools.partial(reshard_input_based_on_rank_fn,
                                        task_p.train.inputs_split_mapping,
                                        mesh_names)
  inputs_partition_spec = get_input_partition_specs(mesh_names, inputs_shape)
  prng_key_partition_spec = base_layer.to_partition_spec((None,), mesh_names)

  def _step_fn(state, prng_key, inputs):
    # Reshard inputs.
    # TODO(pax): Since we rely on xla auto-sharding for automatically figure
    # out the proper sharding of intermediate notes, we can get rid of this
    # manual sharding now?
    inputs = jax.tree_map(reshard_inputs_fn, inputs)

    # When auto sharding is enabled, uneven sharding is not supported. This is
    # because the way padding is added is dependent on the
    # `model_state_partition_specs`. This is not available during auto sharding
    # until after the compilation is done.

    fn = eval_step_single_learner if is_eval else train_step_single_learner
    fn_out = fn(
        jax_task, state, prng_key, inputs, fprop_dtype=model_p.fprop_dtype)

    assert len(fn_out) > 1
    return fn_out

  # pjit-ed step function.
  # provide inputs_partition_spec because GDA creation is specialized to the
  # input partition specs created here. If we use partition specs returned by
  # XLA, it errors out.
  step_fn = pjit.pjit(
      _step_fn,
      in_axis_resources=(pjit.AUTO, prng_key_partition_spec,
                         inputs_partition_spec),
      out_axis_resources=pjit.AUTO,
      donate_argnums=() if is_eval else (0,))
  return step_fn, None


def get_partitioned_spmd_model_decode_fn(jax_task, init_key,
                                         model_state_partition_specs,
                                         inputs_shape: NestedShapeDtypeStruct):
  """Return sharded decode step function and input partition spec.

  Args:
    jax_task: Task instance.
    init_key: PRNGKey for initializing the model variables.
    model_state_partition_specs: A TrainState contains PartitionSpecs for all
      the variables.
    inputs_shape: Shape of the inputs for use in pjit sharding.

  Returns:
    (decode_step_fn, inputs_partition_spec):
    The decode step function, and input partition spec.
  """
  task_p = jax_task.hparams
  model_p = task_p.model
  mesh_names = task_p.model.mesh_axis_names
  model = jax_task.model

  # Compute inputs PartitionSpec from inputs_shape
  inputs_partition_spec_fn = functools.partial(
      shard_on_batch_dim_partition_spec, mesh_names)
  reshard_inputs_fn = functools.partial(reshard_input_based_on_rank_fn,
                                        task_p.train.inputs_split_mapping,
                                        mesh_names)

  inputs_partition_spec = tf.nest.map_structure(inputs_partition_spec_fn,
                                                inputs_shape)

  # TODO(b/198356509): Fix this so that prng_key is no longer replicated, as
  # we want each core to not have identical random behavior.
  prng_key_partition_spec = base_layer.to_partition_spec((None,), mesh_names)

  def _maybe_pad(x, pspec, shape):
    if py_utils.is_optax_masked_node(x):
      return pspec
    return py_utils.maybe_pad_uneven_sharding(x, pspec, shape,
                                              model_p.mesh_shape,
                                              model_p.mesh_axis_names)

  vars_weight_params = jax_task.model.abstract_init_with_metadata(init_key)
  model_state_unpadded_shapes = jax.tree_map(
      lambda x: x.shape,
      jax_task.create_train_state_unpadded_shapes(
          vars_weight_params, discard_opt_states=True))

  def _decode_step(states, prng_key, inputs):
    inputs = jax.tree_map(reshard_inputs_fn, inputs)
    states = jax.tree_map(py_utils.maybe_slice_uneven_sharding, states,
                          model_state_partition_specs,
                          model_state_unpadded_shapes)
    # Right now we only pad the vars, and decode doesn't output vars so we do
    # not need to pad at the end.
    return decode_step(
        model, states, prng_key, inputs, fprop_dtype=task_p.model.fprop_dtype)

  def init_model_from_seed(init_key):
    outs = initialize_model_state(
        jax_task,
        init_key,
        discard_opt_states=True,
        do_init_checkpoint_rules=False)
    return jax.tree_map(_maybe_pad, outs, model_state_partition_specs,
                        model_state_unpadded_shapes,
                        is_leaf=py_utils.is_optax_masked_node)

  var_padded_shapes = jax.eval_shape(init_model_from_seed, init_key)

  decode_out_shapes = jax.eval_shape(_decode_step, var_padded_shapes, init_key,
                                     inputs_shape)

  decode_fn_in_partition_specs = (model_state_partition_specs,
                                  prng_key_partition_spec,
                                  inputs_partition_spec)
  # decoder output are always replicated at the moment.
  decode_fn_out_partition_specs = tf.nest.map_structure(lambda _: None,
                                                        decode_out_shapes)
  decode_step_fn = pjit.pjit(
      _decode_step,
      in_axis_resources=decode_fn_in_partition_specs,
      out_axis_resources=decode_fn_out_partition_specs)

  return decode_step_fn, inputs_partition_spec


def check_unique_names(inputs: Sequence[base_input.BaseInput]) -> None:
  names = set()
  for inp in inputs:
    name = inp.hparams.name
    if name in names:
      raise ValueError(f'Duplicate param name found in list: "{name}"')
    names.add(name)
