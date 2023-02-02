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

import abc
import dataclasses
import enum
import functools
import json
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

from absl import logging
from clu import platform
from etils import epath
from flax.core import frozen_dict
import jax
from jax import numpy as jnp
from jax.experimental import pjit
from jax.interpreters import pxla
from paxml import base_metrics
from paxml import sgf
from paxml import summary_utils
from paxml import tasks_lib
from paxml import train_states
from praxis import asserts
from praxis import base_hyperparams
from praxis import base_input
from praxis import base_layer
from praxis import base_model
from praxis import py_utils
from praxis import pytypes

from paxml import checkpoints  # mapped to internal

PartitionSpec = jax.sharding.PartitionSpec

JTensor = pytypes.JTensor
NestedJTensor = pytypes.NestedJTensor
NestedMap = py_utils.NestedMap
NestedNpTensor = pytypes.NestedNpTensor
NestedShape = NestedMap
PRNGKey = pytypes.PRNGKey
ParamsT = pytypes.HParamsT
PyTreeDef = pytypes.PyTreeDef
Nested = pytypes.Nested
NestedPartitionSpec = pytypes.NestedPartitionSpec
NestedShapeDtypeStruct = pytypes.NestedShapeDtypeStruct
NestedShapeDtypeLike = pytypes.NestedShapeDtypeLike
NestedPartitionSpec = pytypes.NestedPartitionSpec
NestedWeightHParams = base_layer.NestedWeightHParams
TrainState = train_states.TrainState
SummaryDict = pytypes.SummaryDict
WeightedScalars = pytypes.WeightedScalars
WeightedScalarsList = pytypes.WeightedScalarsList
TrainStepFn = Callable[[TrainState, JTensor, NestedJTensor], Tuple[TrainState,
                                                                   ...]]
EvalStepFn = Callable[[NestedJTensor, JTensor, JTensor, NestedJTensor], Tuple]
DecodeFn = Callable[[NestedJTensor, JTensor, JTensor, NestedJTensor],
                    NestedJTensor]
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


def update_nestedmap(full_set, partial_set):
  # Update full_set according to partial_set when name matches.

  if type(full_set) is not dict:
    return partial_set
  ret = NestedMap() if type(full_set) is NestedMap else {}
  for i in full_set.keys():
    if i in partial_set.keys():
      ret[i] = filter_nestedmap(full_set[i], partial_set[i])
    else:
      ret[i] = full_set[i]
  return ret


def filter_nestedmap(full_set, partial_set):
  # Project full_set into partial set
  if type(full_set) is not dict:
    return full_set
  ret = NestedMap() if type(full_set) is NestedMap else {}
  for i in partial_set.keys():
    ret[i] = filter_nestedmap(full_set[i], partial_set[i])
  return ret


class RunningMode(enum.Flag):
  """Running mode."""
  UNKNOWN = 0
  TRAIN = enum.auto()
  EVAL = enum.auto()
  DECODE = enum.auto()

  @classmethod
  def detect(cls, has_train_metrics: bool, has_eval_metrics: bool,
             has_decode_metrics: bool) -> 'RunningMode':
    """Detects running mode from generated metrics."""
    mode = RunningMode.UNKNOWN
    if has_train_metrics:
      mode |= RunningMode.TRAIN
    if has_eval_metrics:
      mode |= RunningMode.EVAL
    if has_decode_metrics:
      mode |= RunningMode.DECODE
    return mode

  @property
  def has_train(self):
    """Returns True if current mode has training."""
    return bool(self & RunningMode.TRAIN)

  @property
  def has_eval(self):
    """Returns True if current mode has evaluation."""
    return bool(self & RunningMode.EVAL)

  @property
  def has_decode(self):
    """Returns True if current mode has decoding."""
    return bool(self & RunningMode.DECODE)


@dataclasses.dataclass(frozen=True)
class TrainStateMetadata:
  """Metadata around the TrainState.

  Specifically, this encapsulates information relevant for model initialization
  as well as train/eval/decode step creation.
  """
  input_shape_dtype: NestedShapeDtypeLike
  var_weight_hparams: NestedWeightHParams
  padded_global_shapes: Optional[TrainState] = None
  unpadded_global_shapes: Optional[TrainState] = None
  partition_specs: Optional[TrainState] = None


def create_train_state_metadata(jax_task: tasks_lib.SingleTask,
                                train_shape_dtype: NestedShapeDtypeLike,
                                discard_opt_states: bool = False,
                                do_eval: bool = False) -> TrainStateMetadata:
  """Creates a TrainStateMetadata instance.

  Args:
    jax_task: The SingleTask instance.
    train_shape_dtype: Training input shape dtype to be used by
      model.abstract_init_with_metadata(). It should have per-core shapes
      for pmap models and global shapes for pjit ones.
    discard_opt_states: Whether to discard the part corresponding to the
      optimizer states or not.
    do_eval: Whether this metadata is used for evaluation.

  Returns:
    A TrainStateMetadata instance.
  """
  var_weight_hparams = jax_task.model.abstract_init_with_metadata(
      train_shape_dtype, do_eval=do_eval)
  padded_global_shapes = jax_task.create_train_state_padded_shapes(
      var_weight_hparams, discard_opt_states=discard_opt_states)
  unpadded_global_shapes = jax_task.create_train_state_unpadded_shapes(
      var_weight_hparams, discard_opt_states=discard_opt_states)
  if jax_task.model.hparams.mesh_shape is not None:
    partition_specs = jax_task.create_train_state_partition_specs(
        var_weight_hparams, discard_opt_states=discard_opt_states
    )
  else:
    partition_specs = None
  return TrainStateMetadata(
      input_shape_dtype=train_shape_dtype,
      var_weight_hparams=var_weight_hparams,
      padded_global_shapes=padded_global_shapes,
      unpadded_global_shapes=unpadded_global_shapes,
      partition_specs=partition_specs,
  )


def compile_for_auto_sharding(step_fn: Any,
                              train_state: train_states.TrainState,
                              step_key: pytypes.PRNGKey,
                              inputs_shape_dtype: NestedShapeDtypeLike):
  """Compiles step_fn ahead of time to extract the shardings.

  The sharding is returned by the auto spmd partitioner and is attached on the
  compiled object.

  Args:
    step_fn: The step_fn function which will be compiled ahead of time.
    train_state: Train state which contains abstract values for ahead of time
      compilation.
    step_key: Prng key.
    inputs_shape_dtype: Inputs with shape/dtype attributes to be used for shape
      inference.

  Returns:
    * A compiled step_fn function
    * The input shardings returned by the auto spmd partitioner.
  """

  def _create_aval(x):
    # canonicalize_dtype is necessary to avoid errors like
    # data types are different when compiling and when being called.
    dtype = jax.dtypes.canonicalize_dtype(x.dtype)
    return jax.ShapedArray(x.shape, dtype)

  inputs_shape_dtype = jax.tree_map(_create_aval, inputs_shape_dtype)
  compiled = step_fn.lower(
      train_state, step_key, inputs_shape_dtype).compile()
  return compiled, compiled.input_shardings[0]


EarlyStoppingFn = Callable[[Dict[str, float], RunningMode, int, bool], bool]


def write_post_init_model_hparams_file(
    model: base_model.BaseModel,
    var_weight_hparams: NestedWeightHParams,
    job_log_dir: epath.Path,
    do_eval: bool = False,
) -> None:
  """Writes a post-init params file into the root `job_log_dir`.

  This file is the source of truth of how model is constructed. It contains two
  parts:
  1) how each layer is configured during layer construction time.
  2) variable WeightHParams for each of the model weight.

  Args:
    model: A BaseModel
    var_weight_hparams: A pytree of WeightHParams
    job_log_dir: The root dir for the training job.
    do_eval: whether this is in eval mode.
  """
  if jax.process_index() == 0:
    params_fpath = job_log_dir / 'post_init_model_params.txt'
    logging.info('post_init_model_params: %s', params_fpath)
    job_log_dir.mkdir(parents=True, exist_ok=True)
    context_p = base_layer.JaxContext.HParams(do_eval=do_eval)
    with params_fpath.open('w') as params_file:
      prng_key = jax.random.PRNGKey(seed=123)

      def gen_post_init_hparams(prng_key):
        with base_layer.JaxContext.new_context(hparams=context_p):
          return model.apply(
              {},
              rngs={base_layer.PARAMS: prng_key},
              method=model.post_init_hparams,
              mutable=True,
          )[1]

      variables_abstract = jax.eval_shape(gen_post_init_hparams, prng_key)
      assert base_layer.HYPER_PARAMS in variables_abstract

      hyper_params = jax.tree_map(
          lambda x: x.meta,
          variables_abstract[base_layer.HYPER_PARAMS],
          is_leaf=lambda x: isinstance(x, base_layer.WrappedHParams))

      hyper_params_dump = base_hyperparams.nested_struct_to_text(hyper_params)
      params_file.write(hyper_params_dump)
      params_file.write('\n\n')

      if var_weight_hparams:
        params_inits_text = base_hyperparams.nested_struct_to_text(
            var_weight_hparams)
        params_file.write(params_inits_text)


def adjust_input_params_for_small_batch(
    inp_p: base_input.BaseInput.HParams,
    global_mesh: jax.sharding.Mesh) -> base_input.BaseInput.HParams:
  """Creates a copy of inp_p adjusted when per-device batch < 1."""
  # Remote input adjusts the params for small batch itself.
  if inp_p.experimental_remote_input:
    return inp_p

  local_device_count = jax.local_device_count()
  batch_size = inp_p.cls.get_batch_size(inp_p)

  if (batch_size % local_device_count == 0 and
      inp_p.num_infeed_hosts == jax.process_count()):
    return inp_p
  copy = inp_p.clone()
  if batch_size > local_device_count:
    if batch_size % local_device_count != 0:
      raise NotImplementedError('Per-host batch size must be a multiple of per-'
                                'host device count, or smaller than it.')
  else:
    copy.batch_padding_size = local_device_count - batch_size

  assert inp_p.num_infeed_hosts <= jax.process_count()
  # LINT.IfChange(PspecSharding)
  if jax.process_count() == 1:
    # If there is only one host, valid examples are already contiguous so we can
    # use default GDA creation.
    # Inputs use pspec sharding (see praxis.BaseInput.reshard_for_spmd).
    return copy
  # LINT.ThenChange(trainer_lib.py:UsePspecOnArrayInputs)
  # Some hosts may produce duplicate data, but they will be discarded.
  copy.infeed_host_index = jax.process_index() % inp_p.num_infeed_hosts
  if copy.infeed_host_index >= inp_p.num_infeed_hosts:
    logging.info('Process %s: infeed data will be dropped.',
                 jax.process_index())

  # Figure out the cores that have valid data, and construct a device order for
  # GSPMD sharding that place the valid data on the left side of the logical
  # input tensor.
  per_host_core_counter = {}
  for pid in range(jax.process_count()):
    per_host_core_counter[pid] = 0
  # We place useful data on the left-aligned subsequence of all devices.
  used_cores = []
  unused_cores = []
  for global_device_idx, device in enumerate(global_mesh.devices.flat):
    process_idx = device.process_index
    core_offset_in_host = per_host_core_counter[device.process_index]
    per_host_core_counter[device.process_index] += 1
    if (process_idx >= inp_p.num_infeed_hosts or
        core_offset_in_host >= batch_size):
      # Not an infeeding host.
      unused_cores.append(global_device_idx)
    else:
      used_cores.append(global_device_idx)
  copy.custom_device_order = used_cores + unused_cores
  return copy


def initialize_model_state(
    jax_task: tasks_lib.SingleTask,
    prng_key: PRNGKey,
    inputs_shape_dtype: NestedShapeDtypeLike,
    discard_opt_states: bool = False,
    do_init_checkpoint_rules: bool = True,
    is_eval: bool = False,
    checkpoint_type: CheckpointType = CheckpointType.CHECKPOINT_UNSPECIFIED
) -> TrainState:
  """Initializes the model states.

  Weights are random initialized first.
  Then we restores weights based on the init_checkpoint_rules.

  Args:
    jax_task: An instance of tasks.SingleTask.
    prng_key: A PRNGKey, of shape [2], of type np.uint32.
    inputs_shape_dtype: A nested ShapeDtype-like structure for shape inference.
      For pmap, this should use per-core batch size, and for pjit, this should
      use global batch size.
    discard_opt_states: Whether to discard optimizer states.
    do_init_checkpoint_rules: Whether to apply init checkpoint rules or not.
    is_eval: whether to initialize in under eval context. Only used under the
      legacy model initialization flow.
    checkpoint_type: The checkpoint type to use when restoring weights based on
      the init_checkpoint_rules.

  Returns:
    TrainState - training states.
  """
  model = jax_task.model
  prng_key, k1, k2, k3 = jax.random.split(prng_key, 4)
  init_key = {PARAMS: k1, RANDOM: k2, NON_PAX_RNG_KEY: k3}
  if jax_task.hparams.train.always_use_train_for_model_init:
    is_eval_for_init = False
  else:
    is_eval_for_init = is_eval
  var_weight_hparams = model.abstract_init_with_metadata(
      inputs_shape_dtype, do_eval=is_eval_for_init)
  logging.info('init_var prng_seed: %s', init_key)
  logging.info('var_weight_hparams: %s', var_weight_hparams)

  # Use jax.jit to reduce model.init memory usage. Required by a few tests after
  # migrating to shape inference.
  @jax.jit
  def init_fn(init_key):
    context_p = base_layer.JaxContext.HParams(do_eval=is_eval_for_init)
    with base_layer.JaxContext.new_context(hparams=context_p):
      inputs = jax.tree_map(lambda x: jnp.zeros(x.shape, x.dtype),
                            inputs_shape_dtype)
      if model.hparams.fprop_dtype == jnp.bfloat16:
        inputs = jax.tree_map(_maybe_to_bfloat16, inputs)
      return model.init(init_key, inputs)

  initial_vars = init_fn(init_key)
  logging.info('initial_vars: %s', jax.tree_map(lambda x: x.shape,
                                                initial_vars))

  # In case jax_task.model wraps a t5x model, let's remove the params_axes
  # variable collection.
  if 'params_axes' in initial_vars:
    del initial_vars['params_axes']
  train_state = jax_task.create_train_state(initial_vars, var_weight_hparams,
                                            discard_opt_states)
  # `do_init_checkpoint_rules` is False for pjit/spmd.
  if do_init_checkpoint_rules:
    if checkpoint_type == CheckpointType.CHECKPOINT_UNSPECIFIED:
      if py_utils.pmap_use_tensorstore():
        checkpoint_type = CheckpointType.CHECKPOINT_GDA
      else:
        checkpoint_type = CheckpointType.CHECKPOINT_FLAX
    # Overwrite some parts if init_checkpoint_rules are set (warm-start)
    # Note that this assumes a pmap model with Flax checkpoint(s).
    train_state, update_opt_states = jax_task.apply_init_checkpoint_rules(
        train_state, checkpoint_type=checkpoint_type)
    if update_opt_states:
      # Free the previous opt_states as it will be re-computed.
      jax.tree_util.tree_map(lambda x: x.delete(), train_state.opt_states)
      # Re-compute opt_states after the model variables are updated.
      opt_states = jax_task.create_opt_states(train_state.mdl_vars,
                                              var_weight_hparams)
      train_state = train_state.replace(opt_states=opt_states)
  return train_state


def replicate_model_state(model_states: TrainState) -> TrainState:
  """Replicates the model states."""
  def _replicate(state):
    # Skip the copy if it's already replicated.
    if (jax.config.jax_array and isinstance(state, jax.Array) and
        len(state.devices()) != 1):
      return state
    elif isinstance(state, pxla.ShardedDeviceArray):
      return state
    else:
      return jax.device_put_replicated(state, jax.local_devices())

  return jax.tree_map(_replicate, model_states)


def initialize_replicate_model_state(
    jax_task: tasks_lib.SingleTask,
    prng_key: PRNGKey,
    inputs_shape_dtype: NestedShapeDtypeLike,
    discard_opt_states: bool = False) -> TrainState:
  """Initializes and replicates the model states."""
  model_states = initialize_model_state(jax_task, prng_key, inputs_shape_dtype,
                                        discard_opt_states)
  return replicate_model_state(model_states)


def _maybe_to_bfloat16(x: JTensor) -> JTensor:
  if x.dtype == jnp.float32:
    return x.astype(jnp.bfloat16)
  return x


def _maybe_to_bfloat16_vars(mdl_vars, var_weight_hparams):
  """Helper for bfloat16 conversion of model vars.

  Args:
    mdl_vars: A nested structure of vars.
    var_weight_hparams: A nested structure of the variable weight params.
      var_weight_hparams must have the same structure as mdl_vars.

  Returns:
    vars with dtype of bfloat16 for every compatible tensor.
  """
  asserts.assert_same_structure(mdl_vars, var_weight_hparams)

  def _maybe_bfloat16_var_fn(var, var_param):
    if base_layer.var_disallow_bfloat16_conversion(var_param):
      return var
    else:
      return _maybe_to_bfloat16(var)

  return jax.tree_util.tree_map(_maybe_bfloat16_var_fn, mdl_vars,
                                var_weight_hparams)


def _maybe_to_float32(x: JTensor) -> JTensor:
  if x.dtype == jnp.bfloat16:
    return x.astype(jnp.float32)
  return x


# TODO(pax): maybe move to metric_utils.py.
def _maybe_aggregate_metrics_summaries(
    loss_aggregator: base_metrics.LossAggregator,
    weighted_scalars: WeightedScalars,
    summary_dict: SummaryDict,
    per_example_out: NestedMap,
) -> Tuple[JTensor, JTensor, Optional[JTensor], WeightedScalars, SummaryDict,
           NestedMap]:
  """If in pmap, aggregate metrics and summaries across model replicas.

  Args:
    loss_aggregator: An instance of a LossAggregator class to aggregate the
      loss. Defaults to the a single Loss weighted loss calculation.
    weighted_scalars: a WeightedScalars.
    summary_dict: a SummaryDict.
    per_example_out: a NestedMap of per example values.

  Returns:
    (weighted_loss, mean_loss, loss_weight, aggregated_metrics,
     aggregated_summaries, per_example_out)
    weighted_loss - the per-replica loss to back-propagate from. Used for
      computing gradients only.
    mean_loss - the avg per-replica loss. This often is the psum of
      weighted_loss.
    loss_weight - if applicable, the factor by which the per-replica loss was
      weighted; otherwise `None`.
    aggregated_scalars - the aggregated weighted scalars dict.
    aggregated_summaries - the aggregated summaries.
    per_example_out - the aggregated per_example_out.
  """
  # compute weighted loss and mean across shards
  weighted_loss, mean_loss, loss_weight = loss_aggregator.aggregate(
      weighted_scalars)

  if base_layer.is_running_under_pmap():
    # aggregate data across devices.
    aggregated_scalars = type(weighted_scalars)()
    for key in weighted_scalars:
      value, weight = weighted_scalars[key]
      sum_value = jax.lax.psum(
          value * weight, axis_name=PMAP_PARALLEL_AXIS_NAME)
      sum_weight = jax.lax.psum(weight, axis_name=PMAP_PARALLEL_AXIS_NAME)
      aggregated_scalars[key] = (sum_value / (sum_weight + 1e-8), sum_weight)
    aggregated_summaries = summary_utils.aggregate_per_replica_summaries(
        summary_dict)
    per_example_out = jax.tree_map(
        lambda x: jax.lax.all_gather(  # pylint: disable=g-long-lambda
            x, axis_name=PMAP_PARALLEL_AXIS_NAME, tiled=True),
        per_example_out)
  else:
    # No aggregation of weighted scalars is needed.
    aggregated_scalars = weighted_scalars
    # No aggregation of summaries is needed.
    aggregated_summaries = summary_dict

  return (weighted_loss, mean_loss, loss_weight, aggregated_scalars,
          aggregated_summaries, per_example_out)


def _zero_gradient_for_non_learnable_vars(grads, var_weight_hparams):
  """A helper function to zero out grads for non-learnable vars.

  Args:
    grads: a nested structure of var gradients.
    var_weight_hparams: a nested structure of the variable weight params.
      var_weight_hparams must have the same structure as grads.

  Returns:
    grads with gradient for non-learnable vars zero-ed out.
  """
  asserts.assert_same_structure(grads, var_weight_hparams)
  var_is_learnable = jax.tree_util.tree_map(
      lambda x: not base_layer.var_not_trainable(x), var_weight_hparams)

  def _maybe_zero_out_grad_fn(var_grad, var_learnable):
    if var_learnable:
      return var_grad
    elif var_grad.dtype == jax.dtypes.float0:
      # Gradient of an integer-valued input cannot be consumed by jnp operation.
      # Zeros dtype should be int32 same as the original input that produced
      # float0.
      return jnp.zeros_like(var_grad, dtype=jnp.int32)
    else:
      return jnp.zeros_like(var_grad)

  # Zero-out gradient for non-learnable vars.
  return jax.tree_util.tree_map(_maybe_zero_out_grad_fn, grads,
                                var_is_learnable)


def _maybe_synchronize_non_learnable_vars(old_vars, new_vars,
                                          var_weight_hparams):
  """A helper function to synchronize non-learnable vars for pmap training.

  Each non-learnable variable declares how it should be synchronized across
  model replicas during training. Currently, we only support mean aggregation.

  If the input is bool_ we will keep new var.
  If the input is integer and mean aggregation is required, we will round it.

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

  asserts.assert_same_structure(old_vars, new_vars)
  asserts.assert_same_structure(old_vars, var_weight_hparams)

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

    if old_var.dtype == jnp.bool_:
      # bool doesn't support subtraction so cannot sum/mean
      return new_var

    if base_layer.var_requires_mean_sync(var_param):
      # Possible to get float from integer
      output = _synchronize_vars_using_mean(old_var, new_var)
      if jnp.issubdtype(old_var, jnp.integer):
        output = jnp.round(output).astype(new_var.dtype)
      return output
    elif base_layer.var_requires_sum_sync(var_param):
      return _synchronize_vars_using_sum(old_var, new_var)
    else:
      raise ValueError('Non-trainable variables must have a cross-replica '
                       'synchronization method specified.')

  if base_layer.is_running_under_pmap():

    def _sync_var(old_var, new_var, var_param):
      return _synchronize_non_learnable_var(old_var, new_var, var_param)

    return jax.tree_util.tree_map(_sync_var, old_vars, new_vars,
                                  var_weight_hparams)
  else:
    # no synchronization is needed.
    return new_vars


# TODO(yonghui): refactor to pass in learner separately.
def train_step_single_learner(
    jax_task: tasks_lib.SingleTask,
    states: TrainState,
    prng_key: PRNGKey,
    inputs: Union[JTensor, NestedMap],
    fprop_dtype: jnp.dtype = jnp.float32,
    var_weight_hparams: Optional[NestedWeightHParams] = None,
) -> Tuple[TrainState, Any, Any, Any, SummaryDict]:
  """Trains a model for a single step.

  This function works for both pmap-ed model and pjit-ed model.

  TODO(yonghui): Maybe refactor pmap and pjit into two functions.

  This utility is specialized for the singler learner case.

  Args:
    jax_task: An instance of tasks.SingleTask.
    states: An instance of model.TrainState.
    prng_key: A PRNGKey, of shape [2], of type np.uint32.
    inputs: Inputs to the model() function.
    fprop_dtype: fprop datatype, can be either jnp.float32 or jnp.bfloat16.
    var_weight_hparams: A pytree of WeightHParams for the model variables.

  Returns:
    A tuple of the following elements.
    updated_states - updated states.
    loss - loss as computed by model.fprop.
    weighted_scalars - a dict of (value, weight) pairs representing simple
      weighted average metrics or losses.
    per_example_out - auxilillary per-example output as computed in model.fprop.
    summary_tensors - A dict or nested map of summary tensors computed in
      forward as well as backward.
  """
  model = jax_task.model
  assert len(jax_task.learners) == 1
  learner = jax_task.learners[0]

  context_p = base_layer.JaxContext.HParams(do_eval=False)
  # Fold in global_step as part of the random seed key, so that random
  # numbers depends on global step.
  #
  # TODO(yonghui): also fold in the replica id.
  prng_key = jax.random.fold_in(prng_key, states.step)

  if not var_weight_hparams:
    with base_layer.JaxContext.new_context(hparams=context_p):
      var_weight_hparams = model.abstract_init_with_metadata(inputs)
  updated_mdl_vars = jax_task.maybe_adjust_train_state(
      states.step, states.mdl_vars, var_weight_hparams, prng_key)

  def _loss_fn(
      mdl_vars: NestedJTensor, inputs: NestedMap, prng_key
  ) -> Tuple[JTensor, sgf.GradAuxInfo]:
    """Computes loss as well as other auxiliary outputs."""
    if fprop_dtype == jnp.float32:
      pass
    elif fprop_dtype == jnp.bfloat16:
      mdl_vars = _maybe_to_bfloat16_vars(mdl_vars, var_weight_hparams)
      inputs = jax.tree_map(_maybe_to_bfloat16, inputs)
    else:
      assert NotImplementedError(f'fprop_dtype {fprop_dtype} not supported.')

    with base_layer.JaxContext.new_context(hparams=context_p):
      prng_key, k1, k2, k3 = jax.random.split(prng_key, 4)
      apply_rng_keys = {PARAMS: k1, RANDOM: k2, NON_PAX_RNG_KEY: k3}
      (weighted_scalars, per_example_output), updated_vars = model.apply(
          mdl_vars,
          inputs,
          mutable=jax_task.hparams.train.apply_mutable_list,
          method=model.__call__,
          rngs=apply_rng_keys)

      # Fetch all the summary tensors.
      summary_tensors = updated_vars.get(SUMMARIES, {})
      # TODO(yonghui): Fetch aux losses and add them to summaries.
      summary_tensors = summary_utils.flatten_flax_summaries(summary_tensors)

      (weighted_loss, mean_loss, loss_weight, aggregated_scalars,
       aggregated_summaries,
       per_example_output) = _maybe_aggregate_metrics_summaries(
           jax_task.loss_aggregator, weighted_scalars, summary_tensors,
           per_example_output)
      # metrics and summary_tensors no longer needed.
      del weighted_scalars
      del summary_tensors

      forward_updated_vars = {}
      for collection in [NON_TRAINABLE] + NON_PAX_VAR_COLLECTION:
        if collection in updated_vars:
          forward_updated_vars[collection] = updated_vars[collection]
    if fprop_dtype == jnp.bfloat16 and weighted_loss.dtype == fprop_dtype:
      weighted_loss = weighted_loss.astype(jnp.float32)
    return weighted_loss, sgf.GradAuxInfo(
        loss_weight=loss_weight,
        aux_info=(mean_loss, aggregated_scalars, forward_updated_vars,
                  aggregated_summaries, per_example_output))

  prng_key, subkey = jax.random.split(prng_key)

  # Layers may have integer-valued non-trainable vars. `allow_int=True` is
  # needed to allow jax.grad to differentiate wrt integer-values.
  # However, the gradient of an integer input will have a trivial vector-space
  # dtype (float0). They cannot be consumed by jnp operations.
  # _zero_gradient_for_non_learnable_vars needs to handle jax.dtypes.float0
  # specially.

  if learner.stochastic_gradient is None:
    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True, allow_int=True)
  else:
    grad_fn = functools.partial(learner.stochastic_gradient.grad_fn, _loss_fn)
  ((weighted_loss, aux_info), grads) = grad_fn(updated_mdl_vars, inputs, subkey)

  (mean_loss, weighted_scalars, fwd_updated_vars, fwd_summary_tensors,
   per_example_out) = aux_info.aux_info

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
        asserts.assert_same_structure(states.mdl_vars[collection],
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

  summary_tensors = NestedMap()
  summary_tensors.update(fwd_summary_tensors)
  summary_tensors.update(bwd_summary_tensors)

  return (new_states, mean_loss, weighted_scalars, per_example_out,
          summary_tensors)


# TODO(laigd): rename - eval has nothing to do with number of learners.
def eval_step_single_learner(
    jax_task: tasks_lib.SingleTask,
    states: TrainState,
    prng_key: JTensor,
    inputs: Union[JTensor, NestedMap],
    fprop_dtype: jnp.dtype = jnp.float32,
    var_weight_hparams: Optional[NestedWeightHParams] = None,
) -> Tuple[Any, Any, Any, SummaryDict]:
  """Evaluates a model for a single step.

  This utility is specialized for the single learner case.

  Args:
    jax_task: An instance of tasks.SingleTask.
    states: An instance of model.TrainState.
    prng_key: A prng seed, of shape [2], of type np.uint32.
    inputs: Inputs to the model() function.
    fprop_dtype: fprop datatype, can be either jnp.float32 or jnp.bfloat16.
    var_weight_hparams: A pytree of WeightHParams for the model variables.

  Returns:
    A tuple of the following elements.
    loss - loss as computed by model.fprop.
    weighted_scalars - a dict of (value, weight) scalar pairs representing
      simple metrics or losses.
    per_example_out - auxilillary per-example output as computed in model.fprop.
    summary_tensors - A nested map or dict of summary tensors computed in
      forward as well as backward pass.
  """
  model = jax_task.model
  context_p = base_layer.JaxContext.HParams(do_eval=True, summary_verbosity=2)
  # Fold in global_step as part of the random seed key, so that random
  # numbers depends on global step.
  prng_key = jax.random.fold_in(prng_key, states.step)
  mdl_vars = states.mdl_vars
  # assert not states.opt_states

  if not var_weight_hparams:
    var_weight_hparams = model.abstract_init_with_metadata(
        inputs,
        do_eval=not jax_task.hparams.train.always_use_train_for_model_init)

  if fprop_dtype == jnp.float32:
    pass
  elif fprop_dtype == jnp.bfloat16:
    mdl_vars = _maybe_to_bfloat16_vars(mdl_vars, var_weight_hparams)
    inputs = jax.tree_map(_maybe_to_bfloat16, inputs)
  else:
    assert NotImplementedError(f'fprop_dtype {fprop_dtype} not supported.')

  enum_keys, inputs = py_utils.filter_by_matching_keys(
      inputs, [py_utils.PROVENANCE_PREFIX])
  with base_layer.JaxContext.new_context(hparams=context_p):
    prng_key, k1, k2, k3 = jax.random.split(prng_key, 4)
    apply_rng_keys = {PARAMS: k1, RANDOM: k2, NON_PAX_RNG_KEY: k3}
    (weighted_scalars, per_example_out), updated_vars = model.apply(
        mdl_vars,
        inputs,
        mutable=jax_task.hparams.evaluate.apply_mutable_list,
        method=model.__call__,
        rngs=apply_rng_keys)

    summary_tensors = updated_vars.get(SUMMARIES, {})
    # TODO(yonghui): Add aux-loss to summaries.
    summary_tensors = summary_utils.flatten_flax_summaries(summary_tensors)

    # merge back, if any, enum keys for eval matching
    per_example_out.update(enum_keys)
    (_, mean_loss, _, aggregated_scalars, aggregated_summaries,
     per_example_out) = _maybe_aggregate_metrics_summaries(
         jax_task.loss_aggregator, weighted_scalars, summary_tensors,
         per_example_out)

    # weighted_scalars and summary_tensors no longer needed.
    del weighted_scalars
    del summary_tensors

  if fprop_dtype == jnp.bfloat16:
    (mean_loss, aggregated_scalars,
     per_example_out, aggregated_summaries) = jax.tree_map(
         _maybe_to_float32,
         (mean_loss, aggregated_scalars, per_example_out, aggregated_summaries))

  return mean_loss, aggregated_scalars, per_example_out, aggregated_summaries


def decode_step(
    model: base_model.BaseModel,
    states: TrainState,
    prng_key: JTensor,
    var_weight_hparams: NestedWeightHParams,
    inputs: Union[JTensor, NestedMap],
    fprop_dtype: jnp.dtype = jnp.float32,
    prng_key_fold_with_global_step: bool = True
) -> Tuple[Tuple[Any, Any, Any], NestedMap]:
  """Decodes a model for a single step.

  Args:
    model: An instance of models.BaseModel.
    states: An instance of TrainState..
    prng_key: A prng seed, of shape [2], of type np.uint32.
    var_weight_hparams: A pytree of WeightHParams for the model variables.
    inputs: A batch of inputs to model.decode().
    fprop_dtype: fprop datatype, can be either jnp.float32 or jnp.bfloat16.
    prng_key_fold_with_global_step: Boolean to decide whether to fold the
      prng_key with the global step.

  Returns:
    A tuple of (weighted_scalars, results, eval_metrics) as computed
      by model.decode() and the updated weights.
  """
  context_p = base_layer.JaxContext.HParams(do_eval=True, summary_verbosity=2)
  if prng_key_fold_with_global_step:
    # Fold in global_step as part of the random seed key, so that random
    # numbers depends on global step.
    prng_key = jax.random.fold_in(prng_key, states.step)
  mdl_vars = states.mdl_vars

  assert not states.opt_states

  if fprop_dtype == jnp.bfloat16:
    mdl_vars = _maybe_to_bfloat16_vars(mdl_vars, var_weight_hparams)
    inputs = jax.tree_map(_maybe_to_bfloat16, inputs)
  elif fprop_dtype != jnp.float32:
    assert NotImplementedError(f'fprop_dtype {fprop_dtype} not supported.')

  enum_keys, inputs = py_utils.filter_by_matching_keys(
      inputs, [py_utils.PROVENANCE_PREFIX])
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

    # If model's decode() function only has two returns, we assume they are
    # weighted_scalars and per_example_outputs, and the models haven't yet
    # added an Metrics return (NestedMap of keys -> clu.metrics).
    # TODO(bencaine): Remove this when all models are updated.
    if len(outputs) == 2:
      weighted_scalars, per_example_out = outputs
      outputs = (weighted_scalars, per_example_out, {})

    # merge back, if any, enum keys for eval matching
    per_example_out = outputs[1]
    per_example_out.update(enum_keys)

    summary_tensors = updated_vars.get(base_layer.SUMMARIES, {})
    if summary_tensors:
      summary_tensors = jax.tree_map(_maybe_to_float32, summary_tensors)
      updated_vars[base_layer.SUMMARIES] = summary_tensors

    return outputs, updated_vars


def _decode_step_for_partitioner(
    task, states, prng_key, inputs, fprop_dtype, var_weight_hparams
):
  """Decode step function used by the partitioner."""
  return decode_step(
      task.model, states, prng_key, var_weight_hparams, inputs, fprop_dtype
  )


def initialize_partitioned_model_states(
    jax_task: tasks_lib.SingleTask,
    prng_key: PRNGKey,
    global_input_shapes: NestedShapeDtypeLike,
    discard_opt_states: bool = False,
    global_mesh: Optional[jax.sharding.Mesh] = None,
    checkpoint_type: CheckpointType = CheckpointType.CHECKPOINT_GDA,
    state_specs: Optional[TrainState] = None,
    do_init_checkpoint_rules: bool = True) -> Tuple[TrainState, TrainState]:
  """Initializes model vars that are partitioned over TPU devices.

  Weights are random initialized first.
  Then we restores weights based on the init_checkpoint_rules.

  This function is equivalent to calling a pjit-ted version of
  InitializesModelStates().

  Args:
    jax_task: The task which is an instance of tasks.SingleTask.
    prng_key: A PRNGKey.
    global_input_shapes: Global shapes of sample inputs for shape inference.
    discard_opt_states: bool, When true, optimizer slot variables are skipped.
    global_mesh: The global mesh to use when restoring weights based on the
      init_checkpoint_rules. Required for GDA-based checkpoints.
    checkpoint_type: The checkpoint type to use when restoring weights based on
      the init_checkpoint_rules.
    state_specs: The TrainState specs when restoring weights based on the
      init_checkpoint_rules. Required for GDA-based checkpoints.
    do_init_checkpoint_rules: If apply init_checkpoint_rules.

  Returns:
    The partitioned specs and the partitioned vars themselves.
  """
  model = jax_task.model
  var_weight_hparams = model.abstract_init_with_metadata(global_input_shapes)

  if state_specs is None:
    train_state_partition_specs = jax_task.create_train_state_partition_specs(
        var_weight_hparams, discard_opt_states)
  else:
    if discard_opt_states:
      train_state_partition_specs = TrainState(
          step=state_specs.step, mdl_vars=state_specs.mdl_vars, opt_states={})
    else:
      train_state_partition_specs = state_specs

  train_state_unpadded_shapes = jax.tree_map(
      lambda x: x.shape,
      jax_task.create_train_state_unpadded_shapes(var_weight_hparams,
                                                  discard_opt_states))
  assert train_state_partition_specs is not None

  def init_model_from_seed(prng_key):
    outs = initialize_model_state(
        jax_task,
        prng_key,
        global_input_shapes,
        discard_opt_states,
        do_init_checkpoint_rules=False)
    return py_utils.maybe_pad_uneven_sharding(outs, train_state_partition_specs,
                                              train_state_unpadded_shapes,
                                              model.hparams.mesh_shape,
                                              model.hparams.mesh_axis_names)

  logging.info('unpadded_out_shape: %s', train_state_unpadded_shapes)
  logging.info('train_state_partition_specs: %s', train_state_partition_specs)
  asserts.assert_same_structure(train_state_unpadded_shapes,
                                train_state_partition_specs)

  mesh_names = model.hparams.mesh_axis_names
  prng_key_partition_spec = base_layer.to_partition_spec((None,), mesh_names)

  init_fn = pjit.pjit(
      init_model_from_seed,
      in_axis_resources=(prng_key_partition_spec),
      out_axis_resources=train_state_partition_specs)
  init_fn = bind_mesh(init_fn, global_mesh)

  partitioned_vars = init_fn(prng_key)
  # Overwrite some parts if init_checkpoint_rules are set (warm-start)
  if (do_init_checkpoint_rules and
      jax_task.hparams.train.init_from_checkpoint_rules):
    # TODO(b/230132535): Note that this application after constructing the
    # partitioned vars is currently inconsistent with what is being performed
    # for pmap models.
    partitioned_vars, _ = jax_task.apply_init_checkpoint_rules(
        partitioned_vars,
        train_state_partition_specs=train_state_partition_specs,
        global_mesh=global_mesh,
        checkpoint_type=checkpoint_type)

  return (train_state_partition_specs, partitioned_vars)


def shard_on_batch_dim_partition_spec(
    mesh_names: Sequence[str], x: jax.ShapeDtypeStruct
) -> jax.sharding.PartitionSpec:
  """Fully shards x on the batch dimension."""
  x_dim = len(x.shape)
  assert x_dim >= 1
  sharding = [-1] * x_dim
  # Assume the first dim is batch, and fully shard the batch dim over the entire
  # mesh.
  sharding[0] = tuple(mesh_names)
  return base_layer.to_partition_spec(sharding, mesh_names)


def reshard_input_based_on_rank_fn(
    mapping_dict: Optional[Dict[str, base_layer.SplitDimsMapping]],
    mesh_names: Sequence[str],
    x: JTensor,
) -> JTensor:
  """Reshards input based on its rank.

  Args:
    mapping_dict: Dictionary which contains the split mapping for different
      shapes. For n-d shape, it must have an entry f'map_{n}d' which tells us
      how to partition tensors of this dimension. If mapping_dict is None,
      no resharding of the tensor.
    mesh_names: List of mesh axis names.
    x: JTensor which to shard.

  Returns:
    Resharded tensor.
  """
  if mapping_dict is None:
    logging.info('No resharding of input as mapping_dict is None.')
    return x
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
) -> Optional[jax.sharding.PartitionSpec]:
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


def get_inputs_shape_dtype(
    input_p: base_input.BaseInput.HParams,
) -> Tuple[NestedShapeDtypeLike, NestedShapeDtypeLike]:
  """Returns the per-host and global shape/dtype information of the input."""
  sample_inputs = instantiate(input_p).get_next_padded()
  # TODO(pax-dev): Retrieve shapes from input specs and compare against real
  # input shapes from training input pipeline.
  perhost_inputs_shape_dtype = jax.tree_map(
      lambda x: jax.ShapeDtypeStruct(shape=x.shape, dtype=x.dtype),
      sample_inputs,
  )
  global_inputs_shape_dtype = jax.tree_map(
      py_utils.get_global_input_shape_dtype, sample_inputs
  )
  return perhost_inputs_shape_dtype, global_inputs_shape_dtype


def get_input_partition_specs(mesh_axis_names, inputs_shape_dtype):
  # Compute inputs PartitionSpec from inputs_shape_dtype
  inputs_partition_spec_fn = functools.partial(
      shard_on_batch_dim_partition_spec, mesh_axis_names)
  return jax.tree_util.tree_map(inputs_partition_spec_fn, inputs_shape_dtype)


def _remove_input_padding(inputs: NestedJTensor,
                          unpadded_global_batch_size: Optional[int] = None,
                          mesh_names: Optional[Sequence[str]] = None):
  """Removes input padding on the batch dimension."""
  if unpadded_global_batch_size is None:
    return inputs
  padded_global_batch_size = jax.tree_util.tree_leaves(inputs)[0].shape[0]
  if padded_global_batch_size == unpadded_global_batch_size:
    return inputs

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
        x, (None,) * x.ndim, mesh_names, unconstrained_dims=range(0, x.ndim))
    x = jax.lax.slice_in_dim(x, 0, 1, axis=0)
    # Partially annotate the sliced dim to be replicated.
    x = base_layer.maybe_shard(
        x, (None,) * x.ndim, mesh_names, unconstrained_dims=range(1, x.ndim))
    x = jnp.squeeze(x, axis=0)
    return x[:unpadded_global_batch_size]

  return jax.tree_map(_remove_padding, inputs)


def _write_input_specs(
    input_specs: NestedShapeDtypeLike, job_log_dir: epath.Path
) -> None:
  """Writes input specs as JSON to a file."""
  if jax.process_index() != 0:
    return

  def _to_dict(array_like: Any) -> Dict[str, Any]:
    return {
        '_array': {
            'shape': list(array_like.shape),
            'dtype': str(array_like.dtype),
        }
    }

  input_specs_dict = frozen_dict.unfreeze(
      jax.tree_util.tree_map(_to_dict, input_specs)
  )
  fpath = job_log_dir / 'input_specs.json'
  with fpath.open('w') as f:
    json.dump(input_specs_dict, f, indent=2, sort_keys=True)

  work_unit = platform.work_unit()
  work_unit.create_artifact(
      platform.ArtifactType.FILE, str(fpath), 'Input specs'
  )


class Partitioner(metaclass=abc.ABCMeta):
  """Interface for partitioning computations.

  Example usage:

  ```
  # Create the partitioner.
  partitioner = create_partitioner(
      jax_task, init_key, train_inputs_shape_dtype, job_log_dir=job_log_dir)

  # [Optional] Set the training input shape/dtype information. Needed only if
  # train_inputs_shape_dtype is not set when creating the partitioner above.
  train_input_p = ...  # The config for training input pipeline.
  train_input_p = partitioner.preprocess_input_params(train_input_p)
  train_input_pipeline = instantiate(train_input_p)
  partitioner.set_train_inputs_shape_dtype(train_input_pipeline)

  # Restore the train state.
  metadata = partitioner.get_train_state_metadata()
  train_state = restore(metadata, ...)

  # Partition the step function and run it.
  partitioned_step_fn = partitioner.partition(
      step_fn, inputs_shape_dtype, is_eval, unpadded_global_batch_size)
  prng_key = ...  # Create the PRNG key.
  prng_key = partitioner.preprocess_prng_key(prng_key)
  inputs = train_input_pipeline.get_next_padded()
  partitioned_step_fn(train_state, prng_key, inputs)
  ```
  """

  def __init__(
      self,
      jax_task: tasks_lib.SingleTask,
      init_key: PRNGKey,
      train_inputs_shape_dtype: Optional[NestedShapeDtypeLike] = None,
      init_is_eval: bool = False,
      job_log_dir: Optional[epath.Path] = None,
  ):
    """Constructor.

    Args:
      jax_task: The task which is an instance of tasks.SingleTask.
      init_key: PRNGKey for initializing the model variables.
      train_inputs_shape_dtype: Shape/dtype attributes of the training inputs to
        model.init, for use in getting params of model variables. Can also be
        set using self.set_train_inputs_shape_dtype() if not provided during
        construction.
      init_is_eval: Whether it should set is_eval=True when running
        abstract_init_with_metadata.
      job_log_dir: Directory for the job logs.
    """
    self._jax_task = jax_task
    self._init_key = init_key
    self._train_inputs_shape_dtype = train_inputs_shape_dtype
    # TODO(laigd): remove this option (it should always be False) once
    # always_use_train_for_model_init is enabled by default.
    self._init_is_eval = init_is_eval
    self._job_log_dir = job_log_dir
    self._train_state_metadata = None

  @property
  def train_inputs_shape_dtype(self) -> Optional[NestedShapeDtypeLike]:
    """Shape/dtype attributes of the training inputs to model.init."""
    return self._train_inputs_shape_dtype

  # TODO(pax-dev): remove this method and switch to train_inputs_shape_dtype
  # provided during construction once all experiments provide input specs.
  @abc.abstractmethod
  def set_train_inputs_shape_dtype(self, train_input_pipeline: Any) -> None:
    """Sets training shape/dtype using sample inputs from the input pipeline.

    Args:
      train_input_pipeline: The training input pipeline that provides a
        peek_padded() or get_next_padded() method to get sample input batches.
        TODO(laigd): consider using a protocol instead.
    """

  @property
  def global_mesh(self) -> Optional[jax.sharding.Mesh]:
    """The global mesh."""
    return None

  @abc.abstractmethod
  def preprocess_input_params(
      self, input_ps: base_input.BaseInput.HParams
  ) -> base_input.BaseInput.HParams:
    """Preprocess input hparam(s) if necessary.

    This is necessary before using the hparam to create the input pipeline, so
    that the generated input batch can be used to run the partitioned function.

    Args:
      input_ps: The input hparam(s) to adjust.

    Returns:
      The adjusted input hparam(s), with the same nested structure.
    """

  @abc.abstractmethod
  def preprocess_prng_key(self, prng_key: PRNGKey) -> PRNGKey:
    """Preprocess the key before using it to run the partitioned function.

    Args:
      prng_key: The prng key to preprocess.

    Returns:
      The preprocessed prng key that's ready to be used to run the partitioned
      function.
    """

  @abc.abstractmethod
  def get_train_state_metadata(
      self,
      discard_opt_states: bool = False,
  ) -> TrainStateMetadata:
    """Gets the TrainStateMetadata used for partitioning.

    Args:
      discard_opt_states: Whether to discard the part corresponding to the
        optimizer states or not.

    Returns:
      The TrainStateMetadata.
    """

  def _get_train_state_metadata_default(self) -> TrainStateMetadata:
    """Helper method to get the TrainStateMetadata."""
    if not self._train_inputs_shape_dtype:
      raise ValueError(
          'Training input spec is not set. It can be set either when creating '
          'the partitioner, or by calling set_train_inputs_shape_dtype().'
      )
    return create_train_state_metadata(
        self._jax_task,
        self._train_inputs_shape_dtype,
        discard_opt_states=False,
        do_eval=False
        if self._jax_task.hparams.train.always_use_train_for_model_init
        else self._init_is_eval,
    )

  def _maybe_discard_opt_states(
      self, metadata: TrainStateMetadata, discard_opt_states: bool
  ) -> TrainStateMetadata:
    if not discard_opt_states:
      # Make sure all the metadata has opt_states.
      for state in [
          metadata.padded_global_shapes,
          metadata.unpadded_global_shapes,
          metadata.partition_specs,
      ]:
        if state:
          assert state.opt_states
      return metadata

    # Discard the opt_states.
    to_eval_state = lambda state: state.to_eval_state() if state else None
    return TrainStateMetadata(
        input_shape_dtype=metadata.input_shape_dtype,
        var_weight_hparams=metadata.var_weight_hparams,
        padded_global_shapes=to_eval_state(metadata.padded_global_shapes),
        unpadded_global_shapes=to_eval_state(metadata.unpadded_global_shapes),
        partition_specs=to_eval_state(metadata.partition_specs),
    )

  @abc.abstractmethod
  def partition(
      self,
      step_fn: Callable[..., Any],
      inputs_shape_dtype: NestedShapeDtypeLike,
      is_eval: bool,
      unpadded_global_batch_size: Optional[int] = None,
  ) -> Tuple[Callable[..., Any], NestedPartitionSpec]:
    """Partitions the step function.

    Args:
      step_fn: Training or eval step function to partition.
      inputs_shape_dtype: Shape/dtype attributes of the inputs of step_fn.
      is_eval: A boolean indicating if it's a eval/decode task or not.
      unpadded_global_batch_size: If not None, the unpadded size of global
        batch, and the padding is on the right side of each input.

    Returns:
      (partitioned_step_fn, input_partition_spec):

      - partitioned_step_fn: The partitioned step function.
      - input_partition_spec: The partition spec for the inputs of the step
        function.
    """


class _PmapPartitioner(Partitioner):

  def set_train_inputs_shape_dtype(self, train_input_pipeline: Any) -> None:
    assert (
        not self._train_inputs_shape_dtype
    ), 'train_inputs_shape_dtype has been set before.'
    input_fn = (
        train_input_pipeline.peek_padded
        if hasattr(train_input_pipeline, 'peek_padded')
        else train_input_pipeline.get_next_padded
    )
    sample_inputs = input_fn()
    self._train_inputs_shape_dtype = jax.tree_map(
        py_utils.get_global_input_shape_dtype, sample_inputs
    )
    _write_input_specs(self._train_inputs_shape_dtype, self._job_log_dir)

  def preprocess_input_params(
      self, input_ps: base_input.BaseInput.HParams
  ) -> base_input.BaseInput.HParams:
    """Preprocess input hparam(s) if necessary."""
    return input_ps

  def preprocess_prng_key(self, prng_key: PRNGKey) -> PRNGKey:
    """Preprocess the key before using it to run the partitioned function."""
    # TODO(laigd): follow train_and_evaluate_pmap() and split the key into
    # jax.local_device_count() pieces.
    return prng_key

  def get_train_state_metadata(
      self,
      discard_opt_states: bool = False,
  ) -> TrainStateMetadata:
    """Gets the TrainStateMetadata used for partitioning."""
    if not self._train_state_metadata:
      self._train_state_metadata = self._get_train_state_metadata_default()
    return self._maybe_discard_opt_states(
        self._train_state_metadata, discard_opt_states
    )

  def partition(
      self,
      step_fn: Callable[..., Any],
      inputs_shape_dtype: NestedShapeDtypeLike,
      is_eval: bool,
      unpadded_global_batch_size: Optional[int] = None,
  ) -> Tuple[Callable[..., Any], NestedPartitionSpec]:
    """Partitions the step function."""
    del inputs_shape_dtype, unpadded_global_batch_size

    def _wrapped_step_fn(state, prng_key, inputs):
      return step_fn(
          self._jax_task,
          state,
          prng_key,
          inputs,
          fprop_dtype=self._jax_task.hparams.model.fprop_dtype,
          var_weight_hparams=self._train_state_metadata.var_weight_hparams,
      )

    return (
        jax.pmap(
            _wrapped_step_fn,
            # For training, TrainState is the first argument and return value.
            # We setup donation/alias to minimize device memory usage.
            donate_argnums=() if is_eval else (0,),
            axis_name=base_layer.PMAP_PARALLEL_AXIS_NAME,
        ),
        # Input partition spec.
        None,  # type: ignore
    )


class _PjitPartitioner(Partitioner):
  """Used for partitioning a step function of a SPMD model."""

  @dataclasses.dataclass
  class AutoShardingInfo:
    """Info needed by auto-sharding to get train state partition specs."""

    # The step function to run auto-sharding on. This will be used to compute
    # the train state partition spec.
    step_fn: Callable[..., Any]

    # Whether step_fn is used for evaluation.
    is_eval: bool

    # Whether to replicate the output when auto sharding is enabled.
    # TODO(pax-dev): support custom output partition spec.
    replicate_output: bool

  @dataclasses.dataclass
  class _AutoShardingResult:
    """Output of auto-sharding and related information."""

    # The partitioned step_fn generated by auto-sharding.
    partitioned_step_fn: Callable[..., Any]

    # Generated partition spec for the TrainState.
    train_state_partition_spec: TrainState

    # Generated partition spec for the data inputs of the step function.
    input_partition_spec: NestedPartitionSpec

    # Shape/dtype information for the inputs to partitioned_step_fn.
    inputs_shape_dtype: NestedShapeDtypeLike

  def __init__(
      self,
      jax_task: tasks_lib.SingleTask,
      init_key: PRNGKey,
      train_inputs_shape_dtype: Optional[NestedShapeDtypeLike] = None,
      init_is_eval: bool = False,
      auto_sharding_info: Optional[AutoShardingInfo] = None,
      job_log_dir: Optional[epath.Path] = None,
  ):
    """Constructor.

    Args:
      jax_task: The task which is an instance of tasks.SingleTask.
      init_key: PRNGKey for initializing the model variables.
      train_inputs_shape_dtype: Shape/dtype attributes of the inputs to
        model.init, for use in getting params of model variables. This is needed
        when always_use_train_for_model_init is True.
      init_is_eval: Whether it should set is_eval=True when running
        abstract_init_with_metadata.
      auto_sharding_info: Information used for XLA auto-sharding. If None, it'll
        use the sharding information provided by the model config instead.
      job_log_dir: Directory for the job logs.
    """
    # Create global mesh.
    model_p = jax_task.hparams.model
    device_mesh = py_utils.create_device_mesh(
        model_p.ici_mesh_shape,
        model_p.dcn_mesh_shape,
        contiguous_submeshes=model_p.contiguous_submeshes,
    )
    logging.info('device_mesh: %s', device_mesh)
    self._global_mesh = jax.sharding.Mesh(device_mesh, model_p.mesh_axis_names)

    # Initialize the remaining parts.
    super().__init__(
        jax_task, init_key, train_inputs_shape_dtype, init_is_eval, job_log_dir
    )
    self._auto_sharding_info = auto_sharding_info
    self._auto_sharding_result = None  # Used to cache auto-sharding results.
    self._enable_auto_sharding = auto_sharding_info is not None
    self._mesh_names = self._jax_task.hparams.model.mesh_axis_names

    self._broadcast_key_fn = None

  def set_train_inputs_shape_dtype(self, train_input_pipeline: Any) -> None:
    assert (
        not self._train_inputs_shape_dtype
    ), 'train_inputs_shape_dtype has been set before.'
    input_fn = (
        train_input_pipeline.peek_padded
        if hasattr(train_input_pipeline, 'peek_padded')
        else train_input_pipeline.get_next_padded
    )
    sample_inputs = input_fn()
    self._train_inputs_shape_dtype = jax.tree_map(
        py_utils.get_global_input_shape_dtype, sample_inputs
    )
    perhost_inputs_shape_dtype = jax.tree_map(
        lambda x: jax.ShapeDtypeStruct(shape=x.shape, dtype=x.dtype),
        sample_inputs,
    )
    _write_input_specs(perhost_inputs_shape_dtype, self._job_log_dir)

    if self._enable_auto_sharding:
      train_input_p = train_input_pipeline.hparams
      if train_input_p.num_infeed_hosts < jax.process_count() or (
          train_input_p.cls.get_batch_size(train_input_p)
          < jax.local_device_count()
      ):
        raise NotImplementedError(
            'Per-device batch size < 1 not supported for auto sharding.'
        )
      logging.info('Auto sharding is enabled in PAX.')

  @property
  def global_mesh(self) -> Optional[jax.sharding.Mesh]:
    return self._global_mesh

  def preprocess_input_params(
      self, input_ps: base_input.BaseInput.HParams
  ) -> base_input.BaseInput.HParams:
    """Preprocess input hparam(s) if necessary."""
    return jax.tree_util.tree_map(
        lambda p: adjust_input_params_for_small_batch(p, self.global_mesh),
        input_ps,
    )

  def preprocess_prng_key(self, prng_key: PRNGKey) -> PRNGKey:
    """Preprocess the key before using it to run the partitioned function."""
    if not self._broadcast_key_fn:
      # The prng keys are already created on device with jax.random.split. We
      # broadcast it with an identity pjit function to avoid doing it in the
      # loop where a multi-slice program could be generated.
      def _broadcast_key(k):
        def _identity(x):
          return x

        with self._global_mesh:
          return pjit.pjit(
              _identity, in_axis_resources=None, out_axis_resources=None
          )(k)

      self._broadcast_key_fn = _broadcast_key

    return self._broadcast_key_fn(prng_key)

  def get_train_state_metadata(
      self,
      discard_opt_states: bool = False,
  ) -> TrainStateMetadata:
    """Gets the TrainStateMetadata used for partitioning.

    Args:
      discard_opt_states: Whether to discard the part corresponding to the
        optimizer states or not.
    """
    if self._train_state_metadata:
      return self._maybe_discard_opt_states(
          self._train_state_metadata, discard_opt_states
      )

    train_state_metadata = self._get_train_state_metadata_default()
    if not self._enable_auto_sharding:
      self._train_state_metadata = train_state_metadata
      return self._maybe_discard_opt_states(
          self._train_state_metadata, discard_opt_states
      )

    assert self._train_inputs_shape_dtype
    # Currently we run auto-sharding only once to get the train state
    # partition spec, and reuse it for all subsequent get_train_state_metadata()
    # and partition() calls.
    # Since the structure of the partition spec needs to match the actual train
    # state passed to the auto-sharded step function, we need to discard the
    # opt_states if AutoShardingInfo.is_eval==True.
    train_state_metadata = self._maybe_discard_opt_states(
        train_state_metadata, self._auto_sharding_info.is_eval
    )
    if not self._auto_sharding_result:
      input_partition_spec = get_input_partition_specs(
          self._mesh_names, self._train_inputs_shape_dtype
      )
      wrapped_step_fn = self._get_step_fn(
          self._auto_sharding_info.step_fn,
          self._auto_sharding_info.is_eval,
          train_state_metadata,
          input_partition_spec=input_partition_spec,
      )
      with self.global_mesh:
        partitioned_step_fn, input_pspec, train_state_pspec = (
            self._partition_auto_shard(
                wrapped_step_fn,
                self._auto_sharding_info.is_eval,
                self._train_inputs_shape_dtype,
                input_partition_spec,
                train_state_metadata,
            )
        )
      self._auto_sharding_result = _PjitPartitioner._AutoShardingResult(
          partitioned_step_fn,
          train_state_pspec,
          input_pspec,
          jax.tree_util.tree_map(lambda x: x, self._train_inputs_shape_dtype),
      )

    partition_specs = self._auto_sharding_result.train_state_partition_spec
    self._train_state_metadata = dataclasses.replace(
        train_state_metadata, partition_specs=partition_specs
    )
    return self._maybe_discard_opt_states(
        self._train_state_metadata, discard_opt_states
    )

  def partition(
      self,
      step_fn: Callable[..., Any],
      inputs_shape_dtype: NestedShapeDtypeLike,
      is_eval: bool,
      unpadded_global_batch_size: Optional[int] = None,
  ) -> Tuple[Callable[..., Any], NestedPartitionSpec]:
    """Gets a sharded (pjit-ed) step function of the SPMD Model.

    Args:
      step_fn: Training or eval step function to partition.
      inputs_shape_dtype: Shape/dtype attributes of the inputs of step_fn.
      is_eval: A boolean indicating if it's a eval/decode task or not.
      unpadded_global_batch_size: If not None, the unpadded size of global
        batch, and the padding is on the right side of each input. Required only
        when auto sharding is disabled.

    Returns:
      (partitioned_step_fn, input_partition_spec):

      - partitioned_step_fn: The partitioned step function.
      - input_partition_spec: The partition spec for the inputs of the step
        function.
    """
    # Auto-sharding result is generated by self.get_train_state_metadata, so we
    # call it first.
    # The step function doesn't need opt_states when is_eval=True.
    metadata = self.get_train_state_metadata(discard_opt_states=is_eval)
    if (
        self._enable_auto_sharding
        and step_fn is self._auto_sharding_info.step_fn
        and inputs_shape_dtype == self._auto_sharding_result.inputs_shape_dtype
    ):
      return (
          self._auto_sharding_result.partitioned_step_fn,
          self._auto_sharding_result.input_partition_spec,
      )

    input_partition_spec = get_input_partition_specs(
        self._mesh_names, inputs_shape_dtype
    )
    logging.info('step_fn inputs_partition_spec=%s', input_partition_spec)
    # Step function to be pjit-ed.
    wrapped_step_fn = self._get_step_fn(
        step_fn,
        is_eval,
        metadata,
        input_partition_spec,
        unpadded_global_batch_size,
    )
    with self.global_mesh:
      return (
          self._partition_manual_shard(
              wrapped_step_fn,
              is_eval,
              inputs_shape_dtype,
              input_partition_spec,
              metadata,
          ),
          input_partition_spec,
      )

  def _get_step_fn(
      self,
      step_fn: Callable[..., Any],
      is_eval: bool,
      metadata: TrainStateMetadata,
      input_partition_spec: NestedPartitionSpec,
      unpadded_global_batch_size: Optional[int] = None,
  ):
    """Returns a step function to apply the SPMD partition (pjit)."""
    task_p = self._jax_task.hparams
    model_p = task_p.model
    reshard_inputs_fn = functools.partial(reshard_input_based_on_rank_fn,
                                          task_p.train.inputs_split_mapping,
                                          self._mesh_names)

    def _wrapped_step_fn(state, prng_key, inputs):
      # When auto-sharding is enabled, we can't pad the variables whose input
      # sharding may get changed by auto-sharding.
      # TODO(pax-dev): Add support for padding and unpadding inputs when auto
      # sharding is enabled.
      if not self._enable_auto_sharding:
        # When there are input padding on multi-host, we use a different device
        # order in the program's input sharding. We now make sure they are
        # resharded back to the device order consistent with the global mesh.
        inputs = jax.tree_util.tree_map(
            lambda x, s: base_layer.maybe_shard(x, s, self._mesh_names), inputs,
            input_partition_spec)
        # Vars/inputs are padded at program entry/exit to avoid uneven sharding.
        # We slice the vars to remove padding before the step computation, and
        # pad them after the step computation to make user code independent of
        # paddings.
        # Internal uneven sharding in the step computation is supported by XLA.
        state = self._unpad_states(metadata, state)
        inputs = self._unpad_inputs(inputs, unpadded_global_batch_size)

      # Reshard inputs.
      # TODO(pax): when xla auto-sharding is enabled, it'll automatically figure
      # out the proper sharding of intermediate nodes, we can get rid of this
      # manual sharding now?
      inputs = jax.tree_map(reshard_inputs_fn, inputs)

      # When auto sharding is enabled, uneven sharding is not supported. This is
      # because the way padding is added is dependent on the provided train
      # state partition specs. This is not available during auto-sharding until
      # after the compilation is done.
      fn_out = step_fn(
          self._jax_task,
          state,
          prng_key,
          inputs,
          fprop_dtype=model_p.fprop_dtype,
          var_weight_hparams=metadata.var_weight_hparams)

      if is_eval:
        return fn_out
      assert len(fn_out) > 1

      # Pad the model states again for training step functions.
      if not self._enable_auto_sharding:
        padded_states = self._pad_states(metadata, fn_out[0])
        fn_out = (padded_states,) + fn_out[1:]
      return fn_out

    return _wrapped_step_fn

  def _pjit(
      self,
      step_fn: Callable[..., Any],
      is_eval: bool,
      fn_in_partition_specs: NestedPartitionSpec,
      fn_out_partition_specs: NestedPartitionSpec,
      use_pspec_on_array_inputs: bool = False,
  ):
    logging.info('step_fn fn_in_partition_specs=%s', fn_in_partition_specs)
    logging.info('step_fn fn_out_partition_specs=%s', fn_out_partition_specs)

    extra_kwargs = dict(in_axis_resources=fn_in_partition_specs)
    if jax.config.jax_array and not use_pspec_on_array_inputs:
      extra_kwargs = {}
    pjitted_fn = pjit.pjit(
        step_fn,
        out_axis_resources=fn_out_partition_specs,
        # For training, TrainState is the first argument and return value. We
        # setup donation/alias to minimize device memory usage.
        donate_argnums=() if is_eval else (0,),
        **extra_kwargs,
    )
    return bind_mesh(pjitted_fn, self.global_mesh)

  def _get_state_unpadded_shapes(self, metadata: TrainStateMetadata):
    return jax.tree_map(lambda x: x.shape, metadata.unpadded_global_shapes)

  def _pad_states(
      self, metadata: TrainStateMetadata, unpadded_state: TrainState
  ):
    """Pad variables to avoid uneven sharding."""
    assert not self._enable_auto_sharding
    model_p = self._jax_task.hparams.model

    # Here the metadata is derived from input_spec which includes all possible
    # inputs. Thus metadata includes the full TrainState. The unpadded_state
    # here could be derived from a eval/decode dataset (e.g.,
    # get_spmd_model_step_fns_from_inputs) so it only includes a subset of
    # TrainState. Here we project the metadata according to the actual state.

    partition_specs = metadata.partition_specs.replace(
        mdl_vars=filter_nestedmap(
            metadata.partition_specs.mdl_vars, unpadded_state.mdl_vars
        )
    )
    state_unpadded_shapes = self._get_state_unpadded_shapes(metadata)
    state_unpadded_shapes = state_unpadded_shapes.replace(
        mdl_vars=filter_nestedmap(
            state_unpadded_shapes.mdl_vars, unpadded_state.mdl_vars
        )
    )

    return py_utils.maybe_pad_uneven_sharding(
        unpadded_state,
        partition_specs,
        state_unpadded_shapes,
        model_p.mesh_shape,
        model_p.mesh_axis_names,
    )

  def _unpad_states(
      self, metadata: TrainStateMetadata, padded_state: TrainState
  ):
    """Remove paddings from variables."""
    assert not self._enable_auto_sharding

    # Similar to _pad_states above we need to project the metadata to match the
    # actual padded_state

    partition_specs = metadata.partition_specs.replace(
        mdl_vars=filter_nestedmap(
            metadata.partition_specs.mdl_vars, padded_state.mdl_vars
        )
    )
    state_unpadded_shapes = self._get_state_unpadded_shapes(metadata)
    state_unpadded_shapes = state_unpadded_shapes.replace(
        mdl_vars=filter_nestedmap(
            state_unpadded_shapes.mdl_vars, padded_state.mdl_vars
        )
    )
    return py_utils.maybe_slice_uneven_sharding(
        padded_state,
        partition_specs,
        state_unpadded_shapes,
        is_leaf=py_utils.is_optax_masked_node,
    )

  def _unpad_inputs(
      self, padded_inputs: NestedJTensor, unpadded_global_batch_size: int
  ):
    """Remove paddings from inputs."""
    assert not self._enable_auto_sharding
    return _remove_input_padding(
        padded_inputs, unpadded_global_batch_size, self._mesh_names
    )

  def _partition_auto_shard(
      self,
      step_fn: Callable[..., Any],
      is_eval: bool,
      inputs_shape_dtype: NestedShapeDtypeLike,
      input_partition_spec: NestedPartitionSpec,
      metadata: TrainStateMetadata,
  ) -> Tuple[Callable[..., Any], NestedPartitionSpec, TrainState]:
    """Generates and returns the train state partition spec automatically."""
    # Workflow: create abstract train state and ahead of time compile the
    # `step_fn`. Then we can extract the input shardings returned by XLA's
    # auto spmd partitioner from the compiled object.
    assert self._enable_auto_sharding

    # We provide input_partition_spec because GDA creation is specialized to the
    # input partition specs created here. If we use partition specs returned by
    # XLA, it errors out.
    prng_key_partition_spec = PartitionSpec(None)
    fn_in_partition_specs = (pjit.AUTO, prng_key_partition_spec,
                             input_partition_spec)
    fn_out_partition_specs = (
        PartitionSpec()
        if self._auto_sharding_info.replicate_output
        else pjit.AUTO
    )

    partitioned_step_fn = self._pjit(
        step_fn,
        is_eval,
        fn_in_partition_specs,
        fn_out_partition_specs,
        use_pspec_on_array_inputs=True)

    # NOTE(pax-dev): The following is currently incompatible with variable
    # uneven-sharding padding.
    (auto_sharded_step_fn, input_shardings,) = compile_for_auto_sharding(
        partitioned_step_fn,
        metadata.unpadded_global_shapes,
        self._init_key,
        inputs_shape_dtype,
    )
    new_train_state_pspec = jax.tree_map(lambda x: x.spec, input_shardings[0])
    new_input_pspec = jax.tree_map(lambda x: x.spec, input_shardings[2])
    return auto_sharded_step_fn, new_input_pspec, new_train_state_pspec

  def _partition_manual_shard(
      self,
      step_fn: Callable[..., Any],
      is_eval: bool,
      inputs_shape_dtype: NestedShapeDtypeLike,
      input_partition_spec: NestedPartitionSpec,
      metadata: TrainStateMetadata,
  ):
    prng_key_partition_spec = PartitionSpec(None)
    fn_in_partition_specs = (
        metadata.partition_specs,
        prng_key_partition_spec,
        input_partition_spec,
    )

    def init_model_from_seed(init_key):
      states = initialize_model_state(
          self._jax_task,
          init_key,
          inputs_shape_dtype,
          discard_opt_states=is_eval,
          do_init_checkpoint_rules=False,
      )
      if not self._enable_auto_sharding:
        states = self._pad_states(metadata, states)
      return states

    var_shapes = jax.eval_shape(init_model_from_seed, self._init_key)
    out_shapes = jax.eval_shape(
        step_fn, var_shapes, self._init_key, inputs_shape_dtype
    )
    # Currently, all the outputs are fully replicated.
    # TODO(yonghui): Somehow fetch the output sharding spec from _eval_step fn.
    fn_out_partition_specs = jax.tree_util.tree_map(
        lambda _: PartitionSpec(), out_shapes
    )
    if not is_eval:
      fn_out_partition_specs = tuple(
          [metadata.partition_specs] + list(fn_out_partition_specs[1:])
      )

    asserts.assert_same_structure(fn_out_partition_specs, out_shapes)

    # LINT.IfChange(UsePspecOnArrayInputs)
    use_pspec_on_array_inputs = (jax.process_count() == 1)
    # LINT.ThenChange(trainer_lib.py:PspecSharding)
    partitioned_step_fn = self._pjit(
        step_fn,
        is_eval,
        fn_in_partition_specs,
        fn_out_partition_specs,
        use_pspec_on_array_inputs=use_pspec_on_array_inputs)
    return partitioned_step_fn


def get_step_fn(mode: RunningMode) -> Tuple[Callable[..., Any], bool]:
  """Returns the step function to partition.

  Args:
    mode: One of TRAIN, EVAL, and DECODE, that determines the step function to
      use.

  Returns:
    (step_fn, is_eval), where step_fn is the step function to partition, and
    is_eval indicates whether step_fn is used for evaluation.
  """
  assert int(mode.has_train) + int(mode.has_eval) + int(mode.has_decode) == 1
  if mode.has_train:
    is_eval = False
    step_fn = train_step_single_learner
  elif mode.has_eval:
    is_eval = True
    step_fn = eval_step_single_learner
  else:
    is_eval = True
    step_fn = _decode_step_for_partitioner

  return step_fn, is_eval


def create_partitioner(
    jax_task: tasks_lib.SingleTask,
    init_key: PRNGKey,
    train_inputs_shape_dtype: Optional[NestedShapeDtypeLike] = None,
    init_is_eval: bool = False,
    auto_sharding_mode: Optional[RunningMode] = None,
    job_log_dir: Optional[epath.Path] = None,
) -> Partitioner:
  """Return sharded train/eval/decode step function of the SPMD Model.

  Args:
    jax_task: The task which is an instance of tasks.SingleTask.
    init_key: PRNGKey for initializing the model variables.
    train_inputs_shape_dtype: Shape/dtype attributes of the inputs to
      model.init, for use in getting params of model variables.
    init_is_eval: Whether it should set is_eval=True when running
      abstract_init_with_metadata.
    auto_sharding_mode: One of TRAIN, EVAL, and DECODE, that determines the step
      function to use for auto-sharding (when pjit is used). If None, it means
      to disable auto-sharding.
    job_log_dir: Directory for the job logs.

  Returns:
    A Partitioner instance.
  """
  if jax_task.hparams.model.ici_mesh_shape is None:
    partitioner = _PmapPartitioner(
        jax_task,
        init_key,
        train_inputs_shape_dtype,
        init_is_eval,
        job_log_dir,
    )
  else:
    auto_sharding_info = None
    if auto_sharding_mode:
      step_fn, step_fn_is_eval = get_step_fn(auto_sharding_mode)
      replicate_output = auto_sharding_mode == RunningMode.DECODE
      auto_sharding_info = _PjitPartitioner.AutoShardingInfo(
          step_fn, step_fn_is_eval, replicate_output
      )
    partitioner = _PjitPartitioner(
        jax_task,
        init_key,
        train_inputs_shape_dtype,
        init_is_eval,
        auto_sharding_info,
        job_log_dir,
    )
  return partitioner


# TODO(laigd): get rid of this method, it's used only in unit tests/colab.
def get_partitioned_spmd_model_step_fn(
    jax_task: tasks_lib.SingleTask,
    mode: RunningMode,
    init_key: PRNGKey,
    inputs_shape_dtype: NestedShapeDtypeLike,
    train_inputs_shape_dtype: Optional[NestedShapeDtypeLike] = None,
    train_state_partition_spec: Optional[TrainState] = None,
    unpadded_global_batch_size: Optional[int] = None,
    enable_auto_sharding: bool = False) -> Tuple[Any, NestedPartitionSpec,
                                                 TrainState]:
  """Return sharded train/eval/decode step function of the SPMD Model.

  Args:
    jax_task: The task which is an instance of tasks.SingleTask.
    mode: One of TRAIN, EVAL, and DECODE, that determines the step function to
      use.
    init_key: PRNGKey for initializing the model variables.
    inputs_shape_dtype: Shape/dtype attributes of the inputs to the step
      function, for use in pjit sharding.
    train_inputs_shape_dtype: Shape/dtype attributes of the inputs to
      model.init, for use in getting params of model variables. If not provided,
      it assumes it's the same as inputs_shape_dtype.
    train_state_partition_spec: A TrainState contains PartitionSpecs for all the
      variables. Used when auto-sharding is disabled.
    unpadded_global_batch_size: If not None, this is the unpadded size of global
      batch, and the padding is on the right side of inputs_shape_dtype. Used
      when auto-sharding is disabled.
    enable_auto_sharding: Whether to enable xla auto-sharding.

  Returns:
    (partitioned_step_fn, input_partition_spec, train_state_partition_spec):
    The partitioned step function and the partition spec for the inputs and
    train state. If auto-sharding is enabled, train_state_partition_spec is
    automatically generated, otherwise it's the same as the provided one.
  """
  del train_state_partition_spec  # TODO(laigd): remove this arg.
  # TODO(laigd): check that train_inputs_shape_dtype is None if
  # always_use_train_for_model_init is not set.
  # Train/eval both use model.__call__ so train_inputs_shape_dtype and
  # inputs_shape_dtype are the same.
  if not train_inputs_shape_dtype and (mode.has_train or mode.has_eval):
    train_inputs_shape_dtype = inputs_shape_dtype

  step_fn, is_eval = get_step_fn(mode)
  partitioner = create_partitioner(
      jax_task,
      init_key,
      train_inputs_shape_dtype,
      init_is_eval=is_eval,
      auto_sharding_mode=mode if enable_auto_sharding else None,
  )
  partitioned_step_fn, input_partition_spec = partitioner.partition(
      step_fn, inputs_shape_dtype, is_eval, unpadded_global_batch_size
  )
  metadata = partitioner.get_train_state_metadata(discard_opt_states=is_eval)
  return partitioned_step_fn, input_partition_spec, metadata.partition_specs


def get_spmd_model_step_fns_from_inputs(
    input_ps: Sequence[base_input.BaseInput.HParams],
    unpadded_input_ps: Sequence[base_input.BaseInput.HParams],
    partitioner: Partitioner,
    mode: RunningMode,
) -> Tuple[Sequence[Callable[..., Any]], Sequence[NestedPartitionSpec]]:
  """Helper for calling `get_partitioned_spmd_model_step_fn` with input_ps.

  Note: This method instantiates all the input pipelines passed in `input_ps` to
    get a sample input.

  Args:
    input_ps: inputs hparams list. May be padded unliked `unpadded_input_ps`.
    unpadded_input_ps: inputs hparams list. Importantly these are inputs
      *before* being called by `adjust_input_params_for_small_batch`. Thus,
      `py_utils.get_global_batch_size(p)` returns the unpadded batch size.
    mode: One of TRAIN, EVAL, and DECODE, that determines the step function to
      use.

  Returns:
    (partitioned_step_fns, input_partition_specs):
    The partitioned step functions, partition specs for the inputs,
    and shape/dtype information for the inputs.
  """
  if len(input_ps) != len(unpadded_input_ps):
    raise ValueError(
        'Length of padded and unpadded inputs must match and be 1-to-1: '
        f'{len(input_ps)} != {len(unpadded_input_ps)}.')

  step_fn, is_eval = get_step_fn(mode)
  partitioned_step_fns = []
  input_partition_specs = []
  for input_p, unpadded_input_p in zip(input_ps, unpadded_input_ps):
    # TODO(pax-dev): Investigate if we can use model input specs
    # instead of instantiating this input pipeline.
    _, inputs_shape_dtype = get_inputs_shape_dtype(input_p)
    partitioned_step_fn, inputs_partition_spec = partitioner.partition(
        step_fn,
        inputs_shape_dtype,
        is_eval,
        unpadded_global_batch_size=(
            unpadded_input_p.cls.get_global_batch_size(unpadded_input_p)
        ),
    )

    partitioned_step_fns.append(partitioned_step_fn)
    input_partition_specs.append(inputs_partition_spec)

  return partitioned_step_fns, input_partition_specs


def check_unique_names(inputs: Sequence[base_input.BaseInput]) -> None:
  names = set()
  for inp in inputs:
    name = inp.hparams.name
    if name in names:
      raise ValueError(f'Duplicate param name found in list: "{name}"')
    names.add(name)


def bind_mesh(pjitted_fn, global_mesh: jax.sharding.Mesh):
  """Wraps a pjitted_fn with a mesh context."""

  def call(*args):
    with global_mesh:
      return pjitted_fn(*args)

  def lower(*args, **kwargs):
    with global_mesh:
      return pjitted_fn.lower(*args, **kwargs)

  call.lower = lower
  return call


class SingleTaskPjitTrainer:
  """Trainer that assumes a single taks on a single dataset."""

  def __init__(self,
               task: tasks_lib.SingleTask,
               train_input: base_input.BaseInput,
               mesh: jax.sharding.Mesh,
               enable_auto_sharding: bool = False):
    self._task = task
    self._train_input = train_input
    self._mesh = mesh
    self._enable_auto_sharding = enable_auto_sharding
    self._train_unpadded_global_batch_size = (
        train_input.HParams.cls.get_global_batch_size(train_input.hparams)
    )
    # Has to initialize before train_state_metadata as the inputs_shape_dtype
    # requires self._sample_inputs.
    self._sample_inputs = train_input.get_next_padded()
    self._train_state_metadata = create_train_state_metadata(
        self._task, self.inputs_shape_dtype)
    # Lazily initialized values after calling `initialize`.
    self._step_fn = None
    self._train_input_pipeline = None
    self._input_pspecs = None
    self._train_state_partition_spec = None

  def compile_step(
      self, prng_key: jax.random.KeyArray
  ) -> Tuple[Any, NestedPartitionSpec, TrainState]:
    """Compiles a partitioned step_fn under pjit.

    Args:
      prng_key: The jax random key to use for initializing model variables.

    Returns:
      A tuple of (partitioned step function, partitioned input spec,
        partitioned train state specs).
    """
    if self._step_fn is None:
      self._step_fn, self._input_pspecs, self._train_state_partition_spec = (
          get_partitioned_spmd_model_step_fn(
              self._task,
              RunningMode.TRAIN,
              prng_key,
              inputs_shape_dtype=self.inputs_shape_dtype,
              train_state_partition_spec=(
                  self._train_state_metadata.partition_specs
              ),
              unpadded_global_batch_size=self._train_unpadded_global_batch_size,
              enable_auto_sharding=self._enable_auto_sharding,
          )
      )
    return self._step_fn, self._input_pspecs, self._train_state_partition_spec

  def train_step(self, state: TrainState, prng_key: jax.random.KeyArray,
                 inputs: Any):
    if self._step_fn is None:
      # Compile and get the step_fn.
      # NOTE: The compilation turns of auto sharding so the partition_spec is
      # not modified upon returning.
      self.compile_step(prng_key)
    return self._step_fn(state, prng_key, inputs)

  @property
  def train_state_metadata(self) -> TrainStateMetadata:
    return self._train_state_metadata

  @property
  def train_unpadded_global_batch_size(self) -> int:
    return self._train_unpadded_global_batch_size

  @property
  def task(self) -> tasks_lib.SingleTask:
    if self._task is None:
      raise ValueError(
          'initialize() method must be called before using Trainer.')
    return self._task

  @property
  def inputs_shape_dtype(self) -> NestedShapeDtypeLike:
    return jax.tree_map(py_utils.get_global_input_shape_dtype,
                        self._sample_inputs)

  @property
  def perhost_inputs_shape_dtype(self) -> NestedShapeDtypeLike:
    return jax.tree_map(
        lambda x: jax.ShapeDtypeStruct(shape=x.shape, dtype=x.dtype),
        self._sample_inputs)
