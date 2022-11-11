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

import dataclasses
import enum
import functools
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

from absl import logging
from etils import epath
import jax
from jax import numpy as jnp
from jax.experimental import maps
from jax.experimental import pjit
from jax.interpreters import pxla
from paxml import base_metrics
from paxml import summary_utils
from paxml import tasks_lib
from praxis import asserts
from praxis import base_hyperparams
from praxis import base_input
from praxis import base_layer
from praxis import base_model
from praxis import py_utils
from praxis import pytypes
from praxis import sgf
from praxis import train_states

from paxml import checkpoints  # mapped to internal

PartitionSpec = pjit.PartitionSpec

JTensor = pytypes.JTensor
NestedJTensor = pytypes.NestedJTensor
NestedMap = py_utils.NestedMap
NestedNpTensor = pytypes.NestedNpTensor
NestedShape = NestedMap
PRNGKey = pytypes.PRNGKey
ParamsT = pytypes.HParamsT
PyTreeDef = pytypes.PyTreeDef
NestedShapeDtypeStruct = pytypes.NestedShapeDtypeStruct
NestedShapeDtypeLike = pytypes.NestedShapeDtypeLike
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


@dataclasses.dataclass
class TrainStateMetadata:
  """Metadata around the TrainState.

  Specifically, this encapsulates information relevant for model initialization
  as well as train/eval/decode step creation.
  """
  input_shape_dtype: NestedShapeDtypeLike
  var_weight_hparams: NestedWeightHParams
  padded_global_shapes: Optional[TrainState] = None
  unpadded_global_shapes: Optional[TrainState] = None
  partitioned_specs: Optional[TrainState] = None


def create_train_state_metadata(
    jax_task: tasks_lib.SingleTask,
    train_shape_dtype: NestedShapeDtypeLike,
    discard_opt_states: bool = False) -> TrainStateMetadata:
  """Creates a TrainStateMetadata instance.

  Args:
    jax_task: The SingleTask instance.
    train_shape_dtype: Training input shape dtype to be used by
      model.abstract_init_with_metadata(). It should have per-core shapes
      for pmap models and global shapes for pjit ones.
    discard_opt_states: Whether to discard the part corresponding to the
      optimizer states or not.

  Returns:
    A TrainStateMetadata instance.
  """
  var_weight_hparams = jax_task.model.abstract_init_with_metadata(
      train_shape_dtype)
  padded_global_shapes = jax_task.create_train_state_padded_shapes(
      var_weight_hparams, discard_opt_states=discard_opt_states)
  unpadded_global_shapes = jax_task.create_train_state_unpadded_shapes(
      var_weight_hparams, discard_opt_states=discard_opt_states)
  if jax_task.model.hparams.mesh_shape is not None:
    partitioned_specs = jax_task.create_train_state_partition_specs(
        var_weight_hparams, discard_opt_states=discard_opt_states)
  else:
    partitioned_specs = None
  return TrainStateMetadata(
      input_shape_dtype=train_shape_dtype,
      var_weight_hparams=var_weight_hparams,
      padded_global_shapes=padded_global_shapes,
      unpadded_global_shapes=unpadded_global_shapes,
      partitioned_specs=partitioned_specs)


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
      train_state, step_key, inputs_shape_dtype, _global_avals=True).compile()
  return compiled, compiled.input_shardings[0]


EarlyStoppingFn = Callable[[Dict[str, float], RunningMode, int, bool], bool]


def write_post_init_model_hparams_file(model: base_model.BaseModel,
                                       var_weight_hparams: NestedWeightHParams,
                                       job_log_dir: epath.Path) -> None:
  """Writes a post-init params file into the root `job_log_dir`.

  This file is the source of truth of how model is constructed. It contains two
  parts:
  1) how each layer is configured during layer construction time.
  2) variable WeightHParams for each of the model weight.

  Args:
    model: A BaseModel
    var_weight_hparams: A pytree of WeightHParams
    job_log_dir: The root dir for the training job.
  """
  if jax.process_index() == 0:
    params_fpath = job_log_dir / 'post_init_model_params.txt'
    logging.info('post_init_model_params: %s', params_fpath)
    job_log_dir.mkdir(parents=True, exist_ok=True)
    with params_fpath.open('w') as params_file:
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

      if var_weight_hparams:
        params_inits_text = base_hyperparams.nested_struct_to_text(
            var_weight_hparams)
        params_file.write(params_inits_text)


def adjust_input_params_for_small_batch(
    inp_p: base_input.BaseInput.HParams,
    global_mesh: maps.Mesh) -> base_input.BaseInput.HParams:
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
  if jax.process_count() == 1:
    # If there is only one host, valid examples are already contiguous so we can
    # use default GDA creation.
    return copy
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


def initialize_model_state(jax_task: tasks_lib.SingleTask,
                           prng_key: PRNGKey,
                           inputs_shape_dtype: NestedShapeDtypeLike,
                           discard_opt_states: bool = False,
                           do_init_checkpoint_rules: bool = True,
                           is_eval: bool = False) -> TrainState:
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

  Returns:
    TrainStates - training states.
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
      # Zerso dtype should be int32 same as the original input that produced
      # float0.
      return jnp.zeros((), dtype=jnp.int32)
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
    fprop_dtype: jnp.dtype = jnp.float32
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

  with base_layer.JaxContext.new_context(hparams=context_p):
    var_weight_hparams = model.abstract_init_with_metadata(inputs)

  updated_mdl_vars = jax_task.maybe_adjust_train_state(states.step,
                                                       states.mdl_vars,
                                                       var_weight_hparams,
                                                       prng_key)

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
          mutable=[AUX_LOSS, SUMMARIES, NON_TRAINABLE] + NON_PAX_VAR_COLLECTION,
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
    fprop_dtype: jnp.dtype = jnp.float32) -> Tuple[Any, Any, Any, SummaryDict]:
  """Evaluates a model for a single step.

  This utility is specialized for the single learner case.

  Args:
    jax_task: An instance of tasks.SingleTask.
    states: An instance of model.TrainState.
    prng_key: A prng seed, of shape [2], of type np.uint32.
    inputs: Inputs to the model() function.
    fprop_dtype: fprop datatype, can be either jnp.float32 or jnp.bfloat16.

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

  if jax_task.hparams.train.always_use_train_for_model_init:
    do_eval_for_init = False
  else:
    do_eval_for_init = True
  var_weight_hparams = model.abstract_init_with_metadata(
      inputs, do_eval=do_eval_for_init)

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
        mutable=[AUX_LOSS, SUMMARIES, NON_TRAINABLE],
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
    fprop_dtype: jnp.dtype = jnp.float32
) -> Tuple[Tuple[Any, Any, Any], NestedMap]:
  """Decodes a model for a single step.

  Args:
    model: An instance of models.BaseModel.
    states: An instance of TrainState..
    prng_key: A prng seed, of shape [2], of type np.uint32.
    var_weight_hparams: A pytree of WeightHParams for the model variables.
    inputs: A batch of inputs to model.decode().
    fprop_dtype: fprop datatype, can be either jnp.float32 or jnp.bfloat16.

  Returns:
    A tuple of (weighted_scalars, results, eval_metrics) as computed
      by model.decode() and the updated weights.
  """
  context_p = base_layer.JaxContext.HParams(do_eval=True, summary_verbosity=2)
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


def initialize_partitioned_model_states(
    jax_task: tasks_lib.SingleTask,
    prng_key: PRNGKey,
    global_input_shapes: NestedShapeDtypeLike,
    discard_opt_states: bool = False,
    global_mesh: Optional[maps.Mesh] = None,
    checkpoint_type: CheckpointType = CheckpointType.CHECKPOINT_FLAX,
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


class _SpmdModelPartitioner:
  """Used for partitioning a step function of a SPMD model."""

  @dataclasses.dataclass
  class ShardingInfo:
    """Relevant information needed by manual (i.e. non-auto) sharding."""
    # A TrainState contains PartitionSpecs for all the variables.
    train_state_partition_spec: TrainState
    # If not None, this is the unpadded size of global batch, and the padding is
    # on the right side of each input.
    unpadded_global_batch_size: Optional[int]

  def __init__(self,
               jax_task: tasks_lib.SingleTask,
               global_mesh: maps.Mesh,
               init_key: PRNGKey,
               inputs_shape_dtype: NestedShapeDtypeLike,
               train_inputs_shape_dtype: Optional[NestedShapeDtypeLike] = None,
               sharding_info: Optional[ShardingInfo] = None,
               auto_sharding_replicate_output: Optional[bool] = False):
    """Constructor.

    Args:
      jax_task: The task which is an instance of tasks.SingleTask.
      global_mesh: The global mesh.
      init_key: PRNGKey for initializing the model variables.
      inputs_shape_dtype: Shape/dtype attributes of the step function input for
        use in pjit sharding.
      train_inputs_shape_dtype: Shape/dtype attributes of the inputs to
        model.init, for use in getting params of model variables. If not
        provided, it assumes it's the same as inputs_shape_dtype.
      sharding_info: Information used for manual sharding. If None, it'll use
        xla auto-sharding instead.
      auto_sharding_replicate_output: Whether to replicate the output when auto
        sharding is enabled. TODO(pax-dev): support custom output partitionspec.
    """
    self._jax_task = jax_task
    self._global_mesh = global_mesh
    self._init_key = init_key
    self._inputs_shape_dtype = inputs_shape_dtype
    self._sharding_info = sharding_info
    self._enable_auto_sharding = (sharding_info is None)
    self._auto_sharding_replicate_output = auto_sharding_replicate_output
    self._mesh_names = self._jax_task.hparams.model.mesh_axis_names

    if train_inputs_shape_dtype is None:
      self._init_shape_dtype = inputs_shape_dtype  # Specs used for vars init.
      self._abstract_init_with_train = False
    else:
      self._init_shape_dtype = train_inputs_shape_dtype
      self._abstract_init_with_train = True

  def partition(self, step_fn, is_eval):
    """Gets a sharded (pjit-ed) step function of the SPMD Model.

    Args:
      step_fn: Training or eval step function.
      is_eval: bool, indicating if it's a eval/decode task or not.

    Returns:
      (partitioned_step_fn, input_partition_spec, train_state_partition_spec):

      - partitioned_step_fn: The partitioned step function.
      - input_partition_spec: The partition spec for the inputs of the step
        function.
      - train_state_partition_spec: The partition spec for the train state. If
        auto-sharding is enabled, this is automatically generated. Otherwise the
        spec provided by ShardingInfo will be returned.
    """
    state_unpadded_shapes = (None if self._enable_auto_sharding else
                             self._get_state_unpadded_shapes(is_eval))
    input_partition_spec = get_input_partition_specs(self._mesh_names,
                                                     self._inputs_shape_dtype)
    # Step function to be pjit-ed.
    wrapped_step_fn = self._get_step_fn(step_fn, is_eval, state_unpadded_shapes,
                                        input_partition_spec)
    with self._global_mesh:
      if self._enable_auto_sharding:
        return self._partition_auto_shard(wrapped_step_fn, is_eval,
                                          input_partition_spec)
      else:
        return self._partition_manual_shard(wrapped_step_fn, is_eval,
                                            input_partition_spec,
                                            state_unpadded_shapes)

  def _get_step_fn(self, step_fn, is_eval, state_unpadded_shapes,
                   input_partition_spec):
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
        state, inputs = self._unpad(state, state_unpadded_shapes, inputs)

      # Reshard inputs.
      # TODO(pax): when xla auto-sharding is enabled, it'll automatically figure
      # out the proper sharding of intermediate nodes, we can get rid of this
      # manual sharding now?
      inputs = jax.tree_map(reshard_inputs_fn, inputs)

      # When auto sharding is enabled, uneven sharding is not supported. This is
      # because the way padding is added is dependent on the
      # `train_state_partition_spec`. This is not available during auto
      # sharding until after the compilation is done.
      fn_out = step_fn(
          self._jax_task,
          state,
          prng_key,
          inputs,
          fprop_dtype=model_p.fprop_dtype)

      if is_eval:
        return fn_out
      assert len(fn_out) > 1

      # Pad the model states again for training step functions.
      if not self._enable_auto_sharding:
        padded_states = self._pad(fn_out[0], state_unpadded_shapes)
        fn_out = (padded_states,) + fn_out[1:]
      return fn_out

    return _wrapped_step_fn

  def _pjit(self,
            step_fn,
            is_eval,
            fn_in_partition_specs,
            fn_out_partition_specs,
            use_pspec_on_array_inputs=False):
    logging.info('step_fn fn_in_partition_specs=%s', fn_in_partition_specs)
    logging.info('step_fn fn_out_partition_specs=%s', fn_out_partition_specs)
    if jax.config.jax_array and not use_pspec_on_array_inputs:
      pjitted_fn = pjit.pjit(
          step_fn,
          out_axis_resources=fn_out_partition_specs,
          # For training we don't need the original TrainState after running the
          # train step.
          donate_argnums=() if is_eval else (0,))
    else:
      pjitted_fn = pjit.pjit(
          step_fn,
          in_axis_resources=fn_in_partition_specs,
          out_axis_resources=fn_out_partition_specs,
          donate_argnums=() if is_eval else (0,))
    return bind_mesh(pjitted_fn, self._global_mesh)

  def _get_var_weight_hparams(self, is_eval):
    return self._jax_task.model.abstract_init_with_metadata(
        self._init_shape_dtype,
        do_eval=False if self._abstract_init_with_train else is_eval)

  def _get_state_unpadded_shapes(self, is_eval):
    return jax.tree_map(
        lambda x: x.shape,
        self._jax_task.create_train_state_unpadded_shapes(
            self._get_var_weight_hparams(is_eval), discard_opt_states=is_eval))

  def _pad(self, unpadded_state, state_unpadded_shapes):
    """Pad variables to avoid uneven sharding."""
    assert not self._enable_auto_sharding
    model_p = self._jax_task.hparams.model
    return py_utils.maybe_pad_uneven_sharding(
        unpadded_state, self._sharding_info.train_state_partition_spec,
        state_unpadded_shapes, model_p.mesh_shape, model_p.mesh_axis_names)

  def _unpad(self, padded_state, state_unpadded_shapes, padded_inputs):
    """Remove paddings from variables/inputs."""
    assert not self._enable_auto_sharding
    state = jax.tree_map(
        py_utils.maybe_slice_uneven_sharding,
        padded_state,
        self._sharding_info.train_state_partition_spec,
        state_unpadded_shapes,
        is_leaf=py_utils.is_optax_masked_node)
    inputs = _remove_input_padding(
        padded_inputs, self._sharding_info.unpadded_global_batch_size,
        self._mesh_names)
    return state, inputs

  def _partition_auto_shard(self, step_fn, is_eval, input_partition_spec):
    # Workflow: create abstract train state and ahead of time compile the
    # `train_step`. Then we can extract the input shardings returned by XLA's
    # auto spmd partitioner from the compiled object.

    # We provide input_partition_spec because GDA creation is specialized to the
    # input partition specs created here. If we use partition specs returned by
    # XLA, it errors out.
    prng_key_partition_spec = PartitionSpec(None)
    fn_in_partition_specs = (pjit.AUTO, prng_key_partition_spec,
                             input_partition_spec)
    fn_out_partition_specs = (
        PartitionSpec() if self._auto_sharding_replicate_output else pjit.AUTO)

    partitioned_step_fn = self._pjit(
        step_fn,
        is_eval,
        fn_in_partition_specs,
        fn_out_partition_specs,
        use_pspec_on_array_inputs=True)

    # NOTE(pax-dev): The following is currently incompatible with variable
    # uneven-sharding padding. When enable_auto_sharding is False,
    # train_state_pspecs correspond to padded train_states.
    abstract_train_state = self._jax_task.create_train_state_unpadded_shapes(
        self._get_var_weight_hparams(is_eval), discard_opt_states=is_eval)
    auto_sharded_partitioned_step_fn, input_shardings = (
        compile_for_auto_sharding(partitioned_step_fn, abstract_train_state,
                                  self._init_key, self._inputs_shape_dtype))
    train_state_partition_spec = jax.tree_map(lambda x: x.spec,
                                              input_shardings[0])
    updated_input_partition_spec = jax.tree_map(lambda x: x.spec,
                                                input_shardings[2])
    logging.info('step_fn inputs_partition_spec=%s',
                 updated_input_partition_spec)
    return (auto_sharded_partitioned_step_fn, updated_input_partition_spec,
            train_state_partition_spec)

  def _partition_manual_shard(self, step_fn, is_eval, input_partition_spec,
                              state_unpadded_shapes):
    prng_key_partition_spec = PartitionSpec(None)
    train_state_partition_spec = (
        self._sharding_info.train_state_partition_spec)

    def init_model_from_seed(init_key):
      outs = initialize_model_state(
          self._jax_task,
          init_key,
          self._inputs_shape_dtype,
          discard_opt_states=is_eval,
          do_init_checkpoint_rules=False)
      return self._pad(outs, state_unpadded_shapes)

    var_padded_shapes = jax.eval_shape(init_model_from_seed, self._init_key)
    out_padded_shapes = jax.eval_shape(step_fn, var_padded_shapes,
                                       self._init_key, self._inputs_shape_dtype)
    fn_in_partition_specs = (train_state_partition_spec,
                             prng_key_partition_spec, input_partition_spec)
    # Currently, all the outputs are fully replicated.
    # TODO(yonghui): Somehow fetch the output sharding spec from _eval_step fn.
    fn_out_partition_specs = jax.tree_util.tree_map(lambda _: PartitionSpec(),
                                                    out_padded_shapes)
    if not is_eval:
      fn_out_partition_specs = tuple([train_state_partition_spec] +
                                     list(fn_out_partition_specs[1:]))

    asserts.assert_same_structure(fn_out_partition_specs, out_padded_shapes)
    partitioned_step_fn = self._pjit(step_fn, is_eval, fn_in_partition_specs,
                                     fn_out_partition_specs)
    logging.info('step_fn inputs_partition_spec=%s', input_partition_spec)
    return (partitioned_step_fn, input_partition_spec,
            train_state_partition_spec)


def get_partitioned_spmd_model_step_fn(
    jax_task: tasks_lib.SingleTask,
    mode: RunningMode,
    global_mesh: maps.Mesh,
    init_key: PRNGKey,
    inputs_shape_dtype: NestedShapeDtypeLike,
    train_inputs_shape_dtype: Optional[NestedShapeDtypeLike] = None,
    train_state_partition_spec: Optional[TrainState] = None,
    unpadded_global_batch_size: Optional[int] = None,
    enable_auto_sharding: bool = False):
  """Return sharded train/eval/decode step function of the SPMD Model.

  Args:
    jax_task: The task which is an instance of tasks.SingleTask.
    mode: One of TRAIN, EVAL, and DECODE, that determines the step function to
      use.
    global_mesh: The global mesh.
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
  assert int(mode.has_train) + int(mode.has_eval) + int(mode.has_decode) == 1
  # TODO(laigd): check that train_inputs_shape_dtype is None if
  # always_use_train_for_model_init is not set.
  if mode.has_train:
    train_inputs_shape_dtype = None
    is_eval = False
    step_fn = train_step_single_learner
    auto_sharding_replicate_output = False
  elif mode.has_eval:
    train_inputs_shape_dtype = None
    is_eval = True
    step_fn = eval_step_single_learner
    auto_sharding_replicate_output = False
  else:
    is_eval = True
    auto_sharding_replicate_output = True

    if jax_task.hparams.train.always_use_train_for_model_init:
      if train_inputs_shape_dtype is None:
        raise ValueError(
            'No training input specs available, while enabling '
            '`task_p.train.always_use_train_for_model_init` requires it.')
      var_weight_hparams = jax_task.model.abstract_init_with_metadata(
          train_inputs_shape_dtype)
    else:
      train_inputs_shape_dtype = None
      var_weight_hparams = jax_task.model.abstract_init_with_metadata(
          inputs_shape_dtype, do_eval=True)

    # TODO(laigd): train/eval/decode step functions should have the same
    # signature.
    def step_fn(task, states, prng_key, inputs, fprop_dtype):
      return decode_step(task.model, states, prng_key, var_weight_hparams,
                         inputs, fprop_dtype)

  if enable_auto_sharding:
    sharding_info = None
  else:
    sharding_info = _SpmdModelPartitioner.ShardingInfo(
        train_state_partition_spec, unpadded_global_batch_size)

  partitioner = _SpmdModelPartitioner(jax_task, global_mesh, init_key,
                                      inputs_shape_dtype,
                                      train_inputs_shape_dtype, sharding_info,
                                      auto_sharding_replicate_output)
  return partitioner.partition(step_fn, is_eval)


def check_unique_names(inputs: Sequence[base_input.BaseInput]) -> None:
  names = set()
  for inp in inputs:
    name = inp.hparams.name
    if name in names:
      raise ValueError(f'Duplicate param name found in list: "{name}"')
    names.add(name)


def bind_mesh(pjitted_fn, global_mesh: maps.Mesh):
  """Wraps a pjitted_fn with a mesh context."""

  def call(*args):
    with global_mesh:
      return pjitted_fn(*args)

  def lower(*args, **kwargs):
    with global_mesh:
      return pjitted_fn.lower(*args, **kwargs)

  call.lower = lower
  return call
