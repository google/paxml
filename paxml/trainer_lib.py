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

"""Shared trainer lib utilities."""

import dataclasses
import enum
import functools
import pprint
from typing import Any, Dict, Optional, Protocol, Sequence, Tuple, Union

from absl import logging
from etils import epath
import fiddle as fdl
from flax import struct as flax_struct
import jax
from jax import numpy as jnp
from jax.experimental import pjit
from paxml import base_metrics
from paxml import checkpoint_types
from paxml import learners as learners_lib
from paxml import sgf
from paxml import tasks_lib
from paxml import train_states
from praxis import asserts
from praxis import base_hyperparams
from praxis import base_input
from praxis import base_layer
from praxis import base_model
from praxis import lazy_loader
from praxis import pax_fiddle
from praxis import py_utils
from praxis import pytypes

# summary_utils is slow to import, so we do it lazily.
summary_utils = lazy_loader.LazyLoader(
    'summary_utils', globals(), 'paxml.summary_utils'
)

PartitionSpec = jax.sharding.PartitionSpec

JTensor = pytypes.JTensor
NestedJTensor = pytypes.NestedJTensor
NestedMap = py_utils.NestedMap
NestedNpTensor = pytypes.NestedNpTensor
NestedShape = NestedMap
PRNGKey = pytypes.PRNGKey
ParamsT = pytypes.HParamsT
Nested = pytypes.Nested
NestedPartitionSpec = pytypes.NestedPartitionSpec
NestedShapeDtypeLike = pytypes.NestedShapeDtypeLike
NestedShapeDtypeStruct = pytypes.NestedShapeDtypeStruct
NestedWeightHParams = base_layer.NestedWeightHParams
TrainState = train_states.TrainState
TensorProvenance = train_states.TensorProvenance
TrainStateProvenance = train_states.TrainStateProvenance
SummaryDict = pytypes.SummaryDict
WeightedScalars = pytypes.WeightedScalars
WeightedScalarsList = pytypes.WeightedScalarsList

CheckpointType = checkpoint_types.CheckpointType

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
  def detect(
      cls,
      has_train_metrics: bool,
      has_eval_metrics: bool,
      has_decode_metrics: bool,
  ) -> 'RunningMode':
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


@dataclasses.dataclass(frozen=True, slots=True)
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


def create_train_state_metadata(
    jax_task: tasks_lib.SingleTask,
    train_shape_dtype: NestedShapeDtypeLike,
    discard_opt_states: bool = False,
    do_eval: bool = False,
) -> TrainStateMetadata:
  """Creates a TrainStateMetadata instance.

  Args:
    jax_task: The SingleTask instance.
    train_shape_dtype: Training input shape dtype to be used by
      model.abstract_init_with_metadata(). It should have per-core shapes for
      pmap models and global shapes for pjit ones.
    discard_opt_states: Whether to discard the part corresponding to the
      optimizer states or not.
    do_eval: Whether this metadata is used for evaluation.

  Returns:
    A TrainStateMetadata instance.
  """
  var_weight_hparams = jax_task.model.abstract_init_with_metadata(
      train_shape_dtype, do_eval=do_eval
  )
  padded_global_shapes = jax_task.create_train_state_padded_shapes(
      var_weight_hparams, discard_opt_states=discard_opt_states
  )
  unpadded_global_shapes = jax_task.create_train_state_unpadded_shapes(
      var_weight_hparams, discard_opt_states=discard_opt_states
  )
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


EarlyStoppingFn = tasks_lib.EarlyStoppingFn


def write_post_init_model_hparams_file(
    model: base_model.BaseModel,
    train_state_metadata: TrainStateMetadata,
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
    train_state_metadata: An instance of TrainStateMetadata.
    job_log_dir: The root dir for the training job.
    do_eval: whether this is in eval mode.
  """
  if jax.process_index() == 0:
    params_fpath = job_log_dir / 'post_init_model_params.txt'
    logging.info('post_init_model_params: %s', params_fpath)
    job_log_dir.mkdir(parents=True, exist_ok=True)
    hyper_params = model.abstract_init_with_mdl_config(
        train_state_metadata.input_shape_dtype, do_eval=do_eval
    )
    with params_fpath.open('w') as params_file:
      hyper_params_dump = base_hyperparams.nested_struct_to_text(hyper_params)
      params_file.write(hyper_params_dump)
      params_file.write('\n\n')
      params_inits_text = base_hyperparams.nested_struct_to_text(
          train_state_metadata.var_weight_hparams
      )
      params_file.write(params_inits_text)


def write_train_provenance_file(
    train_state_provenance: train_states.TrainStateProvenance,
    job_log_dir: epath.Path,
) -> None:
  """Writes a file with train state provenance into the root `job_log_dir`."""
  if jax.process_index() == 0:
    filename = job_log_dir / 'train_state_provenance.txt'
    if filename.exists():
      return
    job_log_dir.mkdir(parents=True, exist_ok=True)
    with filename.open('w') as provenance_file:
      output = ''
      for attr, val in vars(train_state_provenance).items():
        attr_formatted = attr.replace('_', ' ').capitalize() + '\n'
        provenance_out = summary_utils.pretty_repr_provenance(val) + '\n\n'
        output += attr_formatted + provenance_out
      provenance_file.write(output)


def adjust_input_params_for_small_batch(
    input_p: pax_fiddle.Config[base_input.BaseInput],
    global_mesh: jax.sharding.Mesh,
) -> pax_fiddle.Config[base_input.BaseInput]:
  """Creates a copy of input_p adjusted when the per-device batch < 1.

  When users specify fractional batch sizes for very large models, it is
  necessary to pad the input batch. This is because JAX requires equally-shaped
  tensors at every program's entrypoint and exitpoint.

  Args:
    input_p: Fiddle config of a BaseInput which could be modified.
    global_mesh: a Mesh needed as reference to construct a device sharding
      order.

  Returns:
    The possibly-modified BaseInput Fiddle config.
  """
  # Remote input adjusts the params for small batch itself.
  if input_p.experimental_remote_input:
    return input_p

  local_device_count = jax.local_device_count()
  batch_size = fdl.get_callable(input_p).get_batch_size(input_p)  # pytype: disable=attribute-error

  if (
      batch_size % local_device_count == 0
      and input_p.num_infeed_hosts == jax.process_count()
  ):
    return input_p

  # Determine correct padding amount.
  copy = input_p.clone()
  if batch_size <= local_device_count:
    copy.batch_padding_size = local_device_count - batch_size
  else:
    if batch_size % local_device_count != 0:
      if jax.process_count() > 1:
        # The custom device order resharding currently works only when
        # batch_size < local_device_count.
        raise NotImplementedError(
            'Per-host batch size must be a multiple of '
            'per-host device count, or smaller than it.'
        )
      else:
        # Single-host input doesn't need to do custom order resharding.
        copy.batch_padding_size = (
            (batch_size + local_device_count)
            // local_device_count
            * local_device_count
        ) - batch_size

  assert input_p.num_infeed_hosts <= jax.process_count()
  if jax.process_count() == 1:
    # If there is only one host, valid examples are already contiguous so we can
    # use default Jax array creation.
    # Inputs use pspec sharding (see praxis.BaseInput.reshard_for_spmd).
    return copy
  # Some hosts may produce duplicate data, but they will be discarded.
  copy.infeed_host_index = jax.process_index() % input_p.num_infeed_hosts
  if copy.infeed_host_index >= input_p.num_infeed_hosts:
    logging.info(
        'Process %s: infeed data will be dropped.', jax.process_index()
    )

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
    if (
        process_idx >= input_p.num_infeed_hosts
        or core_offset_in_host >= batch_size
    ):
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
    checkpoint_type: CheckpointType = CheckpointType.UNSPECIFIED,
    var_weight_hparams: Optional[NestedWeightHParams] = None,
) -> Tuple[TrainState, TrainStateProvenance]:
  """Initializes the model states.

  Weights are random initialized first.
  Then we restore weights based on the init_checkpoint_rules.

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
    var_weight_hparams: A pytree of WeightHParams for the model variables.

  Returns:
    Training state and train state provenance.
  """
  model = jax_task.model
  prng_key, k1, k2, k3 = jax.random.split(prng_key, 4)
  init_key = {PARAMS: k1, RANDOM: k2, NON_PAX_RNG_KEY: k3}
  if jax_task.hparams.train.always_use_train_for_model_init:
    is_eval_for_init = False
  else:
    is_eval_for_init = is_eval
  if not var_weight_hparams:
    var_weight_hparams = model.abstract_init_with_metadata(
        inputs_shape_dtype, do_eval=is_eval_for_init
    )
  logging.info('init_var prng_seed: %s', init_key)
  logging.info('var_weight_hparams: %s', var_weight_hparams)

  # Use jax.jit to reduce model.init memory usage. Required by a few tests after
  # migrating to shape inference.
  @jax.jit
  def init_fn(init_key):
    context_p = base_layer.JaxContext.HParams(
        do_eval=is_eval_for_init,
        summary_verbosity=jax_task.summary_verbosity,
    )
    with base_layer.JaxContext.new_context(hparams=context_p):
      inputs = jax.tree_map(
          lambda x: jnp.zeros(x.shape, x.dtype), inputs_shape_dtype
      )
      if model.hparams.fprop_dtype == jnp.bfloat16:
        inputs = jax.tree_map(_maybe_to_bfloat16, inputs)
      return model.init(init_key, inputs)

  initial_vars = init_fn(init_key)
  logging.info(
      'initial_vars: %s', jax.tree_map(lambda x: x.shape, initial_vars)
  )

  # In case jax_task.model wraps a t5x model, let's remove the params_axes
  # variable collection.
  if 'params_axes' in initial_vars:
    del initial_vars['params_axes']
  train_state = jax_task.create_train_state(
      initial_vars, var_weight_hparams, discard_opt_states
  )
  train_state_provenance = train_states.build_train_state_provenance(
      train_state
  )
  # `do_init_checkpoint_rules` is False for pjit/spmd.
  if do_init_checkpoint_rules:
    if checkpoint_type == CheckpointType.UNSPECIFIED:
      if py_utils.pmap_use_tensorstore():
        checkpoint_type = CheckpointType.GDA
      else:
        checkpoint_type = CheckpointType.FLAX
    # Overwrite some parts if init_checkpoint_rules are set (warm-start)
    # Note that this assumes a pmap model with Flax checkpoint(s).
    train_state, train_state_provenance, update_opt_states = (
        jax_task.apply_init_checkpoint_rules(
            train_state, train_state_provenance, checkpoint_type=checkpoint_type
        )
    )
    if update_opt_states:
      # Free the previous opt_states as it will be re-computed.
      jax.tree_util.tree_map(lambda x: x.delete(), train_state.opt_states)
      # Re-compute opt_states after the model variables are updated.
      opt_states = jax_task.create_opt_states(
          train_state.mdl_vars, var_weight_hparams
      )
      train_state = train_state.replace(opt_states=opt_states)
  return train_state, train_state_provenance


def replicate_model_state(model_states: TrainState) -> TrainState:
  """Replicates the model states."""

  def _replicate(state):
    # Skip the copy if it's already replicated.
    if isinstance(state, jax.Array) and len(state.devices()) != 1:
      return state
    else:
      return jax.device_put_replicated(state, jax.local_devices())

  return jax.tree_map(_replicate, model_states)


# TODO(laigd): remove this since it's used only by tests.
def initialize_replicate_model_state(
    jax_task: tasks_lib.SingleTask,
    prng_key: PRNGKey,
    inputs_shape_dtype: NestedShapeDtypeLike,
    discard_opt_states: bool = False,
) -> TrainState:
  """Initializes and replicates the model states."""
  model_states, _ = initialize_model_state(
      jax_task, prng_key, inputs_shape_dtype, discard_opt_states
  )
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

  # Match the definition of leaf to asserts.assert_same_structure.
  is_leaf = lambda x: not isinstance(x, (tuple, dict, list))
  return jax.tree_util.tree_map(
      _maybe_bfloat16_var_fn, mdl_vars, var_weight_hparams, is_leaf=is_leaf
  )


def _maybe_to_float32(x: JTensor) -> JTensor:
  if hasattr(x, 'dtype') and x.dtype == jnp.bfloat16:
    return x.astype(jnp.float32)
  return x


# TODO(pax): maybe move to metric_utils.py.
def _maybe_aggregate_metrics_summaries(
    loss_aggregator: base_metrics.LossAggregator,
    weighted_scalars: WeightedScalars,
    summary_dict: SummaryDict,
    per_example_out: dict[str, Any],
) -> Tuple[
    JTensor,
    JTensor,
    Optional[JTensor],
    WeightedScalars,
    SummaryDict,
    dict[str, Any],
]:
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
      weighted_scalars
  )

  if base_layer.is_running_under_pmap():
    # aggregate data across devices.
    aggregated_scalars = type(weighted_scalars)()
    for key in weighted_scalars:
      value, weight = weighted_scalars[key]
      sum_value = jax.lax.psum(
          value * weight, axis_name=PMAP_PARALLEL_AXIS_NAME
      )
      sum_weight = jax.lax.psum(weight, axis_name=PMAP_PARALLEL_AXIS_NAME)
      aggregated_scalars[key] = (sum_value / (sum_weight + 1e-8), sum_weight)
    aggregated_summaries = summary_utils.aggregate_per_replica_summaries(
        summary_dict
    )
    per_example_out = jax.lax.all_gather(
        per_example_out, axis_name=PMAP_PARALLEL_AXIS_NAME, tiled=True
    )
  else:
    # No aggregation of weighted scalars is needed.
    aggregated_scalars = weighted_scalars
    # No aggregation of summaries is needed.
    aggregated_summaries = summary_dict

  return (  # pytype: disable=bad-return-type  # jax-ndarray
      weighted_loss,
      mean_loss,
      loss_weight,
      aggregated_scalars,
      aggregated_summaries,
      per_example_out,
  )


def _maybe_synchronize_non_learnable_vars(
    old_vars, new_vars, var_weight_hparams
):
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

  def _synchronize_vars_using_mean(
      old_var: JTensor, new_var: JTensor
  ) -> JTensor:
    """Synchronize a variable across replicas by averaging."""
    delta = new_var - old_var
    delta_mean = jax.lax.pmean(delta, axis_name=PMAP_PARALLEL_AXIS_NAME)
    updated_var = old_var + delta_mean
    return updated_var

  def _synchronize_vars_using_sum(
      old_var: JTensor, new_var: JTensor
  ) -> JTensor:
    """Synchronize a variable across replicas by summing."""
    delta = new_var - old_var
    delta_total = jax.lax.psum(delta, axis_name=PMAP_PARALLEL_AXIS_NAME)
    updated_var = old_var + delta_total
    return updated_var

  def _synchronize_non_learnable_var(
      old_var: JTensor, new_var: JTensor, var_param: ParamsT
  ) -> JTensor:
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
      raise ValueError(
          'Non-trainable variables must have a cross-replica '
          'synchronization method specified.'
      )

  if base_layer.is_running_under_pmap():

    def _sync_var(old_var, new_var, var_param):
      return _synchronize_non_learnable_var(old_var, new_var, var_param)

    return jax.tree_util.tree_map(
        _sync_var, old_vars, new_vars, var_weight_hparams
    )
  # no synchronization is needed.
  return new_vars


@flax_struct.dataclass
class BaseStepFnStaticArgs:
  """Dataclass encapsulating all static arguments of a step function.

  We can define subclass and use it in the custom step function without changing
  its API:

    >>> @flax.struct.dataclass
    ... class MyStaticArgs(BaseStepFnStaticArgs):
    ...   adjust_loss: bool

    >>> def my_step_fn(
    ...     jax_task: tasks_lib.SingleTask,
    ...     states: TrainState,
    ...     prng_key: pytypes.PRNGKey,
    ...     inputs: NestedMap,
    ...     fprop_dtype: jnp.dtype,
    ...     var_weight_hparams: NestedWeightHParams,
    ...     static_args: MyStaticArgs,
    ... ) -> Tuple[Optional[TrainState], StepFnOutput]:
    ...   loss = ...
    ...   if static_args.adjust_loss:
    ...     loss *= 2.0
    ...   return new_state, StepFnOutput(loss=loss, ...)

    >>> partitioned_step_fn = partitioner.partition(
    ...     my_step_fn, input_shape_dtype)
    >>> partitioned_step_fn(
    ...     train_state, prng_key, inputs,
    ...     MyStaticArgs(adjust_loss=False))
    >>> partitioned_step_fn(
    ...     train_state, prng_key, inputs,
    ...     MyStaticArgs(adjust_loss=True))  # Re-compilation.

  Note all static arguments added as class members must be hashable, meaning
  both __hash__ and __eq__ are implemented, and should be immutable.
  See the description for argument static_broadcasted_argnums (pmap) and
  static_argnums (pjit) for more information:

  - pmap: https://jax.readthedocs.io/en/latest/_autosummary/jax.pmap.html
  - pjit: https://jax.readthedocs.io/en/latest/jax.experimental.pjit.html
  """

  # The unpadded size of global batch, and the padding is on the right side of
  # each input. Needed only when padding is used.
  unpadded_global_batch_size: Optional[int]


@flax_struct.dataclass
class StepFnOutput:
  """Dataclass encapsulating the output of a step function."""

  # Mean loss computed by the model.
  loss: Optional[JTensor]

  # A dict of (value, weight) pairs representing simple weighted average metrics
  # or losses.
  weighted_scalars: WeightedScalars

  # Auxilillary per-example output computed by the model.
  per_example_out: NestedMap

  # A dict or nested map of summary tensors computed in forward/backward pass.
  summary_tensors: SummaryDict

  # A dict of clu.metrics. Used by decode step function currently.
  clu_metrics: Optional[Dict[str, Any]] = None


class ApplyFnProtocol(Protocol):
  """Protocol for a function that can perform model inference."""

  def __call__(
      self,
      model: base_model.BaseModel,
      variables: NestedJTensor,
      inputs: NestedMap,
      mutable: bool,
      rngs: dict[str, PRNGKey],
  ) -> tuple[tuple[base_model.WeightedScalars, dict[str, Any]], NestedJTensor]:
    pass


def _default_apply_fn(
    model: base_model.BaseModel,
    variables: NestedJTensor,
    inputs: NestedMap,
    mutable: bool,
    rngs: dict[str, PRNGKey],
) -> tuple[tuple[base_model.WeightedScalars, dict[str, Any]], NestedJTensor]:
  """Just calls model.__call__ with the given vars, inputs, and other params."""
  return model.apply(
      variables,
      inputs,
      method=model.__call__,
      mutable=mutable,
      rngs=rngs,
  )


class LossFnProtocol(Protocol):

  def __call__(
      self, mdl_vars: NestedJTensor, inputs: NestedMap, prng_key: PRNGKey
  ) -> tuple[JTensor, sgf.GradAuxInfo]:
    """Produces losses and grad info by passing the inputs through a model."""


def _get_default_loss_fn(
    jax_task: tasks_lib.SingleTask,
    context_p: base_layer.JaxContext.HParams,
    fprop_dtype: jnp.dtype,
    var_weight_hparams: NestedWeightHParams,
    apply_fn: ApplyFnProtocol | None = None,
) -> LossFnProtocol:
  """Get the default loss function."""
  if apply_fn is None:
    apply_fn = _default_apply_fn

  def _loss_fn(
      mdl_vars: NestedJTensor, inputs: NestedMap, prng_key: PRNGKey
  ) -> tuple[JTensor, sgf.GradAuxInfo]:
    """Computes loss as well as other auxiliary outputs."""
    if fprop_dtype == jnp.float32:
      pass
    elif fprop_dtype == jnp.bfloat16:
      mdl_vars = _maybe_to_bfloat16_vars(mdl_vars, var_weight_hparams)
      inputs = jax.tree_map(_maybe_to_bfloat16, inputs)
    else:
      assert NotImplementedError(f'fprop_dtype {fprop_dtype} not supported.')

    with base_layer.JaxContext.new_context(hparams=context_p):
      k1, k2, k3 = jax.random.split(prng_key, 3)
      (weighted_scalars, per_example_output), updated_vars = apply_fn(
          model=jax_task.model,
          variables=mdl_vars,
          inputs=inputs,
          mutable=jax_task.hparams.train.apply_mutable_list,
          rngs={PARAMS: k1, RANDOM: k2, NON_PAX_RNG_KEY: k3},
      )

      # Fetch all the summary tensors.
      assert isinstance(updated_vars, dict)
      summary_tensors = updated_vars.get(SUMMARIES, {})
      # TODO(yonghui): Fetch aux losses and add them to summaries.
      summary_tensors = summary_utils.flatten_flax_summaries(summary_tensors)

      (
          weighted_loss,
          mean_loss,
          loss_weight,
          aggregated_scalars,
          aggregated_summaries,
          per_example_output,
      ) = _maybe_aggregate_metrics_summaries(
          loss_aggregator=jax_task.loss_aggregator_inst,
          weighted_scalars=weighted_scalars,
          summary_dict=summary_tensors,
          per_example_out=per_example_output,
      )
      # metrics and summary_tensors no longer needed.
      del weighted_scalars
      del summary_tensors

      forward_updated_vars = {
          collection: updated_vars[collection]
          for collection in [NON_TRAINABLE] + NON_PAX_VAR_COLLECTION
          if collection in updated_vars
      }

    if fprop_dtype == jnp.bfloat16 and weighted_loss.dtype == fprop_dtype:
      weighted_loss = weighted_loss.astype(jnp.float32)

    return weighted_loss, sgf.GradAuxInfo(
        loss_weight=loss_weight,
        aux_info=(
            mean_loss,
            aggregated_scalars,
            forward_updated_vars,
            aggregated_summaries,
            per_example_output,
        ),
    )

  return _loss_fn


class GradFnProtocol(Protocol):
  """Protocol for a gradient function, used in train_step_single_learner."""

  def __call__(
      self,
      loss_fn: LossFnProtocol,
      learner: learners_lib.Learner,
      mdl_vars: pytypes.PyTree,
      inputs: NestedMap,
      prng_key: PRNGKey,
  ) -> tuple[Any, Any]:
    """Computes gradients given an input batch.

    Args:
      loss_fn: A callable conforming to the LossFn protocol.
      learner: A pax learner.
      mdl_vars: A PyTree of tensors
      inputs: A NestedMap of model inputs.
      prng_key: a root PRNG key.

    Returns:
      A tuple of (values, grads).
    """


def _get_default_grad_fn(
    excluded_for_grad: NestedMap, excluded_for_opt: NestedMap
) -> GradFnProtocol:
  """Returns the default grad function, used for training."""

  def grad_fn(
      loss_fn: LossFnProtocol,
      learner: learners_lib.Learner,
      mdl_vars: NestedJTensor,
      inputs: NestedMap,
      prng_key: PRNGKey,
  ):
    with_grad = tasks_lib.filter_vars_for_grad_or_opt(
        mdl_vars, excluded_for_grad
    )
    no_grad = jax.tree_map(
        lambda x, e: x if e else {}, mdl_vars, excluded_for_grad
    )

    def _loss(
        mdl_vars_grad: NestedJTensor,
        mdl_vars_nograd_and_inputs: Tuple[NestedJTensor, NestedMap],
        prng_key: PRNGKey,
    ):
      mdl_vars_nograd, inputs = mdl_vars_nograd_and_inputs
      merged_vars = jax.tree_map(
          lambda e, x, y: y if e else x,
          excluded_for_grad,
          mdl_vars_grad,
          mdl_vars_nograd,
      )
      return loss_fn(merged_vars, inputs, prng_key)

    if learner.stochastic_gradient is None:
      g = jax.value_and_grad(_loss, has_aux=True, allow_int=True)
    else:
      g = functools.partial(learner.stochastic_gradient.grad_fn, _loss)
    values, grads = g(with_grad, (no_grad, inputs), prng_key)
    grads = jax.tree_map(
        lambda eo, eg, m, g: jnp.zeros_like(m) if eg and not eo else g,
        excluded_for_opt,
        excluded_for_grad,
        mdl_vars,
        grads,
    )
    return values, grads

  return grad_fn


def _log_bprop_include_exclude_list(
    var_weight_hparams: NestedWeightHParams, excluded_for_grad: NestedMap
) -> None:
  flat_var_prefix = jax.tree_flatten(
      py_utils.extract_prefixed_keys_from_nested_map(var_weight_hparams)
  )[0]
  flat_mask = jax.tree_flatten(excluded_for_grad)[0]
  for prefix, excluded in zip(flat_var_prefix, flat_mask):
    if excluded:
      logging.info('Bprop excluded var: %s', prefix)
  for prefix, excluded in zip(flat_var_prefix, flat_mask):
    if not excluded:
      logging.info('Bprop included var: %s', prefix)


# TODO(yonghui): refactor to pass in learner separately.
def train_step_single_learner(
    jax_task: tasks_lib.SingleTask,
    states: TrainState,
    prng_key: PRNGKey,
    inputs: JTensor | NestedMap,
    fprop_dtype: jnp.dtype = jnp.float32,
    var_weight_hparams: NestedWeightHParams | None = None,
    static_args: BaseStepFnStaticArgs | None = None,
    *,
    grad_fn: GradFnProtocol | None = None,
    apply_fn: ApplyFnProtocol | None = None,
    expose_updated_nontrainables_to_learner=True,
) -> Tuple[TrainState, StepFnOutput]:
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
    static_args: Unused.
    grad_fn: A custom gradient function to be used instead of a default one.
    apply_fn: A custom apply function to be used instead of a default one. Note
      that when `grad_fn` is not `None`, this argument has no effect and you
      don't need it, because the custom `grad_fn` can choose to use whatever
      "apply" function. `apply_fn` is only used by the default gradient
      function.
    expose_updated_nontrainables_to_learner: Whether to make updated
      non-trainable variables visible to the learner. Some optimizers such as
      `optimizers.DynamicAccumulator` assume special non-trainable variables
      such as EMA (Exponential Moving Average) being set during forward-prop for
      controlling their behavior.

  Returns:
    A tuple (updated_states, StepFnOutput).
  """
  del static_args

  model = jax_task.model
  assert len(jax_task.learners) == 1
  learner = jax_task.learners[0]

  context_p = base_layer.JaxContext.HParams(
      do_eval=False,
      summary_verbosity=jax_task.summary_verbosity,
  )
  # Fold in global_step as part of the random seed key, so that random
  # numbers depends on global step.
  #
  # TODO(yonghui): also fold in the replica id.
  prng_key = jax.random.fold_in(prng_key, states.step)  # pytype: disable=wrong-arg-types  # jax-ndarray

  if not var_weight_hparams:
    with base_layer.JaxContext.new_context(hparams=context_p):
      var_weight_hparams = model.abstract_init_with_metadata(inputs)
  updated_model_vars = jax_task.maybe_adjust_train_state(  # pytype: disable=wrong-arg-types  # jax-ndarray
      step=states.step,
      mdl_vars=states.mdl_vars,
      var_weight_hparams=var_weight_hparams,
      prng_key=prng_key,
  )

  _, subkey = jax.random.split(prng_key)

  # Skip variables for gradients.
  excluded_for_grad = tasks_lib.get_excluded_var_mask_for_grad(
      var_weight_hparams, learner
  )
  _log_bprop_include_exclude_list(var_weight_hparams, excluded_for_grad)

  # Excluded for optimizer states.
  excluded_for_opt = tasks_lib.get_excluded_var_mask_for_opt(
      var_weight_hparams,
      learner,
  )

  # Construct and call the grad function.
  if not grad_fn:
    grad_fn = _get_default_grad_fn(excluded_for_grad, excluded_for_opt)
  (weighted_loss, aux_info), grads = grad_fn(
      loss_fn=_get_default_loss_fn(
          jax_task=jax_task,
          context_p=context_p,
          fprop_dtype=fprop_dtype,
          var_weight_hparams=var_weight_hparams,
          apply_fn=apply_fn,
      ),
      learner=learner,
      mdl_vars=updated_model_vars,
      inputs=inputs,
      prng_key=subkey,
  )
  (
      mean_loss,
      weighted_scalars,
      fwd_updated_vars,
      fwd_summary_tensors,
      per_example_out,
  ) = aux_info.aux_info

  # weighted_loss is only needed for computing gradients, but otherwise, not
  # needed.
  del weighted_loss

  # Carry out backward computation under a JaxContext.
  with base_layer.JaxContext.new_context(hparams=context_p):
    if base_layer.is_running_under_pmap():
      # Aggregate grads across different model replicas.
      grads = jax.lax.psum(grads, axis_name=PMAP_PARALLEL_AXIS_NAME)
    else:
      # No gradient aggregation is needed.
      pass

    # Add a summary for learning rate
    learner.plot_learning_rate(states.step)  # pytype: disable=wrong-arg-types  # jax-ndarray

    # Apply gradient transformations.
    mdl_vars = states.mdl_vars.copy()  # pytype: disable=attribute-error  # jax-ndarray
    if (
        expose_updated_nontrainables_to_learner
        and NON_TRAINABLE in fwd_updated_vars
    ):
      # Make updated non-trainable vars visible to EMA.
      mdl_vars[NON_TRAINABLE] = fwd_updated_vars[NON_TRAINABLE]
    vars_with_opt = tasks_lib.filter_vars_for_grad_or_opt(
        mdl_vars, excluded_for_opt
    )
    wps_with_opt = tasks_lib.filter_vars_for_grad_or_opt(
        var_weight_hparams, excluded_for_opt
    )
    transformed_grads, new_opt_states = learner.update_states(
        grads, states.opt_states[0], vars_with_opt, wps_with_opt
    )
    vars_with_opt = learner.apply_gradient(
        vars_with_opt, transformed_grads, wps_with_opt
    )
    mdl_vars = jax.tree_map(
        lambda e, old, new: old if e else new,
        excluded_for_grad,
        mdl_vars,
        vars_with_opt,
    )

    for collection in [NON_TRAINABLE] + NON_PAX_VAR_COLLECTION:
      if collection in states.mdl_vars:
        # We need to update the non-trainable vars.
        asserts.assert_same_structure(
            states.mdl_vars[collection], fwd_updated_vars[collection]
        )
        mdl_vars[collection] = _maybe_synchronize_non_learnable_vars(
            states.mdl_vars[collection],
            fwd_updated_vars[collection],
            var_weight_hparams[collection],
        )

    # We may have updated non-trainable vars that have been explicitly excluded.
    mdl_vars = jax.tree_map(
        lambda e, old, new: old if e else new,
        # Filter out only the explicitly masked non-trainables.
        tasks_lib.get_excluded_var_mask_for_grad_or_opt(
            var_weight_hparams, learner, mask_all_non_trainable=False
        ),
        states.mdl_vars,
        mdl_vars,
    )
    new_states = states.new_state(
        mdl_vars=mdl_vars, opt_states=[new_opt_states], extra_state=()
    )
    # Finally fetch all backward summary tensors. We do not aggregate the scalar
    # summaries with pmean because the grads are already psum-ed.
    if jax_task.hparams.train.variable_norm_summary:
      var_summary_tensors = summary_utils.l2_mean(
          new_states.mdl_vars, prefix='vars', max_level=20
      )
      for name, norm in var_summary_tensors.items():
        base_layer.add_global_summary(name, norm)
    if isinstance(
        learner.stochastic_gradient,
        (sgf.DpSgdStochasticGradient, sgf.PercoreClippedDpSgdGradient),
    ):
      if base_layer.is_running_under_pmap():
        frac_clipped = jax.lax.pmean(
            aux_info.dp_aux_info['frac_clipped'],
            axis_name=PMAP_PARALLEL_AXIS_NAME,
        )
      else:
        frac_clipped = aux_info.dp_aux_info['frac_clipped']
      base_layer.add_global_summary('dp_metrics/frac_clipped', frac_clipped)
    if isinstance(learner.stochastic_gradient, sgf.PercoreClippedDpSgdGradient):
      if base_layer.is_running_under_pmap():
        mean_per_core_grad_norm = jax.lax.psum(
            aux_info.dp_aux_info['per_core_grad_norm'],
            axis_name=PMAP_PARALLEL_AXIS_NAME,
        )
      else:
        mean_per_core_grad_norm = aux_info.dp_aux_info['per_core_grad_norm']
      base_layer.add_global_summary(
          'dp_metrics/mean_per_core_grad_norm', mean_per_core_grad_norm
      )
    bwd_summary_tensors = base_layer.all_global_summaries()

  summary_tensors = NestedMap()
  summary_tensors.update(fwd_summary_tensors)
  summary_tensors.update(bwd_summary_tensors)

  return new_states, StepFnOutput(
      loss=mean_loss,
      weighted_scalars=weighted_scalars,
      per_example_out=per_example_out,
      summary_tensors=summary_tensors,
  )


# TODO(laigd): rename - eval has nothing to do with number of learners.
def eval_step_single_learner(
    jax_task: tasks_lib.SingleTask,
    states: TrainState,
    prng_key: JTensor,
    inputs: Union[JTensor, NestedMap],
    fprop_dtype: jnp.dtype = jnp.float32,
    var_weight_hparams: Optional[NestedWeightHParams] = None,
    static_args: Optional[BaseStepFnStaticArgs] = None,
) -> Tuple[None, StepFnOutput]:
  """Evaluates a model for a single step.

  This utility is specialized for the single learner case.

  Args:
    jax_task: An instance of tasks.SingleTask.
    states: An instance of model.TrainState.
    prng_key: A prng seed, of shape [2], of type np.uint32.
    inputs: Inputs to the model() function.
    fprop_dtype: fprop datatype, can be either jnp.float32 or jnp.bfloat16.
    var_weight_hparams: A pytree of WeightHParams for the model variables.
    static_args: Unused.

  Returns:
    A tuple (None, StepFnOutput) to be compatible with the general step function
    output format (updated_train_state, StepFnOutput), where updated_train_state
    is None since it doesn't update the train state.
  """
  del static_args

  model = jax_task.model
  context_p = base_layer.JaxContext.HParams(
      do_eval=True,
      summary_verbosity=jax_task.summary_verbosity,
  )
  # Fold in global_step as part of the random seed key, so that random
  # numbers depends on global step.
  prng_key = jax.random.fold_in(prng_key, states.step)  # pytype: disable=wrong-arg-types  # jax-ndarray
  mdl_vars = states.mdl_vars
  # assert not states.opt_states

  if not var_weight_hparams:
    var_weight_hparams = model.abstract_init_with_metadata(
        inputs,
        do_eval=not jax_task.hparams.train.always_use_train_for_model_init,
    )

  if fprop_dtype == jnp.float32:
    pass
  elif fprop_dtype == jnp.bfloat16:
    mdl_vars = _maybe_to_bfloat16_vars(mdl_vars, var_weight_hparams)
    inputs = jax.tree_map(_maybe_to_bfloat16, inputs)
  else:
    assert NotImplementedError(f'fprop_dtype {fprop_dtype} not supported.')

  enum_keys, inputs = py_utils.filter_by_matching_keys(
      inputs, [py_utils.PROVENANCE_PREFIX]
  )
  with base_layer.JaxContext.new_context(hparams=context_p):
    prng_key, k1, k2, k3 = jax.random.split(prng_key, 4)
    apply_rng_keys = {PARAMS: k1, RANDOM: k2, NON_PAX_RNG_KEY: k3}
    (weighted_scalars, per_example_out), updated_vars = model.apply(
        mdl_vars,
        inputs,
        mutable=jax_task.hparams.evaluate.apply_mutable_list,
        method=model.__call__,
        rngs=apply_rng_keys,
    )

    summary_tensors = updated_vars.get(SUMMARIES, {})
    # TODO(yonghui): Add aux-loss to summaries.
    summary_tensors = summary_utils.flatten_flax_summaries(summary_tensors)

    # merge back, if any, enum keys for eval matching
    per_example_out.update(enum_keys)
    (
        _,
        mean_loss,
        _,
        aggregated_scalars,
        aggregated_summaries,
        per_example_out,
    ) = _maybe_aggregate_metrics_summaries(
        jax_task.loss_aggregator_inst,
        weighted_scalars,
        summary_tensors,
        per_example_out,
    )

    # weighted_scalars and summary_tensors no longer needed.
    del weighted_scalars
    del summary_tensors

  if fprop_dtype == jnp.bfloat16:
    (mean_loss, aggregated_scalars, per_example_out, aggregated_summaries) = (
        jax.tree_map(
            _maybe_to_float32,
            (
                mean_loss,
                aggregated_scalars,
                per_example_out,
                aggregated_summaries,
            ),
        )
    )

  return None, StepFnOutput(
      loss=mean_loss,
      weighted_scalars=aggregated_scalars,
      per_example_out=per_example_out,
      summary_tensors=aggregated_summaries,
  )


def decode_step(
    model: base_model.BaseModel,
    states: TrainState,
    prng_key: JTensor,
    var_weight_hparams: NestedWeightHParams,
    inputs: Union[JTensor, NestedMap],
    fprop_dtype: jnp.dtype = jnp.float32,
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
  prng_key = jax.random.fold_in(prng_key, states.step)  # pytype: disable=wrong-arg-types  # jax-ndarray
  mdl_vars = states.mdl_vars

  assert not states.opt_states

  if fprop_dtype == jnp.bfloat16:
    mdl_vars = _maybe_to_bfloat16_vars(mdl_vars, var_weight_hparams)
    inputs = jax.tree_map(_maybe_to_bfloat16, inputs)
  elif fprop_dtype != jnp.float32:
    assert NotImplementedError(f'fprop_dtype {fprop_dtype} not supported.')

  enum_keys, inputs = py_utils.filter_by_matching_keys(
      inputs, [py_utils.PROVENANCE_PREFIX]
  )
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
        ],
    )
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
    task,
    states,
    prng_key,
    inputs,
    fprop_dtype,
    var_weight_hparams,
    static_args: Optional[BaseStepFnStaticArgs] = None,
) -> Tuple[None, StepFnOutput]:
  """Decode step function used by the partitioner."""
  (weighted_scalars, per_example_out, updated_metrics), updated_vars = (
      decode_step(
          task.model, states, prng_key, var_weight_hparams, inputs, fprop_dtype
      )
  )
  # TODO(laigd): move this inside decode_step(). None of the existing callers
  # need more than summary_tensors.
  summary_tensors = updated_vars.get(base_layer.SUMMARIES, {})
  summary_tensors = summary_utils.flatten_flax_summaries(summary_tensors)

  # TODO(laigd): this logic is similar to _maybe_aggregate_metrics_summaries,
  # consider unify them.
  if base_layer.is_running_under_pmap():
    # Similar to (train|eval)_step_single_learner, we aggregate all outputs from
    # each device using an all_gather, so that the results are fully replicated.
    weighted_scalars = task.metrics_aggregator.aggregate(weighted_scalars)
    per_example_out = jax.lax.all_gather(
        per_example_out, axis_name=PMAP_PARALLEL_AXIS_NAME, tiled=True
    )
    summary_tensors = summary_utils.aggregate_per_replica_summaries(
        summary_tensors
    )

    # Gathers the metrics across workers, and then aggregate them.
    # Note that it's important to disable tiling (tiled=False by default) when
    # calling all_gather(), so that clu_metrics.Metric.reduce() can aggregate
    # the value correctly using jax.lax.scan.
    aggregated_metrics = {}
    for metric_name, metric in updated_metrics.items():
      aggregated_metrics[metric_name] = jax.lax.all_gather(
          metric, axis_name=PMAP_PARALLEL_AXIS_NAME
      ).reduce()
    updated_metrics = aggregated_metrics

  return None, StepFnOutput(
      loss=None,
      weighted_scalars=weighted_scalars,
      per_example_out=per_example_out,
      summary_tensors=summary_tensors,
      clu_metrics=updated_metrics,
  )


def initialize_partitioned_model_states(
    jax_task: tasks_lib.SingleTask,
    prng_key: PRNGKey,
    global_input_shapes: NestedShapeDtypeLike,
    state_specs: TrainState,
    discard_opt_states: bool = False,
    global_mesh: Optional[jax.sharding.Mesh] = None,
    checkpoint_type: CheckpointType = CheckpointType.GDA,
    do_init_checkpoint_rules: bool = True,
    var_weight_hparams: Optional[NestedWeightHParams] = None,
    is_eval: bool = False,
) -> Tuple[TrainState, TrainStateProvenance]:
  """Initializes model vars that are partitioned over TPU devices.

  Weights are random initialized first. Then we restore weights based on the
  init_checkpoint_rules.

  This function is equivalent to calling a pjit-ted version of
  InitializesModelStates().

  Args:
    jax_task: The task which is an instance of tasks.SingleTask.
    prng_key: A PRNGKey.
    global_input_shapes: Global shapes of sample inputs for shape inference.
    state_specs: The TrainState specs when restoring weights based on the
      init_checkpoint_rules.
    discard_opt_states: bool, When true, optimizer slot variables are skipped.
    global_mesh: The global mesh to use when restoring weights based on the
      init_checkpoint_rules. Required for GDA-based checkpoints.
    checkpoint_type: The checkpoint type to use when restoring weights based on
      the init_checkpoint_rules.
    do_init_checkpoint_rules: If apply init_checkpoint_rules.
    var_weight_hparams: A pytree of WeightHParams for the model variables.
    is_eval: Whether to load the model in eval mode (useful for models which
      have different inference graphs than training).

  Returns:
    The partitioned vars themselves.
  """
  model = jax_task.model
  if not var_weight_hparams:
    var_weight_hparams = model.abstract_init_with_metadata(
        global_input_shapes, do_eval=is_eval
    )

  train_state_partition_specs = (
      state_specs.to_eval_state() if discard_opt_states else state_specs
  )
  train_state_unpadded_shapes = jax.tree_map(
      lambda x: x.shape,
      jax_task.create_train_state_unpadded_shapes(
          var_weight_hparams, discard_opt_states
      ),
  )
  assert train_state_partition_specs is not None

  def init_model_from_seed(prng_key):
    outs, _ = initialize_model_state(
        jax_task,
        prng_key,
        global_input_shapes,
        discard_opt_states,
        # `do_init_checkpoint_rules` is False for pjit/spmd here because
        # checkpoint loading has to be done after partitioning. See below.
        do_init_checkpoint_rules=False,
        var_weight_hparams=var_weight_hparams,
        is_eval=is_eval,
    )
    return py_utils.maybe_pad_uneven_sharding(
        outs,
        train_state_partition_specs,  # pytype: disable=wrong-arg-types  # jax-ndarray
        train_state_unpadded_shapes,
        model.hparams.mesh_shape,
        model.hparams.mesh_axis_names,
    )

  logging.info('unpadded_out_shape: %s', train_state_unpadded_shapes)
  logging.info('train_state_partition_specs: %s', train_state_partition_specs)
  asserts.assert_same_structure(
      train_state_unpadded_shapes, train_state_partition_specs
  )

  mesh_names = model.hparams.mesh_axis_names
  prng_key_partition_spec = base_layer.to_partition_spec((None,), mesh_names)

  init_fn = pjit.pjit(
      init_model_from_seed,
      in_shardings=prng_key_partition_spec,
      out_shardings=train_state_partition_specs,
  )
  init_fn = bind_mesh(init_fn, global_mesh)

  partitioned_vars = init_fn(prng_key)
  train_state_provenance = train_states.build_train_state_provenance(
      partitioned_vars
  )
  # Overwrite some parts if init_checkpoint_rules are set (warm-start)
  if (
      do_init_checkpoint_rules
      and jax_task.hparams.train.init_from_checkpoint_rules
  ):
    # TODO(b/230132535): Note that this application after constructing the
    # partitioned vars is currently inconsistent with what is being performed
    # for pmap models.
    partitioned_vars, train_state_provenance, _ = (
        jax_task.apply_init_checkpoint_rules(
            partitioned_vars,
            train_state_provenance,
            train_state_partition_specs=train_state_partition_specs,
            global_mesh=global_mesh,
            checkpoint_type=checkpoint_type,
        )
    )

  return partitioned_vars, train_state_provenance


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
      how to partition tensors of this dimension. If mapping_dict is None, no
      resharding of the tensor.
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
    raise ValueError(
        f'Split mapping must be provided for {len(x.shape)}-d '
        f'in the form of key map_{len(x.shape)} in '
        f'{mapping_dict}.'
    )
  if mapping_dict[key] is not None:
    return base_layer.maybe_shard(x, mapping_dict[key], mesh_names)
  else:
    return x


def get_inputs_shape_dtype(
    input_config: pax_fiddle.Config[base_input.BaseInput],
) -> tuple[NestedShapeDtypeLike, NestedShapeDtypeLike]:
  """Returns the per-host and global shape/dtype information of the input.

  WARNING: this does instantiate the dataset and fetches a batch, so is costly.
    This method of obtaining shapes can be avoided by specifying an input spec
    provider in your experiment.

  Args:
    input_config: a Fiddle config parameterizing a BaseInput.

  Returns:
    A tuple consisting of the per-host input shapes and the global input shapes.
  """
  sample_inputs = instantiate(input_config).get_next_padded()

  perhost_inputs_shape_dtype = jax.tree_map(
      lambda x: jax.ShapeDtypeStruct(shape=x.shape, dtype=x.dtype),
      sample_inputs,
  )
  global_inputs_shape_dtype = jax.tree_map(
      py_utils.get_global_input_shape_dtype, sample_inputs
  )
  return perhost_inputs_shape_dtype, global_inputs_shape_dtype


def get_input_partition_specs(mesh_axis_names, inputs_shape_dtype):
  logging.info(
      'get_input_partition_specs from mesh_axis_names=%s and '
      'inputs_shape_dtype=%s',
      mesh_axis_names,
      inputs_shape_dtype,
  )
  # Compute inputs PartitionSpec from inputs_shape_dtype
  inputs_partition_spec_fn = functools.partial(
      shard_on_batch_dim_partition_spec, mesh_axis_names
  )
  return jax.tree_util.tree_map(inputs_partition_spec_fn, inputs_shape_dtype)


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


def get_train_input_specs_for_model_init(
    task: tasks_lib.SingleTask,
    input_specs_provider: base_input.BaseInputSpecsProvider,
) -> NestedShapeDtypeStruct:
  """Returns the shape/dtypes needed to initialize the partitioner.

  This will have the per-device batch size for pmap cases (when there is no mesh
  shape defined in the model) and the global batch size for the pjit case.

  Args:
    task: The task parameters
    input_specs_provider: The BaseInputSpecsProvider that provides the train
      input shapes/dtypes, which will have per-device batch size for pmap, and
      per-process batch size for pjit.

  Returns:
    The input spec needed to initialize the partitioner.
  """
  train_input_specs = input_specs_provider.get_input_specs()
  logging.info(
      'Spec yielded by InputSpecProvider for model init: %s',
      pprint.pformat(train_input_specs),
  )

  # All pjit models specify at least the ICI mesh shape.
  if task.model.mesh_shape is not None:
    # Pjit - handle the fractional batch size case.
    batch_size = jax.tree_util.tree_leaves(train_input_specs)[0].shape[0]
    num_devices = jax.local_device_count()

    # Each device needs at least one example.
    batch_size = max(num_devices, batch_size)

    # And each device needs to receive identically-shape tensors.
    if batch_size % num_devices != 0:
      batch_size = ((batch_size + num_devices) // num_devices) * num_devices

    # The batch size we assign here is the global batch size (the convention for
    # models parallelized with pjit).
    train_input_specs = jax.tree_map(
        lambda x: jax.ShapeDtypeStruct(  # pylint: disable=g-long-lambda
            shape=[batch_size * jax.process_count(), *x.shape[1:]],
            dtype=x.dtype,
        ),
        train_input_specs,
    )
    logging.info(
        'Modified spec for pjit global batch size: %s',
        pprint.pformat(train_input_specs),
    )

  return train_input_specs
