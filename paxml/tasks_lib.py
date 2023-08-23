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

"""Implementation of tasks.

Note: The module name is suffixed with `_lib` to avoid the name conflict with
the `tasks` Python submodule.
"""

from __future__ import annotations

import copy
import dataclasses
import enum
import itertools
import re
import typing
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

from absl import logging
from etils import epath
import flax
import jax
from jax import numpy as jnp
import numpy as np
import optax
from paxml import base_inference_runner
from paxml import base_metrics
from paxml import base_task
from paxml import checkpoint_types
from paxml import io_utils
from paxml import learners as learners_lib
from paxml import train_states
from praxis import asserts
from praxis import base_hyperparams
from praxis import base_input
from praxis import base_layer
from praxis import base_model
from praxis import lazy_loader
from praxis import optimizer_prefix_vectorization
from praxis import optimizers
from praxis import pax_fiddle
from praxis import py_utils
from praxis import pytypes
from praxis import trees

# Those modules are slow to import, so we do it lazily.
ocp = lazy_loader.LazyLoader('ocp', globals(), 'orbax.checkpoint')
checkpoints = lazy_loader.LazyLoader(
    'checkpoints', globals(), 'paxml.checkpoints'  # mapped to internal
)

BaseInferenceRunner = base_inference_runner.BaseInferenceRunner
CheckpointType = checkpoint_types.CheckpointType
Nested = pytypes.Nested
NestedMap = py_utils.NestedMap
NestedJTensor = base_layer.NestedJTensor
NestedWeightHParams = base_layer.NestedWeightHParams
JTensor = base_layer.JTensor
PartitionSpec = jax.sharding.PartitionSpec
TrainState = train_states.TrainState
TensorProvenance = train_states.TensorProvenance
TrainStateProvenance = train_states.TrainStateProvenance
NO_PREFIX_KEY = optimizer_prefix_vectorization.NO_PREFIX_KEY
EarlyStoppingFn = Callable[[Dict[str, float], enum.Flag, int, bool], bool]

PRNGKey = pytypes.PRNGKey
RegexStr = str

instantiate = base_hyperparams.instantiate

# Shorthand for a loading rule that loads everything as is.
# e.g. load_rules = [LOAD_ALL]
LOAD_ALL = ('(.*)', '{}')

TRAIN_DEFAULT_MUTABLE_LIST = [
    base_layer.AUX_LOSS,
    base_layer.SUMMARIES,
    base_layer.NON_TRAINABLE,
] + base_layer.NON_PAX_VAR_COLLECTION


EVAL_DEFAULT_MUTABLE_LIST = [
    base_layer.AUX_LOSS,
    base_layer.SUMMARIES,
    base_layer.NON_TRAINABLE,
]


def _flatten_dict(
    node: Dict[str, Any],
    prefix: str = '',
) -> List[Tuple[str, Any]]:
  """Flattens a given nested dict and returns <key, value> pairs in a list."""
  ret = []
  if isinstance(node, dict):
    for k, v in node.items():
      res = _flatten_dict(v, prefix + '.' + k if prefix else k)
      ret.extend(res)
  else:
    ret = [(prefix, node)]
  return ret


def is_vectorized(states: TrainState) -> bool:
  """Determines whether it is a vectorized model."""
  if not states.opt_states:
    raise ValueError(
        'cannot decide if it is vectorized model without opt_states'
    )
  return NO_PREFIX_KEY in states.opt_states[0]


def has_ema(task: SingleTask) -> bool:
  """Determines whether ema is used or not."""
  return task.train.learner.optimizer.ema_decay > 0.0


def extract_ema(
    model_states: train_states.TrainState,
    merge_for_bprop_exclusion: bool = True,
) -> train_states.TrainState:
  """Finds the ema state from optimizer states."""
  if len(model_states.opt_states) != 1:
    raise ValueError(
        'EMA currently only supports a single learner (got '
        f'`{len(model_states.opt_states)}`).'
    )
  vectorized = is_vectorized(model_states)
  extracted = None
  if not vectorized:
    for v in model_states.opt_states[0]:
      if isinstance(v, dict) and 'ema' in v:
        extracted = v.ema
        break
  else:
    extracted = None
    # For vectorized model, the structure looks like this:
    # opt_states: [{'no_prefix': ({'count': '', 'ema': {'params': {'ctcloss':
    # It is a list of dictionaries. The key corresponds to the #stages.
    # Here the ema is constructed by combining the ema state from all those
    # dictionaries. Each parameter belongs to one dictionary and is labelled as
    # masked node in others.
    for item in model_states.opt_states[0].values():  # pytype: disable=attribute-error  # jax-ndarray
      if isinstance(item, tuple):
        for v in item:
          if isinstance(v, dict) and 'ema' in v:
            if extracted is None:
              extracted = v.ema
            else:
              extracted = jax.tree_map(
                  lambda x, y: y if py_utils.is_optax_masked_node(x) else x,
                  extracted,
                  v.ema,
                  is_leaf=py_utils.is_optax_masked_node,
              )
  if extracted is None:
    raise ValueError(
        'Could not find EMA states in `%r`.' % model_states.opt_states
    )
  extracted = jax.tree_map(
      lambda x: None if py_utils.is_optax_masked_node(x) else x,
      extracted,
      is_leaf=py_utils.is_optax_masked_node,
  )

  def _replace_bprop_masked(x, from_mdl_vars):
    if not py_utils.is_bprop_masked_node(x):
      return x
    if py_utils.is_optax_masked_node(from_mdl_vars):
      return None
    return from_mdl_vars

  if merge_for_bprop_exclusion:
    extracted = jax.tree_util.tree_map(
        _replace_bprop_masked,
        extracted,
        model_states.mdl_vars,
        is_leaf=py_utils.is_bprop_masked_node,
    )
  return TrainState(
      step=model_states.step,
      mdl_vars=extracted,
      opt_states=[],
      extra_state=(),
  )


def _set_nested_dict_value(node: Dict[str, Any], path: str, value: Any) -> None:
  """Sets the value for a nested key.

  Based on py_utils.NestedMap.Set(...).

  Args:
    node: A nested dict.
    path: str of the form key_part1.key_part2...key_partN.
    value: The value to insert.

  Raises:
    ValueError if a sub key is not a  dict.
  """

  current = node
  sub_paths = path.split('.')

  for i, k in enumerate(sub_paths):
    if not isinstance(current, dict):
      raise ValueError('Error while setting key {}. Sub key "{}" is of type'
                       ' {} but must be a dict or NestedMap.'
                       ''.format(path, k, type(current)))
    if i == (len(sub_paths) - 1):
      current[k] = value
    else:
      if k not in current:
        current[k] = NestedMap()
      current = current[k]


# These rules are needed in order to load optimizer states from checkpoint with
# Praxis optimizer to new Optax based optimizer.
# For example, the internal state for Adam optimizer in praxis used variables
# 'm' and 'v', whereas optax based optimizer stores the internal state in
# variables 'mu' and 'nu' within ScaleByAdamState state. Thus, we replace these
# state variables when loading from checkpoint using variable mapping. To
# avoid each user from specifying these common patterns, we keep all the
# required mappings for optimizer states in one place. Mapping for each
# optimizer will be introduced one by one after testing.
def get_optax_opt_load_rules() -> Sequence[Tuple[re.Pattern[str], str]]:
  rules = [
      # Rules for Adam optimizer.
      # Note extra '/0/' is to account for ScaleByAdamState tuple.
      (r'(.*)/0/mu/params/(.*)', '{}/m/params/{}'),
      (r'(.*)/0/nu/params/(.*)', '{}/v/params/{}'),
      # Note this will be populated one by one for each optimizer.
  ]
  loading_rules = [(re.compile(pattern), ref) for pattern, ref in rules]
  return loading_rules


def get_praxis_opt_state_regex() -> Sequence[re.Pattern[str]]:
  opt_state_patterns = [
      # Regex for matching opt state for Praxis Adam optimizer.
      r'(.*)/m/params/(.*)',
      r'(.*)/v/params/(.*)',
  ]
  opt_state_patterns = [re.compile(pattern) for pattern in opt_state_patterns]
  return opt_state_patterns


def _get_var_mapping(
    varnames: List[str],
    loading_rules: Sequence[Tuple[re.Pattern[str], str]],
    ignore_rules: Optional[Sequence[re.Pattern[str]]],
    is_initialized: Dict[str, str],
    ckpt_path: str,
    kind: str,
    safe_load: bool,
    target_partition_specs: Optional[TrainState] = None
) -> Tuple[Dict[str, str], Dict[str, Any]]:
  """Creates a mapping from model var names to corresponding var names in ckpt.
  """

  if target_partition_specs is not None:
    flat_mdl_vars_pspecs = dict(
        NestedMap(target_partition_specs.mdl_vars).FlattenItems())
  else:
    flat_mdl_vars_pspecs = None

  def _get_var_pspec(varname):
    if flat_mdl_vars_pspecs is None:
      return None
    return flat_mdl_vars_pspecs[varname]

  matched_pspecs = {}
  var_mapping = {}
  if safe_load:
    all_dest_patterns_in_loading_rules = set()
    matched_dest_patterns_in_loading_rules = set()
  for varname in varnames:
    varname_orig = varname
    varname = varname.replace('.', '/')  # dot is reserved for regex
    for pattern, refname in loading_rules:
      if safe_load:
        all_dest_patterns_in_loading_rules.add(pattern.pattern)
      mo = pattern.match(varname)
      if mo is None:
        continue
      if any(pat.match(varname) is not None for pat in ignore_rules):
        logging.info(
            '%s Initialization by external checkpoint: '
            '%s match ignore rule, skip.', kind, varname)
        continue
      if varname in is_initialized:
        logging.info(
            '%s Initialization by external checkpoint: '
            '%s is already initialized by %s, skip.', kind, varname,
            is_initialized[varname])
        continue
      refname = refname.format(*mo.groups())
      refname = refname.replace('/', '.')
      if safe_load:
        matched_dest_patterns_in_loading_rules.add(pattern.pattern)

      # Only for logging, keep name of ckpt that initialized the variable
      is_initialized[varname] = ckpt_path + '/' + refname
      logging.info(
          '%s Initialization by external checkpoint: '
          '%s is overwritten by %s in %s', kind, varname, refname, ckpt_path)
      pspec = _get_var_pspec(varname_orig)
      if refname in matched_pspecs:
        if matched_pspecs[refname] != pspec:
          raise ValueError('Not supported: multiple vars initialized from '
                           'the same init checkpoint variable but have '
                           f'different pspecs. {matched_pspecs[refname]} vs '
                           f'{pspec} in {varname}')
      else:
        matched_pspecs[refname] = pspec
      var_mapping[varname_orig] = refname
      break
    else:
      logging.info(
          '%s Initialization by external checkpoint: '
          "%s doesn't match rule, skip.",
          kind,
          varname,
      )

  if safe_load:
    # Check that all source names have been matched; if they have not then the
    # loading rules do not serve the intended purpose.
    diff = all_dest_patterns_in_loading_rules.difference(
        matched_dest_patterns_in_loading_rules)
    if diff:
      logging.info('Difference all-matched load_rule patterns=%r', diff)
      raise ValueError(f'The checkpoint loading rule(s) {loading_rules} '
                       'do not serve the intended purpose; some model '
                       'variables that were meant to be loaded from '
                       'checkpoint are left to their initial (random) values '
                       f'due to wrong pattern(s): {diff}.')
  return var_mapping, matched_pspecs


def _assign_model_vars(
    model_vars: Union[NestedMap, Dict[str, Any]],
    loaded_vars: Dict[str, Any],
    model_vars_mapping: Dict[str, str],
    model_provenance: Union[
        NestedMap, Dict[str, Any]
    ],
    loaded_provenance: Dict[str, Any],
) -> None:
  """Sets current model vars from loaded model vars using provided mapping."""
  used_vars = set()
  for var_name, init_var_name in model_vars_mapping.items():
    loaded_var = loaded_vars[init_var_name]
    loaded_var_provenance = loaded_provenance[init_var_name]
    if init_var_name not in used_vars:
      used_vars.add(init_var_name)
    else:
      # Allow the copy of cross-host Jax arrays.
      with jax.spmd_mode('allow_all'):
        loaded_var = jnp.copy(loaded_var)
    if isinstance(
        model_vars, NestedMap
    ):
      model_vars.Set(var_name, loaded_var)
    else:
      _set_nested_dict_value(model_vars, var_name, loaded_var)
    if isinstance(model_provenance, NestedMap):
      model_provenance.Set(var_name, loaded_var_provenance)
    else:
      _set_nested_dict_value(model_provenance, var_name, loaded_var_provenance)


def _make_train_state(
    rules: CheckpointLoadingRules,
    ckpt_train_state: TrainState,
    train_state_pspecs: TrainState,
    matched_pspecs: Dict[str, Any],
    load_ema_states: bool,
) -> Tuple[TrainState, TrainState]:
  """Makes changes to checkpoint train states for GDA ckpts."""
  # For GDA checkpoint type, skip loading step / opt states from the
  # checkpoint if rules are set to False. FLAX checkpoint type doesn't support
  # loading and assigning partial saved vars.
  if not rules.load_step:
    ckpt_train_state = ckpt_train_state.replace(step={})
    if train_state_pspecs is not None:
      train_state_pspecs = train_state_pspecs.replace(step={})
  if (
      not rules.load_opt_states
      and not rules.partial_load_opt_states
      and not load_ema_states
  ):
    ckpt_train_state = ckpt_train_state.replace(opt_states={})
    if train_state_pspecs is not None:
      train_state_pspecs = train_state_pspecs.replace(opt_states={})

  def is_masked(x):
    return py_utils.is_optax_masked_node(x) or py_utils.is_bprop_masked_node(x)

  def _filter_vars_and_get_pspecs(variables, must_include_for_ema=None):
    # must_include_for_ema is a mask indicating if the non-EMA var must be
    # included to be used as the EMA of a bprop-excluded var.
    prefix = py_utils.extract_prefixed_keys_from_nested_map(
        # extract_prefixed_keys_from_nested_map doesn't work with mask nodes.
        jax.tree_map(
            lambda x: True if is_masked(x) else x, variables, is_leaf=is_masked
        ),
        key_separator='.',
    )
    flatten_prefix, treedef = jax.tree_util.tree_flatten(prefix)
    flatten_variable, _ = jax.tree_util.tree_flatten(
        variables, is_leaf=is_masked
    )
    if must_include_for_ema is None:
      must_include = [False] * len(flatten_prefix)
    else:
      must_include, _ = jax.tree_util.tree_flatten(
          must_include_for_ema, is_leaf=is_masked
      )

    if len(flatten_prefix) != len(flatten_variable):
      raise ValueError('variables and its prefix has different length')

    for i in range(len(flatten_prefix)):
      k = flatten_prefix[i]
      if is_masked(flatten_variable[i]):
        assert not must_include[i]
        if k in matched_pspecs:
          # Preserve original type. If it's bprop-excluded in ema it will be
          # checked later.
          flatten_prefix[i] = flatten_variable[i]
        else:
          flatten_prefix[i] = flatten_variable[i] = optax.MaskedNode()
      elif k in matched_pspecs:
        flatten_prefix[i] = matched_pspecs[k]
      elif must_include[i]:
        flatten_prefix[i] = matched_pspecs['ema.' + k]
      else:
        flatten_prefix[i] = optax.MaskedNode()
        flatten_variable[i] = optax.MaskedNode()
    return jax.tree_util.tree_unflatten(
        treedef, flatten_variable
    ), jax.tree_util.tree_unflatten(treedef, flatten_prefix)

  # Tracking vars requested but missing from ema due to bprop exclusion.
  missing_in_ema = None
  # TODO(nanxinchen): move this to a helper function
  if load_ema_states:
    new_states = []
    new_states_pspecs = []
    vectorized = is_vectorized(ckpt_train_state)

    missing_in_ema = jax.tree_map(
        lambda _: True, ckpt_train_state.mdl_vars, is_leaf=is_masked
    )

    if not vectorized:
      for i, v in enumerate(ckpt_train_state.opt_states[0]):
        if 'ema' not in v:
          new_states.append(v)
          if train_state_pspecs is not None:
            new_states_pspecs.append(train_state_pspecs.opt_states[0][i])
        else:
          filtered_ema, ema_pspecs = _filter_vars_and_get_pspecs(v)
          # is_bprop_masked_node means matched but excluded.
          missing_in_ema = jax.tree_map(
              lambda x, y: x and py_utils.is_bprop_masked_node(y),
              missing_in_ema,
              filtered_ema['ema'],
          )
          v = (
              filtered_ema  # pytype: disable=unsupported-operands  # jax-ndarray
          )
          new_states.append(v)
          if train_state_pspecs is not None:
            new_states_pspecs.append(ema_pspecs)
      tuple_type = type(ckpt_train_state.opt_states[0])
      outer_tuple_type = type(ckpt_train_state.opt_states)
      new_states0 = outer_tuple_type([tuple_type(new_states)])
      ckpt_train_state.replace(
          opt_states=new_states0 + ckpt_train_state.opt_states[1:]
      )
      if train_state_pspecs is not None:
        new_states_pspecs0 = outer_tuple_type([tuple_type(new_states_pspecs)])
        train_state_pspecs.replace(
            opt_states=new_states_pspecs0 + train_state_pspecs.opt_states[1:]
        )
    else:

      new_states0 = ckpt_train_state.opt_states[0]
      new_states_pspecs0 = None
      if train_state_pspecs is not None:
        new_states_pspecs0 = train_state_pspecs.opt_states[0]

      for key, item in ckpt_train_state.opt_states[0].items():  # pytype: disable=attribute-error  # jax-ndarray
        if isinstance(item, tuple):
          # (dict, dict, dict, ...). One or more dicts contain an 'ema' key

          def update_for_ema(v, update_pspecs=False):
            if isinstance(v, dict) and 'ema' in v:
              filtered_vars, ema_pspecs = _filter_vars_and_get_pspecs(v)
              v = ema_pspecs if update_pspecs else filtered_vars
            return v

          new_states0[key] = tuple(update_for_ema(v) for v in item)  # pytype: disable=unsupported-operands  # jax-ndarray
          for v in new_states0[key]:
            if isinstance(v, dict) and 'ema' in v:
              # is_bprop_masked_node means matched but excluded.
              missing_in_ema = jax.tree_map(
                  lambda x, y: x and py_utils.is_bprop_masked_node(y),
                  missing_in_ema,
                  v['ema'],
                  is_leaf=is_masked,
              )
          if new_states_pspecs0 is not None:
            new_states_pspecs0[key] = tuple(  # pytype: disable=unsupported-operands  # jax-ndarray
                update_for_ema(v, update_pspecs=True)
                for v in new_states_pspecs0[key]
            )

      outer_tuple_type = type(ckpt_train_state.opt_states)
      ckpt_train_state.replace(
          opt_states=outer_tuple_type(
              new_states0,
          )
          + ckpt_train_state.opt_states[1:]
      )
      if train_state_pspecs is not None:
        train_state_pspecs.replace(
            opt_states=outer_tuple_type(
                [
                    new_states_pspecs0,
                ]
                + train_state_pspecs.opt_states[1:]
            )
        )

  filtered_vars, pspecs = _filter_vars_and_get_pspecs(
      ckpt_train_state.mdl_vars, missing_in_ema
  )
  ckpt_train_state = ckpt_train_state.replace(mdl_vars=filtered_vars)
  if train_state_pspecs is not None:
    train_state_pspecs = train_state_pspecs.replace(mdl_vars=pspecs)

  return ckpt_train_state, train_state_pspecs


def _load_partial_opt_states(
    train_state: TrainState,
    train_state_provenance: TrainStateProvenance,
    loaded_train_state: TrainState,
    loaded_state_provenance: TrainStateProvenance,
    loading_rules: Sequence[Tuple[re.Pattern[str], str]],
    ignore_rules: Optional[Sequence[re.Pattern[str]]],
    is_opt_states_initialized: Dict[str, str],
    ckpt_path: str,
) -> Tuple[TrainState, TrainStateProvenance]:
  """Loads optimizer state from given checkpoint based on specified rules."""
  opt_states_serialized = flax.serialization.to_state_dict(
      train_state.opt_states)
  opt_states_flat = _flatten_dict(opt_states_serialized)

  loaded_opt_states_flat = _flatten_dict(
      flax.serialization.to_state_dict(loaded_train_state.opt_states))
  opt_provenance_serialized = flax.serialization.to_state_dict(
      train_state_provenance.opt_states
  )
  loaded_opt_provenance_flat = _flatten_dict(
      flax.serialization.to_state_dict(loaded_state_provenance.opt_states)
  )

  opt_state_names = [x[0] for x in opt_states_flat]
  opt_state_mapping, _ = _get_var_mapping(
      opt_state_names,
      loading_rules,
      ignore_rules,
      is_opt_states_initialized,
      ckpt_path,
      kind='Opt State',
      safe_load=False,
      target_partition_specs=None)
  logging.info('opt_state_mapping: %r', opt_state_mapping)

  _assign_model_vars(
      opt_states_serialized,
      dict(loaded_opt_states_flat),
      opt_state_mapping,
      opt_provenance_serialized,
      dict(loaded_opt_provenance_flat),
  )

  restored_opt_states = flax.serialization.from_state_dict(
      train_state.opt_states, opt_states_serialized)
  train_state = train_state.replace(opt_states=restored_opt_states)

  restored_opt_provenance = flax.serialization.from_state_dict(
      train_state_provenance.opt_states, opt_provenance_serialized)
  train_state_provenance = train_state_provenance.replace(
      opt_states=restored_opt_provenance
  )
  # Verify that:
  # 1) the tree structure of the restored opt states match the original opt
  #    states structure.
  # 2) each leaf tensor node in the restored opt states has the same size as
  #    the corresponding leaf tensor node in the original opt states.
  updated_opt_states_flat = _flatten_dict(
      flax.serialization.to_state_dict(train_state.opt_states))
  assert len(opt_states_flat) == len(updated_opt_states_flat)
  for orig_item, updated_item in zip(opt_states_flat, updated_opt_states_flat):
    assert orig_item[0] == updated_item[0]
    assert orig_item[1].size == updated_item[1].size
  logging.info('Partially restored opt_state verified.')
  return train_state, train_state_provenance


# TODO(pax-dev): Move this function when `pmap_use_tensorstore` flag is deleted.
def restore_pmap_from_tensorstore(
    global_shapes,
    checkpoint_dir: epath.PathLike,
    step=None,
    global_mesh=None,
    checkpoint_type=CheckpointType.GDA,
    enforce_restore_shape_check: bool = False,
    tensorstore_use_ocdbt: bool = False,
    restore_transformations: Optional[dict[str, Any]] = None,
):
  """Restores pmap checkpoints from tensorstore.

  The model_states returned are of type `DeviceArray`, `GlobalDeviceArray` or
  `ShardedDeviceArray`.

  Args:
    global_shapes: Global shapes of the tensors to be restored.
    checkpoint_dir: Checkpoint directory where the tensorstore checkpoints are
      present.
    step: Step to restore checkpoint from.
    global_mesh: If set, use this mesh to restore the checkpoint (meaning that
      the checkpoint is restored as part of an init_checkpoint_rules() call for
      a pjit model) and return a Jax Array. If unset, use a dummy mesh and
      return a regular `DeviceArray` or `ShardedDeviceArray` to be used with
      pmap.
    checkpoint_type: The type of checkpoint to use.
    enforce_restore_shape_check: Raises an error if restore shapes do not match
      checkpoint shapes.
    tensorstore_use_ocdbt: Enables Tensorstore OCDBT format.

  Returns:
    Restored model states of type `DeviceArray`, `GlobalDeviceArray` or
    ShardedDeviceArray`.
  """
  if global_mesh is None:
    restore_global_mesh = jax.sharding.Mesh(
        np.array(jax.devices()), axis_names=('x',)
    )
  else:
    restore_global_mesh = global_mesh

  def _get_spec(shape):
    if shape.shape:
      return jax.sharding.PartitionSpec(None)
    else:
      return jax.sharding.PartitionSpec()

  fully_replicated_state_specs = jax.tree_map(_get_spec, global_shapes)
  with restore_global_mesh:
    fully_replicated_gda_model_states = checkpoints.restore_checkpoint(
        global_shapes,
        checkpoint_dir,
        global_mesh=restore_global_mesh,
        checkpoint_type=checkpoint_type,
        state_specs=fully_replicated_state_specs,
        step=step,
        enforce_restore_shape_check=enforce_restore_shape_check,
        tensorstore_use_ocdbt=tensorstore_use_ocdbt,
        restore_transformations=restore_transformations,
    )
  if global_mesh is not None:
    return fully_replicated_gda_model_states
  if checkpoint_type == CheckpointType.PERSISTENCE:
    return jax.tree_map(
        py_utils.convert_fully_replicated_array_to_pmap_array,
        fully_replicated_gda_model_states,
    )
  # model_states is jax.Array; we convert back to DA or jax.Array with
  # single device sharding for pmap.
  return jax.tree_map(
      lambda x: x.addressable_data(0), fully_replicated_gda_model_states
  )


class CheckpointLoadingRules(NamedTuple):
  """Utility class for representing how to read the checkpoint.

  This class is for describing the rules for initializing a model using an
  external checkpoint directory.

  Variables are addressed via variable names that is a dot-separated path of
  key strings of a nested `NestedMap`s. For regex-friendliness and backward
  compatibility with Lingvo, slash ('/') can also be used instead of dot ('.').

  For example, to initialize variables 'params/decoder/lm/...' with
  'params/lm/...' but to keep bias 'b' parameters randomly set,
  `CheckpointLoadingRules` should be defined as follows:
  ```
    CheckpointLoadingRules(
        task_p=task_p,
        load_rules=[(r'params/decoder/lm/(.*)$', 'params/lm/{}')]
        ignore_rules=[r'^.*/b$'])
  ```

  Typically instances of this class are used to define
  `TrainHParams.init_from_checkpoint_rules`, where we specify a mapping from
  a string (a checkpoint dir) to its corresponding CheckpointLoadingRules.

  When initializing an instance of TrainState using this, we iterate over this
  dict, in the same order as insertion. The iteration order matters because
  the **first** time an element matches the selection rule wins.

  We follow the following procedure to load `TrainState` with the `TrainState`s
  from the provided checkpoints:

  - `mdl_vars`: For each element in the flattened model variables, value from
    the first matching CheckpointLoadingRules will be used.
  - `step`: we take the value of the first checkpoint with `self.load_step`
    set; otherwise step is left untouched.
  - `opt_states`: we take the value of the first checkpoint with
    `self.load_opt_states` set; otherwise `opt_states` is left untouched.

  Note how `self.load_rules` and `self.ignore_rules` support a fine-grained
  control
  on the loading behavior of `mdl_vars`, but `step` and `opt_states` are always
  loaded atomically (all or nothing).

  Attributes:
    task_p: A Task config used for producing checkpoints to be loaded.
    load_rules: List of pairs of variable-name regex patterns and variable names
      in the external checkpoint.  For each item `(pattern, source)` in the
      list, if the model to be trained has a variable that matches to the regex
      `pattern`, the initial value of that variable is overwritten by a variable
      with name `source.format(*match_object.groups())` in the external
      checkpoint. In particular, the constant tuple LOAD_ALL is one rule that
      load everything as is.
    safe_load: Boolean controlling safety checks on whether all load_rules
      patterns are in fact used. Typos, or wrong variable names in the pattern
      result on those variables being initialized randomly when the intent was
      probably different. Turning safe_load = True checks that all patterns in
      the set of load_rules were used for matching model parameters, trainable
      or not. Warning: safe_load only applies to load_rules, not ignore_rules.
    ignore_rules: If the variable name matches with one of the regexs in the
      list, the checkpoint variables are not used even if the name matches with
      `load_rules`.
    step: Step specifier used when the directory name is provided as a
      checkpoint path.
    load_step: whether to load the step from this checkpoint.
    load_opt_states: whether to load opt_states (in its entirety) from this
      checkpoint.
    load_ema_states: whether to load EMA state.
    partial_load_opt_states: whether to enable experimental partial opt_states
      loading from this checkpoint.
    input_specs_provider_p: A BaseInputSpecsProvider config used to provide
      input specs information for the pre-trained model initialization.
  """
  task_p: pax_fiddle.Config[SingleTask]
  load_rules: Sequence[Tuple[RegexStr, str]]
  safe_load: bool = False
  ignore_rules: Optional[Sequence[RegexStr]] = None
  step: Optional[int] = None
  load_step: bool = False
  load_opt_states: bool = False
  load_ema_states: bool = True
  partial_load_opt_states: bool = False
  input_specs_provider_p: Optional[
      pax_fiddle.Config[base_input.BaseInputSpecsProvider]
  ] = None


def get_excluded_var_mask_for_grad_or_opt(
    var_weight_hparams: NestedJTensor,
    learner: learners_lib.Learner,
    mask_all_non_trainable: bool,
) -> NestedMap:
  """Returns whether each var should be excluded for grad/optimizer."""
  if learner.keep_optimizer_state_for_excluded_vars:
    return jax.tree_map(lambda _: False, var_weight_hparams)
  # Skip variables for gradients.
  if learner.bprop_variable_inclusion:
    assert not learner.bprop_variable_exclusion
    included_for_grad = py_utils.match_variable_names(
        var_weight_hparams, learner.bprop_variable_inclusion
    )
    excluded_for_grad = jax.tree_map(lambda x: not x, included_for_grad)
  else:
    excluded_for_grad = py_utils.match_variable_names(
        var_weight_hparams, learner.bprop_variable_exclusion
    )
  if mask_all_non_trainable:
    excluded_for_grad = jax.tree_util.tree_map(
        lambda x, e: base_layer.var_not_trainable(x) or e,
        var_weight_hparams,
        excluded_for_grad,
    )
  return excluded_for_grad


def get_excluded_var_mask_for_opt(
    var_weight_hparams: NestedJTensor,
    learner: learners_lib.Learner,
) -> NestedMap:
  """Returns whether each var should be excluded for optimizer."""
  return get_excluded_var_mask_for_grad_or_opt(
      var_weight_hparams, learner, learner.optimizer.ema_decay == 0.0
  )


def get_excluded_var_mask_for_grad(
    var_weight_hparams: NestedJTensor,
    learner: learners_lib.Learner,
) -> NestedMap:
  """Returns whether each var should be excluded for grad."""
  return get_excluded_var_mask_for_grad_or_opt(
      var_weight_hparams, learner, True
  )


def filter_vars_for_grad_or_opt(
    mdl_vars: NestedMap, excluded_for_grad: NestedMap
) -> NestedMap:
  """Filters out vars that should be excluded for grad or optimizer."""
  return jax.tree_map(
      lambda v, e: py_utils.BpropMaskedNode() if e else v,
      mdl_vars,
      excluded_for_grad,
  )


def create_state_partition_specs(
    var_weight_hparams: NestedJTensor,
    mesh_shape: Sequence[int],
    mesh_axis_names: Sequence[str],
    discard_opt_states: bool,
    learners: Optional[Sequence[learners_lib.Learner]],
    opt_states: Optional[optax.OptState] = None,
):
  """Creates partition specs for all variables used in training.

  Args:
    var_weight_hparams: a nested map of variable params for all the forward
      variables.
    mesh_shape: shape of the logical mesh.
    mesh_axis_names: axis names of each mesh axis.
    discard_opt_states: when true, optimizer slot variables are skipped.
    learners: learners of the optimizer. Cannot be None if discard_opt_states
      is false.

  Returns:
    A TrainState that contains PartitionSpecs.
  """

  step_partition_spec = PartitionSpec()
  var_partition_specs = base_layer.var_partition_specs(
      var_weight_hparams,
      mesh_shape=mesh_shape,
      device_axis_names=mesh_axis_names)
  if discard_opt_states:
    opt_var_partition_specs = []
  else:
    opt_var_weight_hparams = []
    index = 0
    for learner in learners:
      excluded = get_excluded_var_mask_for_opt(
          var_weight_hparams,
          learner,
      )
      var_weight_hparams_for_opt = filter_vars_for_grad_or_opt(
          var_weight_hparams, excluded
      )
      grad_tx = learner.get_grad_tx(var_weight_hparams_for_opt)
      assert isinstance(grad_tx, optimizers.ShardedGradientTransformation)
      if isinstance(grad_tx, optimizers.ShardedGradientTransformation):
        opt_var_weight_hparams.append(
            grad_tx.init_partition_spec(var_weight_hparams_for_opt)
        )
      elif isinstance(grad_tx, optax.GradientTransformationExtraArgs):
        opt_var_weight_hparams.append(
            optimizer_prefix_vectorization.partition_params(
                grad_tx, var_weight_hparams_for_opt, opt_states[index]
            )
        )
      index += 1

    opt_var_partition_specs = base_layer.var_partition_specs(
        opt_var_weight_hparams,
        mesh_shape=mesh_shape,
        device_axis_names=mesh_axis_names,
    )

    # Note that due to the double nesting of sharded chain we need to un-mask
    # the outer MaskedState() if present. If this is only a single optimizer
    # the tree_map() will go through with a no-op.
    def _is_instance_masked_state(x):
      return isinstance(x, optax.MaskedState)

    def _maybe_unmask_outer_masked_state(x):
      if _is_instance_masked_state(x):
        return x.inner_state
      return x

    opt_var_partition_specs = jax.tree_map(
        _maybe_unmask_outer_masked_state,
        opt_var_partition_specs,
        is_leaf=_is_instance_masked_state)
  return TrainState(
      step=step_partition_spec,
      mdl_vars=var_partition_specs,
      opt_states=opt_var_partition_specs,
      extra_state=(),
  )


def _create_opt_states(
    mdl_vars: NestedJTensor,
    var_weight_hparams: NestedJTensor,
    learners: Sequence[learners_lib.Learner],
) -> List[NestedJTensor]:
  """Creates opt_states by applying gradient transformations.

  Args:
    mdl_vars: A nested structure of model vars to in TrainState.
    var_weight_hparams: WeightHParams for each of the variable in mdl_vars.
      var_weight_hparams must be of the same structure as mdl_vars. Each model
      weight variable is associated with some WeightHParams which contains all
      the meta information about the weight variable.
    learners: learners of the optimizer.

  Returns:
    A list of NestedJTensor to update `opt_states` in TrainState.
  """
  asserts.assert_same_structure(mdl_vars, var_weight_hparams)
  opt_states = []
  for learner in learners:
    excluded = get_excluded_var_mask_for_opt(
        var_weight_hparams,
        learner,
    )
    var_weight_hparams = filter_vars_for_grad_or_opt(
        var_weight_hparams, excluded
    )
    filtered_mdl_vars = filter_vars_for_grad_or_opt(mdl_vars, excluded)
    grad_tx = learner.get_grad_tx(var_weight_hparams)
    opt_states.append(grad_tx.init(filtered_mdl_vars))
  return opt_states  # pytype: disable=bad-return-type


def create_state(
    mdl_vars: NestedJTensor,
    var_weight_hparams: NestedJTensor,
    discard_opt_states: bool,
    learners: Optional[Sequence[learners_lib.Learner]],
) -> TrainState:
  """Creates train states that holds all the forward/backward variables.

  Args:
    mdl_vars: A nested structure of model vars to create TrainState for.
      'mdl_vars' can be a sub-set of self.vars.
    var_weight_hparams: WeightHParams for each of the variable in mdl_vars.
      var_weight_hparams must be of the same structure as mdl_vars. Each model
      weight variable is associated with some WeightHParams which contains all
      the meta information about the weight variable.
    discard_opt_states: bool, When true, optimizer slot variables are skipped.
    learners: learners of the optimizer. Cannot be None if discard_opt_states is
      false.

  Returns:
    a TrainState.
  """
  # Make a private copy of mdl_vars and var_weight_hparams structures that are
  # not shared with the caller.
  mdl_vars = trees.copy(mdl_vars)
  var_weight_hparams = trees.copy(var_weight_hparams)
  if discard_opt_states:
    opt_states = []
  else:
    opt_states = _create_opt_states(mdl_vars, var_weight_hparams, learners)

  return TrainState(
      # The global step for the model.
      step=jnp.array(0, dtype=jnp.uint32),
      mdl_vars=mdl_vars,
      opt_states=opt_states,
      extra_state=(),
  )


def create_state_unpadded_shapes(
    var_weight_hparams: NestedJTensor,
    discard_opt_states: bool,
    learners: Optional[Sequence[learners_lib.Learner]],
) -> TrainState:
  """Creates shapes for all variables used in training without padding...

  due to uneven sharding.

  Args:
    var_weight_hparams: a nested map of variable params for all the forward
      variables.
    discard_opt_states: bool, When true, optimizer slot variables are skipped.
    learners: learners of the optimizer. Cannot be None if discard_opt_states is
      false.

  Returns:
    A TrainState contains jax.ShapeDtypeStruct for all the forward and
      backward variables.
  """

  def _get_shape(var_param):
    shape = tuple(var_param.repeat_prefix or ()) + tuple(var_param.shape)
    return jax.ShapeDtypeStruct(shape, var_param.dtype)

  var_shapes = jax.tree_map(_get_shape, var_weight_hparams)

  def _create_train_state_from_shape(mdl_vars):
    return create_state(
        mdl_vars, var_weight_hparams, discard_opt_states, learners
    )

  return jax.eval_shape(_create_train_state_from_shape, var_shapes)


def create_state_padded_shapes(
    var_weight_hparams: NestedJTensor,
    mesh_shape: Sequence[int],
    mesh_axis_names: Sequence[str],
    discard_opt_states: bool,
    learners: Optional[Sequence[learners_lib.Learner]],
) -> TrainState:
  """Creates shapes for all variables used in training with padding...

  due to uneven sharding.

  Args:
    var_weight_hparams: a nested map of variable params for all the forward
      variables.
    mesh_shape: shape of the logical mesh.
    mesh_axis_names: axis names of each mesh axis.
    discard_opt_states: bool, When true, optimizer slot variables are skipped.
    learners: learners of the optimizer. Cannot be None if discard_opt_states is
      false.

  Returns:
    A TrainState contains jax.ShapeDtypeStruct for all the forward and
      backward variables.
  """
  unpadded_shapes = create_state_unpadded_shapes(
      var_weight_hparams, discard_opt_states, learners
  )

  if mesh_shape is None:
    return unpadded_shapes

  model_state_partition_specs = create_state_partition_specs(
      var_weight_hparams,
      mesh_shape,
      mesh_axis_names,
      discard_opt_states,
      learners,
      unpadded_shapes.opt_states,
  )
  asserts.assert_same_structure(model_state_partition_specs, unpadded_shapes)

  def _maybe_pad(shape_dtype, pspec):
    if py_utils.is_optax_masked_node(shape_dtype):
      return shape_dtype
    unpadded_shape = shape_dtype.shape
    paddings = py_utils.get_uneven_sharding_paddings(
        pspec, unpadded_shape, mesh_shape, mesh_axis_names
    )
    padded_shape = [s + p for (s, p) in zip(unpadded_shape, paddings)]
    return jax.ShapeDtypeStruct(padded_shape, shape_dtype.dtype)

  padded_shapes = jax.tree_map(
      _maybe_pad,
      unpadded_shapes,
      model_state_partition_specs,
      is_leaf=py_utils.is_optax_masked_node,
  )
  return padded_shapes


class SingleTask(base_task.BaseTask):
  """A JAX task.

  Attributes:
    name: Name of this task object, must be a valid identifier.
    model: The underlying JAX model encapsulating all the layers.
    train: HParams to control how this task should be trained.
    decode: HParams to control how this task should be decoded.
    metrics: A BaseMetrics aggregator class to determine how metrics are
      computed.
    loss_aggregator: A LossAggregator aggregator class to derermine how the
      losses are aggregated (e.g single or MultiLoss)
    vn: HParams to control variational noise.
    infer_writer: specifies how to generate and write some output with a model
    early_stopping_fn: Function to control whether to stop the training loop
      early; the instantiated class should be callable with signature matching
      trainer_lib.EarlyStoppingFn.
    summary_verbosity: Summary verbosity to be used for logging summaries. The
      following are some notes on summary verbosity levels: * The larger the
      verbosity value, the more verbose. * The convention is to use non-negative
      integers. * The default verbosity level at the context level is 3, meaning
      that we'll log any summary written with verbosity <= 3 by default. *
      Summaries are written if context_verbosity >= callsite_verbosity.
  """

  @dataclasses.dataclass(frozen=True)
  class InferWriter:
    """Parameters for generating and writing outputs from a model.

    Attributes:
      restore_checkpoint_dir: The directory from which to restore checkpoint.
      restore_checkpoint_step: If set, the checkpoint step to restore. If unset,
        it will try to restore from the latest checkpoint, if any.
      inference_runner: an instance of BaseInferenceRunner config that defines
        how to run the model and the schema of the corresponding output.
      output_format: the io_utils.OutputFormatType which describes the container
        format to write to.
      output_num_shards: the number of shards for the output container.
    """
    restore_checkpoint_dir: str = ''
    restore_checkpoint_step: Optional[int] = None
    inference_runner: Optional[pax_fiddle.Config[BaseInferenceRunner]] = None
    output_format: io_utils.OutputFormatType = (
        io_utils.OutputFormatType.TFRECORD
    )
    output_num_shards: int = 32

  InferWriterHParams = base_hyperparams.FiddleHParamsClassStub(InferWriter)  # pylint: disable=invalid-name

  @dataclasses.dataclass(frozen=True)
  class VariationalNoise:
    """Parameters for variational noise.

    Attributes:
      vn_scale: The variational noise add to the network weights.
      vn_regex: The reg exp rule used to select the variables that require
        variational noise.
      vn_start_step: Step starting from which variational noise is added.
    """

    vn_scale: float = 0.0
    vn_regex: str = ''
    vn_start_step: int = 0

  VariationalNoiseHParams = base_hyperparams.FiddleHParamsClassStub(  # pylint: disable=invalid-name
      VariationalNoise
  )

  @dataclasses.dataclass(frozen=True)
  class Train:
    """Parameters for training.

    Attributes:
      learner: One or a list of learners.
      num_train_steps: Maximum number of training steps to run.
      save_interval_steps: How frequently to save a model checkpoint in terms of
        the number of training steps.
      save_keep_interval_duration: How frequently to keep a saved model
        checkpoint as a duration string such as `1h` for one hour or `90m` for
        90 minutes. This is performed in addition to keeping the most recent
        `max_to_keep` checkpoint files.
      save_max_to_keep: The maximum number of recent checkpoints to keep.
      max_inflight_steps: Maximum number of inflight train steps. If <= 0, no
        limit.
      summary_interval_steps: How frequently to generate summaries in terms of
        the number of training steps.
      log_train_output_interval_steps:  How frequently to log training output to
        the INFO stream. If set to None, use the same value as for
        `summary_interval_steps`.
      summary_accumulate_interval_steps: How frequently to accumulate summary
        values across steps before writing them to disk. If unset, no
        accumulation is performed and summaries will be written solely based on
        the current step's values.
      async_summary_writing: Whether to log loss and write summaries in a
        separate thread.
      variable_norm_summary: Whether to compute variable norm summaries.
      eval_interval_steps: How frequently to evaluate the model on the
        evaluation splits in terms of the number of training steps. Set to 0 to
        disable eval steps.
      eval_skip_train: By default, we also run eval on the training data input
        (`eval_train`), specifically on a batch not yet used for training. When
        set to True, this is skipped.
      eval_use_ema_states: If True, use ema states to run eval during train,
        note that in this case ema MUST be enabled in the learner.
      inputs_split_mapping: The PartitionSpec for inputs such as inputs, labels,
        targets, paddings, num words etc. This is only relevant for SPMD sharded
        models. By default it is None, which means all the inputs are
        replicated. For sharding inputs, this is a `NestedMap` with keys
        `map_1d`, `map_2d`, ..., etc., which specifies how to shard the inputs
        of that dimension.
      init_from_checkpoint_rules: A dict with str-valued keys corresponding to
        checkpoint dir paths and values corresponding to instances of
        `CheckpointLoadingRules`. See doc string on CheckpointLoadingRules on
        how these rules are interpreted.
      decode_interval_steps: How frequently to run decode on the model on the
        decoder_datasets() in terms of the number of training steps. Skipped if
        this value is not a positive int. Set to 0 to disable decode steps.
      decode_start_after_n_steps: Starts decoder after N steps, only used in
        continuous decoding.
      decode_use_ema_states: If True, use ema states to run decode during train,
        note that in this case ema MUST be enabled in the learner.
      profiler_num_steps: The number of steps to be captured by the profiler
        based on the step time estimate.
      profiler_min_duration_sec: The minimum duration to be captured by the
        profiler in seconds. This is used when the estimate step duration times
        profiler_num_steps is smaller than this value.
      profiler_capture_step: The step index at which to capture a code profile.
        No trace is captured if set to None.
      profiler_max_num_hosts: If set, limit profiling only on the specified
        number of hosts.
      always_use_train_for_model_init: Boolean indicating whether to use the new
        flow for model initialization. With this new flow, dedicated evaluation
        and decoding-only jobs rely on training inputs for model initialization.
      enforce_input_specs: Boolean indicating if the input specs check is
        performed at the first data batch, which raises exception if the input
        and data specs don't match.
      random_seed: Random seed to use at the beginning of the training.
      apply_mutable_list: A list of allowed collections to be mutated during
        train apply.
      tensorstore_metadata_key: The name applied to metadata files created by
        Tensorstore. Uses Tensorstore default if not specified.
      enable_input_checkpointing: Whether to checkpoint training input. Must be
        supported by the BaseInput implementation.
      restore_transformations: Orbax-style transformations. See Orbax
        documentation. `tensorstore_use_ocdbt` must be enabled. Note that some
        shape checking may be disabled when using this option. Use
        `enforce_restore_shape_check` to counteract this, though this may not
        necessarily be suitable for all cases, particularly when
        padding/truncating is involved.
      external_checkpoint_path: A path from which to restore an external
        checkpoint. The checkpoint is used for restoration if there are no
        checkpoints present in the main directory.
      external_checkpoint_handler: An ocp.CheckpointHandler defining logic for
        loading the checkpoint.
    """

    learner: pax_fiddle.Config[learners_lib.Learner] = (
        pax_fiddle.template_field(learners_lib.Learner)
    )
    num_train_steps: float = 1e7
    save_interval_steps: int = 5000
    save_keep_interval_duration: str = '12h'
    save_max_to_keep: int = 10
    max_inflight_steps: int = 2
    summary_interval_steps: int = 100
    log_train_output_interval_steps: Optional[int] = None
    summary_accumulate_interval_steps: Optional[int] = None
    async_summary_writing: bool = True
    variable_norm_summary: bool = True
    eval_interval_steps: int = 100
    eval_skip_train: bool = False
    eval_use_ema_states: bool = False
    inputs_split_mapping: Optional[PartitionSpec] = None
    init_from_checkpoint_rules: Dict[str, CheckpointLoadingRules] = (
        pax_fiddle.instance_field(default_factory=dict)
    )
    decode_interval_steps: Optional[int] = None
    decode_start_after_n_steps: int = 0
    # TODO(zhishuai): verify this for a pjit model.
    decode_use_ema_states: bool = False
    profiler_num_steps: int = 2
    profiler_min_duration_sec: float = 1.0
    profiler_capture_step: Optional[int] = None
    profiler_max_num_hosts: Optional[int] = None
    always_use_train_for_model_init: bool = True
    enforce_input_specs: bool = True
    random_seed: int = 1234
    apply_mutable_list: List[str] = pax_fiddle.instance_field(
        default_factory=lambda: TRAIN_DEFAULT_MUTABLE_LIST[:]
    )
    tensorstore_metadata_key: Optional[str] = None
    enable_input_checkpointing: Optional[bool] = False
    restore_transformations: Optional[Dict[str, Any]] = None
    external_checkpoint_path: Optional[epath.Path] = None
    external_checkpoint_handler: Optional[ocp.CheckpointHandler] = None

  TrainHParams = base_hyperparams.FiddleHParamsClassStub(Train)  # pylint: disable=invalid-name

  @dataclasses.dataclass(frozen=True)
  class Decode:
    """Parameters for decoding.

    Attributes:
      prng_key_fold_with_batch_index: if True, folds the decode prng key per
        decoding batch index.
      random_seed: Random seed to use at the beginning of the decoding.
      profiler_num_steps: The number of steps to be captured by the profiler
        based on the step time estimate.
      profiler_min_duration_sec: The minimum duration to be captured by the
        profiler in seconds. This is used when the estimate step duration times
        profiler_num_steps is smaller than this value.
      profiler_capture_step: The step index at which to capture a code profile.
      profiler_max_num_hosts: If set, limit profiling only on the specified
        number of hosts.
    """

    prng_key_fold_with_batch_index: bool = False
    random_seed: int = 1234
    profiler_num_steps: int = 0
    profiler_min_duration_sec: float = 1.0
    profiler_capture_step: int = 1
    profiler_max_num_hosts: Optional[int] = None

  DecodeHParams = base_hyperparams.FiddleHParamsClassStub(Decode)  # pylint: disable=invalid-name

  @dataclasses.dataclass(frozen=True)
  class Evaluate:
    """Parameters for evaluation.

    Attributes:
      random_seed: Random seed to use at the beginning of the evaluation.
      apply_mutable_list: A list of allowed collections to be mutated during
        evaluation apply.
    """

    random_seed: int = 1234
    apply_mutable_list: List[str] = pax_fiddle.instance_field(
        default_factory=lambda: EVAL_DEFAULT_MUTABLE_LIST[:]
    )

  EvaluateHParams = base_hyperparams.FiddleHParamsClassStub(Evaluate)  # pylint: disable=invalid-name

  @dataclasses.dataclass(frozen=True)
  class Infer:
    """Parameters for inference.

    Attributes:
      random_seed: Random seed to use at the beginning of the inference.
    """

    random_seed: int = 1234

  InferHParams = base_hyperparams.FiddleHParamsClassStub(Infer)  # pylint: disable=invalid-name

  @enum.unique
  class TrackDecoderMetricMode(str, enum.Enum):
    """Two different modes for tracking a metric: min or max."""

    MAX = 'max'
    MIN = 'min'

  model: base_model.BaseModel = None

  # Implementation note: `SingleTask` is not defined in the interpreter
  # context here, so we need to wrap it in a lambda which will look it up from
  # the global scope later.
  train: pax_fiddle.Config[SingleTask.Train] = pax_fiddle.template_field(Train)
  decode: pax_fiddle.Config[SingleTask.Decode] = pax_fiddle.template_field(
      Decode
  )
  evaluate: pax_fiddle.Config[SingleTask.Evaluate] = pax_fiddle.template_field(
      Evaluate
  )
  infer: pax_fiddle.Config[SingleTask.Infer] = pax_fiddle.template_field(Infer)

  metrics: Optional[pax_fiddle.Config[base_layer.BaseLayer]] = None
  loss_aggregator: Optional[pax_fiddle.Config[base_layer.BaseLayer]] = None
  vn: pax_fiddle.Config[SingleTask.VariationalNoise] = (
      pax_fiddle.template_field(VariationalNoise)
  )
  infer_writer: Optional[pax_fiddle.Config[SingleTask.InferWriter]] = None
  early_stopping_fn: Optional[EarlyStoppingFn] = None
  _learners: Any = dataclasses.field(init=False, repr=False)
  _metrics_aggregator: Any = dataclasses.field(init=False, repr=False)
  _loss_aggregator_inst: Any = dataclasses.field(init=False, repr=False)
  _inference_runner: Any = dataclasses.field(init=False, repr=False)
  summary_verbosity: int = 3

  def __post_init__(self):
    super().__post_init__()

    assert self.train.learner is not None
    # TODO(yonghui): implement multiple learners.
    assert not isinstance(self.train.learner, (tuple, list))
    learner_params = [self.train.learner]
    learner_params = NestedMap.FromNestedDict(learner_params)
    uid = itertools.count()

    def _build(
        p: pax_fiddle.Config[learners_lib.Learner],
    ) -> learners_lib.Learner:
      p = p.clone().set(name='learner_%d' % next(uid))
      return pax_fiddle.build(p)

    self._learners = NestedMap(sub=learner_params).Transform(_build).sub

    assert self.model is not None
    if isinstance(self.model, pax_fiddle.Config):
      self.model = instantiate(self.model)
    elif isinstance(self.model, base_layer.BaseLayer):
      self.model = self.model
    else:
      raise ValueError(
          'Expected `model` to be a BaseModel or a Config[BaseModel].')

    # instantiate the metrics aggregation helper
    if self.metrics:
      self._metrics_aggregator = instantiate(self.metrics)
    else:
      metrics_p = pax_fiddle.Config(base_metrics.MeanMetrics)
      self._metrics_aggregator = instantiate(metrics_p)

    # instantiate the loss aggregation helper
    if self.loss_aggregator:
      if any([learner.loss_name is not None for learner in self._learners]):
        raise ValueError('If a `loss_aggregator` is specified, all '
                         '`loss_names` on the learner are expected to be None.')
      self._loss_aggregator_inst = instantiate(self.loss_aggregator)
    else:
      if self._learners[0].loss_name is None:
        raise ValueError('`loss_name` on the learner is None. Must be set.')
      loss_p = pax_fiddle.Config(
          base_metrics.LossAggregator, loss_key=self._learners[0].loss_name
      )
      self._loss_aggregator_inst = instantiate(loss_p)

    if self.infer_writer:
      self._inference_runner = self.infer_writer.inference_runner.Instantiate(
          model=self.model
      )

  @property
  def learners(self) -> Sequence[learners_lib.Learner]:
    return self._learners

  @property
  def metrics_aggregator(self) -> base_metrics.MeanMetrics:
    return self._metrics_aggregator

  @property
  def loss_aggregator_inst(self) -> base_metrics.LossAggregator:
    return self._loss_aggregator_inst

  @property
  def has_ema_decay(self):
    return bool(self.learners[0].optimizer and
                self.learners[0].optimizer.ema_decay > 0)

  @property
  def inference_runner(self) -> BaseInferenceRunner:
    return self._inference_runner

  def create_opt_states(
      self, mdl_vars: NestedJTensor,
      var_weight_hparams: NestedJTensor) -> List[NestedJTensor]:
    """Creates opt_states by applying gradient transformations.

    Args:
      mdl_vars: A nested structure of model vars to in TrainState.
      var_weight_hparams: WeightHParams for each of the variable in mdl_vars.
        var_weight_hparams must be of the same structure as mdl_vars. Each model
        weight variable is associated with some WeightHParams which contains all
        the meta information about the weight variable.

    Returns:
      A list of NestedJTensor to update `opt_states` in TrainState.
    """
    return _create_opt_states(mdl_vars, var_weight_hparams, self.learners)

  def create_train_state(self,
                         mdl_vars: NestedJTensor,
                         var_weight_hparams: NestedJTensor,
                         discard_opt_states=False) -> TrainState:
    """Creates train states that holds all the forward/backward variables.

    Args:
      mdl_vars: A nested structure of model vars to create TrainState for.
        'mdl_vars' can be a sub-set of self.vars.
      var_weight_hparams: WeightHParams for each of the variable in mdl_vars.
        var_weight_hparams must be of the same structure as mdl_vars. Each model
        weight variable is associated with some WeightHParams which contains all
        the meta information about the weight variable.
      discard_opt_states: bool, When true, optimizer slot variables are skipped.

    Returns:
      a TrainState.
    """
    return create_state(
        mdl_vars, var_weight_hparams, discard_opt_states, self.learners
    )

  def create_train_state_padded_shapes(self,
                                       var_weight_hparams,
                                       discard_opt_states=False) -> TrainState:
    """Creates shapes for all variables used in training with padding...

    due to uneven sharding.

    Args:
      var_weight_hparams: a nested map of variable params for all the forward
        variables.
      discard_opt_states: bool, When true, optimizer slot variables are skipped.

    Returns:
      A TrainState contains jax.ShapeDtypeStruct for all the forward and
        backward variables.
    """
    mesh_shape = self.model.mesh_shape
    mesh_axis_names: Sequence[str] = self.model.mesh_axis_names  # pytype: disable=annotation-type-mismatch
    return create_state_padded_shapes(
        var_weight_hparams,
        mesh_shape,
        mesh_axis_names,
        discard_opt_states,
        self.learners,
    )

  def create_train_state_unpadded_shapes(self,
                                         var_weight_hparams,
                                         discard_opt_states=False
                                        ) -> TrainState:
    """Creates shapes for all variables used in training without padding...

    due to uneven sharding.

    Args:
      var_weight_hparams: a nested map of variable params for all the forward
        variables.
      discard_opt_states: bool, When true, optimizer slot variables are skipped.

    Returns:
      A TrainState contains jax.ShapeDtypeStruct for all the forward and
        backward variables.
    """

    return create_state_unpadded_shapes(
        var_weight_hparams, discard_opt_states, self.learners
    )

  def create_train_state_partition_specs(self,
                                         var_weight_hparams,
                                         discard_opt_states=False):
    """Creates partition specs for all variables used in training.

    Args:
      var_weight_hparams: a nested map of variable params for all the forward
        variables.
      discard_opt_states: bool, When true, optimizer slot variables are skipped.

    Returns:
      A TrainState contains PartitionSpecs for all the forward and/or backward
        variables depending on the value of is_eval, or None.
    """
    mesh_shape = self.model.mesh_shape
    if mesh_shape is None:
      return None
    mesh_axis_names = self.model.mesh_axis_names
    unpadded_shapes = self.create_train_state_unpadded_shapes(
        var_weight_hparams, discard_opt_states
    )

    return create_state_partition_specs(
        var_weight_hparams,
        mesh_shape,
        mesh_axis_names,
        discard_opt_states,
        self.learners,
        unpadded_shapes.opt_states,
    )

  def maybe_adjust_train_state(self, step: int, mdl_vars: Dict[
      str, JTensor], var_weight_hparams: Nested[base_layer.WeightHParams],
                               prng_key: PRNGKey) -> Dict[str, JTensor]:
    """Maybe adjust train state.

    This is the place to hook in various weight transformations. For example,
    one can override this function to hook in logic for adding variational noise
    on selected parts of the model (e.g. based on a regular expression of the
    path to model weights). As another example, one can implement quantized
    training by adding weight quantization logic here.

    The variational noise is implemented here to support all models. One can
    specify vn_scale > 0.0 to enable it and vn_regex to filter variables.

    Args:
      step: the current step.
      mdl_vars: the train params.
      var_weight_hparams: weight params for all forward variables.
      prng_key: A prng key to use for random number generations.

    Returns:
      mdl_vars with all parameters

    Raises:
      RuntimeError: if vn_scale > 0 but vn_regex doesn't match any variables.
    """
    if self.vn.vn_scale > 0.0:
      names = py_utils.extract_prefixed_keys_from_nested_map(var_weight_hparams)
      regexp = re.compile(self.vn.vn_regex)

      # This is the mask of variational noise
      # True: apply vn; False: without vn
      vn_mask = jax.tree_map(lambda x: bool(regexp.match(x) is not None), names)

      # Check if any variables match the rule
      # If vn_scale > 0.0 but none of variables match, throw error
      if not any(jax.tree_util.tree_leaves(vn_mask)):
        raise RuntimeError('Variational noise is enabled but rules don\'t '
                           'match any variables. Please disable vn by specify'
                           ' vn.vn_scale = 0. or check vn.vn_regex. One common'
                           ' issue is that it should start with params,'
                           ' i.e., decoder -> params/decoder.')
      else:
        logging.info('Variational noise applies to: %s', vn_mask)

      params_flat, params_def = jax.tree_util.tree_flatten(names)

      rng_flat = jax.random.split(prng_key, len(params_flat))
      rng_tree = jax.tree_util.tree_unflatten(params_def, rng_flat)

      def add_vn(params, rng, mask):
        if mask:
          return params + self.vn.vn_scale * jax.random.normal(
              shape=params.shape, key=rng
          )
        else:
          return params

      # VN only updates trainable part and copy non-trainable
      ret = jax.tree_map(add_vn, mdl_vars, rng_tree, vn_mask)
      return jax.tree_map(
          lambda x, y: jnp.where(step >= self.vn.vn_start_step, x, y),
          ret,
          mdl_vars,
      )
    else:
      return mdl_vars

  def _maybe_delete_unneeded_opt_vars(
      self,
      rules,
      opt_states,
      is_opt_states_initialized,
      loading_rules,
      ignore_rules,
      ckpt_path,
      loading_to_optax_opt_states,
  ):
    # pylint: disable=g-doc-args
    """Delete some opt vars before they get are loaded from the checkpoint.

    This reduces the maximum HBM usage in the beginning of the experiment.
    Otherwise, when the variables are being loadded, at the same time there
    could exist two copies of opt variables on the same device - the first copy
    initialized randomly, and the other one loaded from the checkpoint. Here the
    randomly initialized variables are released before checkpoint loading. This
    avoids OOMing in some large models.
    """
    # pylint: enable=g-doc-args
    if rules.partial_load_opt_states or rules.load_opt_states:
      opt_states_serialized = flax.serialization.to_state_dict(opt_states)
      opt_states_flat = _flatten_dict(opt_states_serialized)
      flattened_opt_vars = dict(opt_states_flat)
      opt_state_names = [x[0] for x in opt_states_flat]
      # Make a copy as we don't want to modify `is_opt_states_initialized`
      # which is going to be used later.
      is_opt_states_initialized_copy = copy.deepcopy(is_opt_states_initialized)
      if rules.partial_load_opt_states:
        # Delete matching vars from the checkpoint
        opt_state_mapping, _ = _get_var_mapping(
            opt_state_names,
            loading_rules,
            ignore_rules,
            is_opt_states_initialized_copy,
            ckpt_path,
            kind='Opt State',
            safe_load=False,
            target_partition_specs=None,
        )
      else:
        if loading_to_optax_opt_states:
          # For loading opt states from old checkpoints to new optax based
          # optimizers, we need to retain the other parameters.
          opt_state_mapping, _ = _get_var_mapping(
              opt_state_names,
              loading_rules,
              ignore_rules,
              is_opt_states_initialized_copy,
              ckpt_path,
              kind='Opt State',
              safe_load=False,
              target_partition_specs=None,
          )
        else:
          # Delete all opt vars
          opt_state_mapping = opt_state_names

      jax.block_until_ready(flattened_opt_vars)
      for k in opt_state_mapping:
        flattened_opt_vars[k].delete()

  def _apply_init_checkpoint_rule(
      self,
      train_state: TrainState,
      train_state_provenance: TrainStateProvenance,
      ckpt_path: str,
      rules: CheckpointLoadingRules,
      load_status: List[Any],
      global_mesh: Optional[jax.sharding.Mesh] = None,
      checkpoint_type: CheckpointType = CheckpointType.FLAX,
      target_partition_specs: Optional[TrainState] = None,
  ) -> Tuple[TrainState, TrainStateProvenance]:
    """Applies one CheckpointLoadingRules to train_state."""
    uses_gda = checkpoint_type in {
        CheckpointType.GDA,
        CheckpointType.PERSISTENCE,
    }
    if uses_gda:
      rules.task_p.model.ici_mesh_shape = self.model.ici_mesh_shape
      rules.task_p.model.dcn_mesh_shape = self.model.dcn_mesh_shape
      rules.task_p.model.mesh_axis_names = self.model.mesh_axis_names
    ckpt_task = typing.cast(SingleTask, instantiate(rules.task_p))
    is_step_loaded, is_var_initialized, is_opt_states_initialized = load_status
    model_vars = train_state.mdl_vars

    input_specs_provider_p = rules.input_specs_provider_p
    input_specs_provider = instantiate(input_specs_provider_p)
    inputs_shape_dtype = input_specs_provider.get_input_specs()
    # TODO(pax-dev): Add better/cleaner API to identify pmap vs. pjit models
    # (and check for dcn_mesh_shape too).
    if (hasattr(ckpt_task.model, 'ici_mesh_shape') and
        ckpt_task.model.ici_mesh_shape is not None):
      inputs_shape_dtype = jax.tree_map(py_utils.get_global_input_shape_dtype,
                                        inputs_shape_dtype)
    # Initialize with a dummy seed
    var_weight_hparams = ckpt_task.model.abstract_init_with_metadata(
        inputs_shape_dtype)
    ckpt_train_state = ckpt_task.create_train_state_padded_shapes(
        var_weight_hparams)
    train_state_pspecs = ckpt_task.create_train_state_partition_specs(
        var_weight_hparams)

    load_ema_states = (
        hasattr(rules.task_p, 'train')
        and rules.task_p.train.learner.optimizer.ema_decay > 0.0
    ) and rules.load_ema_states

    loading_rules = [
        (re.compile(pattern), ref) for pattern, ref in rules.load_rules
    ]
    ignore_rules = rules.ignore_rules if rules.ignore_rules is not None else []
    ignore_rules = [re.compile(pattern) for pattern in ignore_rules]
    flattened_model_vars = dict(model_vars.FlattenItems())  # pytype: disable=attribute-error
    # matched_pspecs: pspecs for the init checkpoint, inferred from model_vars.
    # model_vars_mapping: Mapping from names in model_vars to names in the init
    #                     checkpoint.
    model_vars_mapping, matched_pspecs = _get_var_mapping(
        list(flattened_model_vars.keys()),
        loading_rules,
        ignore_rules,
        is_var_initialized,
        ckpt_path,
        kind='Var',
        safe_load=rules.safe_load,
        target_partition_specs=target_partition_specs)

    # TODO(b/276310871): Avoid initialization of the variables in the first
    # place.
    # Free vars that will be replaced by the checkpoint to save device memory.
    jax.block_until_ready(flattened_model_vars)
    for k in model_vars_mapping:
      flattened_model_vars[k].delete()

    # When loading from old opt checkpoint with current praxis optimizer to
    # new optax basd optimizers, we need to apply loading rules to match the
    # optimizer states.
    # Below we check whether we are loading the checkpoint to optax based
    # optimizer using regex match. If yes, we want to prevent prematurely
    # deleting unneeded opt states.
    # We perform this check only for load_opt_states since for
    # partial_load_opt_states, only the loading rules from user are honored.
    loading_to_optax_opt = False
    if rules.load_opt_states and not rules.partial_load_opt_states:
      opt_states_serialized = flax.serialization.to_state_dict(
          train_state.opt_states
      )
      opt_states_flat = _flatten_dict(opt_states_serialized)
      for opt_state_name in opt_states_flat:
        var_name = opt_state_name[0].replace(
            '.', '/'
        )  # dot is reserved for regex
        for rule, _ in get_optax_opt_load_rules():
          if rule.match(var_name) is not None:
            loading_to_optax_opt = True
            break

    self._maybe_delete_unneeded_opt_vars(
        rules,
        train_state.opt_states,
        is_opt_states_initialized,
        loading_rules,
        ignore_rules,
        ckpt_path,
        loading_to_optax_opt,
    )

    if uses_gda:
      ckpt_train_state, train_state_pspecs = _make_train_state(
          rules,
          ckpt_train_state,
          train_state_pspecs,
          matched_pspecs,
          load_ema_states,
      )

    if (py_utils.pmap_use_tensorstore() and
        ckpt_task.model.ici_mesh_shape is None):
      assert checkpoint_type in {
          CheckpointType.GDA,
          CheckpointType.PERSISTENCE,
      }
      loaded_train_state = restore_pmap_from_tensorstore(
          ckpt_train_state,
          ckpt_path,
          step=rules.step,
          global_mesh=global_mesh,
          checkpoint_type=checkpoint_type)
    else:
      loaded_train_state = checkpoints.restore_checkpoint(
          ckpt_train_state,
          checkpoint_dir=ckpt_path,
          global_mesh=global_mesh,
          checkpoint_type=checkpoint_type,
          state_specs=train_state_pspecs,
          step=rules.step)

    # Use NestedMap's utility accessors
    loaded_vars = dict(NestedMap(loaded_train_state.mdl_vars).FlattenItems())
    loaded_state_provenance = train_states.build_train_state_provenance(
        loaded_train_state, ckpt_path, rules.step
    )
    provenance_model_vars = train_state_provenance.mdl_vars
    flat_loaded_vars_provenance = dict(
        NestedMap(loaded_state_provenance.mdl_vars).FlattenItems()
    )
    # Load EMA state if specified
    if load_ema_states:
      ema_required = False
      for _, ref in rules.load_rules:
        if ref.startswith('ema/'):
          ema_required = True

      if ema_required:
        loaded_vars.update(
            NestedMap.FromNestedDict(
                {'ema': extract_ema(loaded_train_state).mdl_vars}
            ).FlattenItems()
        )
        flat_loaded_vars_provenance.update(
            NestedMap.FromNestedDict(
                {'ema': extract_ema(loaded_train_state).mdl_vars}
            ).FlattenItems()
        )
    else:
      # Check if rules use ema state
      for _, ref in rules.load_rules:
        if ref.startswith('ema/'):
          raise RuntimeError('Load ema state but ema is not enabled for ckpt')

    _assign_model_vars(
        model_vars,
        loaded_vars,
        model_vars_mapping,
        provenance_model_vars,
        flat_loaded_vars_provenance,
    )
    train_state = train_state.replace(mdl_vars=model_vars)

    if rules.partial_load_opt_states:
      train_state, train_state_provenance = _load_partial_opt_states(
          train_state,
          train_state_provenance,
          loaded_train_state,
          loaded_state_provenance,
          loading_rules,
          ignore_rules,
          is_opt_states_initialized,
          ckpt_path,
      )

    if rules.load_step:
      if is_step_loaded:
        logging.info(
            'train_state.step is already initialized by %s, skip.',
            is_step_loaded,
        )
      else:
        loaded_step = loaded_train_state.step
        train_state = train_state.replace(step=loaded_step)
        train_state_provenance = train_state_provenance.replace(
            step=loaded_step
        )
        load_status[0] = ckpt_path
        logging.info(
            (
                'Initialization by external checkpoint: step is overwritten by '
                'value from %s with value %s'
            ),
            ckpt_path,
            train_state.step,
        )

    if rules.load_opt_states and not rules.partial_load_opt_states:
      if is_opt_states_initialized:
        logging.info(
            'train_state.opt_states is already initialized by %s, skip.',
            is_opt_states_initialized,
        )
      else:
        # Confirm that we are loading from loading from old opt checkpoint to
        # new optax based checkpoints. If yes, apply loading rules to match the
        # optimizer states.
        # We do not perform this automatic application of rules for
        # partial_load_opt_states, since the rules might conflict with user
        # provided rules. In case of partial_load_opt_states, expectation is for
        # the user to send the rules correctly for loading from old checkpoint
        # that uses praxis optimizers.
        loading_from_praxis_opt_state = False
        loaded_opt_states_flat = _flatten_dict(
            flax.serialization.to_state_dict(loaded_train_state.opt_states)
        )

        for opt_state_name in loaded_opt_states_flat:
          var_name = opt_state_name[0].replace(
              '.', '/'
          )  # dot is reserved for regex
          for rule in get_praxis_opt_state_regex():
            if rule.match(var_name) is not None:
              loading_from_praxis_opt_state = True
              break

        if loading_from_praxis_opt_state and loading_to_optax_opt:
          train_state, train_state_provenance = _load_partial_opt_states(
              train_state,
              train_state_provenance,
              loaded_train_state,
              loaded_state_provenance,
              get_optax_opt_load_rules(),
              ignore_rules,
              is_opt_states_initialized,
              ckpt_path,
          )

          logging.info(
              (
                  'Initialization by external checkpoint: '
                  'with train_state.opt_states replaced from praxis optimizer '
                  'to optax optimizer opt_state variables using checkpoint '
                  '%s'
              ),
              ckpt_path,
          )
        else:
          train_state = train_state.replace(
              opt_states=loaded_train_state.opt_states
          )
          train_state_provenance = train_state_provenance.replace(
              opt_states=loaded_state_provenance.opt_states
          )
          load_status[2] = {'all': ckpt_path}
          logging.info(
              (
                  'Initialization by external checkpoint:'
                  ' train_state.opt_states is overwritten by value from %s'
              ),
              ckpt_path,
          )

    return train_state, train_state_provenance

  def apply_init_checkpoint_rules(
      self,
      train_state: TrainState,
      train_state_provenance: TrainStateProvenance,
      train_state_partition_specs: Optional[TrainState] = None,
      global_mesh: Optional[jax.sharding.Mesh] = None,
      checkpoint_type: CheckpointType = CheckpointType.FLAX,
  ) -> Tuple[TrainState, TrainStateProvenance, bool]:
    """Applies self.train.init_from_checkpoint_rules to update train_state.

    Args:
      train_state: initialized train_state.
      train_state_provenance: initialized train_state provenance
      train_state_partition_specs: The TrainState specs for initialized
        train_state. Required for GDA-based checkpoints.
      global_mesh: optional mesh used to restore checkpoint if needed.
      checkpoint_type: used to restore checkpoint.

    Returns:
      A tuple of the updated new train state, the train state provenance, and
      whether caller needs
      to recompute opt_states after mdl_vars are updated.
    """
    all_rules = self.train.init_from_checkpoint_rules
    if not all_rules:
      return train_state, train_state_provenance, False

    # mdl_vars are Python dict. First, convert it to NestedMap for convenience.
    train_state = train_state.replace(
        mdl_vars=NestedMap.FromNestedDict(train_state.mdl_vars))
    # record which checkpoint initialized which var.
    is_var_initialized = dict()
    # record which checkpoint which opt state.
    is_opt_states_initialized = dict()
    # record which checkpoint loaded step.
    is_step_loaded = None

    load_status = [
        is_step_loaded, is_var_initialized, is_opt_states_initialized
    ]
    for ckpt_path, rules in all_rules.items():
      train_state, train_state_provenance = self._apply_init_checkpoint_rule(
          train_state,
          train_state_provenance,
          ckpt_path,
          rules,
          load_status,
          global_mesh,
          checkpoint_type,
          target_partition_specs=train_state_partition_specs,
      )

    # Convert mdl_vars back to Python dict for compatibility.
    train_state = train_state.replace(
        mdl_vars=train_state.mdl_vars.ToNestedDict())  # pytype: disable=attribute-error  # jax-ndarray
    return train_state, train_state_provenance, not load_status[2]
