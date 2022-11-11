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

"""Implementation of tasks.

Note: The module name is suffixed with `_lib` to avoid the name conflict with
the `tasks` Python submodule.
"""

from __future__ import annotations

import dataclasses
import enum
import itertools
import re
import typing
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

from absl import logging
from etils import epath
import flax
import jax
from jax import numpy as jnp
from jax.experimental import global_device_array
from jax.experimental import maps
from jax.experimental import pjit
import numpy as np
import optax
from paxml import base_inference_runner
from paxml import base_metrics
from paxml import base_task
from paxml import io_utils
from praxis import asserts
from praxis import base_hyperparams
from praxis import base_input
from praxis import base_layer
from praxis import base_model
from praxis import learners as learners_lib
from praxis import optimizers
from praxis import py_utils
from praxis import pytypes
from praxis import train_states

from paxml import checkpoints  # mapped to internal

BaseInferenceRunner = base_inference_runner.BaseInferenceRunner
CheckpointType = checkpoints.CheckpointType
Nested = pytypes.Nested
NestedMap = py_utils.NestedMap
NestedJTensor = base_layer.NestedJTensor
JTensor = base_layer.JTensor
PartitionSpec = pjit.PartitionSpec
TrainState = train_states.TrainState

PRNGKey = pytypes.PRNGKey
PyTreeDef = pytypes.PyTreeDef
sub_config_field = base_hyperparams.sub_config_field
RegexStr = str

instantiate = base_hyperparams.instantiate

# Shorthand for a loading rule that loads everything as is.
# e.g. load_rules = [LOAD_ALL]
LOAD_ALL = ('(.*)', '{}')


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
        logging.info(
            '%s Initialization by external checkpoint: '
            '%s doesn\'t match rule, skip.', kind, varname)
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


def _assign_model_vars(model_vars: Union[NestedMap, Dict[str, Any]],
                       loaded_vars: dict[str, Any],
                       model_vars_mapping: dict[str, str]) -> None:
  """Sets current model vars from loaded model vars using provided mapping."""
  used_vars = set()
  for var_name, init_var_name in model_vars_mapping.items():
    loaded_var = loaded_vars[init_var_name]
    if init_var_name not in used_vars:
      used_vars.add(init_var_name)
    # Copy the var if it's used more than once.
    elif isinstance(loaded_var, global_device_array.GlobalDeviceArray):
      loaded_var = py_utils.copy_gda(loaded_var)
    else:
      loaded_var = jnp.copy(loaded_var)
    if isinstance(model_vars, NestedMap):
      model_vars.Set(var_name, loaded_var)
    else:
      _set_nested_dict_value(model_vars, var_name, loaded_var)


def _make_gda_train_state(
    rules: CheckpointLoadingRules, ckpt_train_state: TrainState,
    train_state_pspecs: TrainState, matched_pspecs: Dict[str, Any],
    load_ema_state: bool) -> Tuple[TrainState, TrainState]:
  """Makes changes to checkpoint train states for GDA ckpts."""
  # For GDA checkpoint type, skip loading step / opt states from the
  # checkpoint if rules are set to False. FLAX checkpoint type doesn't support
  # loading and assigning partial saved vars.
  if not rules.load_step:
    ckpt_train_state = ckpt_train_state.replace(step={})
    if train_state_pspecs is not None:
      train_state_pspecs = train_state_pspecs.replace(step={})
  if (not rules.load_opt_states and not rules.partial_load_opt_states and
      not load_ema_state):
    ckpt_train_state = ckpt_train_state.replace(opt_states={})
    if train_state_pspecs is not None:
      train_state_pspecs = train_state_pspecs.replace(opt_states={})

  def _filter_vars_and_get_pspecs(variables):
    filtered_vars = []
    pspecs = []
    for k, v in variables.FlattenItems():
      if k in matched_pspecs:
        filtered_vars.append(v)
        pspecs.append(matched_pspecs[k])
      else:
        filtered_vars.append(())
        pspecs.append(())
    return variables.Pack(filtered_vars), variables.Pack(pspecs)

  filtered_vars, pspecs = _filter_vars_and_get_pspecs(
      NestedMap(ckpt_train_state.mdl_vars))
  ckpt_train_state = ckpt_train_state.replace(mdl_vars=filtered_vars)
  if train_state_pspecs is not None:
    train_state_pspecs = train_state_pspecs.replace(mdl_vars=pspecs)
  if load_ema_state:
    new_states = []
    new_states_pspecs = []
    # TODO(pax-dev): This doesn't work with prefix dims.
    for i, v in enumerate(ckpt_train_state.opt_states[0]):
      if 'ema' not in v:
        new_states.append(v)
        if train_state_pspecs is not None:
          new_states_pspecs.append(train_state_pspecs.opt_states[0][i])
      else:
        v = NestedMap.FromNestedDict(v)
        filtered_ema, ema_pspecs = _filter_vars_and_get_pspecs(v.ema)
        v.ema = filtered_ema
        new_states.append(v)
        if train_state_pspecs is not None:
          v_pspecs = NestedMap.FromNestedDict(
              train_state_pspecs.opt_states[0][i])
          v_pspecs.ema = ema_pspecs
          new_states_pspecs.append(v_pspecs)
    tuple_type = type(ckpt_train_state.opt_states[0])
    outer_tuple_type = type(ckpt_train_state.opt_states)
    new_states0 = outer_tuple_type([tuple_type(new_states)])
    ckpt_train_state.replace(opt_states=new_states0 +
                             ckpt_train_state.opt_states[1:])
    if train_state_pspecs is not None:
      new_states_pspecs0 = outer_tuple_type([tuple_type(new_states_pspecs)])
      train_state_pspecs.replace(opt_states=new_states_pspecs0 +
                                 train_state_pspecs.opt_states[1:])
  return ckpt_train_state, train_state_pspecs


def _load_partial_opt_states(train_state: TrainState,
                             loaded_train_state: TrainState,
                             loading_rules: Sequence[Tuple[re.Pattern[str],
                                                           str]],
                             ignore_rules: Optional[Sequence[re.Pattern[str]]],
                             is_opt_states_initialized: Dict[str, str],
                             ckpt_path: str) -> TrainState:
  """Loads optimizer state from given checkpoint based on specified rules."""
  opt_states_serialized = flax.serialization.to_state_dict(
      train_state.opt_states)
  opt_states_flat = _flatten_dict(opt_states_serialized)

  loaded_opt_states_flat = _flatten_dict(
      flax.serialization.to_state_dict(loaded_train_state.opt_states))

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

  _assign_model_vars(opt_states_serialized, dict(loaded_opt_states_flat),
                     opt_state_mapping)

  restored_opt_states = flax.serialization.from_state_dict(
      train_state.opt_states, opt_states_serialized)
  train_state = train_state.replace(opt_states=restored_opt_states)

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
  return train_state


# TODO(pax-dev): Move this function when `pmap_use_tensorstore` flag is deleted.
def restore_pmap_from_tensorstore(
    global_shapes,
    checkpoint_dir: epath.PathLike,
    step=None,
    global_mesh=None,
    checkpoint_type=CheckpointType.CHECKPOINT_GDA):
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
       a pjit model) and return a GDA. If unset, use a dummy mesh and return
       a regular `DeviceArray` or `ShardedDeviceArray` to be used with pmap.
    checkpoint_type: The type of checkpoint to use.

  Returns:
    Restored model states of type `DeviceArray`, `GlobalDeviceArray` or
    ShardedDeviceArray`.
  """
  if global_mesh is None:
    restore_global_mesh = maps.Mesh(np.array(jax.devices()), axis_names=('x',))
  else:
    restore_global_mesh = global_mesh

  def _get_spec(shape):
    if shape.shape:
      return pjit.PartitionSpec(None)
    else:
      return pjit.PartitionSpec()

  fully_replicated_state_specs = jax.tree_map(_get_spec, global_shapes)
  with restore_global_mesh:
    fully_replicated_gda_model_states = checkpoints.restore_checkpoint(
        global_shapes,
        checkpoint_dir,
        global_mesh=restore_global_mesh,
        checkpoint_type=checkpoint_type,
        state_specs=fully_replicated_state_specs,
        step=step)
  if global_mesh is not None:
    return fully_replicated_gda_model_states
  if checkpoint_type == CheckpointType.CHECKPOINT_PERSISTENCE:
    if jax.config.jax_array:
      fully_replicated_array_model_states = jax.tree_map(
          py_utils.convert_fully_replicated_array_to_pmap_array,
          fully_replicated_gda_model_states)
      return fully_replicated_array_model_states
    else:
      fully_replicated_sda_model_states = jax.tree_map(
          py_utils.convert_fully_replicated_gda_to_sda,
          fully_replicated_gda_model_states)
    return fully_replicated_sda_model_states
  # model_states is GDA or jax.Array; we convert back to DA or jax.Array with
  # single device sharding for pmap.
  if jax.config.jax_array:
    model_states = jax.tree_map(lambda x: x.addressable_data(0),
                                fully_replicated_gda_model_states)
  else:
    model_states = jax.tree_map(lambda x: x.addressable_data(0),
                                fully_replicated_gda_model_states)
  return model_states


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
  - `step`: we take the value of the first checkpoint with `p.load_step`
    set; otherwise step is left untouched.
  - `opt_states`: we take the value of the first checkpoint with
    `p.load_opt_states` set; otherwise `opt_states` is left untouched.

  Note how `p.load_rules` and `p.ignore_rules` support a fine-grained control
  on the loading behavior of `mdl_vars`, but `step` and `opt_states` are always
  loaded atomically (all or nothing).

  Attributes:
    task_p: An `Task.HParams` used for producing checkpoints to be loaded.
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
    partial_load_opt_states: whether to enable experimental partial opt_states
      loading from this checkpoint.
    input_specs_provider_p: A `BaseInputSpecsProvider.HParams` used to provide
      input specs information for the pre-trained model initialization.
  """
  task_p: SingleTask.HParams
  load_rules: Sequence[Tuple[RegexStr, str]]
  safe_load: bool = False
  ignore_rules: Optional[Sequence[RegexStr]] = None
  step: Optional[int] = None
  load_step: bool = False
  load_opt_states: bool = False
  partial_load_opt_states: bool = False
  input_specs_provider_p: Optional[
      base_input.BaseInputSpecsProvider.HParams] = None


class SingleTask(base_task.BaseTask):
  """A JAX task."""

  class InferWriterHParams(base_hyperparams.BaseHyperParams):
    """Parameters for generating and writing outputs from a model.

    Attributes:
      restore_checkpoint_dir: The directory from which to restore checkpoint.
      restore_checkpoint_step: If set, the checkpoint step to restore. If unset,
        it will try to restore from the latest checkpoint, if any.
      inference_runner: an instance of BaseInferenceRunner.HParams that defines
        how to run the model and the schema of the corresponding output.
      output_format: the io_utils.OutputFormatType which describes the container
        format to write to.
      output_num_shards: the number of shards for the output container.
    """
    restore_checkpoint_dir: str = ''
    restore_checkpoint_step: Optional[int] = None
    inference_runner: Optional[BaseInferenceRunner.HParams] = None
    output_format: io_utils.OutputFormatType = (
        io_utils.OutputFormatType.TFRECORD)
    output_num_shards: int = 32

  class VariationalNoiseHParams(base_hyperparams.BaseHyperParams):
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

  class TrainHParams(base_hyperparams.BaseHyperParams):
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
      summary_interval_steps: How frequently to generate summaries in terms of
        the number of training steps.
      device_sync_interval_steps: How many train steps to dispatch before
        explicit device sync. If set, log loss and write summaries in a separate
        thread.
      log_train_output_interval_steps:  How frequently to log training output to
        the INFO stream. If set to None, use the same value as for
        `summary_interval_steps`.
      summary_accumulate_interval_steps: How frequently to accumulate summary
        values across steps before writing them to disk. If unset, no
        accumulation is performed and summaries will be written solely based on
        the current step's values.
      variable_norm_summary: Whether to compute variable norm summaries.
      eval_interval_steps: How frequently to evaluate the model on the
        evaluation splits in terms of the number of training steps. Set to 0 to
        disable eval steps.
      eval_skip_train: By default, we also run eval on the training data input
        (`eval_train`), specifically on a batch not yet used for training. When
        set to True, this is skipped.
      inputs_split_mapping: The PartitionSpec for inputssuch as inputs, labels,
        targets, paddings, num words etc. This is onlyrelevant for SPMD sharded
        models. By default it is None, which meansall the inputs are replicated.
        For sharding inputs, this is a `NestedMap` with keys `map_1d`, `map_2d`,
        ..., etc.,which specifies how to shard the inputs of that dimension.
      init_from_checkpoint_rules: A dict with str-valued keys corresponding to
        checkpoint dir paths and values corresponding to instances of
        `CheckpointLoadingRules`. See doc string on CheckpointLoadingRules on
        how these rules are interpreted.
      decode_interval_steps: How frequently to run decode on the model on the
        decoder_datasets() in terms of the number of training steps. Skipped if
        this value is not a positive int. Set to 0 to disable decode steps.
      decode_start_after_n_steps: Starts decoder after N steps, only used in
        continuous decoding.
      decode_use_ema_state: If True, use ema states to run decode during train,
        note that in this case ema MUST be enabled in the learner.
      profiler_num_steps: The number of steps to be captured by the profiler
        based on the step time estimate.
      profiler_min_duration_sec: The minimum duration to be captured by the
        profiler in seconds. This is used when the estimate step duration times
        profiler_num_steps is smaller than this value.
      profiler_capture_step: The step index at which to capture a code profile.
        No trace is captured if set to None.
      always_use_train_for_model_init: Boolean indicating whether to use the new
        flow for model initialization. With this new flow, dedicated evaluation
        and decoding-only jobs rely on training inputs for model initialization.
    """
    learner: learners_lib.Learner.HParams = sub_config_field(
        learners_lib.Learner.HParams)
    num_train_steps: float = 1e7
    save_interval_steps: int = 5000
    save_keep_interval_duration: str = '12h'
    save_max_to_keep: int = 10
    summary_interval_steps: int = 100
    device_sync_interval_steps: Optional[int] = None
    log_train_output_interval_steps: Optional[int] = None
    summary_accumulate_interval_steps: Optional[int] = None
    variable_norm_summary: bool = True
    eval_interval_steps: int = 100
    eval_skip_train: bool = False
    inputs_split_mapping: Optional[PartitionSpec] = None
    init_from_checkpoint_rules: Dict[
        str, CheckpointLoadingRules] = dataclasses.field(default_factory=dict)
    decode_interval_steps: Optional[int] = None
    decode_start_after_n_steps: int = 0
    # TODO(zhishuai): verify this for a pjit model.
    decode_use_ema_state: bool = False
    profiler_num_steps: int = 2
    profiler_min_duration_sec: float = 1.
    profiler_capture_step: Optional[int] = None
    always_use_train_for_model_init: bool = True

  @enum.unique
  class TrackDecoderMetricMode(str, enum.Enum):
    """Two different modes for tracking a metric: min or max."""
    MAX = 'max'
    MIN = 'min'

  class HParams(base_task.BaseTask.HParams):
    """Task parameters.

    Attributes:
      name: Name of this task object, must be a valid identifier.
      model: The underlying JAX model encapsulating all the layers.
      train: HParams to control how this task should be trained.
      metrics: A BaseMetrics aggregator class to determine how metrics are
        computed.
      loss_aggregator: A LossAggregator aggregator class to derermine how the
        losses are aggregated (e.g single or MultiLoss)
      vn: HParams to control variational noise.
      track_decoder_metric: which decoding metric to track, e.g. 'wer'.
      track_decoder_metric_min_or_max: track min or max metric value.
      infer_writer: specifies how to generate and write some output with a model
      fold_decode_prng_key_per_batch: if True, folds the decode prng key per
        decoding batch index. Only effective for pmap decoding.
    """
    # TODO(b/249483164) Change this to just `= sub_config_field(None)` after
    # Fiddle migration is complete.
    model: Optional[base_model.BaseModel.HParams] = (
        None if issubclass(base_model.BaseModel, base_layer.BaseLayer) else
        sub_config_field(None))

    # Implementation note: `SingleTask` is not defined in the interpreter
    # context here, so we need to wrap it in a lambda which will look it up from
    # the global scope later.
    train: SingleTask.TrainHParams = sub_config_field(
        lazy_ref=lambda: SingleTask.TrainHParams)
    metrics: Any = None
    loss_aggregator: Any = None
    vn: SingleTask.VariationalNoiseHParams = sub_config_field(
        lazy_ref=lambda: SingleTask.VariationalNoiseHParams)
    track_decoder_metric: Optional[str] = None
    track_decoder_metric_min_or_max: Optional[
        SingleTask.TrackDecoderMetricMode] = None
    infer_writer: Optional[SingleTask.InferWriterHParams] = None
    fold_decode_prng_key_per_batch: bool = False

  def __init__(self, hparams: SingleTask.HParams) -> None:
    super().__init__(hparams)
    p = self.hparams

    assert p.train.learner is not None
    # TODO(yonghui): implement multiple learners.
    assert not isinstance(p.train.learner, (tuple, list))
    learner_params = [p.train.learner]
    learner_params = NestedMap.FromNestedDict(learner_params)
    uid = itertools.count()

    def _instantiate(p: learners_lib.Learner.HParams) -> learners_lib.Learner:
      p = p.clone().set(name='learner_%d' % next(uid))
      return instantiate(p)

    self._learners = NestedMap(sub=learner_params).Transform(_instantiate).sub

    assert p.model is not None
    self._model: base_model.BaseModel = instantiate(p.model)

    # instantiate the metrics aggregation helper
    if p.metrics:
      self._metrics_aggregator = instantiate(p.metrics)
    else:
      metrics_p = base_metrics.MeanMetrics.HParams()
      self._metrics_aggregator = instantiate(metrics_p)

    # instantiate the loss aggregation helper
    if p.loss_aggregator:
      if any([learner.loss_name is not None for learner in self._learners]):
        raise ValueError('If a `loss_aggregator` is specified, all '
                         '`loss_names` on the learner are expected to be None.')
      self._loss_aggregator = instantiate(p.loss_aggregator)
    else:
      if self._learners[0].loss_name is None:
        raise ValueError('`loss_name` on the learner is None. Must be set.')
      loss_p = base_metrics.LossAggregator.HParams(
          loss_key=self._learners[0].loss_name)
      self._loss_aggregator = instantiate(loss_p)

    if p.infer_writer:
      self._inference_runner = p.infer_writer.inference_runner.Instantiate(
          model=self._model)

  @property
  def learners(self) -> Sequence[learners_lib.Learner]:
    return self._learners

  @property
  def model(self) -> base_model.BaseModel:
    return self._model

  @property
  def metrics_aggregator(self) -> base_metrics.MeanMetrics:
    return self._metrics_aggregator

  @property
  def loss_aggregator(self) -> base_metrics.LossAggregator:
    return self._loss_aggregator

  @property
  def has_ema_decay(self):
    return bool(self.learners[0].hparams.optimizer and
                self.learners[0].hparams.optimizer.ema_decay > 0)

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
    grad_txs = [x.get_grad_tx(var_weight_hparams) for x in self.learners]
    asserts.assert_same_structure(mdl_vars, var_weight_hparams)
    return [x.init(mdl_vars) for x in grad_txs]

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
    # Make a private copy of mdl_vars and var_weight_hparams structures that are
    # not shared with the caller.
    mdl_vars = jax.tree_util.tree_map(lambda x: x, mdl_vars)
    var_weight_hparams = jax.tree_util.tree_map(lambda x: x, var_weight_hparams)
    if discard_opt_states:
      opt_states = {}
    else:
      opt_states = self.create_opt_states(mdl_vars, var_weight_hparams)

    return TrainState(
        # The global step for the model.
        step=jnp.array(0, dtype=jnp.uint32),
        mdl_vars=mdl_vars,
        opt_states=opt_states)

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
    unpadded_shapes = self.create_train_state_unpadded_shapes(
        var_weight_hparams, discard_opt_states)
    mesh_shape = self.hparams.model.mesh_shape
    mesh_axis_names = self.hparams.model.mesh_axis_names
    if mesh_shape is None:
      return unpadded_shapes

    model_state_partition_specs = self.create_train_state_partition_specs(
        var_weight_hparams, discard_opt_states)
    asserts.assert_same_structure(model_state_partition_specs, unpadded_shapes)

    def _maybe_pad(shape_dtype, pspec):
      if py_utils.is_optax_masked_node(shape_dtype):
        return shape_dtype
      unpadded_shape = shape_dtype.shape
      paddings = py_utils.get_uneven_sharding_paddings(pspec, unpadded_shape,
                                                       mesh_shape,
                                                       mesh_axis_names)
      padded_shape = [s + p for (s, p) in zip(unpadded_shape, paddings)]
      return jax.ShapeDtypeStruct(padded_shape, shape_dtype.dtype)

    padded_shapes = jax.tree_map(
        _maybe_pad,
        unpadded_shapes,
        model_state_partition_specs,
        is_leaf=py_utils.is_optax_masked_node)
    return padded_shapes

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

    def _get_shape(var_param):
      shape = tuple(var_param.repeat_prefix or ()) + tuple(var_param.shape)
      return jax.ShapeDtypeStruct(shape, var_param.dtype)

    var_shapes = jax.tree_map(_get_shape, var_weight_hparams)

    def _create_train_state_from_shape(mdl_vars):
      return self.create_train_state(mdl_vars, var_weight_hparams,
                                     discard_opt_states)

    return jax.eval_shape(_create_train_state_from_shape, var_shapes)

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
    p = self.hparams
    mesh_shape = p.model.mesh_shape
    if mesh_shape is None:
      return None
    step_partition_spec = PartitionSpec()
    var_partition_specs = base_layer.var_partition_specs(
        var_weight_hparams,
        mesh_shape=mesh_shape,
        device_axis_names=p.model.mesh_axis_names)
    if discard_opt_states:
      opt_var_partition_specs = {}
    else:
      grad_txs = [x.get_grad_tx(var_weight_hparams) for x in self.learners]
      opt_var_weight_hparams = []
      for grad_tx in grad_txs:
        assert isinstance(grad_tx, optimizers.ShardedGradientTransformation)
        opt_var_weight_hparams.append(
            grad_tx.init_partition_spec(var_weight_hparams))
      opt_var_partition_specs = base_layer.var_partition_specs(
          opt_var_weight_hparams,
          mesh_shape=mesh_shape,
          device_axis_names=p.model.mesh_axis_names)

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
        opt_states=opt_var_partition_specs)

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
    p = self.hparams

    if p.vn.vn_scale > 0.0:
      names = py_utils.extract_prefixed_keys_from_nested_map(var_weight_hparams)
      regexp = re.compile(p.vn.vn_regex)

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

      # VN only updates trainable part and copy non-trainable
      ret = mdl_vars.copy()

      def add_vn(params, rng, mask):
        if mask:
          return params + p.vn.vn_scale * jax.random.normal(
              shape=params.shape, key=rng)
        else:
          return params

      ret = jax.tree_map(add_vn, mdl_vars, rng_tree, vn_mask)
      return jax.tree_map(
          lambda x, y: jnp.where(step >= p.vn.vn_start_step, x, y), ret,
          mdl_vars)
    else:
      return mdl_vars

  def _apply_init_checkpoint_rule(
      self,
      train_state: TrainState,
      ckpt_path: str,
      rules: CheckpointLoadingRules,
      load_status: List[Any],
      global_mesh: Optional[maps.Mesh] = None,
      checkpoint_type: CheckpointType = CheckpointType.CHECKPOINT_FLAX,
      target_partition_specs: Optional[TrainState] = None):
    """Applies one CheckpointLoadingRules to train_state."""
    uses_gda = (
        checkpoint_type == CheckpointType.CHECKPOINT_GDA or
        checkpoint_type == CheckpointType.CHECKPOINT_PERSISTENCE)
    if uses_gda:
      rules.task_p.model.ici_mesh_shape = self.model.hparams.ici_mesh_shape
      rules.task_p.model.dcn_mesh_shape = self.model.hparams.dcn_mesh_shape
      rules.task_p.model.mesh_axis_names = self.model.hparams.mesh_axis_names
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

    load_ema_state = (
        hasattr(rules.task_p, 'train') and
        rules.task_p.train.learner.optimizer.ema_decay > 0.0)

    loading_rules = [
        (re.compile(pattern), ref) for pattern, ref in rules.load_rules
    ]
    ignore_rules = rules.ignore_rules if rules.ignore_rules is not None else []
    ignore_rules = [re.compile(pattern) for pattern in ignore_rules]
    var_names = [x[0] for x in model_vars.FlattenItems()]
    # matched_pspecs: pspecs for the init checkpoint, inferred from model_vars.
    # model_vars_mapping: Mapping from names in model_vars to names in the init
    #                     checkpoint.
    model_vars_mapping, matched_pspecs = _get_var_mapping(
        var_names,
        loading_rules,
        ignore_rules,
        is_var_initialized,
        ckpt_path,
        kind='Var',
        safe_load=rules.safe_load,
        target_partition_specs=target_partition_specs)

    if uses_gda:
      ckpt_train_state, train_state_pspecs = _make_gda_train_state(
          rules, ckpt_train_state, train_state_pspecs, matched_pspecs,
          load_ema_state)

    if (py_utils.pmap_use_tensorstore() and
        ckpt_task.model.hparams.ici_mesh_shape is None):
      assert (checkpoint_type in {
          CheckpointType.CHECKPOINT_GDA, CheckpointType.CHECKPOINT_PERSISTENCE
      })
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

    if loaded_train_state is None:
      raise RuntimeError(f'Cannot find checkpoint from {ckpt_path}')
    # Use NestedMap's utility accessors
    loaded_vars = dict(NestedMap(loaded_train_state.mdl_vars).FlattenItems())

    # Load EMA state if specified
    if load_ema_state:
      # TODO(pax-dev): This doesn't work with prefix dims.
      for v in loaded_train_state.opt_states[0]:
        if 'ema' in v:
          loaded_vars.update(
              NestedMap.FromNestedDict({
                  'ema': v.ema
              }).FlattenItems())
    else:
      # Check if rules use ema state
      for _, ref in rules.load_rules:
        if ref.startswith('ema/'):
          raise RuntimeError('Load ema state but ema is not enabled for ckpt')

    _assign_model_vars(model_vars, loaded_vars, model_vars_mapping)
    train_state = train_state.replace(mdl_vars=model_vars)

    if rules.partial_load_opt_states:
      train_state = _load_partial_opt_states(train_state, loaded_train_state,
                                             loading_rules, ignore_rules,
                                             is_opt_states_initialized,
                                             ckpt_path)

    if rules.load_step:
      if is_step_loaded:
        logging.info('train_state.step is already initialized by %s, skip.',
                     is_step_loaded)
      else:
        train_state = train_state.replace(step=loaded_train_state.step)
        load_status[0] = ckpt_path
        logging.info(
            'Initialization by external checkpoint: step is overwritten by '
            'value from %s with value %s', ckpt_path, train_state.step)

    if rules.load_opt_states and not rules.partial_load_opt_states:
      if is_opt_states_initialized:
        logging.info(
            'train_state.opt_states is already initialized by %s, skip.',
            is_opt_states_initialized)
      else:
        train_state = train_state.replace(
            opt_states=loaded_train_state.opt_states)
        is_opt_states_initialized = {'all': ckpt_path}
        logging.info(
            'Initialization by external checkpoint: train_state.opt_states is '
            'overwritten by value from %s', ckpt_path)

    return train_state

  def apply_init_checkpoint_rules(
      self,
      train_state: TrainState,
      train_state_partition_specs: Optional[TrainState] = None,
      global_mesh: Optional[maps.Mesh] = None,
      checkpoint_type: CheckpointType = CheckpointType.CHECKPOINT_FLAX,
  ) -> Tuple[TrainState, bool]:
    """Applies p.train.init_from_checkpoint_rules to update train_state.

    Args:
      train_state: initialized train_state.
      train_state_partition_specs: The TrainState specs for initialized
        train_state. Required for GDA-based checkpoints.
      global_mesh: optional mesh used to restore checkpoint if needed.
      checkpoint_type: used to restore checkpoint.

    Returns:
      A tuple of the updated new train state, and whether caller needs
      to recompute opt_states after mdl_vars are updated.
    """
    all_rules = self.hparams.train.init_from_checkpoint_rules
    if not all_rules:
      return train_state, False

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
      train_state = self._apply_init_checkpoint_rule(
          train_state,
          ckpt_path,
          rules,
          load_status,
          global_mesh,
          checkpoint_type,
          target_partition_specs=train_state_partition_specs)

    # Convert mdl_vars back to Python dict for compatibility.
    train_state = train_state.replace(
        mdl_vars=train_state.mdl_vars.ToNestedDict())
    return train_state, not load_status[2]
