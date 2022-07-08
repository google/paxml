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
import itertools
import re
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

from absl import logging
import jax
from jax import numpy as jnp
from jax.experimental import maps
from jax.experimental import pjit
import optax
from paxml import base_inference_runner
from paxml import base_metrics
from paxml import base_task
from paxml import io_utils
from praxis import base_hyperparams
from praxis import base_layer
from praxis import base_model
from praxis import learners as learners_lib
from praxis import optimizers
from praxis import py_utils
from praxis import pytypes
from praxis import train_states
import tensorflow.compat.v2 as tf

from paxml import checkpoints  # mapped to internal

BaseInferenceRunner = base_inference_runner.BaseInferenceRunner
CheckpointType = checkpoints.CheckpointType
NestedMap = py_utils.NestedMap
NestedJTensor = base_layer.NestedJTensor
JTensor = base_layer.JTensor
PartitionSpec = pjit.PartitionSpec
TrainState = train_states.TrainState

PRNGKey = pytypes.PRNGKey
sub_config_field = base_hyperparams.sub_config_field
RegexStr = str

instantiate = base_hyperparams.instantiate


# Shorthand for a loading rule that loads everything as is.
# e.g. load_rules = [LOAD_ALL]
LOAD_ALL = ('(.*)', '{}')


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
    ignore_rules: If the variable name matches with one of the regexs in the
      list, the checkpoint variables are not used even if the name matches with
      `load_rules`.
    step: Step specifier used when the directory name is provided as a
      checkpoint path.
    load_step: whether to load the step from this checkpoint.
    load_opt_states: whether to load opt_states (in its entirety) from this
      checkpoint.
  """
  task_p: SingleTask.HParams
  load_rules: Sequence[Tuple[RegexStr, str]]
  ignore_rules: Optional[Sequence[RegexStr]] = None
  step: Optional[int] = None
  load_step: bool = False
  load_opt_states: bool = False


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
        decoder_datasets() in terms of the number of training steps. Skipped
        if this value is not a positive int. Set to 0 to disable decode steps.
    """
    learner: learners_lib.Learner.HParams = sub_config_field(
        learners_lib.Learner.HParams)
    num_train_steps: float = 1e7
    save_interval_steps: int = 5000
    save_keep_interval_duration: str = '12h'
    save_max_to_keep: int = 10
    summary_interval_steps: int = 100
    variable_norm_summary: bool = True
    eval_interval_steps: int = 100
    eval_skip_train: bool = False
    inputs_split_mapping: Optional[PartitionSpec] = None
    init_from_checkpoint_rules: Dict[
        str, CheckpointLoadingRules] = dataclasses.field(default_factory=dict)
    decode_interval_steps: Optional[int] = None

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
      infer_writer: specifies how to generate and write some output with a model
    """
    model: Optional[base_model.BaseModel.HParams] = None

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
    infer_writer: Optional[SingleTask.InferWriterHParams] = None

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

  def get_model_name_for_step(self, step_i):
    del step_i
    return 'base_model'

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
    tf.nest.assert_same_structure(mdl_vars, var_weight_hparams)
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
    mdl_vars = tf.nest.map_structure(lambda x: x, mdl_vars)
    var_weight_hparams = tf.nest.map_structure(lambda x: x, var_weight_hparams)
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
      padded_shapes = unpadded_shapes
    else:
      model_state_partition_specs = self.create_train_state_partition_specs(
          var_weight_hparams, discard_opt_states)
      tf.nest.assert_same_structure(model_state_partition_specs,
                                    unpadded_shapes)

    def _maybe_pad(shape_dtype, pspec):
      if py_utils.is_optax_masked_node(shape_dtype):
        return shape_dtype
      unpadded_shape = shape_dtype.shape
      paddings = py_utils.get_uneven_sharding_paddings(
          pspec, unpadded_shape, mesh_shape, mesh_axis_names)
      padded_shape = [s + p for (s, p) in zip(unpadded_shape, paddings)]
      return jax.ShapeDtypeStruct(padded_shape, shape_dtype.dtype)

    padded_shapes = jax.tree_map(_maybe_pad, unpadded_shapes,
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

      opt_var_partition_specs = jax.tree_map(_maybe_unmask_outer_masked_state,
                                             opt_var_partition_specs,
                                             is_leaf=_is_instance_masked_state)
    return TrainState(
        step=step_partition_spec,
        mdl_vars=var_partition_specs,
        opt_states=opt_var_partition_specs)

  def maybe_adjust_train_state(self, step: int, mdl_vars: Dict[
      str, JTensor], var_weight_hparams: Dict[str, base_layer.WeightHParams],
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
      if not any(jax.tree_leaves(vn_mask)):
        raise RuntimeError('Variational noise is enabled but rules don\'t '
                           'match any variables. Please disable vn by specify'
                           ' vn.vn_scale = 0. or check vn.vn_regex. One common'
                           ' issue is that it should start with params,'
                           ' i.e., decoder -> params/decoder.')
      else:
        logging.info('Variational noise applies to: %s', vn_mask)

      params_flat, params_def = jax.tree_util.tree_flatten(names)

      rng_flat = jax.random.split(prng_key, len(params_flat))
      rng_tree = jax.tree_unflatten(params_def, rng_flat)

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
      checkpoint_type: CheckpointType = CheckpointType.CHECKPOINT_FLAX):
    """Applies one CheckpointLoadingRules to train_state."""
    ckpt_task = instantiate(rules.task_p)
    is_step_loaded, is_initialized, is_opt_states_loaded = load_status
    model_vars = train_state.mdl_vars

    # Initialize with a dummy seed
    vars_weight_params = ckpt_task.model.abstract_init_with_metadata(
        jax.random.PRNGKey(0))
    ckpt_train_state = ckpt_task.create_train_state_unpadded_shapes(
        vars_weight_params)
    train_state_pspecs = ckpt_task.create_train_state_partition_specs(
        vars_weight_params)
    loaded_train_state = checkpoints.restore_checkpoint(
        ckpt_train_state,
        ckpt_path,
        global_mesh=global_mesh,
        checkpoint_type=checkpoint_type,
        state_specs=train_state_pspecs,
        step=rules.step)
    if loaded_train_state is None:
      raise RuntimeError(f'Cannot find checkpoint from {ckpt_path}')

    # Use NestedMap's utility accessors
    loaded_vars = dict(
        NestedMap.FromNestedDict(loaded_train_state.mdl_vars).FlattenItems())

    # Load EMA state if specified
    if hasattr(rules.task_p,
               'train') and rules.task_p.train.learner.optimizer.ema_decay > 0.:
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

    loading_rules = [
        (re.compile(pattern), ref) for pattern, ref in rules.load_rules
    ]
    ignore_rules = rules.ignore_rules if rules.ignore_rules is not None else []
    ignore_rules = [re.compile(pattern) for pattern in ignore_rules]
    for varname, unused_val in model_vars.FlattenItems():
      varname_orig = varname
      varname = varname.replace('.', '/')  # dot is reserved for regex
      for pattern, refname in loading_rules:
        mo = pattern.match(varname)
        if mo is None:
          logging.info(
              'Initialization by external checkpoint: '
              '%s doesn\'t match rule, skip.', varname)
          continue
        if any(pat.match(varname) is not None for pat in ignore_rules):
          logging.info(
              'Initialization by external checkpoint: '
              '%s match ignore rule, skip.', varname)
          continue
        if varname in is_initialized:
          logging.info(
              'Initialization by external checkpoint: '
              '%s is already initialized by %s, skip.', varname,
              is_initialized[varname])
          continue
        refname = refname.format(*mo.groups())
        refname = refname.replace('/', '.')

        # Only for logging, keep name of ckpt that initialized the variable
        is_initialized[varname] = ckpt_path + '/' + refname
        logging.info(
            'Initialization by external checkpoint: '
            '%s is overwritten by %s in %s', varname, refname, ckpt_path)
        model_vars.Set(varname_orig, loaded_vars[refname])
    train_state = train_state.replace(mdl_vars=model_vars)

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

    if rules.load_opt_states:
      if is_opt_states_loaded:
        logging.info(
            'train_state.opt_states is already initialized by %s, skip.',
            is_opt_states_loaded)
      else:
        train_state = train_state.replace(
            opt_states=loaded_train_state.opt_states)
        load_status[2] = ckpt_path
        logging.info(
            'Initialization by external checkpoint: train_state.opt_states is '
            'overwritten by value from %s', ckpt_path)

    return train_state

  def apply_init_checkpoint_rules(
      self,
      train_state: TrainState,
      global_mesh: Optional[maps.Mesh] = None,
      checkpoint_type: CheckpointType = CheckpointType.CHECKPOINT_FLAX,
  ) -> Tuple[TrainState, bool]:
    """Applies p.train.init_from_checkpoint_rules to update train_state.

    Args:
      train_state: initialized train_state.
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
    is_initialized = dict()  # record which checkpoint initialized which var.
    is_step_loaded = None  # record which checkpoint loaded step.
    is_opt_states_loaded = None  # record which checkpoint loaded opt_states
    load_status = [is_step_loaded, is_initialized, is_opt_states_loaded]
    for ckpt_path, rules in all_rules.items():
      train_state = self._apply_init_checkpoint_rule(train_state, ckpt_path,
                                                     rules, load_status,
                                                     global_mesh,
                                                     checkpoint_type)

    # Convert mdl_vars back to Python dict for compatibility.
    train_state = train_state.replace(
        mdl_vars=train_state.mdl_vars.ToNestedDict())
    return train_state, load_status[2] is None
