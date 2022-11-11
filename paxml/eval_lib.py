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

"""Evaluation loop for Pax model."""

import abc
import collections
import contextlib
import functools
import sys
import time
import typing
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from absl import flags
from absl import logging
from clu import platform
from etils import epath
import jax
import jax.numpy as jnp
from jax.experimental import maps
from jax.experimental import multihost_utils
import numpy as np
from paxml import base_experiment
from paxml import base_metrics
from paxml import checkpoint_pb2
from paxml import io_utils
from paxml import metric_tracker_utils as trk_utils
from paxml import metric_utils
from paxml import seqio_input
from paxml import summary_utils
from paxml import tasks_lib
from paxml import trainer_lib
from paxml import tuning_lib
from praxis import base_hyperparams
from praxis import base_input
from praxis import base_layer
from praxis import optimizer_prefix_vectorization
from praxis import py_utils
from praxis import pytypes
from praxis import train_states
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

from paxml import checkpoints  # mapped to internal

instantiate = base_hyperparams.instantiate
CheckpointType = checkpoint_pb2.CheckpointType
EvaluationMode = io_utils.EvaluationMode
JTensor = pytypes.JTensor
Metrics = pytypes.Metrics
NestedMap = py_utils.NestedMap
JTensor = pytypes.JTensor
NestedJTensor = pytypes.NestedJTensor
NestedPartitionSpec = pytypes.NestedPartitionSpec
NestedShapeDtypeLike = pytypes.NestedShapeDtypeLike
NestedWeightHParams = base_layer.NestedWeightHParams
SummaryWriter = tf.summary.SummaryWriter
TrainState = train_states.TrainState
WeightedScalars = pytypes.WeightedScalars
WeightedScalarsList = pytypes.WeightedScalarsList
PMAP_PARALLEL_AXIS_NAME = base_layer.PMAP_PARALLEL_AXIS_NAME
PRNGKey = pytypes.PRNGKey
NestedShapeDtypeLike = pytypes.NestedShapeDtypeLike
NO_PREFIX_KEY = optimizer_prefix_vectorization.NO_PREFIX_KEY


def _is_vectorized(states: train_states.TrainState) -> bool:
  """Determines whether it is a vectorized model."""
  if not states.opt_states:
    raise ValueError(
        'cannot decide if it is vectorized model without opt_states')
  return NO_PREFIX_KEY in states.opt_states[0]


def _get_dir_names(
    input_p: Sequence[base_input.BaseInput.HParams]) -> Sequence[epath.Path]:
  """Returns a list of same length for parent dir names for each dataset."""
  return [epath.Path(p.name) for p in input_p]


def _get_filename(step: Union[base_layer.JTensorOrPartitionSpec, int],
                  prefix: str) -> str:
  """Returns a filename for the given step."""
  step_num = py_utils.maybe_unreplicate_for_fully_replicated(step)
  return f'{prefix}_out_{step_num}_shard_{jax.process_index()}'


def _can_load_written_outputs(basedir: epath.Path, pname: str,
                              mode: EvaluationMode, step: int) -> bool:
  """Returns whether we can load the eval/decoder outputs already."""
  success = np.array([0], dtype=np.int32)
  if jax.process_index() == 0:
    try:
      outputs = io_utils.load_outputs(basedir, pname, mode.value, step)
      success[0] = len(outputs)
    except Exception:  # pylint: disable=broad-except
      pass
  out = multihost_utils.broadcast_one_to_all(success)
  return out[0] > 0


def _maybe_write_scoring_outputs(
    job_log_dir: epath.Path, step: int, input_name: str,
    scoring_outputs: Sequence[Dict[str, Any]]) -> None:
  """Write model scoring outputs to disk from leader process."""
  if (jax.process_index() != 0 or flags.FLAGS.pax_only_aggregate_summaries):
    return

  mode_str = EvaluationMode.EVAL.value
  basedir = job_log_dir / f'{mode_str}_out'
  fq_fname = basedir / input_name / _get_filename(step, mode_str)
  fq_fname.parent.mkdir(parents=True, exist_ok=True)

  # convert list of batches into list of example kv pairs
  flat_scoring_outputs = []
  for batch in scoring_outputs:
    for ex in py_utils.tree_unstack(batch, 0):
      flat_scoring_outputs.append((py_utils.get_enumeration_id(ex), ex))

  logging.info('Writing eval outputs to %s with %d entries',
               fq_fname, len(scoring_outputs))

  io_utils.write_key_value_pairs(fq_fname, flat_scoring_outputs)


def _wait_until_step(checkpointer, start_step):
  """Waits until start_step is reached."""
  if not start_step:
    return

  while True:
    cur_step = checkpointer.retrieve_latest_checkpoint_step()
    if cur_step is not None and start_step <= cur_step:
      break
    time.sleep(300)


def has_ema(task_p: tasks_lib.SingleTask.HParams) -> bool:
  """Determines whether ema is used or not."""
  return task_p.train.learner.optimizer.ema_decay > 0.


def extract_ema(
    model_states: train_states.TrainState) -> train_states.TrainState:
  """Finds the ema state from optimizer states."""
  if len(model_states.opt_states) != 1:
    raise ValueError('EMA currently only supports a single learner (got '
                     f'`{len(model_states.opt_states)}`).')
  is_vectorized = _is_vectorized(model_states)
  if not is_vectorized:
    for v in model_states.opt_states[0]:
      if isinstance(v, dict) and 'ema' in v:
        return TrainState(step=model_states.step, mdl_vars=v.ema, opt_states={})
  else:
    ret = None
    # For vectorized model, the structure looks like this:
    # opt_states: [{'no_prefix': ({'count': '', 'ema': {'params': {'ctcloss':
    # It is a list of dictionaries. The key corresponds to the #stages.
    # Here the ema is constructed by combining the ema state from all those
    # dictionaries. Each parameter belongs to one dictionary and is labelled as
    # masked node in others.
    for item in model_states.opt_states[0].values():
      if isinstance(item, tuple):
        for v in item:
          if isinstance(v, dict) and 'ema' in v:
            if ret is None:
              ret = v.ema
            else:
              ret = jax.tree_map(
                  lambda x, y: y if py_utils.is_optax_masked_node(x) else x,
                  ret,
                  v.ema,
                  is_leaf=py_utils.is_optax_masked_node)
    if ret is not None:
      return TrainState(step=model_states.step, mdl_vars=ret, opt_states={})
  raise ValueError('Could not find EMA states in `%r`.' %
                   model_states.opt_states)


class _EvalCheckpointer(metaclass=abc.ABCMeta):
  """Adapts particular implementations of checkpointing into a common API."""

  restore_checkpoint_dir: epath.Path

  def retrieve_latest_checkpoint_step(self) -> Optional[int]:
    return checkpoints.retrieve_latest_checkpoint_step(
        self.restore_checkpoint_dir)

  def wait_for_new_step(self, last_checkpoint_step: int) -> int:
    new_checkpoint_step = self.retrieve_latest_checkpoint_step()
    while new_checkpoint_step == last_checkpoint_step:
      logging.info('Sleep before checking for new latest checkpoint.')
      time.sleep(60)
      new_checkpoint_step = self.retrieve_latest_checkpoint_step()
    # There must be a new checkpoint here.
    assert new_checkpoint_step is not None
    logging.info('Found new checkpoint at step: %d', new_checkpoint_step)
    return new_checkpoint_step

  @abc.abstractmethod
  def load_checkpoint_for_step(
      self, new_checkpoint_step: int,
      train_state_global_shapes: train_states.TrainState,
      partitioned_specs: Optional[train_states.TrainState],
      global_mesh: Optional[maps.Mesh]) -> train_states.TrainState:
    raise NotImplementedError


class _SpmdEvalCheckpointer(_EvalCheckpointer):

  def __init__(self, task_p: tasks_lib.SingleTask.HParams,
               job_log_dir: epath.Path,
               maybe_use_persistence_checkpointing: bool, mode: EvaluationMode,
               restore_checkpoint_dir: epath.Path,
               restore_checkpoint_step: int):
    del mode
    self.checkpoint_type = checkpoints.retrieve_checkpoint_type(
        maybe_use_persistence_checkpointing, task_p)
    self.task_p = task_p
    self.job_log_dir = job_log_dir
    self.maybe_use_persistence_checkpointing = maybe_use_persistence_checkpointing
    self.restore_checkpoint_dir: epath.Path = restore_checkpoint_dir
    self.restore_checkpoint_step: int = restore_checkpoint_step
    self.use_ema: bool = has_ema(task_p)

  def load_checkpoint_for_step(
      self, step: int, train_state_global_shapes: train_states.TrainState,
      partitioned_specs: Optional[train_states.TrainState],
      global_mesh: Optional[maps.Mesh]) -> train_states.TrainState:
    partitioned_train_state = checkpoints.restore_checkpoint(
        train_state_global_shapes,
        self.restore_checkpoint_dir,
        global_mesh=global_mesh,
        checkpoint_type=self.checkpoint_type,
        state_specs=partitioned_specs,
        step=step)
    assert partitioned_train_state
    if self.use_ema:
      partitioned_train_state = extract_ema(partitioned_train_state)
    py_utils.sync_global_devices(
        f'checkpointer:restored:{self.restore_checkpoint_dir}')
    return partitioned_train_state


class _PmapEvalCheckpointer(_EvalCheckpointer):

  def __init__(self, task_p: tasks_lib.SingleTask.HParams,
               job_log_dir: epath.Path,
               maybe_use_persistence_checkpointing: bool, mode: EvaluationMode,
               restore_checkpoint_dir: epath.Path,
               restore_checkpoint_step: int):
    self.checkpoint_type = checkpoints.retrieve_checkpoint_type(
        maybe_use_persistence_checkpointing, task_p)
    self.task_p = task_p
    self.job_log_dir = job_log_dir
    self.maybe_use_persistence_checkpointing = maybe_use_persistence_checkpointing
    self.restore_checkpoint_dir: epath.Path = restore_checkpoint_dir
    self.restore_checkpoint_step: int = restore_checkpoint_step
    self.use_ema: bool = has_ema(task_p)
    self.track_metric: bool = (mode != EvaluationMode.EVAL) and bool(
        task_p.track_decoder_metric)

  def _restore(
      self, step: int, train_state_global_shapes: train_states.TrainState
  ) -> Optional[train_states.TrainState]:
    if py_utils.pmap_use_tensorstore():
      return tasks_lib.restore_pmap_from_tensorstore(
          train_state_global_shapes,
          self.restore_checkpoint_dir,
          step=step,
          checkpoint_type=self.checkpoint_type)
    else:
      return checkpoints.restore_checkpoint(
          train_state_global_shapes,
          self.restore_checkpoint_dir,
          checkpoint_type=self.checkpoint_type,
          step=step)

  def load_checkpoint_for_step(
      self, step: int, train_state_global_shapes: train_states.TrainState,
      partitioned_specs: Optional[train_states.TrainState],
      global_mesh: Optional[maps.Mesh]) -> train_states.TrainState:
    model_states = self._restore(step, train_state_global_shapes)
    if self.use_ema:
      model_states = extract_ema(model_states)
    elif not self.track_metric:
      model_states = model_states.to_eval_state()
    replicated_model_states = trainer_lib.replicate_model_state(model_states)
    del model_states  # Unused at that point.
    return replicated_model_states

  def get_model_states(
      self, jax_task: tasks_lib.SingleTask, prng_key: PRNGKey,
      input_shape_dtypes: Optional[NestedShapeDtypeLike],
      train_state_metadata: Optional[trainer_lib.TrainStateMetadata]
  ) -> Tuple[train_states.TrainState, train_states.TrainState, PRNGKey]:
    if jax_task.hparams.train.always_use_train_for_model_init:
      assert train_state_metadata is not None
      var_weight_hparams = train_state_metadata.var_weight_hparams
      global_shapes = train_state_metadata.unpadded_global_shapes
      is_eval_for_init = False
      input_specs = train_state_metadata.input_shape_dtype
    else:
      assert input_shape_dtypes is not None
      prng_key, init_key = jax.random.split(prng_key)
      input_specs = input_shape_dtypes
      # Restore flax checkpoints still required bak variables in TrainState
      var_weight_hparams = jax_task.model.abstract_init_with_metadata(
          input_specs, do_eval=True)
      # Note: `discard_opt_states` is not supported when restoring pmap
      # checkpoints. We must restore the entire checkpoint and then trim the
      # unrelevant states.
      global_shapes = jax_task.create_train_state_unpadded_shapes(
          var_weight_hparams)
      is_eval_for_init = True
    # Pmap does not use GDA, and so global_mesh and mesh_axes are None.
    model_states = self._restore(self.restore_checkpoint_step, global_shapes)
    if model_states is None:
      prng_key, init_key = jax.random.split(prng_key)
      model_states = trainer_lib.initialize_model_state(
          jax_task,
          init_key,
          input_specs,
          discard_opt_states=not self.use_ema,
          is_eval=is_eval_for_init)
    elif not self.use_ema and not self.track_metric:
      model_states = model_states.to_eval_state()
    if self.use_ema:
      model_states = extract_ema(model_states)
    replicated_model_states = trainer_lib.replicate_model_state(model_states)
    return replicated_model_states, global_shapes, prng_key


def _create_checkpointer(
    task_p: tasks_lib.SingleTask.HParams, job_log_dir: epath.Path,
    maybe_use_persistence_checkpointing: bool, mode: Optional[EvaluationMode],
    restore_checkpoint_dir: Optional[epath.PathLike],
    restore_checkpoint_step: Optional[int]) -> _EvalCheckpointer:
  if not restore_checkpoint_dir:
    # bool(Path(''))==True, so guarding against this odd Optional explicitly ^
    restore_checkpoint_dir = job_log_dir / 'checkpoints'

  if restore_checkpoint_step is None and mode is not None:
    restore_checkpoint_step = io_utils.get_checkpoint_step(
        job_log_dir, restore_checkpoint_dir, mode)
    # TODO(pax-team): Enforce that a checkpoint exists / a checkpoint step was
    # retrieved.

  model_p = task_p.model
  if model_p.mesh_shape is not None:
    return _SpmdEvalCheckpointer(task_p, job_log_dir,
                                 maybe_use_persistence_checkpointing, mode,
                                 restore_checkpoint_dir,
                                 restore_checkpoint_step)
  else:
    return _PmapEvalCheckpointer(task_p, job_log_dir,
                                 maybe_use_persistence_checkpointing, mode,
                                 restore_checkpoint_dir,
                                 restore_checkpoint_step)


def run_eval_loop_over_test_splits(
    num_steps: List[int],
    eval_step: Callable[[NestedJTensor], Any],
    summary_writers: List[SummaryWriter],
    step: int,
    model_inputs: List[base_input.BaseInput],
    job_log_dir: epath.Path,
    eval_inputs_pspecs=None,
    eval_inputs_shape=None,
    global_mesh=None,
    reshard_inputs: Optional[bool] = False,
    create_gda_for_inputs: bool = False,
) -> Tuple[List[Optional[Dict[str, float]]],  # eval metrics.
           List[Optional[Dict[str, float]]],  # eval scoring metrics.
           List[int]  # performed eval steps.
          ]:
  """Run evaluation in a loop over a list of test sets.

  Args:
    num_steps: A list of steps for each test split to evaluate on.
    eval_step: The eval step function which to call to evaluate the model.
    summary_writers: The summary writer objects to log summaries.
    step: The step at which we are evaling the model.
    model_inputs: List of BaseInput instances.
    job_log_dir: Job's log directory in which scoring outputs will be written.
    eval_inputs_pspecs: PartitionSpec for eval inputs.
    eval_inputs_shape: Global shape of eval inputs
    global_mesh: Device mesh used by pjit.
    reshard_inputs: Whether to reshard inputs.
    create_gda_for_inputs: Whether to create GDAs for model inputs.

  Returns:
    A tuple of (a list of eval metrics,
                a list of optional scoring metrics (seqio)
                a list of integer as performed evaluation steps).
      Items from each list are aligned with the `model_inputs`.
  """
  # If reshard_inputs = True, meaning this is called from pmap, hence we need to
  # unreplicate metrics for reporting.
  eval_metrics_list = []
  eval_scoring_metrics_list = []
  num_eval_steps = []
  for split, num_split_steps in enumerate(num_steps):
    if _can_load_written_outputs(job_log_dir, model_inputs[split].hparams.name,
                                 EvaluationMode.EVAL, step):
      logging.info('Eval on input %s at step %d already done, skipping.',
                   model_inputs[split].hparams.name, step)
      eval_metrics_list.append(None)
      eval_scoring_metrics_list.append(None)
      num_eval_steps.append(0)
      continue

    logging.info('Starting eval data split=%d (%s) with num_steps=%d',
                 split, model_inputs[split].hparams.name, num_split_steps)
    # Reset loss and summary tensors for each test split.
    loss = []
    summary_tensors = {}
    metrics = collections.defaultdict(list)
    step_num = 0
    per_example_scores = []
    # Use num_split_steps < 0 to indicate running all of the input until
    # out of range.
    while num_split_steps < 0 or step_num < num_split_steps:
      step_num += 1
      try:
        eval_inputs = model_inputs[split].get_next_padded()
      except (tf.errors.OutOfRangeError, StopIteration):
        if num_split_steps > 0:
          raise
        logging.info('Exhausted eval data split=%d after %d steps', split,
                     step_num - 1)
        model_inputs[split].reset()
        break

      if (global_mesh and
          (create_gda_for_inputs or
           model_inputs[split].hparams.experimental_remote_input)):
        eval_inputs = model_inputs[split].reshard_for_spmd(
            eval_inputs, eval_inputs_shape, global_mesh, eval_inputs_pspecs)
      elif reshard_inputs:
        eval_inputs = model_inputs[split].reshard_for_pmap(eval_inputs)
      # TODO(bencaine): Rename eval_metrics here weighted scalars?
      (eval_loss, eval_metrics, per_example_output,
       eval_summary_tensors) = eval_step(eval_inputs)

      logging.info('Finished eval step on input batch %d for %s',
                   step_num, model_inputs[split].hparams.name)

      eval_loss = py_utils.maybe_unreplicate_for_fully_replicated(eval_loss)
      eval_metrics = py_utils.maybe_unreplicate_for_fully_replicated(
          eval_metrics)
      per_example_output = py_utils.maybe_unreplicate_for_fully_replicated(
          per_example_output)
      eval_summary_tensors = py_utils.maybe_unreplicate_for_fully_replicated(
          eval_summary_tensors)
      per_example_scores.append(jax.tree_map(np.asarray, per_example_output))
      loss += [eval_loss]
      eval_summary_tensors = summary_utils.flatten_summary_dict(
          eval_summary_tensors)
      for k, v in eval_summary_tensors:
        if k in summary_tensors:
          summary_tensors[k] += [v]
        else:
          summary_tensors[k] = [v]
      for k in eval_metrics:
        metrics[k].append(eval_metrics[k])

    logging.info('Finished eval on input %s', model_inputs[split].hparams.name)

    eval_scoring_metrics = None
    if seqio_input.should_process_outputs(model_inputs[split]):
      eval_scoring_metrics = seqio_input.process_outputs(
          model_inputs[split], per_example_scores, summary_writers[split],
          seqio_input.MetricType.SCORE, step)

    loss = np.array(loss)
    for k in summary_tensors:
      summary_tensors[k] = np.array([np.asarray(t) for t in summary_tensors[k]])
    loss = np.mean(loss, axis=0)
    logging.info('step_i: %d, eval test split %s loss: %s', step, split, loss)
    for key, values in metrics.items():
      # `metric_utils.as_float` computes the average from a list of weighted
      # scalars.
      weighted_average = metric_utils.as_float(values)
      sum_metric_weights = np.sum(np.stack([v[1] for v in values]))
      logging.info('  %s=%f (weight=%f)', key, weighted_average,
                   sum_metric_weights.item())
    summary_utils.write_summary_entry(summary_writers[split], step, loss,
                                      metrics, summary_tensors)
    eval_metrics_list.append(metric_utils.as_float_dict(metrics))
    eval_scoring_metrics_list.append(eval_scoring_metrics)
    num_eval_steps.append(step_num)

    _maybe_write_scoring_outputs(
        job_log_dir, step, model_inputs[split].hparams.name, per_example_scores)

  return (eval_metrics_list, eval_scoring_metrics_list, num_eval_steps)


def evaluate(experiment_config: base_experiment.BaseExperiment,
             job_log_dir: epath.Path,
             maybe_use_persistence_checkpointing: bool,
             restore_checkpoint_dir: Optional[epath.Path] = None,
             restore_checkpoint_step: Optional[int] = None,
             early_stopping_fn: Optional[trainer_lib.EarlyStoppingFn] = None,
             enable_auto_sharding: bool = False) -> None:
  """Runs the evaluation loop on the entire eval data set.

  Args:
    experiment_config: an instance of BaseExperiment for the experiment to
      evaluate.
    job_log_dir: The directory for the job logs.
    maybe_use_persistence_checkpointing: If set, it will try to use
      persistence-based checkpointing if suitable.
    restore_checkpoint_dir: Optional directory from which to restore a
      checkpoint.
    restore_checkpoint_step: If set, the checkpoint step to restore.
    early_stopping_fn: An optional callable object for reporting eval metrics
      and determining whether to early stop current training. The callable
      object has signature: (metrics, running_mode, ckpt_step, is_final_ckpt) ->
      should_stop_early.
    enable_auto_sharding: Enables the XLA AutoSharding pass to generate SPMD
      shardings.
  """
  del enable_auto_sharding
  task_p = experiment_config.task()
  task_p = typing.cast(tasks_lib.SingleTask.HParams, task_p)
  checkpointer = _create_checkpointer(
      task_p,
      job_log_dir,
      maybe_use_persistence_checkpointing,
      EvaluationMode.EVAL,
      restore_checkpoint_dir=restore_checkpoint_dir,
      restore_checkpoint_step=restore_checkpoint_step)

  model_p = task_p.model
  eval_input_p = [v for v in experiment_config.datasets() if not v.is_training]
  if not eval_input_p:
    logging.info('No eval datasets defined. Returning early.')
    return
  for inp in eval_input_p:
    if inp.num_infeed_hosts == 0:
      inp.num_infeed_hosts = jax.process_count()
    inp.infeed_host_index = jax.process_index()

  train_input_specs = None
  if task_p.train.always_use_train_for_model_init:
    input_specs_provider = instantiate(
        experiment_config.get_input_specs_provider_params())
    train_input_specs = input_specs_provider.get_input_specs()

  if model_p.mesh_shape is not None:
    train_input_specs = jax.tree_map(py_utils.get_global_input_shape_dtype,
                                     train_input_specs)
    evaluate_spmd_model(task_p, train_input_specs, eval_input_p, job_log_dir,
                        typing.cast(_SpmdEvalCheckpointer, checkpointer),
                        early_stopping_fn)
  else:
    evaluate_pmap_model(task_p, train_input_specs, eval_input_p, job_log_dir,
                        typing.cast(_PmapEvalCheckpointer, checkpointer),
                        early_stopping_fn)


class _PmapEvalRunner:
  """A runner class that runs evaluate with pmap.

  Example usage:

    (replicated_model_states, train_state_global_shapes,
     prng_key) = _PmapEvalRunner.get_model_states(
        jax_task, prng_key, input_shape_dtypes, checkpointer)

    runner = _PmapEvalRunner(task_p, eval_input_params, jax_task, prng_key)
    metrics_list, eval_scoring_metrics_list, num_eval_steps = (
        runner.run_one_step(
            replicated_model_states, sample_inputs, eval_summary_writers))
  """

  def __init__(self, task_p: tasks_lib.SingleTask.HParams,
               eval_input_p: Sequence[base_input.BaseInput.HParams],
               jax_task: tasks_lib.SingleTask, pmap_prng_key: PRNGKey,
               job_log_dir: epath.Path):
    self._eval_input_p = eval_input_p
    self._task_p = task_p
    self._job_log_dir = job_log_dir
    if not self._eval_input_p:
      return
    self._jax_task = jax_task
    self._eval_input_pipelines = [
        instantiate(input_p) for input_p in eval_input_p
    ]
    trainer_lib.check_unique_names(self._eval_input_pipelines)
    self._eval_num_steps = [
        -1 if p.reset_for_eval else p.eval_loop_num_batches
        for p in eval_input_p
    ]
    self._run_pmap(pmap_prng_key)

  @classmethod
  def get_model_states(
      cls, jax_task: tasks_lib.SingleTask, prng_key: PRNGKey,
      input_shape_dtypes: Optional[NestedShapeDtypeLike],
      train_state_metadata: Optional[trainer_lib.TrainStateMetadata],
      checkpointer: _EvalCheckpointer
  ) -> Tuple[train_states.TrainState, train_states.TrainState, PRNGKey]:
    """Returns the (replicated) model states."""
    checkpointer = typing.cast(_PmapEvalCheckpointer, checkpointer)
    replicated_model_states, global_shapes, prng_key = (
        checkpointer.get_model_states(jax_task, prng_key, input_shape_dtypes,
                                      train_state_metadata))
    logging.info('replicated_model_states: %s',
                 jax.tree_map(lambda x: x.shape, replicated_model_states))
    # From now on, different replicas should use different random seeds.
    # Here, each process will have its unique prng_key.
    # prng_key will be further split so that each core on a host will get
    # different prng_key.
    prng_key = jax.random.fold_in(prng_key, jax.process_index())
    logging.info('root prng_key: %s', prng_key)
    return replicated_model_states, global_shapes, prng_key

  def _run_pmap(self, prng_key: PRNGKey):
    """Calls pmap on the eval one step function."""
    if not self._eval_input_p:
      return

    def eval_step(mdl_states, prng_key, inputs):
      return trainer_lib.eval_step_single_learner(
          self._jax_task,
          mdl_states,
          prng_key,
          inputs,
          fprop_dtype=self._jax_task.model.fprop_dtype)

    num_devices = jax.local_device_count()
    prng_key, eval_key = jax.random.split(prng_key)
    self._eval_prng_seed = jax.random.split(eval_key, num=num_devices)
    logging.info('eval prng_seed: %s', self._eval_prng_seed)

    self._pmap_eval_step = jax.pmap(
        eval_step, axis_name=PMAP_PARALLEL_AXIS_NAME)

  def run_one_step(
      self,
      replicated_model_states: train_states.TrainState,
      eval_summary_writers: List[SummaryWriter],
  ) -> Tuple[List[Optional[Dict[str, float]]],  # eval metrics list.
             List[Optional[Dict[str, float]]],  # seqio metrics list.
             List[int]  # actual eval steps.
            ]:
    """Runs evaluate for one step for all test splits."""
    if not self._eval_input_p:
      return [], [], []
    step_i = int(
        py_utils.maybe_unreplicate_for_fully_replicated(
            replicated_model_states.step))

    def eval_step_fn(inputs):
      # TODO(pax): shall we eval all sub-models during eval?
      return self._pmap_eval_step(replicated_model_states, self._eval_prng_seed,
                                  inputs)

    # Run the eval loop.
    return run_eval_loop_over_test_splits(
        self._eval_num_steps,
        eval_step_fn,
        eval_summary_writers,
        step_i,
        self._eval_input_pipelines,
        self._job_log_dir,
        reshard_inputs=True)

  def partition_run_one_step(self, eval_input_p):

    def eval_one_step_fn(replicated_model_states, eval_summary_writers):
      with py_utils.timeit() as eval_period:
        eval_metrics_list, eval_scoring_metrics_list, num_eval_steps = (
            self.run_one_step(replicated_model_states, eval_summary_writers))

      return tuning_lib.EvalMetrics(
          input_p=eval_input_p,
          metrics_list=eval_metrics_list,
          scoring_metrics_list=eval_scoring_metrics_list,
          steps_per_sec=sum(num_eval_steps) / eval_period.elapsed)

    return eval_one_step_fn


def evaluate_pmap_model(
    task_p: tasks_lib.SingleTask.HParams,
    train_input_specs: Optional[NestedShapeDtypeLike],
    eval_input_p: Sequence[base_input.BaseInput.HParams],
    job_log_dir: epath.Path,
    checkpointer: _PmapEvalCheckpointer,
    early_stopping_fn: Optional[trainer_lib.EarlyStoppingFn] = None) -> None:
  """Runs the evaluation loop on the entire test dataset for PMAP model.

  Args:
    task_p: Params for the task encapsulating the data parallel model.
    train_input_specs: Training input specs for model initialization.
    eval_input_p: List of params for the eval data input pipelines.
    job_log_dir: Directory for the job logs.
    early_stopping_fn: An optional callable object for reporting metrics and
      determining whether to early stop current training. The callable object
      has signature: (metrics, running_mode, ckpt_step, is_final_ckpt) ->
      should_stop_early.
  """
  logging.info('Using pmap for data parallelism.')

  if not eval_input_p:
    return

  jax_task = instantiate(task_p)

  partitioned_specs = None
  global_mesh = None

  prng_key = jax.random.PRNGKey(1234)
  if task_p.train.always_use_train_for_model_init:
    if train_input_specs is None:
      raise ValueError(
          'No training input specs available, while enabling '
          '`task_p.train.always_use_train_for_model_init` requires it.')
    train_state_metadata = trainer_lib.create_train_state_metadata(
        jax_task, train_input_specs)
    input_shape_dtypes = None
  else:
    # TODO(pax-dev): Investigate if we can use model input specs
    # instead of instantiating this input pipeline.
    train_state_metadata = None
    input_shape_dtypes = instantiate(eval_input_p[0]).get_next_padded()

  (partitioned_train_state, train_state_global_shapes,
   prng_key) = _PmapEvalRunner.get_model_states(jax_task, prng_key,
                                                input_shape_dtypes,
                                                train_state_metadata,
                                                checkpointer)

  eval_one_step_fn = _PmapEvalRunner(
      task_p, eval_input_p, jax_task, prng_key,
      job_log_dir).partition_run_one_step(eval_input_p)
  decode_once_fn = None
  input_p = None
  continuous_decode = True
  _common_eval_or_decode_loop(EvaluationMode.EVAL, checkpointer, task_p,
                              job_log_dir, global_mesh, input_p, eval_input_p,
                              eval_one_step_fn, decode_once_fn,
                              partitioned_train_state,
                              train_state_global_shapes, partitioned_specs,
                              early_stopping_fn, continuous_decode)


class _SpmdEvalRunner:
  """A runner class that runs evaluate with spmd.

  Example usage:

    model_states_out = (
        _SpmdEvalRunner.get_model_states(
            jax_task, global_mesh, init_key, input_shape_dtypes,
            train_state_metadata, checkpoint_dir, checkpoint_type))
    runner = _SpmdEvalRunner(
        task_p, sample_input_params, jax_task, global_mesh,
        init_key, partitioned_specs, job_log_dir, unpadded_global_batch_size)
    eval_metrics_list, eval_scoring_metrics_list, num_eval_steps = (
        runner.run_one_step(partitioned_train_state, eval_summary_writers,
                            eval_key, create_gda_for_inputs))
  """

  def __init__(self,
               task_p: tasks_lib.SingleTask.HParams,
               eval_input_p: Sequence[base_input.BaseInput.HParams],
               jax_task: tasks_lib.SingleTask,
               global_mesh: maps.Mesh,
               init_key: PRNGKey,
               partitioned_specs: train_states.TrainState,
               job_log_dir: epath.Path,
               use_ema: bool = False,
               unpadded_global_batch_size: Optional[int] = None):
    self._eval_input_p = eval_input_p
    if not self._eval_input_p:
      return
    self._jax_task = jax_task
    self._eval_input_pipelines = [
        instantiate(input_p) for input_p in eval_input_p
    ]
    trainer_lib.check_unique_names(self._eval_input_pipelines)
    self._eval_num_steps = [
        -1 if p.reset_for_eval else p.eval_loop_num_batches
        for p in eval_input_p
    ]
    self._task_p = task_p
    self._job_log_dir = job_log_dir

    # TODO(pax-dev): Investigate if we can use model input specs
    # instead of instantiating this input pipeline.
    # setup input_shape.
    sample_model_inputs = instantiate(self._eval_input_p[0]).get_next_padded()
    self._inputs_shape = jax.tree_util.tree_map(
        py_utils.get_global_input_shape_dtype, sample_model_inputs)
    self.global_mesh = global_mesh

    # Will be populated by self.run_pjit() below.
    self._eval_step = None
    self._inputs_partition_specs = None
    if use_ema:
      partitioned_specs = partitioned_specs.to_eval_state()
    self._run_pjit(init_key, partitioned_specs, unpadded_global_batch_size)

  @classmethod
  def get_model_states(
      cls,
      jax_task: tasks_lib.SingleTask,
      global_mesh: maps.Mesh,
      init_key: PRNGKey,
      input_shape_dtypes: Optional[NestedShapeDtypeLike],
      train_state_metadata: Optional[trainer_lib.TrainStateMetadata],
      checkpoint_dir: epath.Path,
      checkpoint_type: CheckpointType,
      checkpoint_step: Optional[int] = None,
      use_ema: bool = False,
      is_decode: bool = False,
      unpadded_global_batch_size: Optional[int] = None,
      enable_auto_sharding: bool = False,
  ) -> Union[train_states.TrainState, Tuple[
      train_states.TrainState, Optional[train_states.TrainState],
      train_states.TrainState, Any, NestedPartitionSpec, NestedJTensor]]:
    """Gets a partitioned model states and the step function."""
    if jax_task.hparams.train.always_use_train_for_model_init:
      assert train_state_metadata is not None
      train_state_global_shapes = train_state_metadata.padded_global_shapes
      partitioned_specs = train_state_metadata.partitioned_specs
      inputs_shape = train_state_metadata.input_shape_dtype
    else:
      assert input_shape_dtypes is not None
      var_weight_hparams = jax_task.model.abstract_init_with_metadata(
          input_shape_dtypes, do_eval=True)
      train_state_global_shapes = (
          jax_task.create_train_state_padded_shapes(
              var_weight_hparams, discard_opt_states=not use_ema))
      partitioned_specs = jax_task.create_train_state_partition_specs(
          var_weight_hparams, discard_opt_states=not use_ema)
    partitioned_train_state = checkpoints.restore_checkpoint(
        train_state_global_shapes,
        checkpoint_dir,
        global_mesh=global_mesh,
        checkpoint_type=checkpoint_type,
        step=checkpoint_step,
        state_specs=partitioned_specs)
    py_utils.sync_global_devices(f'checkpointer:restored:{checkpoint_dir}')

    init_key, step_key = jax.random.split(init_key)

    if not jax_task.hparams.train.always_use_train_for_model_init:
      inputs_shape = input_shape_dtypes
      step_fn, inputs_partition_specs, partitioned_specs = (
          trainer_lib.get_partitioned_spmd_model_step_fn(
              jax_task,
              trainer_lib.RunningMode.DECODE
              if is_decode else trainer_lib.RunningMode.EVAL,
              global_mesh,
              step_key,
              inputs_shape,
              train_state_partition_spec=partitioned_specs.to_eval_state(),  # pytype: disable=attribute-error
              unpadded_global_batch_size=unpadded_global_batch_size,
              enable_auto_sharding=enable_auto_sharding))

    if partitioned_train_state is None:
      _, partitioned_train_state = (
          trainer_lib.initialize_partitioned_model_states(
              jax_task,
              step_key,
              inputs_shape,
              global_mesh=global_mesh,
              # Note: We currently enforce that the checkpoint to reload via
              # init_checkpoint_rules are in the same format as the checkpoint
              # solution used by the experiment.
              checkpoint_type=checkpoint_type,
              state_specs=partitioned_specs,
              discard_opt_states=True))
    elif use_ema:
      partitioned_train_state = extract_ema(partitioned_train_state)

    if jax_task.hparams.train.always_use_train_for_model_init:
      return partitioned_train_state
    else:
      return (partitioned_train_state, partitioned_specs,
              train_state_global_shapes, step_fn, inputs_partition_specs,
              input_shape_dtypes)  # pytype:disable=bad-return-type

  def _run_pjit(self, init_key: PRNGKey,
                partitioned_specs: train_states.TrainState,
                unpadded_global_batch_size: Optional[int] = None) -> None:
    """Run pjit on the single step evaluation function."""
    if not self._eval_input_p:
      return
    eval_step, inputs_partition_specs, _ = (
        trainer_lib.get_partitioned_spmd_model_step_fn(
            self._jax_task,
            trainer_lib.RunningMode.EVAL,
            self.global_mesh,
            init_key,
            self._inputs_shape,
            train_state_partition_spec=partitioned_specs.to_eval_state(),
            unpadded_global_batch_size=unpadded_global_batch_size))
    self._eval_step = eval_step
    self._inputs_partition_specs = inputs_partition_specs

  def run_one_step(
      self, partitioned_train_state: train_states.TrainState,
      eval_summary_writers: List[SummaryWriter], eval_key: PRNGKey,
      use_gda: bool
  ) -> Tuple[List[Optional[Dict[str, float]]],  # eval metrics list.
             List[Optional[Dict[str, float]]],  # eval scoring metrics list.
             List[int]  # performed eval steps.
            ]:
    """Runs evaluate for one step. Requires calling run_pjit() prior."""
    if not self._eval_input_p:
      return [], [], []
    step_i = int(
        py_utils.maybe_unreplicate_for_fully_replicated(
            partitioned_train_state.step))
    eval_step_fn = functools.partial(self._eval_step,
                                     partitioned_train_state.to_eval_state(),
                                     eval_key)
    # Run the eval loop.
    with self.global_mesh:
      return run_eval_loop_over_test_splits(
          self._eval_num_steps,
          eval_step_fn,
          eval_summary_writers,
          step_i,
          self._eval_input_pipelines,
          self._job_log_dir,
          self._inputs_partition_specs,
          self._inputs_shape,
          self.global_mesh,
          reshard_inputs=False,
          create_gda_for_inputs=use_gda)

  def partition_run_one_step(self, eval_key, eval_input_p, use_gda):

    def eval_one_step_fn(partitioned_train_state, eval_summary_writers):
      with py_utils.timeit() as eval_period:
        (eval_metrics_list, eval_scoring_metrics_list,
         num_eval_steps) = self.run_one_step(partitioned_train_state,
                                             eval_summary_writers, eval_key,
                                             use_gda)

      return tuning_lib.EvalMetrics(
          input_p=eval_input_p,
          metrics_list=eval_metrics_list,
          scoring_metrics_list=eval_scoring_metrics_list,
          steps_per_sec=sum(num_eval_steps) / eval_period.elapsed)

    return eval_one_step_fn


def evaluate_spmd_model(task_p: tasks_lib.SingleTask.HParams,
                        train_input_specs: Optional[NestedShapeDtypeLike],
                        eval_input_p: Sequence[base_input.BaseInput.HParams],
                        job_log_dir: epath.Path,
                        checkpointer: _SpmdEvalCheckpointer,
                        early_stopping_fn: Optional[
                            trainer_lib.EarlyStoppingFn] = None,
                        enable_auto_sharding: bool = False) -> None:
  """Runs the evaluation loop on the entire test dataset for SPMD model.

  Args:
    task_p: Params of the task encapsulating an SPMD model.
    train_input_specs: Training input specs for model initialization.
    eval_input_p: List of Params for the eval data pipelines.
    job_log_dir: Directory for the job logs.
    checkpointer: The model checkpointing method to use.
    early_stopping_fn: An optional callable object for reporting metrics and
      determining whether to early stop current training. The callable object
      has signature: (metrics, running_mode, ckpt_step, is_final_ckpt) ->
      should_stop_early.
    enable_auto_sharding: Enables the XLA AutoSharding pass to generate SPMD
      shardings.
  """
  logging.info('Using SPMD sharding for model parallelism.')

  if not eval_input_p:
    return

  model_p = task_p.model
  device_mesh = py_utils.create_device_mesh(model_p.ici_mesh_shape,
                                            model_p.dcn_mesh_shape)
  global_mesh = maps.Mesh(device_mesh, model_p.mesh_axis_names)
  jax_task = instantiate(task_p)

  # TODO(bf-jax): Retrieve the seeds from the model definition instead.
  prng_key = jax.random.PRNGKey(1234)
  prng_key, init_key = jax.random.split(prng_key)
  # We do not fold in jax.process_index in contrast to the pmap version and
  # use a single global key instead to rely on pjit to split for different
  # replicas.
  logging.info('root prng_key: %s', prng_key)
  _, eval_key = jax.random.split(prng_key)
  logging.info('eval prng_key: %s', eval_key)

  eval_input_p_0 = eval_input_p[0]
  eval_unpadded_global_batch_size = (
      eval_input_p_0.cls.get_batch_size(eval_input_p_0) *
      eval_input_p_0.num_infeed_hosts)
  eval_input_p = [
      trainer_lib.adjust_input_params_for_small_batch(inp, global_mesh)
      for inp in eval_input_p
  ]

  if task_p.train.always_use_train_for_model_init:
    if train_input_specs is None:
      raise ValueError(
          'No training input specs available, while enabling '
          '`task_p.train.always_use_train_for_model_init` requires it.')
    train_state_metadata = trainer_lib.create_train_state_metadata(
        jax_task,
        train_input_specs,
        discard_opt_states=not checkpointer.use_ema)
    input_shape_dtypes = None
  else:
    train_state_metadata = None
    # TODO(pax-dev): Investigate if we can use model input specs
    # instead of instantiating this input pipeline.
    input_shape_dtypes = instantiate(eval_input_p[0]).get_next_padded()
    input_shape_dtypes = jax.tree_util.tree_map(
        py_utils.get_global_input_shape_dtype, input_shape_dtypes)

  model_states_out = _SpmdEvalRunner.get_model_states(
      jax_task,
      global_mesh,
      init_key,
      input_shape_dtypes,
      train_state_metadata,
      checkpointer.restore_checkpoint_dir,
      checkpointer.checkpoint_type,
      checkpoint_step=checkpointer.restore_checkpoint_step,
      use_ema=checkpointer.use_ema,
      enable_auto_sharding=enable_auto_sharding)

  if task_p.train.always_use_train_for_model_init:
    assert train_state_metadata is not None
    partitioned_train_state = model_states_out
    train_state_global_shapes = train_state_metadata.padded_global_shapes
    partitioned_specs = train_state_metadata.partitioned_specs
  else:
    assert isinstance(model_states_out, tuple)
    (partitioned_train_state, partitioned_specs, train_state_global_shapes, _,
     _, input_shape_dtypes) = model_states_out

  logging.info('partitioned_train_state: %s',
               jax.tree_map(lambda x: x.shape, partitioned_train_state))

  use_gda = (
      py_utils.gda_or_jax_array() and
      checkpointer.checkpoint_type != CheckpointType.CHECKPOINT_PERSISTENCE)
  eval_one_step_fn = _SpmdEvalRunner(
      task_p, eval_input_p, jax_task, global_mesh, init_key, partitioned_specs,
      job_log_dir, eval_unpadded_global_batch_size).partition_run_one_step(
          eval_key, eval_input_p, use_gda)

  decode_once_fn = None
  input_p = None
  continuous_decode = True
  _common_eval_or_decode_loop(EvaluationMode.EVAL, checkpointer, task_p,
                              job_log_dir, global_mesh, input_p, eval_input_p,
                              eval_one_step_fn, decode_once_fn,
                              partitioned_train_state,
                              train_state_global_shapes, partitioned_specs,
                              early_stopping_fn, continuous_decode)


def decode(experiment_config: base_experiment.BaseExperiment,
           job_log_dir: epath.PathLike,
           maybe_use_persistence_checkpointing: bool,
           restore_checkpoint_dir: Optional[epath.PathLike],
           restore_checkpoint_step: Optional[int],
           continuous_decode: bool,
           run_eval: Optional[bool] = False,
           early_stopping_fn: Optional[trainer_lib.EarlyStoppingFn] = None,
           enable_auto_sharding: bool = False,
           enable_checkpoint_saving: bool = True,
           output_pickle: bool = True) -> None:
  """Runs decoding on the decoder datasets.

  Args:
    experiment_config: an instance of BaseExperiment for the experiment to
      decode.
    job_log_dir: The directory for the job logs.
    maybe_use_persistence_checkpointing: If set, it will try to use
      persistence-based checkpointing if suitable.
    restore_checkpoint_dir: The directory from which to restore checkpoint.
    restore_checkpoint_step: If set, the checkpoint step to restore. If unset,
      try to restore from the latest checkpoint if any.
    continuous_decode: whether to continuously decode on the latest ckpt.
    run_eval: whether to run evaluate() (i.e. to obtain scoring based metrics)
      as well.
    early_stopping_fn: An optional callable object for reporting metrics and
      determining whether to early stop current training. The callable object
      has signature: (metrics, running_mode, ckpt_step, is_final_ckpt) ->
      should_stop_early.
    enable_auto_sharding: Enables the XLA AutoSharding pass to generate SPMD
      shardings.
    enable_checkpoint_saving: Whether to perform checkpoint saving or not.
    output_pickle: Output .pickle file alongside the .jsonl file when decoding.
  """
  job_log_dir = epath.Path(job_log_dir)
  if restore_checkpoint_dir:
    restore_checkpoint_dir = epath.Path(restore_checkpoint_dir)

  task_p = experiment_config.task()
  task_p = typing.cast(tasks_lib.SingleTask.HParams, task_p)
  model_p = task_p.model

  checkpointer = _create_checkpointer(task_p, job_log_dir,
                                      maybe_use_persistence_checkpointing,
                                      EvaluationMode.DECODE,
                                      restore_checkpoint_dir,
                                      restore_checkpoint_step)

  if continuous_decode:
    logging.info('running continuous_decode from %s',
                 checkpointer.restore_checkpoint_dir)
  else:
    logging.info('running decode_once restored from %s',
                 checkpointer.restore_checkpoint_dir)

  decoder_inputs = experiment_config.decoder_datasets()
  eval_inputs = [v for v in experiment_config.datasets() if not v.is_training]
  if not run_eval:
    eval_inputs = []
  if not decoder_inputs and not eval_inputs:
    logging.info('No input datasets defined.')
    return

  for inp in (decoder_inputs + eval_inputs):
    if inp.num_infeed_hosts == 0:
      inp.num_infeed_hosts = jax.process_count()
    inp.infeed_host_index = jax.process_index()

  try:
    input_specs_provider = instantiate(
        experiment_config.get_input_specs_provider_params())
    train_input_specs = input_specs_provider.get_input_specs()
  except ValueError:
    logging.warning('No training input specs defined.')
    train_input_specs = None

  if model_p.mesh_shape is not None:
    train_input_specs = jax.tree_map(py_utils.get_global_input_shape_dtype,
                                     train_input_specs)
    decode_spmd_model(
        task_p,
        train_input_specs,
        decoder_inputs,
        eval_inputs,
        job_log_dir,
        typing.cast(_SpmdEvalCheckpointer, checkpointer),
        continuous_decode,
        early_stopping_fn,
        enable_auto_sharding=enable_auto_sharding)
  else:
    decode_pmap_model(
        task_p,
        train_input_specs,
        decoder_inputs,
        eval_inputs,
        job_log_dir,
        typing.cast(_PmapEvalCheckpointer, checkpointer),
        continuous_decode,
        early_stopping_fn,
        output_pickle,
        enable_checkpoint_saving=enable_checkpoint_saving)


def _merge_clu_metrics(metrics: Metrics, updated_metrics: Metrics) -> Metrics:
  """Merges existing eval metrics with updated metric data."""
  if metrics:
    if set(metrics.keys()) != set(updated_metrics.keys()):
      raise ValueError('metrics and updated_metrics keys don`t match. '
                       f'metrics keys: {metrics.keys()} '
                       f'updated_metrics keys: {updated_metrics.keys()}')

    for key in metrics:
      metrics[key] = metrics[key].merge(updated_metrics[key])
  else:
    metrics = updated_metrics
  return metrics


def decode_pmap_model(task_p: tasks_lib.SingleTask.HParams,
                      train_input_specs: Optional[NestedShapeDtypeLike],
                      input_p: Sequence[base_input.BaseInput.HParams],
                      eval_input_p: Sequence[base_input.BaseInput.HParams],
                      job_log_dir: epath.Path,
                      checkpointer: _PmapEvalCheckpointer,
                      continuous_decode: bool,
                      early_stopping_fn: Optional[
                          trainer_lib.EarlyStoppingFn] = None,
                      output_pickle: bool = True,
                      enable_checkpoint_saving: bool = True) -> None:
  """Runs the decoding on the entire decoder datasets for a PMAP model.

  Args:
    task_p: Params of the task encapsulating a the data parallel model.
    train_input_specs: Training input specs for model initialization.
    input_p: List of input params to be decoded.
    eval_input_p: List of input params to be evaluated.
    job_log_dir: Directory for the job logs.
    checkpointer: Callbacks for checkpointing.
    continuous_decode: whether to continuously decode on the latest ckpt.
    early_stopping_fn: An optional callable object for reporting metrics and
      determining whether to early stop current training. The callable object
      has signature: (metrics, running_mode, ckpt_step, is_final_ckpt) ->
      should_stop_early.
    enable_checkpoint_saving: Whether to perform checkpoint saving or not.
    output_pickle: Output .pickle file alongside the .jsonl file when decoding.
  """
  jax_task = instantiate(task_p)

  partitioned_specs = None
  global_mesh = None

  # TODO(shafey): Retrieve the seeds from the model definition instead.
  prng_key = jax.random.PRNGKey(1234)
  prng_key, eval_key = jax.random.split(prng_key)

  if task_p.train.always_use_train_for_model_init:
    train_state_metadata = trainer_lib.create_train_state_metadata(
        jax_task, train_input_specs)
    input_shape_dtypes = None
    var_weight_hparams = train_state_metadata.var_weight_hparams
  else:
    train_state_metadata = None
    # Either decoder or eval inputs is not empty.
    assert list(input_p) + list(eval_input_p)
    # _PmapEvalRunner requires drawing a sample input for restoring checkpoints.
    # We assume that either eval_input or decoder_input can be used to retrieve
    # all the model variable shapes.
    # TODO(zhangqiaorjc): If we can no longer assume variable shapes will be the
    # same regardless of which eval_input or decoder_input we use to draw the
    # sample inputs, we need to revisit the design here.
    sample_input_p = input_p[0] if input_p else eval_input_p[0]
    # TODO(pax-dev): Investigate if we can use model input specs
    # instead of instantiating this input pipeline.
    input_shape_dtypes = instantiate(sample_input_p).get_next_padded()
    var_weight_hparams = jax_task.model.abstract_init_with_metadata(
        input_shape_dtypes, do_eval=True)

  eval_one_step_fn = _PmapEvalRunner(
      task_p, eval_input_p, jax_task, eval_key,
      job_log_dir).partition_run_one_step(eval_input_p)

  # JaxContext needed for parameter sharing.
  context_p = base_layer.JaxContext.HParams(do_eval=True)
  with base_layer.JaxContext.new_context(hparams=context_p):
    trainer_lib.write_post_init_model_hparams_file(
        jax_task.model,
        var_weight_hparams,
        job_log_dir / 'decoder_out',
    )

  if continuous_decode:
    # Waits until train.decode_start_after_n_steps is reached.
    _wait_until_step(checkpointer,
                     jax_task.hparams.train.decode_start_after_n_steps)

  (partitioned_train_state, train_state_global_shapes,
   prng_key) = _PmapEvalRunner.get_model_states(jax_task, prng_key,
                                                input_shape_dtypes,
                                                train_state_metadata,
                                                checkpointer)
  prng_key, decode_key = jax.random.split(prng_key)
  prng_seed = jax.random.split(decode_key, num=jax.local_device_count())
  logging.info('decoder prng_seed: %s', prng_seed)

  inputs = [instantiate(p) for p in input_p]
  trainer_lib.check_unique_names(inputs)
  decode_once_fn = partition_decode_once_pmap_model(
      jax_task,
      task_p,
      var_weight_hparams,
      inputs,
      input_p,
      prng_seed,
      job_log_dir,
      output_pickle,
      enable_checkpoint_saving=enable_checkpoint_saving)

  _common_eval_or_decode_loop(EvaluationMode.DECODE, checkpointer, task_p,
                              job_log_dir, global_mesh, input_p, eval_input_p,
                              eval_one_step_fn, decode_once_fn,
                              partitioned_train_state,
                              train_state_global_shapes, partitioned_specs,
                              early_stopping_fn, continuous_decode)


def partition_decode_once_pmap_model(
    jax_task: tasks_lib.SingleTask,
    task_p: tasks_lib.SingleTask.HParams,
    var_weight_hparams: NestedWeightHParams,
    inputs: List[base_input.BaseInput],
    input_p: Sequence[base_input.BaseInput.HParams],
    prng_seed: JTensor,
    job_log_dir: epath.Path,
    output_pickle: bool = True,
    enable_checkpoint_saving: bool = True,
) -> Callable[[train_states.TrainState, List[SummaryWriter]],
              tuning_lib.DecodeMetrics]:

  def decode_once_fn(partitioned_train_state, summary_writers):
    with py_utils.timeit() as decode_period:
      (decode_metrics_list, processed_decode_metrics_list,
       decode_seqio_metrics_list, num_decode_steps) = (
           decode_once_pmap_model(
               jax_task,
               task_p,
               var_weight_hparams,
               inputs,
               input_p,
               prng_seed,
               job_log_dir,
               partitioned_train_state,
               summary_writers,
               output_pickle,
               enable_checkpoint_saving=enable_checkpoint_saving))
    decode_steps_per_sec = sum(num_decode_steps) / decode_period.elapsed
    return tuning_lib.DecodeMetrics(
        input_p=input_p,
        metrics_list=decode_metrics_list,
        processed_metrics_list=processed_decode_metrics_list,
        seqio_metrics_list=decode_seqio_metrics_list,
        steps_per_sec=decode_steps_per_sec)

  return decode_once_fn


def decode_once_pmap_model(
    jax_task: tasks_lib.SingleTask,
    task_p: tasks_lib.SingleTask.HParams,
    var_weight_hparams: NestedWeightHParams,
    inputs: List[base_input.BaseInput],
    input_p: Sequence[base_input.BaseInput.HParams],
    prng_seed: JTensor,
    job_log_dir: epath.Path,
    replicated_model_states: train_states.TrainState,
    summary_writers: List[SummaryWriter],
    output_pickle: bool = True,
    enable_checkpoint_saving: bool = True,
) -> Tuple[List[Optional[Dict[str, float]]],  # decode metrics.
           List[Optional[Dict[str, float]]],  # processed decode metrics.
           List[Optional[Dict[str, float]]],  # decode (seqio) metrics.
           List[int]  # performed decode steps.
          ]:
  """Runs the decoding on the entire decoder datasets for a PMAP model.

  Args:
    jax_task: instantiated model from task_p.
    task_p: Params for the task encapsulating a data parallel model.
    var_weight_hparams: Nested structure of HParams for the model weights.
    inputs: instantiated inputs.
    input_p: List of input params to be decoded.
    prng_seed: The prng seed used for decoding.
    job_log_dir: Directory for the job logs.
    replicated_model_states: A TrainState object.
    summary_writers: The summary writer objects to log summaries.
    enable_checkpoint_saving: Whether to perform checkpoint saving or not.

  Returns:
    A tuple of (a list of decode metrics,
                a list of processed decode metrics,
                a list of optional decoder (seqio) metrics.
                 list of integers as performed decode steps for each input).
      Items from each list are aligned with each input from input_p.
  """
  if not input_p:
    return [], [], [], []
  work_unit = platform.work_unit()
  model = jax_task.model
  model_p = task_p.model
  metrics_p = task_p.metrics
  if not metrics_p:
    metrics_p = base_metrics.MeanMetrics.HParams()

  step_i = int(
      py_utils.maybe_unreplicate_for_fully_replicated(
          replicated_model_states.step))

  logging.info('step=%d', step_i)

  def decode_step(mdl_states, prng_key, inputs, batch_idx):
    if task_p.fold_decode_prng_key_per_batch:
      prng_seed_decode = jax.random.fold_in(prng_key, batch_idx)
    else:
      prng_seed_decode = prng_key
    mdl_states = mdl_states.to_eval_state()
    (weighted_scalars, per_example_out,
     updated_metrics), updated_vars = trainer_lib.decode_step(
         model, mdl_states, prng_seed_decode, var_weight_hparams, inputs,
         model_p.fprop_dtype)

    weighted_scalars = decode_metrics.aggregate(weighted_scalars)
    aggregated_per_example_out = jax.lax.all_gather(
        per_example_out, axis_name=PMAP_PARALLEL_AXIS_NAME, tiled=True)

    summary_tensors = updated_vars.get(base_layer.SUMMARIES, {})
    summary_tensors = summary_utils.flatten_flax_summaries(summary_tensors)
    aggregated_summaries = summary_utils.aggregate_per_replica_summaries(
        summary_tensors)

    # We want to aggregate metrics across workers.
    # In pmap we do an all gather of the metric state across workers, and then
    # call reduce() on the metric which by default calls merge across workers.
    aggregated_metrics = {}
    for metric_name, metric in updated_metrics.items():
      aggregated_metrics[metric_name] = jax.lax.all_gather(
          metric, axis_name=PMAP_PARALLEL_AXIS_NAME).reduce()

    return (weighted_scalars, aggregated_per_example_out, aggregated_summaries,
            aggregated_metrics)

  # As an example, suppose the output leaf from trainer_lib.decoder_step()
  # for each core has shape: [per_core_batch_size, decoding_length].
  # In the all_gather we set tiled=True, so the output chunks are all
  # concatenated into the existing batch axis, so we get shape
  # [num_cores x per_core_batch_size, decoding_length].
  # In the pmap call we set out_axes=None to not have to manually unreplicate,
  # so the output of pmap_decode_step() will have the same shape.
  #
  # Example code snippet showing this:
  #   # shape (8, 3, 2)
  #   x = jnp.tile(jnp.arange(8)[:, None, None],[1, 3, 2])
  #   # shape (24, 2)
  #   z = jax.pmap(
  #       lambda y: jax.lax.all_gather(y+1, axis_name='i', tiled=True),
  #       axis_name='i', out_axes=None)(x)
  #
  # We aggregate all outputs from decode_step.
  pmap_decode_step = jax.pmap(
      decode_step,
      axis_name=PMAP_PARALLEL_AXIS_NAME,
      out_axes=(None, None, None, None))

  def decode_step_func(inputs, batch_idx):
    # TODO(pax): shall we eval all sub-models during eval?
    return pmap_decode_step(replicated_model_states, prng_seed, inputs,
                            batch_idx * jnp.ones((jax.local_device_count(),)))

  num_steps_per_input = [
      -1 if p.reset_for_eval else p.eval_loop_num_batches for p in input_p
  ]
  basedir = job_log_dir / f'{EvaluationMode.DECODE.value}_out'
  dirnames = _get_dir_names(input_p)
  filename = _get_filename(
      replicated_model_states.step, EvaluationMode.DECODE.value)
  filenames = [basedir / s / filename for s in dirnames]

  decode_metrics_list = []
  processed_decode_metrics_list = []
  seqio_metrics_list = []
  num_decode_steps = []

  for split, num_split_steps in enumerate(num_steps_per_input):
    if _can_load_written_outputs(job_log_dir, input_p[split].name,
                                 EvaluationMode.DECODE, step_i):
      logging.info('Decoding on input %s at step %d already done, skipping.',
                   input_p[split].name, step_i)
      decode_metrics_list.append(None)
      processed_decode_metrics_list.append(None)
      seqio_metrics_list.append(None)
      num_decode_steps.append(0)
      continue
    logging.info('Start decoding on input %s', input_p[split].name)
    step_num = 0
    # decode_metrics and process_decode_metrics work on WeightedScalars
    # which are string -> (value, weight) pairs where value and weight
    # scalars. These metrics are configured on the task.
    decode_metrics = instantiate(metrics_p)
    process_decode_metrics = instantiate(metrics_p)

    # metrics and processed_metrics are dictionaries of
    # strings -> clu_metrics.Metric objects. metrics is returned from decode()
    # and processed_metrics is returned from process_decode_out.
    metrics = {}
    processed_metrics = {}
    processed_decodes = []
    all_summary_tensors = collections.defaultdict(list)
    while num_split_steps < 0 or step_num < num_split_steps:
      step_num += 1
      try:
        batch = inputs[split].get_next()
      except (tf.errors.OutOfRangeError, StopIteration):
        inputs[split].reset()
        break
      batch = inputs[split].reshard_for_pmap(batch)
      (batch_metrics, out, summary_tensors,
       updated_metrics) = decode_step_func(batch, batch_idx=step_num)
      for key, tensor in summary_utils.flatten_summary_dict(summary_tensors):
        all_summary_tensors[key].append(tensor)
      # we store the metric directly as it has already been aggregated in
      # side decode_step_fun
      decode_metrics.store(batch_metrics)
      logging.info('Finished decoding input batch %d for %s',
                   step_num, input_p[split].name)

      # Merge clu.metrics to update for each minibatch.
      metrics = _merge_clu_metrics(metrics, updated_metrics)

      # Run `process_decode_out` on CPU device as its implementation is not
      # expected to be JIT friendly. Since we keep track of its outputs, we also
      # don't want on-device allocation as would eventually lead to HBM OOM.
      if jax.process_index() == 0:
        with jax.default_device(jax.devices('cpu')[0]):
          process_decode_output = model.process_decode_out(
              inputs[split], jax.tree_map(np.asarray, out))

        (processed_scalars, processed_out,
         processed_metric_updates) = process_decode_output
        process_decode_metrics.store(processed_scalars)
        processed_decodes.extend(processed_out)
        if processed_metric_updates:
          processed_metrics = _merge_clu_metrics(processed_metrics,
                                                 processed_metric_updates)

        logging.info('Finished processing decoded input batch %d', step_num)

      work_unit.set_task_status(
          f'Finished decoding on {input_p[split].name} (batches={step_num})')
      logging.info('Finished decoding on %s (batches=%s)',
                   input_p[split].name, step_num)

    # Now the decode loop of multiple batches on current dataset is done,
    # we start to aggregate copmuted metrics and put them in summary.
    seqio_metric_values = None
    if seqio_input.should_process_outputs(inputs[split]):
      logging.info('Finished processing all %d examples.',
                   len(processed_decodes))
      seqio_metric_values = seqio_input.process_outputs(
          inputs[split],
          processed_decodes,
          summary_writers[split],
          seqio_input.MetricType.PREDICT,
          step_i,
          plain_text_output_fname=f'{filenames[split]}.txt')

    # Convert metrics to Dict[str, clu_values.Value] for summary writing.
    metric_values = metric_utils.compute_metric_values(metrics)
    process_metric_values = metric_utils.compute_metric_values(
        processed_metrics)

    with summary_writers[split].as_default():
      logging.info('Summarizing of decode_metrics.')
      decode_metric_dict = decode_metrics.summarize(step_i, 'decode_metrics')
      logging.info('Summarizing of process_decode_metrics.')
      processed_metric_dict = process_decode_metrics.summarize(
          step_i, 'process_decode_metrics')
      for key, tensor in all_summary_tensors.items():
        summary_type = base_layer.get_summary_type_from_key(key)
        summary_utils.write_summary_tensor(step_i, key, np.array(tensor),
                                           summary_type)
      metric_utils.write_clu_metric_summaries(metric_values, step_i)
      metric_utils.write_clu_metric_summaries(process_metric_values, step_i)

    if (jax.process_index() == 0 and
        not flags.FLAGS.pax_only_aggregate_summaries):
      dir_path = basedir / dirnames[split]
      dir_path.mkdir(parents=True, exist_ok=True)
      output_file = filenames[split]
      logging.info('Writing decoder output to %s with %d entries', output_file,
                   len(processed_decodes))
      io_utils.write_key_value_pairs(
          output_file, processed_decodes, output_pickle)

    merged_decode_metrics = metric_utils.update_float_dict(
        metric_utils.as_float_dict(decode_metric_dict),
        metric_utils.as_float_dict(metric_values))
    decode_metrics_list.append(merged_decode_metrics)

    merged_processed_decode_metrics = metric_utils.update_float_dict(
        metric_utils.as_float_dict(processed_metric_dict),
        metric_utils.as_float_dict(process_metric_values))
    processed_decode_metrics_list.append(merged_processed_decode_metrics)
    seqio_metrics_list.append(seqio_metric_values)
    num_decode_steps.append(step_num)

    # Track metric specified by task_p.track_decoder_metric.
    if task_p.track_decoder_metric:
      _find_and_maybe_update_tracked_metric(
          basedir,
          split,
          dirnames,
          step_i,
          input_p,
          replicated_model_states,
          task_p, [merged_decode_metrics, merged_processed_decode_metrics],
          enable_checkpoint_saving=enable_checkpoint_saving)

  return (decode_metrics_list, processed_decode_metrics_list,
          seqio_metrics_list, num_decode_steps)


def decode_spmd_model(task_p: tasks_lib.SingleTask.HParams,
                      train_input_specs: Optional[NestedShapeDtypeLike],
                      input_p: Sequence[base_input.BaseInput.HParams],
                      eval_input_p: Sequence[base_input.BaseInput.HParams],
                      job_log_dir: epath.Path,
                      checkpointer: _SpmdEvalCheckpointer,
                      continuous_decode: bool,
                      early_stopping_fn: Optional[
                          trainer_lib.EarlyStoppingFn] = None,
                      enable_auto_sharding: bool = False) -> None:
  """Runs the decoding on the entire decoder datasets for SPMD model.

  Args:
    task_p: Params for the task that encapsulates an SPMD model.
    train_input_specs: Training input specs for model initialization.
    input_p: List of input params to be decoded.
    eval_input_p: List of input params to be evaluated.
    job_log_dir: Directory for the job logs.
    checkpointer: The checkpointer to use
    continuous_decode: whether to continuously decode on the latest ckpt.
    early_stopping_fn: An optional callable object for reporting metrics and
      determining whether to early stop current training. The callable object
      has signature: (metrics, running_mode, ckpt_step, is_final_ckpt) ->
      should_stop_early.
    enable_auto_sharding: Enables the XLA AutoSharding pass to generate SPMD
      shardings.
  """
  # TODO(bf-jax): Retrieve the seeds from the model definition instead.
  prng_key = jax.random.PRNGKey(1234)
  prng_key, init_key, eval_key = jax.random.split(prng_key, 3)

  jax_task = instantiate(task_p)

  model_p = task_p.model
  device_mesh = py_utils.create_device_mesh(model_p.ici_mesh_shape,
                                            model_p.dcn_mesh_shape)
  global_mesh = maps.Mesh(device_mesh, model_p.mesh_axis_names)

  unpadded_global_batch_size = None
  if input_p:
    input_p_0 = input_p[0]
    unpadded_global_batch_size = (
        input_p_0.cls.get_batch_size(input_p_0) * input_p_0.num_infeed_hosts)
  input_p = [
      trainer_lib.adjust_input_params_for_small_batch(inp, global_mesh)
      for inp in input_p
  ]
  eval_unpadded_global_batch_size = None
  if eval_input_p:
    eval_input_p_0 = eval_input_p[0]
    eval_unpadded_global_batch_size = (
        eval_input_p_0.cls.get_batch_size(eval_input_p_0) *
        eval_input_p_0.num_infeed_hosts)
  eval_input_p = [
      trainer_lib.adjust_input_params_for_small_batch(inp, global_mesh)
      for inp in eval_input_p
  ]

  # Either decoder or eval inputs is not empty.
  assert list(input_p) + list(eval_input_p)
  sample_input_p = input_p[0] if input_p else eval_input_p[0]
  input_shape_dtypes = instantiate(sample_input_p).get_next_padded()
  input_shape_dtypes = jax.tree_util.tree_map(py_utils.get_global_input_shape_dtype,
                                        input_shape_dtypes)
  inputs = [instantiate(p) for p in input_p]
  trainer_lib.check_unique_names(inputs)

  use_gda = (
      py_utils.gda_or_jax_array() and
      checkpointer.checkpoint_type != CheckpointType.CHECKPOINT_PERSISTENCE)

  if jax_task.hparams.train.always_use_train_for_model_init:
    if train_input_specs is None:
      raise ValueError(
          'No training input specs available, while enabling '
          '`task_p.train.always_use_train_for_model_init` requires it.')
    train_state_metadata = trainer_lib.create_train_state_metadata(
        jax_task,
        train_input_specs,
        discard_opt_states=not checkpointer.use_ema)

    var_weight_hparams = train_state_metadata.var_weight_hparams
  else:
    train_state_metadata = None
    var_weight_hparams = jax_task.model.abstract_init_with_metadata(
        input_shape_dtypes, do_eval=True)

  if continuous_decode:
    # Waits until train.decode_start_after_n_steps is reached.
    _wait_until_step(checkpointer,
                     jax_task.hparams.train.decode_start_after_n_steps)

  # _SpmdEvalRunner requires drawing a sample input for restoring checkpoints.
  # We assume that either eval_input or decoder_input can be used to retrieve
  # all the model variable shapes.

  prng_key, init_key = jax.random.split(prng_key, 2)
  # TODO(zhangqiaorjc): If we can no longer assume variable shapes will be the
  # same regardless of which eval_input or decoder_input we use to draw the
  # sample inputs, we need to revisit the design here.
  model_states_out = _SpmdEvalRunner.get_model_states(
      jax_task,
      global_mesh,
      init_key,
      input_shape_dtypes,
      train_state_metadata,
      checkpointer.restore_checkpoint_dir,
      checkpointer.checkpoint_type,
      checkpoint_step=checkpointer.restore_checkpoint_step,
      is_decode=True,
      use_ema=checkpointer.use_ema,
      unpadded_global_batch_size=unpadded_global_batch_size,
      enable_auto_sharding=enable_auto_sharding)

  if jax_task.hparams.train.always_use_train_for_model_init:
    assert train_state_metadata is not None
    prng_key, init_key = jax.random.split(prng_key, 2)
    partitioned_train_state = model_states_out
    partitioned_specs = train_state_metadata.partitioned_specs
    train_state_global_shapes = train_state_metadata.padded_global_shapes
    decode_step_fn, inputs_partition_specs, _ = (
        trainer_lib.get_partitioned_spmd_model_step_fn(
            jax_task, trainer_lib.RunningMode.DECODE, global_mesh, init_key,
            input_shape_dtypes, train_state_metadata.input_shape_dtype,
            train_state_metadata.partitioned_specs.to_eval_state()))
  else:
    assert isinstance(model_states_out, tuple)
    (partitioned_train_state, partitioned_specs, train_state_global_shapes,
     decode_step_fn, inputs_partition_specs,
     input_shape_dtypes) = model_states_out

  decode_once_fn = partition_decode_once_spmd_model(
      jax_task, task_p, inputs, input_p, job_log_dir, prng_key, global_mesh,
      decode_step_fn, use_gda, input_shape_dtypes, inputs_partition_specs)
  eval_one_step_fn = _SpmdEvalRunner(
      task_p, eval_input_p, jax_task, global_mesh, init_key,
      partitioned_specs, job_log_dir,
      eval_unpadded_global_batch_size).partition_run_one_step(
          eval_key, eval_input_p, use_gda)
  trainer_lib.write_post_init_model_hparams_file(jax_task.model,
                                                 var_weight_hparams,
                                                 job_log_dir / 'decoder_out')

  _common_eval_or_decode_loop(EvaluationMode.DECODE, checkpointer, task_p,
                              job_log_dir, global_mesh, input_p, eval_input_p,
                              eval_one_step_fn, decode_once_fn,
                              partitioned_train_state,
                              train_state_global_shapes, partitioned_specs,
                              early_stopping_fn, continuous_decode)

def partition_decode_once_spmd_model(
    jax_task: tasks_lib.SingleTask,
    task_p: tasks_lib.SingleTask.HParams,
    inputs: List[base_input.BaseInput],
    input_p: Sequence[base_input.BaseInput.HParams],
    job_log_dir: epath.Path,
    prng_key: JTensor,
    global_mesh: maps.Mesh,
    decode_step_fn: Callable[[NestedJTensor, JTensor, NestedJTensor],
                             Tuple[Tuple[NestedMap, NestedMap], NestedMap]],
    use_gda: bool,
    inputs_shape: pytypes.NestedShapeDtypeStruct,
    inputs_partition_specs: NestedPartitionSpec,
) -> Callable[[train_states.TrainState, List[SummaryWriter]],
              tuning_lib.DecodeMetrics]:
  def decode_once_fn(partitioned_train_state, summary_writers):
    with py_utils.timeit() as decode_period:
      (decode_metrics_list, processed_decode_metrics_list,
       decode_seqio_metrics_list, num_decode_steps) = decode_once_spmd_model(
           jax_task, task_p, inputs, input_p, job_log_dir,
           partitioned_train_state, summary_writers, prng_key, global_mesh,
           decode_step_fn, use_gda, inputs_shape, inputs_partition_specs)
    decode_steps_per_sec = sum(num_decode_steps) / decode_period.elapsed
    return tuning_lib.DecodeMetrics(
        input_p=input_p,
        metrics_list=decode_metrics_list,
        processed_metrics_list=processed_decode_metrics_list,
        seqio_metrics_list=decode_seqio_metrics_list,
        steps_per_sec=decode_steps_per_sec)

  return decode_once_fn


def _is_shape_dtype_struct(x):
  """Indicates whether the input is of type ShapeDtypeStruct or not."""
  return isinstance(x, jax.ShapeDtypeStruct)


def _extract_common_specs(x, y, is_leaf=_is_shape_dtype_struct):
  """Extracts common specs given two pytrees."""
  if is_leaf(x):
    return y
  elif isinstance(x, dict):
    if not isinstance(y, dict):
      raise ValueError(f'Tree mismatch (`x={x}` vs. `y={y}`.')
    out = {}
    for k in x:
      if k in y:
        out[k] = _extract_common_specs(x[k], y[k], is_leaf)
    return type(x)(out)
  elif isinstance(x, (list, tuple)):
    if not isinstance(y, (list, tuple)) or len(x) != len(y):
      raise ValueError(f'Tree mismatch (`x={x}` vs. `y={y}`.')
    out = []
    for a, b in zip(x, y):
      out.append(_extract_common_specs(a, b, is_leaf))
    return type(x)(out)
  else:
    raise ValueError(f'Unsupported type `{type(x)}` in tree `{x}`.')


def decode_once_spmd_model(
    jax_task: tasks_lib.SingleTask,
    task_p: tasks_lib.SingleTask.HParams,
    inputs: List[base_input.BaseInput],
    input_p: Sequence[base_input.BaseInput.HParams],
    job_log_dir: epath.Path,
    train_state: train_states.TrainState,
    summary_writers: List[SummaryWriter],
    prng_key: JTensor,
    global_mesh: maps.Mesh,
    decode_step_fn: Callable[[NestedJTensor, JTensor, NestedJTensor],
                             Tuple[Tuple[NestedMap, NestedMap], NestedMap]],
    use_gda: bool,
    inputs_shape: pytypes.NestedShapeDtypeStruct,
    inputs_partition_specs: NestedPartitionSpec,
) -> Tuple[List[Optional[Dict[str, float]]],  # decode metrics.
           List[Optional[Dict[str, float]]],  # processed decode metrics.
           List[Optional[Dict[str, float]]],  # decode (seqio) metrics.
           List[int]  # performed decode steps.
          ]:
  """Runs the decoding once on the entire decoder datasets for an SPMD model.

  Args:
    jax_task: instantiated model from task_p.
    task_p: Params for the task that encapsulates an SPMD model.
    inputs: instantiated inputs.
    input_p: List of input params to be decoded.
    job_log_dir: Directory for the job logs.
    train_state: A TrainState object.
    summary_writers: The summary writer objects to log summaries.
    prng_key: The prng key used for decoding.
    global_mesh: the global mesh.
    decode_step_fn: pjit'ed decode function.
    use_gda: bool, whether GDA is used.
    inputs_shape: nested map of shapes of inputs.
    inputs_partition_specs: Partition spec for inputs.

  Returns:
    A tuple of (a list of decode metrics,
                a list of processed decode metrics,
                a list of optional decoder (seqio) metrics.
                 list of integers as performed decode steps for each input).
      Items from each list are aligned with each input from input_p.
  """
  work_unit = platform.work_unit()
  metrics_p = task_p.metrics
  if not metrics_p:
    metrics_p = base_metrics.MeanMetrics.HParams()

  step_i = int(
      py_utils.maybe_unreplicate_for_fully_replicated(train_state.step))
  basedir = job_log_dir / f'{EvaluationMode.DECODE.value}_out'
  dirnames = _get_dir_names(input_p)
  filenames = [
      basedir / s / _get_filename(step_i, EvaluationMode.DECODE.value)
      for s in dirnames
  ]

  logging.info('partitioned_train_state: %s',
               jax.tree_map(lambda x: x.shape, train_state))
  # We do not fold in jax.process_index in contrast to the pmap version and
  # use a single global key instead to rely on pjit to split for different
  # replicas.
  logging.info('decode prng_key: %s', prng_key)
  spmd_decode_step_fn = functools.partial(decode_step_fn,
                                          train_state.to_eval_state(), prng_key)

  num_steps_per_input = [
      -1 if p.reset_for_eval else p.eval_loop_num_batches for p in input_p
  ]
  decode_metrics_list = []
  processed_decode_metrics_list = []
  seqio_metrics_list = []
  num_decode_steps = []

  for split, num_split_steps in enumerate(num_steps_per_input):
    if _can_load_written_outputs(job_log_dir, input_p[split].name,
                                 EvaluationMode.DECODE, step_i):
      logging.info('Decoding on input %s at step %d already done, skipping.',
                   input_p[split].name, step_i)
      decode_metrics_list.append(None)
      processed_decode_metrics_list.append(None)
      seqio_metrics_list.append(None)
      num_decode_steps.append(0)
      continue
    logging.info('Start decoding on input %s', input_p[split].name)
    step_num = 0
    # decode_metrics and process_decode_metrics work on WeightedScalars
    # which are string -> (value, weight) pairs where value and weight
    # scalars. These metrics are configured on the task.
    decode_metrics = instantiate(metrics_p)
    process_decode_metrics = instantiate(metrics_p)

    # metrics and processed_metrics are dictionaries of
    # strings -> clu_metrics.Metric objects. metrics is returned from decode()
    # and processed_metrics is returned from process_decode_out.
    metrics = {}
    processed_metrics = {}
    processed_decodes = []
    all_summary_tensors = collections.defaultdict(list)
    while num_split_steps < 0 or step_num < num_split_steps:
      step_num += 1
      try:
        batch = inputs[split].get_next_padded()
      except (tf.errors.OutOfRangeError, StopIteration):
        inputs[split].reset()
        break
      if jax_task.hparams.train.always_use_train_for_model_init:
        if use_gda:
          inputs_shape = jax.tree_util.tree_map(
              py_utils.get_global_input_shape_dtype, batch)
          inputs_partition_specs_common = _extract_common_specs(
              inputs_shape, inputs_partition_specs)
          batch = py_utils.create_gda(batch, inputs_shape, global_mesh,
                                      inputs_partition_specs_common)
      else:
        if use_gda or inputs[split].hparams.experimental_remote_input:
          batch = inputs[split].reshard_for_spmd(batch, inputs_shape,
                                                 global_mesh,
                                                 inputs_partition_specs)
      (weighted_scalars, out,
       updated_metrics), updated_vars = spmd_decode_step_fn(batch)

      # Cross host synchronization happens at this point.
      py_utils.sync_global_devices(f'spmd_decode_step_fn{step_num}')
      # Output is fully replicated now, so it's ok to unreplicate it by
      # retrieving from device 0 only.
      out = py_utils.maybe_unreplicate_for_fully_replicated(out)
      weighted_scalars = py_utils.maybe_unreplicate_for_fully_replicated(
          weighted_scalars)

      # Because outputs of the decode step in pjit are annotated to be on the
      # GDA, they are already fully replicated across shards and we can just
      # unreplicate.
      # This also means we don't need to call an all_gather and a reduce()
      # on each clu.metric like we do in pmap mode.
      updated_metrics = py_utils.maybe_unreplicate_for_fully_replicated(
          updated_metrics)

      # Merge clu.metrics to update for each minibatch.
      metrics = _merge_clu_metrics(metrics, updated_metrics)

      summary_tensors = updated_vars.get(base_layer.SUMMARIES, {})
      summary_tensors = summary_utils.flatten_flax_summaries(summary_tensors)
      del updated_vars  # release GDA memory allocations

      summary_tensors = py_utils.maybe_unreplicate_for_fully_replicated(
          summary_tensors)
      for key, tensor in summary_utils.flatten_summary_dict(summary_tensors):
        all_summary_tensors[key].append(tensor)

      logging.info('Finished decoding input batch %d for %s',
                   step_num, input_p[split].name)
      if jax.process_index() != 0:
        continue
      weighted_scalars = jax.tree_map(np.array, weighted_scalars)
      decode_metrics.store(weighted_scalars)

      # Run `process_decode_out` on CPU device as its implementation is not
      # expected to be JIT friendly. Since we keep track of its outputs, we also
      # don't want on-device allocation as would eventually lead to HBM OOM.
      with jax.default_device(jax.devices('cpu')[0]):
        process_decode_output = jax_task.model.process_decode_out(
            inputs[split], jax.tree_map(np.asarray, out))

      (process_weighted_scalars, processed,
       processed_metric_updates) = process_decode_output

      process_decode_metrics.store(process_weighted_scalars)
      processed_decodes.extend(processed)
      if processed_metric_updates:
        processed_metrics = _merge_clu_metrics(processed_metrics,
                                               processed_metric_updates)

      logging.info('Finished processing decoded input batch %d', step_num)

    logging.info('Finished decoding on %s (batches=%s)',
                 input_p[split].name, step_num)

    # Now the decode loop of multiple batches on current dataset is done,
    # we start to aggregate copmuted metrics and put them in summary.
    seqio_metric_values = None
    if seqio_input.should_process_outputs(inputs[split]):
      logging.info('Finished processing all %d examples.',
                   len(processed_decodes))
      seqio_metric_values = seqio_input.process_outputs(
          inputs[split],
          processed_decodes,
          summary_writers[split],
          seqio_input.MetricType.PREDICT,
          step_i,
          plain_text_output_fname=f'{filenames[split]}.txt')

    # Convert metrics to Dict[str, clu_values.Value] for summary writing.
    metric_values = metric_utils.compute_metric_values(metrics)
    process_metric_values = metric_utils.compute_metric_values(
        processed_metrics)

    with summary_writers[split].as_default():
      logging.info('Summarizing of decode_metrics.')
      decode_metric_dict = decode_metrics.summarize(step_i, 'decode_metrics')
      logging.info('Summarizing of process_decode_metrics.')
      processed_metric_dict = process_decode_metrics.summarize(
          step_i, 'process_decode_metrics')
      for key, tensor in all_summary_tensors.items():
        summary_type = base_layer.get_summary_type_from_key(key)
        summary_utils.write_summary_tensor(step_i, key, np.array(tensor),
                                           summary_type)
      metric_utils.write_clu_metric_summaries(metric_values, step_i)
      metric_utils.write_clu_metric_summaries(process_metric_values, step_i)

    if jax.process_index() == 0:
      dir_path = basedir / dirnames[split]
      dir_path.mkdir(parents=True, exist_ok=True)
      output_file = filenames[split]
      logging.info('Writing decoder output to %s with %d entries', output_file,
                   len(processed_decodes))
      io_utils.write_key_value_pairs(output_file, processed_decodes)

    work_unit.set_task_status(f'Finished processing decoded input batch for '
                              f'{input_p[split].name}')

    decode_metrics_list.append(
        metric_utils.update_float_dict(
            metric_utils.as_float_dict(decode_metric_dict),
            metric_utils.as_float_dict(metric_values)))
    processed_decode_metrics_list.append(
        metric_utils.update_float_dict(
            metric_utils.as_float_dict(processed_metric_dict),
            metric_utils.as_float_dict(process_metric_values)))
    seqio_metrics_list.append(seqio_metric_values)
    num_decode_steps.append(step_num)

    # Track metric specified by task_p.track_decoder_metric.
    if task_p.track_decoder_metric:
      logging.warn('Decoder metric tracking is not implemented yet for pjit '
                   'models. Ignoring metric tracking.')

  return (decode_metrics_list, processed_decode_metrics_list,
          seqio_metrics_list, num_decode_steps)


def _common_eval_or_decode_loop(
    mode: io_utils.EvaluationMode,
    checkpointer: _EvalCheckpointer,
    task_p: tasks_lib.SingleTask.HParams,
    job_log_dir: epath.Path,
    global_mesh: Optional[maps.Mesh],
    input_p: Optional[Sequence[base_input.BaseInput.HParams]],
    eval_input_p: Sequence[base_input.BaseInput.HParams],
    eval_one_step_fn: Callable[..., tuning_lib.EvalMetrics],
    decode_once_fn: Optional[Callable[..., tuning_lib.DecodeMetrics]],
    partitioned_train_state: train_states.TrainState,
    train_state_global_shapes: train_states.TrainState,
    partitioned_specs: Optional[train_states.TrainState],
    early_stopping_fn: Optional[trainer_lib.EarlyStoppingFn],
    continuous_decode: bool,
):
  last_checkpoint_step = checkpointer.retrieve_latest_checkpoint_step()
  logging.info('Evaluation loop starting...')
  summary_base_dir = job_log_dir / 'summaries'
  if input_p:
    summary_decode_dirs = [
        summary_base_dir / f'decode_test_{p.name}' for p in input_p
    ]
  summary_eval_dirs = [
      summary_base_dir / f'eval_test_{p.name}' for p in eval_input_p
  ]
  with contextlib.ExitStack() as exit_stack:
    if input_p:
      summary_writers = [
          exit_stack.enter_context(summary_utils.get_summary_writer(d))
          for d in summary_decode_dirs
      ]
    eval_summary_writers = [
        exit_stack.enter_context(summary_utils.get_summary_writer(d))
        for d in summary_eval_dirs
    ]

    while True:
      with io_utils.checkpoint_progress(job_log_dir, last_checkpoint_step,
                                        mode):
        decode_metrics = None
        if input_p:
          logging.info('Decoding step %s ckpt ...', last_checkpoint_step)
          decode_metrics = decode_once_fn(partitioned_train_state,
                                          summary_writers)

        logging.info('Evaling step %s ckpt ...', last_checkpoint_step)
        eval_metrics = eval_one_step_fn(partitioned_train_state,
                                        eval_summary_writers)

      if not continuous_decode:
        break

      if last_checkpoint_step is not None:
        exceeded_ckpt = last_checkpoint_step + task_p.train.save_interval_steps
        is_last_ckpt = exceeded_ckpt > task_p.train.num_train_steps
        if tuning_lib.should_early_stop(
            early_stopping_fn,
            last_checkpoint_step,
            is_last_ckpt,
            eval_metrics=eval_metrics,
            decode_metrics=decode_metrics):
          logging.info(
              'Early stopped at checkpoint step %d by the'
              'tuner, while the num_train_steps is %d', last_checkpoint_step,
              task_p.train.num_train_steps)
          break
        if is_last_ckpt:
          break
      # Release partitioned_train_state.
      jax.tree_util.tree_map(lambda x: x.delete(), partitioned_train_state)
      del partitioned_train_state
      new_checkpoint_step = checkpointer.wait_for_new_step(last_checkpoint_step)
      partitioned_train_state = checkpointer.load_checkpoint_for_step(
          new_checkpoint_step, train_state_global_shapes, partitioned_specs,
          global_mesh)
      last_checkpoint_step = new_checkpoint_step


def _maybe_update_tracked_metric(
    m_value: float,
    step: int,
    tracker_dir_path: epath.Path,
    tracked_metric: str,
    min_or_max: tasks_lib.SingleTask.TrackDecoderMetricMode,
    data_partition_name: str,
    replicated_model_states: train_states.TrainState,
    enable_checkpoint_saving: bool = True) -> None:
  """Update tracked metric if new value (m_value) is lower that the stored one.

  Also updates the status file maintained by the tracker and writes
  new checkpoint assets in the same tracker directory.

  Args:
    m_value: new metric value.
    step: current training step.
    tracker_dir_path: directory where the tracker should store the status file
      and also write and garbage collect checkpoint assets.
    tracked_metric: name of metric being tracked, e.g. 'wer'.
    min_or_max: min or max tracker.
    data_partition_name: data partition on which the value of the metric is
      being tracked.
    replicated_model_states: replicated model states used to save the best
      checkpoint.
    enable_checkpoint_saving: Whether to perform checkpoint saving or not.
  """
  if jax.process_index() == 0:
    tracker_dir_path.mkdir(parents=True, exist_ok=True)
    initial_value = sys.float_info.max
    if min_or_max == tasks_lib.SingleTask.TrackDecoderMetricMode.MAX:
      initial_value = -sys.float_info.max
    tracker = trk_utils.MetricTracker(
        dir_name=tracker_dir_path,
        metric_name=tracked_metric,
        metric_partition=data_partition_name,
        initial_metric_value=initial_value)
    if ((min_or_max == tasks_lib.SingleTask.TrackDecoderMetricMode.MIN and
         m_value < tracker.metric_value) or
        (min_or_max == tasks_lib.SingleTask.TrackDecoderMetricMode.MAX and
         m_value > tracker.metric_value)):
      logging.info('Updating tracked %s value and checkpoint.', tracked_metric)
      tracker.update(value=m_value, global_step=step)
      # Also save checkpoint; we just need to save the first model replica.
      # WARNING: the checkpoint saved here will not contain optimizer state
      # if it is written by a separate decoding job; if decoding is done
      # interleaved with training as part of the trainer then it will
      # contain them.
      # Decoding with this checkpoint may thus produce different results
      # than those obtained during training if the model state cannot be
      # fully recovered due to the missing optimizer state, e.g. when using
      # EMA during training and separate decoding jobs.
      # TODO(ciprianchelba): specify the checkpoint format and/or async
      # checkpointing.
      if enable_checkpoint_saving:
        unreplicated_model_states = jax.tree_map(lambda x: x[0],
                                                 replicated_model_states)
        checkpoints.save_checkpoint(unreplicated_model_states, tracker_dir_path)


def _find_and_maybe_update_tracked_metric(
    basedir: epath.Path,
    split: int,
    dirnames: Sequence[epath.Path],
    step_i: int,
    input_p: Sequence[base_input.BaseInput.HParams],
    replicated_model_states: train_states.TrainState,
    task_p: tasks_lib.SingleTask.HParams,
    decode_metrics_list: List[Dict[str, float]],
    enable_checkpoint_saving: bool = True) -> None:
  tracked_metric = task_p.track_decoder_metric
  track_min_or_max = task_p.track_decoder_metric_min_or_max
  if not track_min_or_max:
    raise ValueError(
        'Must also set track_decoder_metric_min_or_max when '
        f'enabling metric tracking: {task_p}')
  m_value = None
  for d in decode_metrics_list:
    if tracked_metric in d:
      m_value = d[tracked_metric]
      break

  if m_value:
    tracker_dir_path = (
        basedir / dirnames[split] /
        f'{tracked_metric}_{track_min_or_max}_tracker')
    _maybe_update_tracked_metric(
        m_value,
        step_i,
        tracker_dir_path,
        tracked_metric,
        track_min_or_max,
        input_p[split].name,
        replicated_model_states,
        enable_checkpoint_saving=enable_checkpoint_saving)
  else:
    logging.info('Cannot track metric %s on input %s.', tracked_metric,
                 input_p[split].name)


def infer_and_write(experiment_config: base_experiment.BaseExperiment,
                    job_log_dir: epath.Path) -> None:
  """Generates output from a model and writes it out.

  Args:
    experiment_config: an instance of BaseExperiment for the experiment with
      output generators configured.
    job_log_dir: The base directory for writing the outputs.
  """
  task_p = experiment_config.task()
  task_p = typing.cast(tasks_lib.SingleTask.HParams, task_p)
  model_p = task_p.model
  inputs_p = experiment_config.decoder_datasets()

  checkpointer = _create_checkpointer(
      task_p,
      job_log_dir,
      maybe_use_persistence_checkpointing=False,
      mode=None,
      restore_checkpoint_dir=task_p.infer_writer.restore_checkpoint_dir,
      restore_checkpoint_step=task_p.infer_writer.restore_checkpoint_step)
  for inp in inputs_p:
    if inp.num_infeed_hosts == 0:
      inp.num_infeed_hosts = jax.process_count()
    inp.infeed_host_index = jax.process_index()

  try:
    input_specs_provider = instantiate(
        experiment_config.get_input_specs_provider_params())
    train_input_specs = input_specs_provider.get_input_specs()
  except ValueError:
    logging.warning('No training input specs defined.')
    train_input_specs = None

  if model_p.mesh_shape is not None:
    # TODO(b/238416854): add support for SPMD models
    raise NotImplementedError('SPMD infer_and_write not implemented yet')
  else:
    checkpointer = typing.cast(_PmapEvalCheckpointer, checkpointer)
    infer_and_write_pmap(task_p, train_input_specs, inputs_p, job_log_dir,
                         checkpointer)


def infer_and_write_pmap(task_p: tasks_lib.SingleTask.HParams,
                         train_input_specs: Optional[NestedShapeDtypeLike],
                         inputs_p: Sequence[base_input.BaseInput.HParams],
                         job_log_dir: epath.Path,
                         checkpointer: _PmapEvalCheckpointer) -> None:
  """Runs the infer_and_write for each of the inputs given task in pmap."""
  task = instantiate(task_p)

  prng_key = jax.random.PRNGKey(0)
  infer_writer_p = task_p.infer_writer

  if not inputs_p:
    return

  if task_p.train.always_use_train_for_model_init:
    if train_input_specs is None:
      raise ValueError(
          'No training input specs available, while enabling '
          '`task_p.train.always_use_train_for_model_init` requires it.')
    train_state_metadata = trainer_lib.create_train_state_metadata(
        task, train_input_specs)
    var_weight_hparams = train_state_metadata.var_weight_hparams
    inputs_sample = None
  else:
    train_state_metadata = None
    # TODO(pax-dev): Investigate if we can use model input specs
    # instead of instantiating this input pipeline or re-using one of the
    # input pipelines below.
    inputs_sample = instantiate(inputs_p[0]).get_next_padded()
    var_weight_hparams = task.model.abstract_init_with_metadata(
        inputs_sample, do_eval=True)

  replicated_model_states, _, prng_key = _PmapEvalRunner.get_model_states(
      task, prng_key, inputs_sample, train_state_metadata, checkpointer)

  @functools.partial(jax.pmap, axis_name=PMAP_PARALLEL_AXIS_NAME, out_axes=None)
  def infer_pmap_step(mdl_states, prng_seeds, input_batch):
    outputs = task.inference_runner.infer(mdl_states, prng_seeds,
                                          var_weight_hparams, input_batch)
    # tiled=True folds in first axis into second axis [2,8,5] -> [2*8,5]
    replicated_outputs = jax.lax.all_gather(
        outputs, axis_name=PMAP_PARALLEL_AXIS_NAME, tiled=True)

    return replicated_outputs

  # Instantiate inputs to infer on
  inputs = [instantiate(p) for p in inputs_p]
  trainer_lib.check_unique_names(inputs)
  num_steps = [
      -1 if p.reset_for_eval else p.eval_loop_num_batches for p in inputs_p
  ]

  for input_p, input_gen, num_steps in zip(inputs_p, inputs, num_steps):
    logging.info('Starting output generation on input "%s"', input_p.name)

    # Feed each (device, input) pair a unique seed
    prng_key, output_seed = jax.random.split(prng_key)
    output_seeds = jax.random.split(output_seed, jax.local_device_count())

    if num_steps > 0:
      logging.info('total number of steps: %d', num_steps)

    # Only write from one process
    dirname = job_log_dir / 'output' / input_p.name
    fq_filename = dirname / 'output'
    if jax.process_index() == 0:
      # Create output dirs if DNE
      if not dirname.exists():
        dirname.mkdir(parents=True, exist_ok=True)

      # Write example schema, metadata, and serialized example protos
      logging.info('writing output to %s', fq_filename)
      features_dict = tfds.features.FeaturesDict(
          task.inference_runner.output_schema)
      features_dict.save_config(dirname.as_posix())
      tfds.core.MetadataDict(
          restore_checkpoint_dir=infer_writer_p.restore_checkpoint_dir,
          restore_checkpoint_step=infer_writer_p.restore_checkpoint_step,
          input_name=input_p.name,
          model_name=task_p.model.name,
      ).save_metadata(dirname)

      writer = io_utils.ShardedParallelWriter(
          fq_filename,
          infer_writer_p.output_num_shards,
          output_format=infer_writer_p.output_format)

    step = 0
    while num_steps < 0 or step < num_steps:
      step += 1
      logging.info('processing input batch %d', step)
      try:
        batch = input_gen.get_next()
      except (tf.errors.OutOfRangeError, StopIteration):
        input_gen.reset()
        break

      pmap_batch = input_gen.reshard_for_pmap(batch)
      outputs = infer_pmap_step(replicated_model_states, output_seeds,
                                pmap_batch)
      # Get first device's output since it's been replicated by all-gather
      outputs = py_utils.maybe_unreplicate_for_fully_replicated(outputs)
      outputs_cpu = jax.tree_map(np.asarray, outputs)

      if jax.process_index() == 0:
        serialized_outputs = task.inference_runner.serialize_outputs(
            outputs_cpu)
        # fire-and-forget writing
        writer.write(serialized_outputs)

    if jax.process_index() == 0:
      writer.close()
