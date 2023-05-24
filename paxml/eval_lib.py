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

"""Evaluation loop for Pax model."""

import abc
import contextlib
import functools
import gc
import time
import typing
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from absl import logging
from clu import metrics as clu_metrics
from etils import epath
import jax
import numpy as np
from paxml import base_experiment
from paxml import base_metrics
from paxml import decode_programs as decode_programs_lib
from paxml import io_utils
from paxml import partitioning
from paxml import programs
from paxml import summary_utils
from paxml import tasks_lib
from paxml import train_states
from paxml import trainer_lib
from paxml import tuning_lib
from praxis import base_hyperparams
from praxis import base_input
from praxis import base_layer
from praxis import optimizer_prefix_vectorization
from praxis import pax_fiddle
from praxis import py_utils
from praxis import pytypes
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

from paxml import checkpoints  # mapped to internal


instantiate = base_hyperparams.instantiate
CheckpointType = checkpoints.CheckpointType
EvaluationMode = io_utils.EvaluationMode
JTensor = pytypes.JTensor
Metrics = pytypes.Metrics
NestedMap = py_utils.NestedMap
JTensor = pytypes.JTensor
NestedJTensor = pytypes.NestedJTensor
NestedPartitionSpec = pytypes.NestedPartitionSpec
NestedShapeDtypeLike = pytypes.NestedShapeDtypeLike
NestedWeightHParams = base_layer.NestedWeightHParams
RunningMode = trainer_lib.RunningMode
SummaryWriter = tf.summary.SummaryWriter
TrainState = train_states.TrainState
WeightedScalars = pytypes.WeightedScalars
WeightedScalarsList = pytypes.WeightedScalarsList
PMAP_PARALLEL_AXIS_NAME = base_layer.PMAP_PARALLEL_AXIS_NAME
PRNGKey = pytypes.PRNGKey
NestedShapeDtypeLike = pytypes.NestedShapeDtypeLike
NO_PREFIX_KEY = optimizer_prefix_vectorization.NO_PREFIX_KEY


def _get_dir_names(
    inputs: Sequence[base_input.BaseInput],
) -> Sequence[epath.Path]:
  """Returns a list of same length for parent dir names for each dataset."""
  return [epath.Path(p.name) for p in inputs]


def _wait_until_checkpoint_step(
    checkpoint_dir: epath.Path, step: int, sleep_interval: int
) -> int:
  """Waits until a checkpoint for the given step is in checkpoint_dir."""
  with py_utils.timeit() as wait_period:
    while True:
      cur_step = checkpoints.retrieve_latest_checkpoint_step_if_exists(
          checkpoint_dir
      )
      if cur_step is not None and step <= cur_step:
        break
      logging.info(
          'Sleeping waiting for a new checkpoint, the current step is %s',
          cur_step if cur_step is not None else 'None',
      )
      time.sleep(sleep_interval)
  logging.info('Found new checkpoint at step: %d', cur_step)
  jax.monitoring.record_event_duration_secs(
      '/jax/pax/eval_or_decode/wait_duration_sec', wait_period.elapsed
  )
  return cur_step


def _get_train_input_specs(
    task_p: pax_fiddle.Config[tasks_lib.SingleTask],
    experiment_config: base_experiment.BaseExperiment,
):
  """Gets the shape/dtype of the inputs to the model."""
  if not task_p.train.always_use_train_for_model_init:
    return None
  train_input_specs = trainer_lib.get_train_input_specs_for_model_init(
      task_p, instantiate(experiment_config.get_input_specs_provider_params())
  )
  if train_input_specs is None:
    raise ValueError(
        'No training input specs available, while enabling '
        '`task_p.train.always_use_train_for_model_init` requires it.'
    )
  return train_input_specs


class _EvalCheckpointer(metaclass=abc.ABCMeta):
  """Adapts particular implementations of checkpointing into a common API."""

  restore_checkpoint_dir: epath.Path

  def __init__(
      self,
      jax_task: tasks_lib.SingleTask,
      job_log_dir: epath.Path,
      checkpoint_type: checkpoints.CheckpointType,
      restore_checkpoint_dir: epath.Path,
      restore_checkpoint_step: int,
      partitioner: partitioning.Partitioner,
      enforce_restore_shape_check: bool = False,
      ocdbt_coordinator_server: Optional[Any] = None,
  ):
    self._jax_task = jax_task
    self._partitioner = partitioner
    self.checkpoint_type = checkpoint_type
    self.job_log_dir = job_log_dir
    self.restore_checkpoint_dir: epath.Path = restore_checkpoint_dir
    self.restore_checkpoint_step: int = restore_checkpoint_step
    self.use_ema: bool = tasks_lib.has_ema(jax_task.hparams)
    self._enforce_restore_shape_check = enforce_restore_shape_check
    self._ocdbt_coordinator_server = ocdbt_coordinator_server

  def retrieve_latest_checkpoint_step(self) -> int:
    return checkpoints.retrieve_latest_checkpoint_step(
        self.restore_checkpoint_dir
    )

  def wait_for_new_step(self, last_checkpoint_step: int) -> int:
    return _wait_until_checkpoint_step(
        self.restore_checkpoint_dir, last_checkpoint_step + 1, sleep_interval=60
    )

  @abc.abstractmethod
  def load_checkpoint_for_step(
      self, step: int, train_state_metadata: trainer_lib.TrainStateMetadata
  ) -> TrainState:
    raise NotImplementedError

  @abc.abstractmethod
  def get_model_states(
      self,
      root_prng_key: PRNGKey,
  ) -> Tuple[TrainState, trainer_lib.TrainStateMetadata, PRNGKey]:
    """Restore the train state from checkpoint or initialize it.

    Args:
      root_prng_key: The root prng key.

    Returns:
      (train_state, train_state_metadata, initialized_root_prng_key).
    """


class _SpmdEvalCheckpointer(_EvalCheckpointer):

  def _restore(
      self, step: int, train_state_metadata: trainer_lib.TrainStateMetadata
  ) -> TrainState:
    partitioned_train_state = checkpoints.restore_checkpoint(
        train_state_metadata.padded_global_shapes,
        self.restore_checkpoint_dir,
        global_mesh=self._partitioner.global_mesh,
        checkpoint_type=self.checkpoint_type,
        state_specs=train_state_metadata.partition_specs,
        step=step,
        enforce_restore_shape_check=self._enforce_restore_shape_check,
        tensorstore_use_ocdbt=(self._ocdbt_coordinator_server is not None),
    )
    py_utils.sync_global_devices(
        f'checkpointer:restored:{self.restore_checkpoint_dir}'
    )
    if self.use_ema:
      partitioned_train_state = tasks_lib.extract_ema(partitioned_train_state)
    return partitioned_train_state

  def load_checkpoint_for_step(
      self, step: int, train_state_metadata: trainer_lib.TrainStateMetadata
  ) -> TrainState:
    return self._restore(step, train_state_metadata)

  def get_model_states(
      self,
      root_prng_key: PRNGKey,
  ) -> Tuple[TrainState, trainer_lib.TrainStateMetadata, PRNGKey]:
    """Gets a partitioned model states and the step function."""
    train_state_metadata = self._partitioner.get_train_state_metadata(
        discard_opt_states=not self.use_ema
    )
    partition_specs = train_state_metadata.partition_specs
    assert partition_specs is not None, 'must be in pjit mode'
    if self.use_ema and not partition_specs.opt_states:
      # Make sure the opt_states exists before restoring.
      # This is combined with the decoding test.
      raise ValueError(
          "The partition spec doesn't include opt states but ema is enabled."
      )

    partitioned_train_state = self._restore(
        self.restore_checkpoint_step, train_state_metadata
    )
    root_prng_key, partitioned_train_state, _ = (
        self._partitioner.initialize_prng_key_and_train_state(
            root_prng_key,
            partitioned_train_state,
            self.checkpoint_type,
            discard_opt_states=True,
        )
    )
    return partitioned_train_state, train_state_metadata, root_prng_key


class _PmapEvalCheckpointer(_EvalCheckpointer):

  def _restore(
      self, step: int, train_state_global_shapes: TrainState
  ) -> TrainState:
    if py_utils.pmap_use_tensorstore():
      model_states = tasks_lib.restore_pmap_from_tensorstore(
          train_state_global_shapes,
          self.restore_checkpoint_dir,
          step=step,
          checkpoint_type=self.checkpoint_type,
          enforce_restore_shape_check=self._enforce_restore_shape_check,
          tensorstore_use_ocdbt=(self._ocdbt_coordinator_server is not None),
      )
    else:
      model_states = checkpoints.restore_checkpoint(
          train_state_global_shapes,
          self.restore_checkpoint_dir,
          checkpoint_type=self.checkpoint_type,
          step=step,
          enforce_restore_shape_check=self._enforce_restore_shape_check,
          tensorstore_use_ocdbt=(self._ocdbt_coordinator_server is not None),
      )
    if self.use_ema:
      model_states = tasks_lib.extract_ema(model_states)
    else:
      model_states = model_states.to_eval_state()
    return model_states

  def load_checkpoint_for_step(
      self, step: int, train_state_metadata: trainer_lib.TrainStateMetadata
  ) -> TrainState:
    model_states = self._restore(
        step, train_state_metadata.unpadded_global_shapes
    )
    replicated_model_states = trainer_lib.replicate_model_state(model_states)
    del model_states  # Unused at that point.
    return replicated_model_states

  def get_model_states(
      self,
      root_prng_key: PRNGKey,
  ) -> Tuple[TrainState, trainer_lib.TrainStateMetadata, PRNGKey]:
    # Note: `discard_opt_states` is not supported when restoring pmap flax ckpt.
    # We must restore the entire checkpoint and then trim the opt states.
    train_state_metadata = self._partitioner.get_train_state_metadata(
        discard_opt_states=py_utils.pmap_use_tensorstore() and not self.use_ema,
    )

    model_states = self._restore(
        self.restore_checkpoint_step,
        train_state_metadata.unpadded_global_shapes,
    )
    root_prng_key, replicated_model_states, _ = (
        self._partitioner.initialize_prng_key_and_train_state(
            root_prng_key,
            model_states,
            self.checkpoint_type,
            discard_opt_states=not self.use_ema,
        )
    )
    return replicated_model_states, train_state_metadata, root_prng_key


def _create_checkpointer(
    jax_task: tasks_lib.SingleTask,
    job_log_dir: epath.Path,
    checkpoint_type: checkpoints.CheckpointType,
    mode: Optional[EvaluationMode],
    restore_checkpoint_dir: Optional[epath.PathLike],
    restore_checkpoint_step: Optional[int],
    partitioner: partitioning.Partitioner,
    enforce_restore_shape_check: bool = False,
    tensorstore_use_ocdbt: bool = False,
    wait_until_step: Optional[int] = None,
) -> _EvalCheckpointer:
  if not restore_checkpoint_dir:
    # bool(Path(''))==True, so guarding against this odd Optional explicitly ^
    restore_checkpoint_dir = job_log_dir / 'checkpoints'

  if wait_until_step is not None:
    logging.info(
        'Waiting for checkpoint step %d within %s',
        wait_until_step,
        restore_checkpoint_dir,
    )
    _wait_until_checkpoint_step(
        restore_checkpoint_dir, wait_until_step, sleep_interval=300
    )

  if restore_checkpoint_step is None and mode is not None:
    restore_checkpoint_step = io_utils.get_checkpoint_step(
        job_log_dir, restore_checkpoint_dir, mode
    )

  ocdbt_coordinator_server = checkpoints.reregister_type_handlers(
      tensorstore_metadata_key=jax_task.hparams.train.tensorstore_metadata_key,
      tensorstore_use_ocdbt=tensorstore_use_ocdbt,
  )
  if jax_task.hparams.model.mesh_shape is not None:
    checkpointer_cls = _SpmdEvalCheckpointer
  else:
    checkpointer_cls = _PmapEvalCheckpointer
  return checkpointer_cls(
      jax_task,
      job_log_dir,
      checkpoint_type,
      restore_checkpoint_dir,
      restore_checkpoint_step,
      partitioner,
      enforce_restore_shape_check=enforce_restore_shape_check,
      ocdbt_coordinator_server=ocdbt_coordinator_server,
  )


def run_eval_loop_over_test_splits(
    test_eval_programs: Sequence[programs.BaseEvalProgram],
    eval_partitioned_train_state: TrainState,
    eval_prng_seed: jax.random.KeyArray,
    step: int,
    job_log_dir: epath.Path,
) -> Tuple[
    List[Optional[Dict[str, float]]],  # eval metrics.
    List[Optional[Dict[str, float]]],  # eval scoring metrics.
    List[int],  # performed eval steps.
]:
  """Run evaluation in a loop over a list of test sets.

  Args:
    test_eval_programs: A list of EvalPrograms to conduct eval.
    eval_partitioned_train_state: Train State to use for eval.
    eval_prng_seed: RNG seed for eval programs.
    step: The step at which we are evaling the model.
    job_log_dir: Job's log directory in which scoring outputs will be written.

  Returns:
    A tuple of (a list of eval metrics,
                a list of optional scoring metrics (seqio)
                a list of integer as performed evaluation steps).
      Items from each list are aligned with the `model_inputs`.
  """
  eval_metrics_list = []
  eval_scoring_metrics_list = []
  num_eval_steps = []
  for eval_program in test_eval_programs:
    program_out = eval_program.run(eval_partitioned_train_state, step)
    eval_metrics_list.append(program_out.aux.eval_metrics)
    eval_scoring_metrics_list.append(program_out.aux.eval_scoring_metrics)
    num_eval_steps.append(program_out.aux.num_eval_steps)

  return (eval_metrics_list, eval_scoring_metrics_list, num_eval_steps)


def evaluate(
    experiment_config: base_experiment.BaseExperiment,
    job_log_dir: epath.Path,
    maybe_use_persistence_checkpointing: bool,
    restore_checkpoint_dir: Optional[epath.Path] = None,
    restore_checkpoint_step: Optional[int] = None,
    early_stopping_fn: Optional[trainer_lib.EarlyStoppingFn] = None,
    enable_auto_sharding: bool = False,
    enforce_restore_shape_check: bool = False,
    tensorstore_use_ocdbt: bool = False,
) -> None:
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
    enforce_restore_shape_check: Raises an error if restore shapes do not match
      checkpoint shapes.
  """
  jax.monitoring.record_event('/jax/pax/evaluate/beacon')
  eval_input_p = [v for v in experiment_config.datasets() if not v.is_training]
  if not eval_input_p:
    logging.info('No eval datasets defined. Returning early.')
    return

  task_p = experiment_config.task()
  task_p = typing.cast(pax_fiddle.Config[tasks_lib.SingleTask], task_p)
  jax_task = instantiate(task_p)
  train_input_specs = _get_train_input_specs(task_p, experiment_config)
  prng_key = jax.random.PRNGKey(task_p.evaluate.random_seed)

  checkpoint_type = checkpoints.retrieve_checkpoint_type(
      maybe_use_persistence_checkpointing, jax_task.hparams
  )
  reshard_inputs = (
      checkpoint_type != CheckpointType.PERSISTENCE or
      eval_input_p[0].experimental_remote_input
  )
  partitioner = partitioning.create_partitioner(
      jax_task,
      init_is_eval=True,
      reshard_inputs=reshard_inputs,
      auto_sharding_mode=RunningMode.EVAL if enable_auto_sharding else None,
      auto_sharding_input_params=eval_input_p[0]
      if enable_auto_sharding
      else None,
  )
  input_for_shape = None
  if not task_p.train.always_use_train_for_model_init:
    assert train_input_specs is None
    # TODO(pax-dev): Investigate if we can use model input specs
    # instead of instantiating this input pipeline.
    input_p = partitioner.preprocess_input_config(eval_input_p[0])
    input_for_shape = instantiate(input_p)
  partitioner.setup(
      jax_task, prng_key, train_input_specs, input_for_shape, job_log_dir
  )

  checkpointer = _create_checkpointer(
      jax_task,
      job_log_dir,
      checkpoint_type,
      EvaluationMode.EVAL,
      restore_checkpoint_dir=restore_checkpoint_dir,
      restore_checkpoint_step=restore_checkpoint_step,
      partitioner=partitioner,
      enforce_restore_shape_check=enforce_restore_shape_check,
      tensorstore_use_ocdbt=tensorstore_use_ocdbt,
      # at least try to wait until checkpoint dir exists (step 0)
      wait_until_step=restore_checkpoint_step or 0,
  )

  partitioned_train_state, train_state_metadata, prng_key = (
      checkpointer.get_model_states(prng_key)
  )
  eval_programs = experiment_config.eval_programs()
  prng_key, eval_key = jax.random.split(prng_key)
  eval_runner = _EvalRunner(
      jax_task=jax_task,
      partitioner=partitioner,
      eval_programs=eval_programs,
      decode_input_ps=[],
      job_log_dir=job_log_dir,
      eval_key=eval_key,
  )

  decode_once_fn = None
  decode_inputs = None
  continuous_decode = True
  _common_eval_or_decode_loop(
      EvaluationMode.EVAL,
      checkpointer,
      jax_task.hparams,
      job_log_dir,
      decode_once_fn,
      partitioned_train_state,
      train_state_metadata,
      early_stopping_fn,
      continuous_decode,
      eval_runner,
      decode_inputs,
  )


class _EvalRunner:
  """A runner class that runs evaluate with pmap or spmd."""

  def __init__(
      self,
      *,
      jax_task: tasks_lib.SingleTask,
      partitioner: partitioning.Partitioner,
      eval_programs: Sequence[programs.BaseEvalProgram],
      decode_input_ps: Sequence[pax_fiddle.Config[base_input.BaseInput]],
      job_log_dir: epath.Path,
      eval_key: PRNGKey,
  ):
    self._jax_task = jax_task
    self._partitioner = partitioner
    self._job_log_dir = job_log_dir
    self._eval_programs = eval_programs
    logging.info('eval prng_key: %s', eval_key)
    self._eval_key = self._partitioner.preprocess_prng_key(eval_key)

    # TODO(wangpeng): Make decode programs configurable.
    create_decode_program = functools.partial(
        decode_programs_lib.SingleTaskDecodeProgram,
        model=jax_task.model,
        partitioner=partitioner,
    )
    self._decode_programs = [
        create_decode_program(decode_input=instantiate(p), input_index=i)
        for i, p in enumerate(decode_input_ps)
    ]
    trainer_lib.check_unique_names(
        [p.decode_input for p in self._decode_programs]
    )

  def setup_eval_programs(self, summary_base_dir: epath.Path):
    for program in self._eval_programs:
      program.setup(
          self._jax_task,
          self._partitioner,
          self._job_log_dir,
          self._eval_key,
          summary_base_dir=summary_base_dir,
      )
    trainer_lib.check_unique_names(
        [program.eval_input for program in self._eval_programs]
    )

  @property
  def decode_programs(self):
    return self._decode_programs

  def run_one_step(self, train_state):
    if not self._eval_programs:
      return tuning_lib.EvalMetrics(
          metrics_list=[],
          scoring_metrics_list=[],
          steps_per_sec=0,
          input_names=[],
      )

    with py_utils.timeit() as eval_period:
      step_i = int(
          py_utils.maybe_unreplicate_for_fully_replicated(train_state.step)
      )
      eval_metrics_list, eval_scoring_metrics_list, num_eval_steps = (
          run_eval_loop_over_test_splits(
              self._eval_programs,
              train_state.to_eval_state(),
              self._eval_key,
              step_i,
              self._job_log_dir,
          )
      )
    jax.monitoring.record_event_duration_secs(
      '/jax/pax/eval/duration_sec', eval_period.elapsed
    )
    return tuning_lib.EvalMetrics(
        metrics_list=eval_metrics_list,
        scoring_metrics_list=eval_scoring_metrics_list,
        steps_per_sec=sum(num_eval_steps) / eval_period.elapsed,
        input_names=[
            program.eval_input.name for program in self._eval_programs
        ],
    )


def decode(
    experiment_config: base_experiment.BaseExperiment,
    job_log_dir: epath.PathLike,
    maybe_use_persistence_checkpointing: bool,
    restore_checkpoint_dir: Optional[epath.PathLike],
    restore_checkpoint_step: Optional[int],
    continuous_decode: bool,
    run_eval: Optional[bool] = False,
    early_stopping_fn: Optional[trainer_lib.EarlyStoppingFn] = None,
    enable_auto_sharding: bool = False,
    enable_checkpoint_saving: bool = True,
    output_pickle: bool = True,
    enforce_restore_shape_check: bool = False,
    tensorstore_use_ocdbt: bool = False,
) -> None:
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
    enforce_restore_shape_check: Raises an error if restore shapes do not match
      checkpoint shapes.
  """
  jax.monitoring.record_event('/jax/pax/decode/beacon')
  job_log_dir = epath.Path(job_log_dir)
  if restore_checkpoint_dir:
    restore_checkpoint_dir = epath.Path(restore_checkpoint_dir)

  decoder_inputs = experiment_config.decoder_datasets()
  eval_inputs = [v for v in experiment_config.datasets() if not v.is_training]
  if not run_eval:
    eval_inputs = []
  combined_input_ps = decoder_inputs + eval_inputs  # Use decode inputs first.

  if not combined_input_ps:
    logging.info('No input datasets defined.')
    return

  # TODO(laigd): the logic below is very similar to the logic in evaluate(),
  # merge them.
  task_p = experiment_config.task()
  task_p = typing.cast(pax_fiddle.Config[tasks_lib.SingleTask], task_p)
  jax_task = instantiate(task_p)
  train_input_specs = _get_train_input_specs(task_p, experiment_config)
  prng_key = jax.random.PRNGKey(task_p.decode.random_seed)

  checkpoint_type = checkpoints.retrieve_checkpoint_type(
      maybe_use_persistence_checkpointing, jax_task.hparams
  )
  reshard_inputs = (
      checkpoint_type != CheckpointType.PERSISTENCE or
      combined_input_ps[0].experimental_remote_input
  )
  partitioner = partitioning.create_partitioner(
      jax_task,
      init_is_eval=True,
      reshard_inputs=reshard_inputs,
      auto_sharding_mode=RunningMode.DECODE if enable_auto_sharding else None,
      auto_sharding_input_params=combined_input_ps[0]
      if enable_auto_sharding
      else None,
  )
  input_for_shape = None
  if not task_p.train.always_use_train_for_model_init:
    assert train_input_specs is None
    # We assume that either eval_input or decoder_input can be used to retrieve
    # all the model variable shapes, which is needed for restoring checkpoints.
    #
    # TODO(zhangqiaorjc): If we can no longer assume variable shapes will be the
    # same regardless of which eval_input or decoder_input we use to draw the
    # sample inputs, we need to revisit the design here.

    # TODO(pax-dev): Investigate if we can use model input specs
    # instead of instantiating this input pipeline.
    input_p = partitioner.preprocess_input_config(combined_input_ps[0])
    input_for_shape = instantiate(input_p)
  partitioner.setup(
      jax_task, prng_key, train_input_specs, input_for_shape, job_log_dir
  )
  # TODO(laigd): move this to decode program when ready.
  decoder_inputs = [
      partitioner.preprocess_input_config(p) for p in decoder_inputs
  ]

  wait_until_step = (
      jax_task.hparams.train.decode_start_after_n_steps
      if continuous_decode
      else None
  )
  checkpointer = _create_checkpointer(
      jax_task,
      job_log_dir,
      checkpoint_type,
      EvaluationMode.DECODE,
      restore_checkpoint_dir,
      restore_checkpoint_step,
      partitioner=partitioner,
      enforce_restore_shape_check=enforce_restore_shape_check,
      tensorstore_use_ocdbt=tensorstore_use_ocdbt,
      wait_until_step=wait_until_step,
  )

  if continuous_decode:
    logging.info(
        'running continuous_decode from %s', checkpointer.restore_checkpoint_dir
    )
  else:
    logging.info(
        'running decode_once restored from %s',
        checkpointer.restore_checkpoint_dir,
    )

  eval_programs = experiment_config.eval_programs() if run_eval else []
  if task_p.model.mesh_shape is not None:
    decode_method = decode_spmd_model
    extra_kwargs = {}
  else:
    decode_method = decode_pmap_model
    extra_kwargs = dict(
        output_pickle=output_pickle,
        enable_checkpoint_saving=enable_checkpoint_saving,
    )
  decode_method(
      jax_task,
      prng_key,
      partitioner,
      checkpointer,
      decoder_inputs,
      eval_programs,
      job_log_dir,
      continuous_decode,
      early_stopping_fn,
      **extra_kwargs,
  )


# TODO(wangpeng): Merge decode_pmap_model and decode_spmd_model.
def decode_pmap_model(
    jax_task: tasks_lib.SingleTask,
    prng_key: PRNGKey,
    partitioner: partitioning.Partitioner,
    checkpointer: _EvalCheckpointer,
    # TODO(wangpeng): Rename to `decode_input_params`
    input_p: Sequence[pax_fiddle.Config[base_input.BaseInput]],
    eval_programs: Sequence[programs.BaseEvalProgram],
    job_log_dir: epath.Path,
    continuous_decode: bool,
    early_stopping_fn: Optional[trainer_lib.EarlyStoppingFn] = None,
    output_pickle: bool = True,
    enable_checkpoint_saving: bool = True,
) -> None:
  """Runs the decoding on the entire decoder datasets for a PMAP model.

  Args:
    jax_task: The task encapsulating a the data parallel model.
    prng_key: Root PRNGKey for the decode pipeline.
    partitioner: The partitioner, will be used to partition the step function.
    checkpointer: The model checkpointing method to use.
    input_p: List of input params to be decoded.
    eval_programs: List of eval programs to run.
    job_log_dir: Directory for the job logs.
    continuous_decode: whether to continuously decode on the latest ckpt.
    early_stopping_fn: An optional callable object for reporting metrics and
      determining whether to early stop current training. The callable object
      has signature: (metrics, running_mode, ckpt_step, is_final_ckpt) ->
      should_stop_early.
    output_pickle: Output .pickle file alongside the .jsonl file when decoding.
    enable_checkpoint_saving: Whether to perform checkpoint saving or not.
  """
  task_p = jax_task.hparams

  partitioned_train_state, train_state_metadata, prng_key = (
      checkpointer.get_model_states(prng_key)
  )
  prng_key, eval_key = jax.random.split(prng_key)
  eval_runner = _EvalRunner(
      jax_task=jax_task,
      partitioner=partitioner,
      eval_programs=eval_programs,
      decode_input_ps=input_p,
      job_log_dir=job_log_dir,
      eval_key=eval_key,
  )

  # JaxContext needed for parameter sharing.
  context_p = base_layer.JaxContext.HParams(do_eval=True)
  with base_layer.JaxContext.new_context(hparams=context_p):
    trainer_lib.write_post_init_model_hparams_file(
        jax_task.model,
        train_state_metadata.var_weight_hparams,
        job_log_dir / 'decoder_out',
        do_eval=True,
    )

  prng_key, decode_key = jax.random.split(prng_key)
  # If prng_key_fold_with_batch_index is True, we need to fold in the step
  # number before preprocessing the key, so preprocessing need to be done at
  # every step.
  if not task_p.decode.prng_key_fold_with_batch_index:
    decode_key = partitioner.preprocess_prng_key(decode_key)
  logging.info('decoder prng_seed: %s', decode_key)

  decode_programs = eval_runner.decode_programs
  decode_once_fn = partitioned_decode_once(
      decode_programs=decode_programs,
      task_p=task_p,
      prng_key=decode_key,
      job_log_dir=job_log_dir,
      use_pmap=True,
      var_weight_params=train_state_metadata.var_weight_hparams,
      output_pickle=output_pickle,
      enable_checkpoint_saving=enable_checkpoint_saving,
  )

  decode_inputs = [p.decode_input for p in decode_programs]
  _common_eval_or_decode_loop(
      EvaluationMode.DECODE,
      checkpointer,
      task_p,
      job_log_dir,
      decode_once_fn,
      partitioned_train_state,
      train_state_metadata,
      early_stopping_fn,
      continuous_decode,
      eval_runner,
      decode_inputs,
  )


def partitioned_decode_once(
    *,
    decode_programs: Sequence[decode_programs_lib.SingleTaskDecodeProgram],
    task_p: pax_fiddle.Config[tasks_lib.SingleTask],
    prng_key: JTensor,
    job_log_dir: epath.Path,
    use_pmap: bool,
    var_weight_params: Optional[NestedWeightHParams] = None,
    output_pickle: bool = True,
    enable_checkpoint_saving: bool = True,
    # TODO(wangpeng): Remove this argument.
    spmd_decode_step: Optional[
        Callable[
            [TrainState, PRNGKey, NestedJTensor, Optional[int]],
            Tuple[None, trainer_lib.StepFnOutput],
        ]
    ] = None,
    inputs_partition_spec: Optional[NestedPartitionSpec] = None,
    train_state_preprocessor: Optional[
        Callable[[TrainState], TrainState]
    ] = None,
) -> Callable[[TrainState, List[SummaryWriter]], tuning_lib.DecodeMetrics]:
  """Returns a function that runs decode over all decoder datasets.

  Args:
    decode_programs: A list of `SingleTaskDecodeProgram`s to do the decoding.
    task_p: Params for the task encapsulating a data parallel model.
    prng_key: The prng key used for decoding.
    job_log_dir: Directory for the job logs.
    use_pmap: Whether to use pmap (instead of SPMD/pjit). If this is True,
      `var_weight_params`, `output_pickle` and `enable_checkpoint_saving` should
      be set; otherwise, `spmd_decode_step`, `inputs_partition_spec` and
      `metrics_p` should be set.
    var_weight_params: Nested structure of HParams for the model weights.
    output_pickle: Whether to write decoding results to a pickle file.
    enable_checkpoint_saving: Whether to perform checkpoint saving or not.
    spmd_decode_step: pjit'ed decode function.
    inputs_partition_spec: Partition specs for inputs.
    train_state_preprocessor: A function to preprocess the train state before
      decoding.
  """
  def decode_once_fn(
      partitioned_train_state: TrainState,
      summary_writers: List[SummaryWriter],
  ) -> tuning_lib.DecodeMetrics:
    if train_state_preprocessor is not None:
      partitioned_train_state = train_state_preprocessor(
          partitioned_train_state
      )
    with py_utils.timeit() as decode_period:
      (
          decode_metrics_list,
          processed_decode_metrics_list,
          decode_seqio_metrics_list,
          num_decode_steps,
      ), _ = _decode_once(
          decode_programs=decode_programs,
          prng_key=prng_key,
          job_log_dir=job_log_dir,
          train_state=partitioned_train_state,
          summary_writers=summary_writers,
          use_pmap=use_pmap,
          task_p=task_p,
          var_weight_params=var_weight_params,
          output_pickle=output_pickle,
          enable_checkpoint_saving=enable_checkpoint_saving,
          spmd_decode_step=spmd_decode_step,
          inputs_partition_spec=inputs_partition_spec,
          metrics_p=task_p.metrics,
      )
    jax.monitoring.record_event_duration_secs(
      '/jax/pax/decode/duration_sec', decode_period.elapsed
    )
    decode_steps_per_sec = sum(num_decode_steps) / decode_period.elapsed
    return tuning_lib.DecodeMetrics(
        metrics_list=decode_metrics_list,
        processed_metrics_list=processed_decode_metrics_list,
        seqio_metrics_list=decode_seqio_metrics_list,
        steps_per_sec=decode_steps_per_sec,
        input_names=[p.decode_input.name for p in decode_programs],
    )

  return decode_once_fn


def _decode_once(
    *,
    decode_programs: Sequence[decode_programs_lib.SingleTaskDecodeProgram],
    prng_key: JTensor,
    job_log_dir: epath.Path,
    train_state: TrainState,
    summary_writers: List[SummaryWriter],
    use_pmap: bool,
    task_p: Optional[pax_fiddle.Config[tasks_lib.SingleTask]] = None,
    var_weight_params: Optional[NestedWeightHParams] = None,
    output_pickle: bool = True,
    enable_checkpoint_saving: bool = True,
    spmd_decode_step: Optional[
        Callable[
            [TrainState, PRNGKey, NestedJTensor, Optional[int]],
            Tuple[None, trainer_lib.StepFnOutput],
        ]
    ] = None,
    inputs_partition_spec: Optional[NestedPartitionSpec] = None,
    metrics_p: Optional[pax_fiddle.Config[base_metrics.BaseMetrics]] = None,
) -> Tuple[
    Tuple[
        List[Optional[Dict[str, float]]],  # decode metrics.
        List[Optional[Dict[str, float]]],  # processed decode metrics.
        List[Optional[Dict[str, float]]],  # decode (seqio) metrics.
        List[int],  # performed decode steps.
    ],
    List[Optional[Mapping[str, clu_metrics.Metric]]],  # raw decode metrics
]:
  """Runs the decoding once on the entire decoder datasets for a PMAP or SPMD model.

  Args:
    decode_programs: A list of `SingleTaskDecodeProgram`s to do the decoding.
    prng_key: The prng key used for decoding.
    job_log_dir: Directory for the job logs.
    train_state: A `TrainState` object.
    summary_writers: The summary writer objects to log summaries.
    use_pmap: Whether to use pmap (instead of SPMD/pjit). If this is True,
      `task_p`, `var_weight_params`, `output_pickle` and
      `enable_checkpoint_saving` should be set; otherwise, `spmd_decode_step`,
      `inputs_partition_spec` and `metrics_p` should be set.
    task_p: Params for the task encapsulating a data parallel model.
    var_weight_params: Nested structure of HParams for the model weights.
    output_pickle: Whether to write decoding results to a pickle file.
    enable_checkpoint_saving: Whether to perform checkpoint saving or not.
    spmd_decode_step: pjit'ed decode function.
    inputs_partition_spec: Partition specs for inputs.
    metrics_p: Parameters to configure how to aggregate the metrics.

  Returns:
    A tuple of(
        tuple of (a list of decode metrics,
                  a list of processed decode metrics,
                  a list of optional decoder (seqio) metrics.
                  list of integers as performed decode steps for each input).
        raw clu metrics).
      Items from each list are aligned with each input from inputs.
  """
  # TODO(wangpeng): Remove unnecessary `use_pmap` branchings.

  if use_pmap and not decode_programs:
    return ([], [], [], []), []

  if use_pmap:
    assert task_p is not None
    metrics_p = task_p.metrics
    model_p = task_p.model
  if not metrics_p:
    metrics_p = pax_fiddle.Config(base_metrics.MeanMetrics)

  step_i = int(
      py_utils.maybe_unreplicate_for_fully_replicated(train_state.step)
  )

  if use_pmap:
    logging.info('step=%d', step_i)
  else:
    logging.info(
        'partitioned_train_state: %s',
        jax.tree_map(lambda x: x.shape, train_state),
    )
    logging.info('decode prng_key: %s', prng_key)

  if use_pmap:
    aggregate_fn = instantiate(metrics_p).aggregate
    model = decode_programs[0].model

    def decode_step(mdl_states, decode_key, inputs):
      (weighted_scalars, per_example_out, updated_metrics), updated_vars = (
          # TODO(wangpeng): Move `decode_step` out of `trainer_lib.py`.
          trainer_lib.decode_step(
              model,
              mdl_states,
              decode_key,
              var_weight_params,
              inputs,
              model_p.fprop_dtype,
              task_p.decode.prng_key_fold_with_global_step,
          )
      )

      weighted_scalars = aggregate_fn(weighted_scalars)
      aggregated_per_example_out = jax.lax.all_gather(
          per_example_out, axis_name=PMAP_PARALLEL_AXIS_NAME, tiled=True
      )

      summary_tensors = updated_vars.get(base_layer.SUMMARIES, {})
      summary_tensors = summary_utils.flatten_flax_summaries(summary_tensors)
      aggregated_summaries = summary_utils.aggregate_per_replica_summaries(
          summary_tensors
      )

      # We want to aggregate metrics across workers.
      # In pmap we do an all gather of the metric state across workers, and then
      # call reduce() on the metric which by default calls merge across workers.
      aggregated_metrics = {}
      for metric_name, metric in updated_metrics.items():
        aggregated_metrics[metric_name] = jax.lax.all_gather(
            metric, axis_name=PMAP_PARALLEL_AXIS_NAME
        ).reduce()

      return (
          weighted_scalars,
          aggregated_per_example_out,
          aggregated_summaries,
          aggregated_metrics,
      )

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
    # TODO(wangpeng): Make this a class attribute of `SingleTaskDecodeProgram`
    pmap_decode_step = jax.pmap(
        decode_step,
        axis_name=PMAP_PARALLEL_AXIS_NAME,
        out_axes=(None, None, None, None),
    )
  else:
    pmap_decode_step = None

  decode_metrics_list = []
  processed_decode_metrics_list = []
  seqio_metrics_list = []
  num_decode_steps = []
  raw_metrics_list = []

  for i, decode_program in enumerate(decode_programs):
    decode_program.setup(
        prng_key=prng_key,
        job_log_dir=job_log_dir,
        summary_writer=summary_writers[i],
        use_pmap=use_pmap,
        pmap_decode_step=pmap_decode_step,
        task_p=task_p,
        output_pickle=output_pickle,
        enable_checkpoint_saving=enable_checkpoint_saving,
        spmd_decode_step=spmd_decode_step,
        inputs_partition_spec=inputs_partition_spec,
        metrics_p=metrics_p,
    )
    decode_output = decode_program.run(train_state, step_i).aux

    decode_metrics_list.append(decode_output.decode_metrics)
    processed_decode_metrics_list.append(decode_output.processed_decode_metrics)
    seqio_metrics_list.append(decode_output.seqio_metrics)
    num_decode_steps.append(decode_output.num_decode_steps)
    raw_metrics_list.append(decode_output.raw_decode_metrics)

  return (
      decode_metrics_list,
      processed_decode_metrics_list,
      seqio_metrics_list,
      num_decode_steps,
  ), raw_metrics_list


def decode_spmd_model(
    jax_task: tasks_lib.SingleTask,
    prng_key: PRNGKey,
    partitioner: partitioning.Partitioner,
    checkpointer: _EvalCheckpointer,
    input_p: Sequence[pax_fiddle.Config[base_input.BaseInput]],
    eval_programs: Sequence[programs.BaseEvalProgram],
    job_log_dir: epath.Path,
    continuous_decode: bool,
    early_stopping_fn: Optional[trainer_lib.EarlyStoppingFn],
) -> None:
  """Runs the decoding on the entire decoder datasets for SPMD model.

  Args:
    jax_task: The task that encapsulates an SPMD model.
    prng_key: Root PRNGKey for the decode pipeline.
    partitioner: The partitioner used to partition the step function.
    checkpointer: The model checkpointing method to use.
    input_p: List of input params to be decoded.
    eval_programs: List of eval programs to run.
    job_log_dir: Directory for the job logs.
    continuous_decode: whether to continuously decode on the latest ckpt.
    early_stopping_fn: An optional callable object for reporting metrics and
      determining whether to early stop current training. The callable object
      has signature: (metrics, running_mode, ckpt_step, is_final_ckpt) ->
      should_stop_early.
  """
  task_p = jax_task.hparams

  partitioned_train_state, train_state_metadata, prng_key = (
      checkpointer.get_model_states(prng_key)
  )
  prng_key, eval_key = jax.random.split(prng_key)
  eval_runner = _EvalRunner(
      jax_task=jax_task,
      partitioner=partitioner,
      eval_programs=eval_programs,
      decode_input_ps=input_p,
      job_log_dir=job_log_dir,
      eval_key=eval_key,
  )
  decode_programs = eval_runner.decode_programs

  if decode_programs:
    # Peek to avoid exhausting the input pipeline.
    sample_inputs = decode_programs[0].decode_input.peek_padded()
    inputs_shape_dtype = jax.tree_map(
        py_utils.get_global_input_shape_dtype, sample_inputs
    )
  else:
    # This means there is no input to run, and currently this happens only
    # when user runs decode() with eval input, i.e. `mode` is set to DECODE
    # above. In that case, the partitioned decode_step_fn won't get used since
    # there is no decode input, and we simply use the training input shapes
    # to partition.
    # TODO(laigd): avoid cases like this.
    inputs_shape_dtype = partitioner.train_inputs_shape_dtype

  decode_step_fn, is_eval = partitioning.get_step_fn(RunningMode.DECODE)
  assert is_eval
  decode_step_fn, inputs_partition_spec = partitioner.partition(
      decode_step_fn, inputs_shape_dtype, is_eval
  )
  decode_once_fn = partitioned_decode_once(
      decode_programs=decode_programs,
      task_p=task_p,
      job_log_dir=job_log_dir,
      prng_key=prng_key,
      use_pmap=False,
      spmd_decode_step=decode_step_fn,
      inputs_partition_spec=inputs_partition_spec,
  )
  trainer_lib.write_post_init_model_hparams_file(
      jax_task.model,
      train_state_metadata.var_weight_hparams,
      job_log_dir / 'decoder_out',
      do_eval=True,
  )

  decode_inputs = [p.decode_input for p in decode_programs]
  _common_eval_or_decode_loop(
      EvaluationMode.DECODE,
      checkpointer,
      task_p,
      job_log_dir,
      decode_once_fn,
      partitioned_train_state,
      train_state_metadata,
      early_stopping_fn,
      continuous_decode,
      eval_runner,
      decode_inputs,
  )


def _is_shape_dtype_struct(x):
  """Indicates whether the input is of type ShapeDtypeStruct or not."""
  return isinstance(x, jax.ShapeDtypeStruct)


def _common_eval_or_decode_loop(
    mode: io_utils.EvaluationMode,
    checkpointer: _EvalCheckpointer,
    task_p: pax_fiddle.Config[tasks_lib.SingleTask],
    job_log_dir: epath.Path,
    decode_once_fn: Optional[Callable[..., tuning_lib.DecodeMetrics]],
    partitioned_train_state: TrainState,
    train_state_metadata: trainer_lib.TrainStateMetadata,
    early_stopping_fn: Optional[trainer_lib.EarlyStoppingFn],
    continuous_decode: bool,
    eval_runner: _EvalRunner,
    decode_inputs: Optional[Sequence[base_input.BaseInput]],
):
  # checkpoint must exist at this point
  last_checkpoint_step = checkpointer.retrieve_latest_checkpoint_step()
  logging.info('Evaluation loop starting...')
  summary_base_dir = job_log_dir / 'summaries'
  if decode_inputs:
    summary_decode_dirs = [
        summary_base_dir / f'decode_test_{inp.name}' for inp in decode_inputs
    ]
  with contextlib.ExitStack() as exit_stack:
    if decode_inputs:
      summary_writers = [
          exit_stack.enter_context(summary_utils.get_summary_writer(d))
          for d in summary_decode_dirs
      ]
    eval_runner.setup_eval_programs(summary_base_dir)

    # Collect then freeze GC, so that GC in the eval loop will not touch the
    # python objects used to initialize the model. Unfreeze at the end of the
    # loop.
    gc.collect()
    gc.freeze()
    while True:
      with io_utils.checkpoint_progress(
          job_log_dir, last_checkpoint_step, mode
      ):
        decode_metrics = None
        if decode_inputs:
          logging.info('Decoding step %s ckpt ...', last_checkpoint_step)
          decode_metrics = decode_once_fn(
              partitioned_train_state, summary_writers
          )

        logging.info('Evaling step %s ckpt ...', last_checkpoint_step)
        eval_metrics = eval_runner.run_one_step(partitioned_train_state)

      exceeded_ckpt = last_checkpoint_step + task_p.train.save_interval_steps
      is_last_ckpt = (
          exceeded_ckpt > task_p.train.num_train_steps or not continuous_decode
      )
      if tuning_lib.should_early_stop(
          early_stopping_fn,
          last_checkpoint_step,
          is_last_ckpt,
          eval_metrics=eval_metrics,
          decode_metrics=decode_metrics,
      ):
        logging.info(
            (
                'Early stopped at checkpoint step %d by the'
                'tuner, while the num_train_steps is %d'
            ),
            last_checkpoint_step,
            task_p.train.num_train_steps,
        )
        break
      if is_last_ckpt:
        break
      # Release partitioned_train_state.
      jax.tree_util.tree_map(lambda x: x.delete(), partitioned_train_state)
      del partitioned_train_state
      new_checkpoint_step = checkpointer.wait_for_new_step(last_checkpoint_step)
      partitioned_train_state = checkpointer.load_checkpoint_for_step(
          new_checkpoint_step, train_state_metadata
      )
      last_checkpoint_step = new_checkpoint_step
    gc.unfreeze()


def infer_and_write(
    experiment_config: base_experiment.BaseExperiment,
    job_log_dir: epath.Path,
    enforce_restore_shape_check: bool = False,
    tensorstore_use_ocdbt: bool = False,
) -> None:
  """Generates output from a model and writes it out.

  Args:
    experiment_config: an instance of BaseExperiment for the experiment with
      output generators configured.
    job_log_dir: The base directory for writing the outputs.
    enforce_restore_shape_check: Raises an error if restore shapes do not match
      checkpoint shapes.
  """
  jax.monitoring.record_event('/jax/pax/infer_and_write/beacon')
  task_p = experiment_config.task()
  task_p = typing.cast(pax_fiddle.Config[tasks_lib.SingleTask], task_p)
  task = instantiate(task_p)
  model_p = task_p.model
  inputs_p = experiment_config.decoder_datasets()
  prng_key = jax.random.PRNGKey(task_p.infer.random_seed)
  train_input_specs = _get_train_input_specs(task_p, experiment_config)

  maybe_use_persistence_checkpointing = False
  checkpoint_type = checkpoints.retrieve_checkpoint_type(
      maybe_use_persistence_checkpointing, task.hparams
  )
  reshard_inputs = (
      checkpoint_type != CheckpointType.PERSISTENCE or
      inputs_p[0].experimental_remote_input
  )
  partitioner = partitioning.create_partitioner(
      task, reshard_inputs=reshard_inputs
  )
  inputs_p = [partitioner.preprocess_input_config(p) for p in inputs_p]
  input_for_shape = None
  if not task_p.train.always_use_train_for_model_init:
    assert train_input_specs is None
    # TODO(pax-dev): Investigate if we can use model input specs
    # instead of instantiating this input pipeline.
    input_for_shape = instantiate(inputs_p[0])
  partitioner.setup(
      task, prng_key, train_input_specs, input_for_shape, job_log_dir
  )

  checkpointer = _create_checkpointer(
      task,
      job_log_dir,
      checkpoint_type,
      mode=None,
      restore_checkpoint_dir=task_p.infer_writer.restore_checkpoint_dir,
      restore_checkpoint_step=task_p.infer_writer.restore_checkpoint_step,
      partitioner=partitioner,
      enforce_restore_shape_check=enforce_restore_shape_check,
      tensorstore_use_ocdbt=tensorstore_use_ocdbt,
  )
  if model_p.mesh_shape is not None:
    # TODO(b/238416854): add support for SPMD models
    raise NotImplementedError('SPMD infer_and_write not implemented yet')
  else:
    infer_and_write_pmap(
        task, prng_key, partitioner, checkpointer, inputs_p, job_log_dir
    )


def infer_and_write_pmap(
    task: tasks_lib.SingleTask,
    prng_key: PRNGKey,
    partitioner: partitioning.Partitioner,
    checkpointer: _EvalCheckpointer,
    inputs_p: Sequence[pax_fiddle.Config[base_input.BaseInput]],
    job_log_dir: epath.Path,
) -> None:
  """Runs the infer_and_write for each of the inputs given task in pmap."""
  task_p = task.hparams
  infer_writer_p = task_p.infer_writer

  if not inputs_p:
    return

  assert isinstance(checkpointer, _PmapEvalCheckpointer)
  replicated_model_states, train_state_metadata, prng_key = (
      checkpointer.get_model_states(prng_key)
  )

  @functools.partial(jax.pmap, axis_name=PMAP_PARALLEL_AXIS_NAME, out_axes=None)
  def infer_pmap_step(mdl_states, prng_seeds, input_batch):
    outputs = task.inference_runner.infer(
        mdl_states,
        prng_seeds,
        train_state_metadata.var_weight_hparams,
        input_batch,
    )
    # tiled=True folds in first axis into second axis [2,8,5] -> [2*8,5]
    replicated_outputs = jax.lax.all_gather(
        outputs, axis_name=PMAP_PARALLEL_AXIS_NAME, tiled=True
    )

    return replicated_outputs

  # Instantiate inputs to infer on
  inputs = [instantiate(p) for p in inputs_p]
  trainer_lib.check_unique_names(inputs)
  num_steps = [
      -1 if p.reset_for_eval else p.eval_loop_num_batches for p in inputs_p
  ]

  for input_gen, num_steps in zip(inputs, num_steps):
    name = input_gen.hparams.name
    logging.info('Starting output generation on input "%s"', name)

    # Feed each (device, input) pair a unique seed
    prng_key, output_seed = jax.random.split(prng_key)
    output_seeds = jax.random.split(output_seed, jax.local_device_count())

    if num_steps > 0:
      logging.info('total number of steps: %d', num_steps)

    # Only write from one process
    dirname = job_log_dir / 'output' / name
    fq_filename = dirname / 'output'
    if jax.process_index() == 0:
      # Create output dirs if DNE
      if not dirname.exists():
        dirname.mkdir(parents=True, exist_ok=True)

      # Write example schema, metadata, and serialized example protos
      logging.info('writing output to %s', fq_filename)
      features_dict = tfds.features.FeaturesDict(
          task.inference_runner.output_schema
      )
      features_dict.save_config(dirname.as_posix())
      tfds.core.MetadataDict(
          restore_checkpoint_dir=infer_writer_p.restore_checkpoint_dir,
          restore_checkpoint_step=infer_writer_p.restore_checkpoint_step,
          input_name=name,
          model_name=task_p.model.name,
      ).save_metadata(dirname)

      writer = io_utils.ShardedParallelWriter(
          fq_filename,
          infer_writer_p.output_num_shards,
          output_format=infer_writer_p.output_format,
      )

    step = 0
    while num_steps < 0 or step < num_steps:
      step += 1
      logging.info('processing input batch %d', step)
      try:
        batch = input_gen.get_next()
      except (tf.errors.OutOfRangeError, StopIteration):
        input_gen.reset()
        break

      pmap_batch = partitioner.preprocess_inputs(input_gen, batch, None)
      outputs = infer_pmap_step(
          replicated_model_states, output_seeds, pmap_batch
      )
      # Get first device's output since it's been replicated by all-gather
      outputs = py_utils.maybe_unreplicate_for_fully_replicated(outputs)
      outputs_cpu = jax.tree_map(np.asarray, outputs)

      if jax.process_index() == 0:
        serialized_outputs = task.inference_runner.serialize_outputs(
            outputs_cpu
        )
        # fire-and-forget writing
        writer.write(serialized_outputs)

    if jax.process_index() == 0:
      writer.close()
