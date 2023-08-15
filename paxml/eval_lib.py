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
import dataclasses
import functools
import gc
import itertools
import time
import typing
from typing import Any, Callable, List, Mapping, Optional, Sequence, Tuple

from absl import logging
from clu import metrics as clu_metrics
from etils import epath
import jax
import numpy as np
from orbax.checkpoint import checkpoint_utils as orbax_checkpoint_utils
from paxml import base_experiment
from paxml import base_metrics
from paxml import checkpoint_paths
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


def _get_train_input_specs(
    task: tasks_lib.SingleTask,
    experiment_config: base_experiment.BaseExperiment,
):
  """Gets the shape/dtype of the inputs to the model."""
  if not task.train.always_use_train_for_model_init:
    return None
  train_input_specs = trainer_lib.get_train_input_specs_for_model_init(
      task, instantiate(experiment_config.get_input_specs_provider_params())
  )
  if train_input_specs is None:
    raise ValueError(
        'No training input specs available, while enabling '
        '`task.train.always_use_train_for_model_init` requires it.'
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
      tensorstore_use_ocdbt: bool = False,
      restore_transformations: Optional[dict[str, Any]] = None,
  ):
    self._jax_task = jax_task
    self._partitioner = partitioner
    self.checkpoint_type = checkpoint_type
    self.job_log_dir = job_log_dir
    self.restore_checkpoint_dir: epath.Path = restore_checkpoint_dir
    self.restore_checkpoint_step: int = restore_checkpoint_step
    self.use_ema: bool = tasks_lib.has_ema(jax_task.hparams)
    self._enforce_restore_shape_check = enforce_restore_shape_check
    self._tensorstore_use_ocdbt = tensorstore_use_ocdbt
    self._restore_transformations = restore_transformations

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
        tensorstore_use_ocdbt=self._tensorstore_use_ocdbt,
        restore_transformations=self._restore_transformations,
    )
    py_utils.sync_global_devices(
        f'checkpointer:restored:{self.restore_checkpoint_dir}'
    )
    if self.use_ema:
      partitioned_train_state = tasks_lib.extract_ema(partitioned_train_state)
    else:
      partitioned_train_state = partitioned_train_state.to_eval_state()
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

  @py_utils.benchmark('[PAX STATUS]: ', first_n=2)
  def _restore(
      self, step: int, train_state_global_shapes: TrainState
  ) -> TrainState:
    if py_utils.pmap_use_tensorstore():
      model_states = tasks_lib.restore_pmap_from_tensorstore(
          global_shapes=train_state_global_shapes,
          checkpoint_dir=self.restore_checkpoint_dir,
          step=step,
          checkpoint_type=self.checkpoint_type,
          enforce_restore_shape_check=self._enforce_restore_shape_check,
          tensorstore_use_ocdbt=self._tensorstore_use_ocdbt,
          restore_transformations=self._restore_transformations,
      )
    else:
      model_states = checkpoints.restore_checkpoint(
          state_global_shapes=train_state_global_shapes,
          checkpoint_dir=self.restore_checkpoint_dir,
          checkpoint_type=self.checkpoint_type,
          step=step,
          enforce_restore_shape_check=self._enforce_restore_shape_check,
          tensorstore_use_ocdbt=self._tensorstore_use_ocdbt,
          restore_transformations=self._restore_transformations,
      )
    if self.use_ema:
      # Note: extract_ema() will remove the opt_states.
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
    *,
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
    # Note: there could still potentially be concurrency issues since the lock
    # will be released, but we do not consider that to be an issue since the
    # first checkpoint is not likely to be cleaned up immediately.
    with orbax_checkpoint_utils.wait_for_new_checkpoint(
        restore_checkpoint_dir,
        until_step=wait_until_step,
        seconds_to_sleep=300,
        timeout=4800,
        step_prefix=checkpoint_paths.checkpoint_prefix(checkpoint_type),
        step_format_fixed_length=checkpoint_paths.checkpoint_name_fixed_length(
            checkpoint_type
        ),
    ):
      pass

  if restore_checkpoint_step is None and mode is not None:
    restore_checkpoint_step = io_utils.get_checkpoint_step(
        job_log_dir, restore_checkpoint_dir, mode
    )

  checkpoints.reregister_type_handlers(
      tensorstore_metadata_key=jax_task.hparams.train.tensorstore_metadata_key,
  )

  restore_transformations = jax_task.train.restore_transformations

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
      tensorstore_use_ocdbt=tensorstore_use_ocdbt,
      restore_transformations=restore_transformations,
  )


def run_eval_programs(
    *,
    eval_programs: Sequence[programs.BaseEvalProgram],
    train_state: TrainState,
    step: int,
) -> Tuple[tuning_lib.EvalMetrics, float]:
  """Run evaluation in a loop over a list of test sets.

  Args:
    eval_programs: A list of EvalPrograms to conduct eval.
    train_state: Train State to use for eval.
    step: The training step at which we are evaling the model.

  Returns:
    A tuning_lib.EvalMetrics instance encapsulating the eval metrics, and the
    time elapsed (in seconds) when running the eval programs.
  """
  eval_metrics = []
  eval_scoring_metrics = []
  num_eval_steps = []

  with py_utils.timeit() as period:
    for program in eval_programs:
      program_out = program.run(train_state, step)
      eval_metrics.append(program_out.eval_metrics)
      eval_scoring_metrics.append(program_out.eval_scoring_metrics)
      num_eval_steps.append(program_out.num_eval_steps)

  eval_steps_per_sec = sum(num_eval_steps) / period.elapsed
  combined_eval_metrics = tuning_lib.EvalMetrics(
      metrics_list=eval_metrics,
      scoring_metrics_list=eval_scoring_metrics,
      steps_per_sec=eval_steps_per_sec,
      input_names=[program.eval_input.name for program in eval_programs],
  )
  return combined_eval_metrics, period.elapsed


@py_utils.benchmark('[PAX STATUS]: ', first_n=2)
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
  train_input_specs = _get_train_input_specs(jax_task, experiment_config)
  prng_key = jax.random.PRNGKey(jax_task.evaluate.random_seed)

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
  if not jax_task.train.always_use_train_for_model_init:
    assert train_input_specs is None
    # TODO(pax-dev): Investigate if we can use model input specs
    # instead of instantiating this input pipeline.
    input_p = partitioner.preprocess_input_config(eval_input_p[0])
    input_for_shape = instantiate(input_p)
  partitioner.setup(
      jax_task, prng_key, train_input_specs, input_for_shape, job_log_dir
  )

  checkpointer = _create_checkpointer(
      jax_task=jax_task,
      job_log_dir=job_log_dir,
      checkpoint_type=checkpoint_type,
      mode=EvaluationMode.EVAL,
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
  eval_runner = _EvalRunner(
      jax_task=jax_task,
      partitioner=partitioner,
      eval_programs=eval_programs,
      decode_programs=[],
      job_log_dir=job_log_dir,
      prng_key=prng_key,
  )

  _common_eval_or_decode_loop(
      mode=EvaluationMode.EVAL,
      checkpointer=checkpointer,
      task=jax_task,
      job_log_dir=job_log_dir,
      partitioned_train_state=partitioned_train_state,
      train_state_metadata=train_state_metadata,
      early_stopping_fn=early_stopping_fn,
      continuous_decode=True,
      eval_runner=eval_runner,
      partitioner=partitioner,
  )


class _EvalRunner:
  """A runner class that runs evaluate with pmap or spmd."""

  def __init__(
      self,
      *,
      jax_task: tasks_lib.SingleTask,
      partitioner: partitioning.Partitioner,
      eval_programs: Sequence[programs.BaseEvalProgram],
      decode_programs: Sequence[decode_programs_lib.SingleTaskDecodeProgram],
      job_log_dir: epath.Path,
      prng_key: PRNGKey,
  ):
    self._jax_task = jax_task
    self._partitioner = partitioner
    self._job_log_dir = job_log_dir
    self._eval_programs = eval_programs
    self._decode_programs = decode_programs

    decode_key, eval_key = jax.random.split(prng_key)
    logging.info('eval prng_key: %s', eval_key)
    self._eval_key = self._partitioner.preprocess_prng_key(eval_key)

    if self._decode_programs:
      # If prng_key_fold_with_batch_index is True, we need to fold in the step
      # number before preprocessing the key, so preprocessing need to be done at
      # every step.
      logging.info('decoder prng_seed: %s', decode_key)
      if not jax_task.decode.prng_key_fold_with_batch_index:
        decode_key = partitioner.preprocess_prng_key(decode_key)
    self._decode_key = decode_key

  def setup_eval_programs(self):
    for program in self._eval_programs:
      program.setup(
          self._jax_task, self._partitioner, self._job_log_dir, self._eval_key
      )
    trainer_lib.check_unique_names(
        [program.eval_input for program in self._eval_programs]
    )

  def setup_decode_programs(self, output_pickle):
    for program in self._decode_programs:
      program.setup(
          self._jax_task,
          self._partitioner,
          self._job_log_dir,
          self._decode_key,
          output_pickle=output_pickle,
      )
    trainer_lib.check_unique_names(
        [p.decode_input for p in self._decode_programs]
    )

  @property
  def eval_programs(self):
    return self._eval_programs

  @property
  def decode_programs(self):
    return self._decode_programs


@py_utils.benchmark('[PAX STATUS]: ', first_n=2)
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
  train_input_specs = _get_train_input_specs(jax_task, experiment_config)
  prng_key = jax.random.PRNGKey(jax_task.decode.random_seed)

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
  if not jax_task.train.always_use_train_for_model_init:
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

  wait_until_step = (
      jax_task.hparams.train.decode_start_after_n_steps
      if continuous_decode
      else None
  )
  checkpointer = _create_checkpointer(
      jax_task=jax_task,
      job_log_dir=job_log_dir,
      checkpoint_type=checkpoint_type,
      mode=EvaluationMode.DECODE,
      restore_checkpoint_dir=restore_checkpoint_dir,
      restore_checkpoint_step=restore_checkpoint_step,
      partitioner=partitioner,
      enforce_restore_shape_check=enforce_restore_shape_check,
      tensorstore_use_ocdbt=tensorstore_use_ocdbt,
      wait_until_step=wait_until_step,
  )
  logging.info(
      'running %s from %s',
      'continuous_decode' if continuous_decode else 'decode_once',
      checkpointer.restore_checkpoint_dir,
  )

  eval_programs = experiment_config.eval_programs() if run_eval else []
  decode_programs = experiment_config.decode_programs()
  partitioned_train_state, train_state_metadata, prng_key = (
      checkpointer.get_model_states(prng_key)
  )

  eval_runner = _EvalRunner(
      jax_task=jax_task,
      partitioner=partitioner,
      eval_programs=eval_programs,
      decode_programs=decode_programs,
      job_log_dir=job_log_dir,
      prng_key=prng_key,
  )

  trainer_lib.write_post_init_model_hparams_file(
      model=jax_task.model,
      train_state_metadata=train_state_metadata,
      job_log_dir=job_log_dir / 'decoder_out',
      do_eval=True,
  )

  _common_eval_or_decode_loop(
      mode=EvaluationMode.DECODE,
      checkpointer=checkpointer,
      task=jax_task,
      job_log_dir=job_log_dir,
      partitioned_train_state=partitioned_train_state,
      train_state_metadata=train_state_metadata,
      early_stopping_fn=early_stopping_fn,
      continuous_decode=continuous_decode,
      eval_runner=eval_runner,
      partitioner=partitioner,
      decode_output_pickle=output_pickle,
  )


def run_decode_programs(
    *,
    decode_programs: Sequence[decode_programs_lib.SingleTaskDecodeProgram],
    train_state: TrainState,
    step: int,
) -> Tuple[tuning_lib.DecodeMetrics, float]:
  """Returns a function that runs decode over all decoder datasets.

  Args:
    decode_programs: A list of `SingleTaskDecodeProgram`s to do the decoding.
    train_state: The partitioned TrainState instance.
    step: The training step at which we are evaling the model.

  Returns:
    A tuning_lib.DecodeMetrics instance encapsulating the decode metrics, and
    the time elapsed (in seconds) running the decode programs.
  """

  decode_metrics = []
  processed_decode_metrics = []
  seqio_metrics = []
  num_decode_steps = []

  with py_utils.timeit() as period:
    for decode_program in decode_programs:
      decode_output = decode_program.run(train_state, step)
      decode_metrics.append(decode_output.decode_metrics)
      processed_decode_metrics.append(decode_output.processed_decode_metrics)
      seqio_metrics.append(decode_output.seqio_metrics)
      num_decode_steps.append(decode_output.num_decode_steps)

  decode_steps_per_sec = sum(num_decode_steps) / period.elapsed
  return (
      tuning_lib.DecodeMetrics(
          metrics_list=decode_metrics,
          processed_metrics_list=processed_decode_metrics,
          seqio_metrics_list=seqio_metrics,
          steps_per_sec=decode_steps_per_sec,
          input_names=[p.decode_input.name for p in decode_programs],
      ),
      period.elapsed,
  )


def _eval_or_decode(
    *,
    step: int,
    partitioned_train_state: TrainState,
    mode: io_utils.EvaluationMode,
    job_log_dir: epath.Path,
    eval_runner: _EvalRunner,
) -> Tuple[
    Optional[tuning_lib.EvalMetrics], Optional[tuning_lib.DecodeMetrics]
]:
  with io_utils.checkpoint_progress(job_log_dir, step, mode):
    decode_metrics = None
    if eval_runner.decode_programs:
      logging.info('Decoding step %s ckpt ...', step)
      decode_metrics, elapsed_secs = run_decode_programs(
          train_state=partitioned_train_state,
          decode_programs=eval_runner.decode_programs,
          step=step,
      )
      jax.monitoring.record_event_duration_secs(
          '/jax/pax/decode/duration_sec', elapsed_secs
      )

    eval_metrics = None
    if eval_runner.eval_programs:
      logging.info('Evaling step %s ckpt ...', step)
      eval_metrics, elapsed_secs = run_eval_programs(
          eval_programs=eval_runner.eval_programs,
          train_state=partitioned_train_state,
          step=step,
      )
      jax.monitoring.record_event_duration_secs(
          '/jax/pax/eval/duration_sec', elapsed_secs
      )
  return eval_metrics, decode_metrics


def _common_eval_or_decode_loop(
    *,
    mode: io_utils.EvaluationMode,
    checkpointer: _EvalCheckpointer,
    task: tasks_lib.SingleTask,
    job_log_dir: epath.Path,
    partitioned_train_state: TrainState,
    train_state_metadata: trainer_lib.TrainStateMetadata,
    early_stopping_fn: Optional[trainer_lib.EarlyStoppingFn],
    continuous_decode: bool,
    eval_runner: _EvalRunner,
    partitioner: partitioning.Partitioner,
    decode_output_pickle: bool = True,
):
  step_prefix = checkpoint_paths.checkpoint_prefix(checkpointer.checkpoint_type)
  step_format_fixed_length = checkpoint_paths.checkpoint_name_fixed_length(
      checkpointer.checkpoint_type
  )
  # If preemption happened during evaluation, some checkpoints may be locked.
  orbax_checkpoint_utils.unlock_existing_checkpoints(
      checkpointer.restore_checkpoint_dir,
      step_prefix=step_prefix,
      step_format_fixed_length=step_format_fixed_length,
  )
  # Retrieve last step from the TrainState directly in case new checkpoints
  # have been written in the mean time.
  last_checkpoint_step = int(
      py_utils.maybe_unreplicate_for_fully_replicated(
          partitioned_train_state.step))
  logging.info('Evaluation loop starting from step `%d`...',
               last_checkpoint_step)

  eval_runner.setup_eval_programs()
  eval_runner.setup_decode_programs(decode_output_pickle)

  # Collect then freeze GC, so that GC in the eval loop will not touch the
  # python objects used to initialize the model. Unfreeze at the end of the
  # loop.
  gc.collect()
  gc.freeze()

  eval_metrics, decode_metrics = _eval_or_decode(
      step=last_checkpoint_step,
      partitioned_train_state=partitioned_train_state,
      mode=mode,
      job_log_dir=job_log_dir,
      eval_runner=eval_runner,
  )

  while True:
    exceeded_ckpt = last_checkpoint_step + task.train.save_interval_steps
    is_last_ckpt = (
        exceeded_ckpt > task.train.num_train_steps or not continuous_decode
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
          task.train.num_train_steps,
      )
      break
    if is_last_ckpt:
      break

    # Release previous partitioned_train_state.
    jax.tree_util.tree_map(lambda x: x.delete(), partitioned_train_state)
    del partitioned_train_state

    # Use context manager to wait for new checkpoint and lock it to prevent
    # cleanup by the training thread.
    with orbax_checkpoint_utils.wait_for_new_checkpoint(
        checkpointer.restore_checkpoint_dir,
        until_step=last_checkpoint_step + 1,
        seconds_to_sleep=60,
        step_prefix=step_prefix,
        step_format_fixed_length=step_format_fixed_length,
    ) as new_checkpoint_step:
      partitioned_train_state = checkpointer.load_checkpoint_for_step(
          new_checkpoint_step, train_state_metadata
      )
      eval_metrics, decode_metrics = _eval_or_decode(
          step=new_checkpoint_step,
          partitioned_train_state=partitioned_train_state,
          mode=mode,
          job_log_dir=job_log_dir,
          eval_runner=eval_runner,
      )
    last_checkpoint_step = new_checkpoint_step
  gc.unfreeze()


@py_utils.benchmark('[PAX STATUS]: ', first_n=2)
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
  model = task.model
  inputs_p = experiment_config.decoder_datasets()
  prng_key = jax.random.PRNGKey(task.infer.random_seed)
  train_input_specs = _get_train_input_specs(task, experiment_config)

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
  if not task.train.always_use_train_for_model_init:
    assert train_input_specs is None
    # TODO(pax-dev): Investigate if we can use model input specs
    # instead of instantiating this input pipeline.
    input_for_shape = instantiate(inputs_p[0])
  partitioner.setup(
      task, prng_key, train_input_specs, input_for_shape, job_log_dir
  )

  checkpointer = _create_checkpointer(
      jax_task=task,
      job_log_dir=job_log_dir,
      checkpoint_type=checkpoint_type,
      mode=None,
      restore_checkpoint_dir=task.infer_writer.restore_checkpoint_dir,
      restore_checkpoint_step=task.infer_writer.restore_checkpoint_step,
      partitioner=partitioner,
      enforce_restore_shape_check=enforce_restore_shape_check,
      tensorstore_use_ocdbt=tensorstore_use_ocdbt,
  )
  if model.mesh_shape is not None:
    # TODO(b/238416854): add support for SPMD models
    raise NotImplementedError('SPMD infer_and_write not implemented yet')

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
  infer_writer_p = task.infer_writer

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
  inputs: Sequence[base_input.BaseInput] = [instantiate(p) for p in inputs_p]
  trainer_lib.check_unique_names(inputs)

  for input_gen in inputs:
    name = input_gen.hparams.name
    num_steps = (
        -1 if input_gen.reset_for_eval else input_gen.eval_loop_num_batches
    )
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
      logging.info('Writing output to %s', fq_filename)
      features_dict = tfds.features.FeaturesDict(
          task.inference_runner.output_schema
      )
      features_dict.save_config(dirname.as_posix())
      tfds.core.MetadataDict(
          restore_checkpoint_dir=infer_writer_p.restore_checkpoint_dir,
          restore_checkpoint_step=infer_writer_p.restore_checkpoint_step,
          input_name=name,
          model_name=task.model.name,
      ).save_metadata(dirname)

      writer = io_utils.ShardedParallelWriter(
          fq_filename,
          infer_writer_p.output_num_shards,
          output_format=infer_writer_p.output_format,
      )

    for step in (range(num_steps) if num_steps >= 0 else itertools.count()):
      logging.info('Evaling input batch %d', step + 1)
      try:
        batch = input_gen.get_next()
      except (tf.errors.OutOfRangeError, StopIteration):
        input_gen.reset()
        break

      outputs = infer_pmap_step(
          replicated_model_states,
          output_seeds,
          partitioner.preprocess_inputs(input_gen, batch, None),
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
