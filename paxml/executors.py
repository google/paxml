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

"""Implementations of program executors."""

import contextlib
import gc
from typing import Any, Callable, Optional, Protocol, Sequence, Tuple

from absl import logging
from etils import epath
import jax
from paxml import base_executor
from paxml import eval_lib
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
from praxis import py_utils
from praxis import pytypes
import tensorflow.compat.v2 as tf

instantiate = base_hyperparams.instantiate
PRNGKey = pytypes.PRNGKey
RunningMode = trainer_lib.RunningMode
SummaryWriter = tf.summary.SummaryWriter
TrainState = train_states.TrainState
TrainStateProvenance = train_states.TrainStateProvenance


def _maybe_update_latest_model_step(
    train_input: base_input.BaseInput,
    train_input_p: base_input.BaseInput.HParams,
    initial_global_step: int,
) -> base_input.BaseInput:
  """Updates `train_input_p` in place its latest model step."""
  if not hasattr(train_input_p, 'deterministic_input_start_index'):
    return train_input
  dp = train_input_p.deterministic_input_start_index
  dp._latest_model_step = (
      initial_global_step  # pylint: disable=protected-access
  )
  logging.info('Reinstantiating input because _latest_model_step is updated.')
  return instantiate(train_input_p)


class _DecodeSummaryWriters(contextlib.ExitStack):
  """Manage decode summary writers."""

  _exit_callbacks = []

  def __init__(
      self, job_log_dir: epath.Path, decode_input_names: Sequence[str]
  ):
    """Initialize context manager.

    Args:
      job_log_dir: Directory for the job logs.
      decode_input_names: list of names for the decode input pipelines.
    """
    super().__init__()
    self.summary_decode_dirs = [
        job_log_dir / 'summaries' / f'decode_test_{name}'
        for name in decode_input_names
    ]

  def __enter__(self) -> Sequence[SummaryWriter]:
    self.decode_summary_writers = [
        self.enter_context(summary_utils.get_summary_writer(d))
        for d in self.summary_decode_dirs
    ]
    return self.decode_summary_writers


class _Checkpointer(Protocol):
  """Checkpointer interface."""

  def save_if_needed(
      self,
      step_i,
      partitioned_train_state,
      train_state_unpadded_shape_dtype_struct,
      train_state_pspecs,
      train_input_pipeline,
  ):
    ...

  def save_final(
      self,
      step_i,
      *,
      partitioned_train_state,
      train_state_unpadded_shape_dtype_struct,
      train_state_pspecs,
      train_input_pipeline,
  ):
    ...

  def get_model_states(
      self,
      partitioner: partitioning.Partitioner,
      metadata: trainer_lib.TrainStateMetadata,
      root_prng_key: PRNGKey,
      train_input_pipeline: Optional[base_input.BaseInput] = None,
  ) -> Tuple[TrainState, Optional[TrainStateProvenance], int, PRNGKey]:
    ...

  def wait_until_finished(self):
    ...


class DefaultExecutor(base_executor.BaseExecutor):
  """The default executor for running programs."""

  def __init__(self):
    super().__init__()

    # States to set in .setup().
    self._job_log_dir: epath.Path = None
    self._early_stopping_fn = None
    self._task: tasks_lib.SingleTask = None
    self._checkpointer: _Checkpointer = None
    self._partitioner: partitioning.Partitioner = None
    self._decode_input_ps = None
    self._train_program: programs.BaseTrainProgram = None
    self._eval_programs: Sequence[programs.BaseEvalProgram] = None

    # States to lazily initialize in .setup().
    self._train_input_pipeline = None
    self._partitioned_train_state = None
    self._train_state_provenance = None
    self._total_num_params = None
    self._prng_key = None
    self._train_prng_seed = None
    self._eval_prng_seed = None

  def setup(
      self,
      jax_task: tasks_lib.SingleTask,
      job_log_dir: epath.Path,
      checkpointer: Any,
      partitioner: partitioning.Partitioner,
      train_input_p: base_input.BaseInput.HParams,
      decode_input_ps: Sequence[base_input.BaseInput.HParams],
      train_program: programs.BaseTrainProgram,
      eval_programs: Sequence[programs.BaseEvalProgram],
      early_stopping_fn: Optional[trainer_lib.EarlyStoppingFn],
  ):
    self._task = jax_task
    self._job_log_dir = job_log_dir
    self._checkpointer = checkpointer
    self._partitioner = partitioner
    self._decode_input_ps = decode_input_ps
    self._train_program = train_program
    self._eval_programs = eval_programs
    self._early_stopping_fn = early_stopping_fn
    task_p = jax_task.hparams

    # Creates the root prng key and train input pipeline.
    root_prng_key = jax.random.PRNGKey(task_p.train.random_seed)
    train_input_p = partitioner.preprocess_input_params(train_input_p)
    train_input_pipeline = instantiate(train_input_p)

    # Sets up the partitioner. Note it only needs shape/dtype information of the
    # prng key.
    # TODO(laigd): let it take ShapeDtypeStruct of prng key instead.
    partitioner.setup(
        jax_task,
        root_prng_key,
        train_inputs_shape_dtype=None,
        train_input_pipeline=train_input_pipeline,
        job_log_dir=job_log_dir,
    )
    train_state_metadata = partitioner.get_train_state_metadata()

    # JaxContext needed for shared layer lookup from global scope.
    with base_layer.JaxContext.new_context():
      # Dump out model meta info for debugging.
      trainer_lib.write_post_init_model_hparams_file(
          jax_task.model, train_state_metadata.var_weight_hparams, job_log_dir
      )

    # Restore TrainState from checkpoint or initialize it.
    train_input_for_checkpoint = (
        train_input_pipeline
        if task_p.train.enable_input_checkpointing
        else None
    )
    (
        partitioned_train_state,
        train_state_provenance,
        total_num_params,
        root_prng_key,
    ) = checkpointer.get_model_states(
        partitioner,
        train_state_metadata,
        root_prng_key,
        train_input_for_checkpoint,
    )
    if train_state_provenance:
      trainer_lib.write_train_provenance_file(
          train_state_provenance, job_log_dir
      )

    # Restore the train input states for deterministic inputs.
    initial_global_step = int(
        py_utils.maybe_unreplicate_for_fully_replicated(
            partitioned_train_state.step
        )
    )
    logging.info('Model initial global_step=%d', initial_global_step)
    if not task_p.train.enable_input_checkpointing:
      train_input_pipeline = _maybe_update_latest_model_step(
          train_input_pipeline, train_input_p, initial_global_step
      )

    # Splits the key.
    prng_key, train_prng_seed, eval_prng_seed = jax.random.split(
        root_prng_key, 3
    )
    logging.info('train prng seed: %s', train_prng_seed)
    logging.info('eval prng seed: %s', eval_prng_seed)
    train_prng_seed = partitioner.preprocess_prng_key(train_prng_seed)
    eval_prng_seed = partitioner.preprocess_prng_key(eval_prng_seed)

    # Sets the lazily initialized states.
    self._train_input_pipeline = train_input_pipeline
    self._partitioned_train_state = partitioned_train_state
    self._train_state_provenance = train_state_provenance
    self._total_num_params = total_num_params
    self._prng_key = prng_key
    self._train_prng_seed = train_prng_seed
    self._eval_prng_seed = eval_prng_seed

  def _partition_decode_once_fns_pmap(self, prng_key, decode_input_p):
    decode_input_pipelines = [
        instantiate(input_p) for input_p in decode_input_p
    ]
    trainer_lib.check_unique_names(decode_input_pipelines)
    prng_key, decode_key = jax.random.split(prng_key, 2)
    logging.info('decode prng_seed: %s', decode_key)
    decode_key = self._partitioner.preprocess_prng_key(decode_key)
    decode_once_fn = eval_lib.partition_decode_once_pmap_model(
        self._task,
        self._partitioner,
        self._task.hparams,
        self._partitioner.get_train_state_metadata().var_weight_hparams,
        decode_input_pipelines,
        decode_key,
        self._job_log_dir,
    )
    decode_input_names = [inp.name for inp in decode_input_pipelines]
    return decode_once_fn, prng_key, decode_input_names

  def _partition_decode_once_fns_spmd(
      self,
      prng_key: jax.random.KeyArray,
      decode_input_ps: Sequence[base_input.BaseInput.HParams],
  ) -> Tuple[
      Callable[..., tuning_lib.DecodeMetrics],
      jax.random.KeyArray,
      Sequence[str],
  ]:
    assert decode_input_ps, 'decode_input_p must not be empty'
    prng_key, decode_key = jax.random.split(prng_key, 2)
    logging.info('decode prng_key: %s', decode_key)
    decode_key = self._partitioner.preprocess_prng_key(decode_key)

    padded_decode_input_ps = [
        trainer_lib.adjust_input_params_for_small_batch(
            input_p, self._partitioner.global_mesh
        )
        for input_p in decode_input_ps
    ]
    padded_decode_input_pipelines = [
        instantiate(input_p) for input_p in padded_decode_input_ps
    ]
    trainer_lib.check_unique_names(padded_decode_input_pipelines)
    _, decode_inputs_shape_dtype = trainer_lib.get_inputs_shape_dtype(
        padded_decode_input_ps[0]
    )

    # TODO(pax-dev): Support auto-sharding for decoder step.
    step_fn, is_eval = partitioning.get_step_fn(RunningMode.DECODE)
    assert is_eval
    decode_step_fn, decode_input_partition_spec = self._partitioner.partition(
        step_fn, decode_inputs_shape_dtype, is_eval
    )

    decode_once_fn = eval_lib.partition_decode_once_spmd_model(
        self._task,
        self._partitioner,
        self._task.hparams,
        padded_decode_input_pipelines,
        self._job_log_dir,
        decode_key,
        decode_step_fn,
        decode_input_partition_spec,
    )

    decode_input_names = [inp.name for inp in padded_decode_input_pipelines]
    return decode_once_fn, prng_key, decode_input_names

  @property
  def partition_decode_once_fns(self):
    if self._task.hparams.model.ici_mesh_shape is not None:
      return self._partition_decode_once_fns_spmd
    else:
      return self._partition_decode_once_fns_pmap

  def start(self):
    is_vars_replicated = self._task.hparams.model.ici_mesh_shape is None
    _train_and_evaluate_common(
        self._task,
        self._partitioner,
        self._train_program,
        self._train_input_pipeline,
        self._partitioned_train_state,
        self._train_state_provenance,
        self._prng_key,
        self._eval_programs,
        self._decode_input_ps,
        self._total_num_params,
        self._early_stopping_fn,
        self._checkpointer,
        self.partition_decode_once_fns,
        self._job_log_dir,
        self._eval_prng_seed,
        is_vars_replicated,
        self._train_prng_seed,
    )

    # Shutdown the programs and run necessary cleanup.
    self._train_program.shutdown()
    for program in self._eval_programs:
      program.shutdown()


def _train_and_evaluate_common(
    task: tasks_lib.SingleTask,
    partitioner: partitioning.Partitioner,
    train_program: programs.BaseTrainProgram,
    train_input: base_input.BaseInput,
    partitioned_train_state: TrainState,
    train_state_provenance: TrainStateProvenance,
    prng_key,
    # TODO(hthu): Take a more generalized form of EvalProgram interface.
    eval_programs: Sequence[programs.BaseEvalProgram],
    decode_input_p,
    total_num_params,
    early_stopping_fn,
    checkpointer,
    partition_decode_once_fns,
    job_log_dir,
    eval_prng_seed,
    is_vars_replicated,
    train_prng_seed,
):
  """Training loop code common to both pmap and spmd."""
  task_p = task.hparams
  train_p = task_p.train
  train_state_metadata = partitioner.get_train_state_metadata()
  train_input_for_checkpoint = (
      train_input if train_p.enable_input_checkpointing else None
  )

  if decode_input_p:
    decode_once_fn, prng_key, decode_input_names = partition_decode_once_fns(
        prng_key, decode_input_p
    )
  else:
    decode_input_names = []

  logging.info('Training loop starting...')

  with _DecodeSummaryWriters(
      job_log_dir, decode_input_names
  ) as decode_summary_writers:
    step_i = int(
        py_utils.maybe_unreplicate_for_fully_replicated(
            partitioned_train_state.step
        )
    )

    # Sets up the programs.
    train_program.setup(
        task,
        train_input,
        partitioner,
        job_log_dir,
        train_prng_seed,
        eval_prng_seed,
        step_i,
    )
    for program in eval_programs:
      program.setup(task, partitioner, job_log_dir, eval_prng_seed)
    trainer_lib.check_unique_names([prog.eval_input for prog in eval_programs])

    train_summary_writer = train_program.summary_writer
    # This only prints the view from the first host machine.
    summary_utils.write_model_structure(
        train_summary_writer, partitioned_train_state, is_vars_replicated
    )
    # train_state_provenance is None when model restored from checkpoint
    if train_state_provenance:
      summary_utils.write_model_provenance(
          train_summary_writer, train_state_provenance
      )
    summary_utils.write_total_num_params(train_summary_writer, total_num_params)
    summary_utils.write_global_batch_size(
        train_summary_writer, train_program.train_unpadded_global_batch_size
    )

    # Start the train loop. Make sure all at the same step.
    py_utils.sync_global_devices(f'Start training loop from step: {step_i}')
    # Collect then freeze GC, so that GC in the training loop will not touch the
    # python objects used to initialize the model. Unfreeze at the end of the
    # loop.
    gc.collect()
    gc.freeze()
    while True:
      logging.debug('step=`%d`: Beginning', step_i)
      checkpointer.save_if_needed(
          step_i,
          partitioned_train_state,
          train_state_metadata.unpadded_global_shapes,
          train_state_metadata.partition_specs,
          train_input_for_checkpoint,
      )

      if not train_program.should_run(partitioned_train_state, step_i):
        logging.info(
            (
                'Training loop completed (step (`%d`) greater than '
                'num_train_step (`%d`).'
            ),
            step_i,
            train_p.num_train_steps,
        )
        break

      program_output = train_program.run(partitioned_train_state, step_i)
      partitioned_train_state = program_output.state
      train_weighted_scalars = program_output.aux.weighted_scalars
      steps_per_sec = program_output.aux.steps_per_sec
      eval_train_metrics = program_output.aux.eval_train_metrics

      # While the eval ones below are post-model weight updates, hence the step
      # counter is incremented in between.
      step_i = program_output.aux.new_train_step

      eval_metrics: Optional[tuning_lib.EvalMetrics] = None
      # Run eval at regular step interval.
      if (
          train_p.eval_interval_steps
          and step_i % train_p.eval_interval_steps == 0
      ):
        logging.debug('  Starting eval_step().')
        eval_partitioned_train_state = programs.get_eval_train_state(
            task, partitioned_train_state
        )
        # If we have eval test then also evaluate on test.
        if eval_programs:
          logging.debug('  Performing eval_step() runs on test splits.')
          with py_utils.timeit() as eval_period:
            eval_metrics_list, eval_scoring_metrics_list, num_eval_steps = (
                eval_lib.run_eval_loop_over_test_splits(
                    eval_programs,
                    eval_partitioned_train_state,
                    eval_prng_seed,
                    step_i,
                    job_log_dir,
                )
            )
          eval_steps_per_sec = sum(num_eval_steps) / eval_period.elapsed
          eval_metrics = tuning_lib.EvalMetrics(
              metrics_list=eval_metrics_list,
              scoring_metrics_list=eval_scoring_metrics_list,
              steps_per_sec=eval_steps_per_sec,
              input_names=[prog.eval_input.name for prog in eval_programs],
          )
          logging.debug(
              '  Completed eval_step() runs on test splits in %f seconds.',
              eval_period.elapsed,
          )

      decode_metrics: Optional[tuning_lib.DecodeMetrics] = None
      if (
          decode_input_p
          and train_p.decode_interval_steps
          and step_i % train_p.decode_interval_steps == 0
      ):
        if train_p.decode_use_ema_states:
          if not tasks_lib.has_ema(task_p):
            raise ValueError(
                'decode_use_ema_states is requested but the '
                'learner does not seem to have ema enabled'
            )
          decode_partitioned_train_state = tasks_lib.extract_ema(
              partitioned_train_state
          )
          logging.debug('  Performing decode_once_fn() with ema states.')
        else:
          decode_partitioned_train_state = partitioned_train_state
        decode_metrics = decode_once_fn(
            decode_partitioned_train_state, decode_summary_writers
        )

      logging.debug('step=`%d`: End', step_i - 1)

      if early_stopping_fn is not None:
        if tuning_lib.should_early_stop(
            early_stopping_fn,
            step_i,
            is_last_ckpt=tuning_lib.is_last_checkpoint(
                RunningMode.detect(
                    has_train_metrics=True,
                    has_eval_metrics=bool(eval_metrics),
                    has_decode_metrics=bool(decode_metrics),
                ),
                step_i,
                task_p.train.num_train_steps,
                task_p.train.eval_interval_steps,
                task_p.train.decode_interval_steps,
                task_p.train.save_interval_steps,
                train_to_end=getattr(
                    early_stopping_fn, 'train_to_end', False)
            ),
            train_weighted_scalars=train_weighted_scalars,
            eval_train_metrics=eval_train_metrics,
            eval_metrics=eval_metrics,
            decode_metrics=decode_metrics,
            train_steps_per_sec=steps_per_sec,
            num_params=total_num_params,
        ):
          logging.info(
              (
                  'Training loop is early stopped at step `%d` by the '
                  'tuner, while num_train_step is `%d`.'
              ),
              step_i,
              train_p.num_train_steps,
          )
          break
    gc.unfreeze()
    # Save checkpoint for the last step.
    checkpointer.save_final(
        step_i,
        partitioned_train_state=partitioned_train_state,
        train_state_unpadded_shape_dtype_struct=train_state_metadata.unpadded_global_shapes,
        train_state_pspecs=train_state_metadata.partition_specs,
        train_input_pipeline=train_input_for_checkpoint,
    )

    checkpointer.wait_until_finished()
