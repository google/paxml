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
import functools
import gc
from typing import Any, Callable, Optional, Sequence, Tuple

from absl import logging
from etils import epath
import jax
from paxml import base_executor
from paxml import base_metrics
from paxml import decode_programs as decode_programs_lib
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
from praxis import pax_fiddle
from praxis import py_utils
import tensorflow.compat.v2 as tf

from paxml import checkpoints  # mapped to internal

instantiate = base_hyperparams.instantiate
RunningMode = trainer_lib.RunningMode
SummaryWriter = tf.summary.SummaryWriter
TrainState = train_states.TrainState
TrainStateProvenance = train_states.TrainStateProvenance

INFO = logging.INFO


def _maybe_update_latest_model_step(
    train_input_p: pax_fiddle.Config[base_input.BaseInput],
    initial_global_step: Optional[int],
    task: tasks_lib.SingleTask,
) -> None:
  """Updates `train_input_p` in place its latest model step."""
  if not hasattr(train_input_p, 'deterministic_input_start_index'):
    # Not deterministic seqio.
    return
  logging.info(
      'Attempting to use global step for deterministic seqio (@ step %r)...',
      initial_global_step,
  )
  if initial_global_step is None:
    if task.train.external_checkpoint_path:
      logging.warning(
          'Disabling deterministic SeqIO since it will restore from external'
          ' checkpoint, and the step number is not known beforehand.'
      )
    # When not restoring from external_checkpoint_path, it means no checkpoint
    # to restore in this case and it'll train from step 0, so no need to update.
    return

  logging.info('Updating _latest_model_step for training input.')
  dp = train_input_p.deterministic_input_start_index
  dp._latest_model_step = (  # pylint: disable=protected-access
      initial_global_step
  )


class DefaultExecutor(base_executor.BaseExecutor):
  """The default executor for running programs."""

  def __init__(self):
    super().__init__()

    # States to set in .setup().
    self._job_log_dir: epath.Path = None
    self._early_stopping_fn = None
    self._task: tasks_lib.SingleTask = None
    self._checkpointer: checkpoints.TrainingCheckpointer = None
    self._partitioner: partitioning.Partitioner = None
    self._train_program: programs.BaseTrainProgram = None
    self._eval_programs: Sequence[programs.BaseEvalProgram] = None
    self._decode_programs: Sequence[
        decode_programs_lib.SingleTaskDecodeProgram
    ] = None

    # States to lazily initialize in .setup().
    self._train_input_pipeline = None
    self._partitioned_train_state = None
    self._train_state_provenance = None
    self._total_num_params = None
    self._train_prng_seed = None
    self._eval_prng_seed = None
    self._decode_prng_seed = None

  def _maybe_create_train_input(
      self,
      task: tasks_lib.SingleTask,
      step: Optional[int],
      train_input_p: pax_fiddle.Config[base_input.BaseInput],
  ) -> Tuple[
      Optional[base_input.BaseInput],
      Optional[base_input.BaseInput],
      Optional[base_input.BaseInput],
  ]:
    """Optionally creates the train input for partitioner and checkpointing.

    Args:
      task: The task.
      step: The step number of the checkpoint to restore from. If None, means no
        checkpoint to restore.
      train_input_p: The config for the train input pipeline.

    Returns:
      A 3-tuple (train_input, train_input_for_partitioner,
      train_input_for_checkpoint), where:

      - train_input: the train input pipeline.
      - train_input_for_partitioner: represents the train_input_pipeline arg
        passed to partitioner.setup(). If set, the partitioner will use it to
        get the shape/dtype information for model.init.
      - train_input_for_checkpoint: represents the train_input_pipeline arg
        passed to checkpointer.get_model_states(). If set, the checkpointer will
        restore its states from checkpoint.
    """
    logging.info(
        '[PAX STATUS]: Instantiating train input pipeline (%s)', train_input_p
    )
    if not task.train.enable_input_checkpointing:
      _maybe_update_latest_model_step(train_input_p, step, task)
    train_input = instantiate(train_input_p)

    train_input_for_partitioner = (
        None if task.train.enforce_input_specs else train_input
    )
    train_input_for_checkpoint = (
        train_input if task.train.enable_input_checkpointing else None
    )
    return train_input, train_input_for_partitioner, train_input_for_checkpoint

  def setup(
      self,
      jax_task: tasks_lib.SingleTask,
      job_log_dir: epath.Path,
      checkpointer: Any,
      partitioner: partitioning.Partitioner,
      input_specs_provider: base_input.BaseInputSpecsProvider,
      train_input_p: pax_fiddle.Config[base_input.BaseInput],
      train_program: programs.BaseTrainProgram,
      eval_programs: Sequence[programs.BaseEvalProgram],
      decode_programs: Sequence[decode_programs_lib.SingleTaskDecodeProgram],
      early_stopping_fn: Optional[trainer_lib.EarlyStoppingFn],
      exit_after_ondemand_checkpoint: bool = False,
  ):
    self._task = jax_task
    self._job_log_dir = job_log_dir
    self._checkpointer = checkpointer
    self._partitioner = partitioner
    self._train_program = train_program
    self._eval_programs = eval_programs
    self._decode_programs = decode_programs
    self._early_stopping_fn = early_stopping_fn
    self._exit_after_ondemand_checkpoint = exit_after_ondemand_checkpoint

    # Creates the root prng key and train input pipeline.
    root_prng_key = jax.random.PRNGKey(self._task.train.random_seed)
    train_input_p = partitioner.preprocess_input_config(train_input_p)
    train_input, train_input_for_partitioner, train_input_for_checkpoint = (
        self._maybe_create_train_input(
            self._task, checkpointer.step_to_restore, train_input_p
        )
    )

    # Sets up the partitioner. Note it only needs shape/dtype information of the
    # prng key.
    # TODO(laigd): let it take ShapeDtypeStruct of prng key instead.
    train_input_specs = trainer_lib.get_train_input_specs_for_model_init(
        self._task, input_specs_provider
    )
    if not train_input_specs:
      raise ValueError(
          'No training input specs available from [%s].'
          % type(input_specs_provider)
      )
    logging.info('[PAX STATUS]: Setting up partitioner')
    partitioner.setup(
        jax_task,
        root_prng_key,
        train_inputs_shape_dtype=train_input_specs,
        job_log_dir=job_log_dir,
    )
    logging.info('[PAX STATUS]: Getting train state metadata.')
    train_state_metadata = partitioner.get_train_state_metadata()

    # JaxContext needed for shared layer lookup from global scope.
    logging.info('[PAX STATUS]: Writing post init model hparams.')
    with base_layer.JaxContext.new_context():
      # Dump out model meta info for debugging.
      trainer_lib.write_post_init_model_hparams_file(
          jax_task.model, train_state_metadata, job_log_dir
      )

    # Restore TrainState from checkpoint or initialize it.
    logging.info('[PAX STATUS]: Starting checkpoint load / variable init.')
    with py_utils.timeit() as checkpoint_load_timer:
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
    logging.info(
        '[PAX STATUS]: Checkpoint load / variable init took %d seconds',
        checkpoint_load_timer.elapsed,
    )
    if train_state_provenance:
      trainer_lib.write_train_provenance_file(
          train_state_provenance, job_log_dir
      )

    # Splits the key.
    train_prng_seed, eval_prng_seed, decode_prng_seed = jax.random.split(
        root_prng_key, 3
    )
    logging.info('train prng seed: %s', train_prng_seed)
    logging.info('eval prng seed: %s', eval_prng_seed)
    logging.info('decode prng seed: %s', decode_prng_seed)
    train_prng_seed = partitioner.preprocess_prng_key(train_prng_seed)
    eval_prng_seed = partitioner.preprocess_prng_key(eval_prng_seed)
    # If prng_key_fold_with_batch_index is True, we need to fold in the step
    # number before preprocessing the key, so preprocessing need to be done at
    # every step.
    if not self._task.decode.prng_key_fold_with_batch_index:
      decode_prng_seed = partitioner.preprocess_prng_key(decode_prng_seed)

    # Sets the lazily initialized states.
    self._train_input_pipeline = train_input
    self._partitioned_train_state = partitioned_train_state
    self._train_state_provenance = train_state_provenance
    self._total_num_params = total_num_params
    self._train_prng_seed = train_prng_seed
    self._eval_prng_seed = eval_prng_seed
    self._decode_prng_seed = decode_prng_seed

  def start(self):
    logging.info('Starting executor.')
    is_vars_replicated = self._task.model.ici_mesh_shape is None
    _train_and_evaluate_common(
        task=self._task,
        partitioner=self._partitioner,
        train_program=self._train_program,
        train_input=self._train_input_pipeline,
        partitioned_train_state=self._partitioned_train_state,
        train_state_provenance=self._train_state_provenance,
        eval_programs=self._eval_programs,
        decode_programs=self._decode_programs,
        total_num_params=self._total_num_params,
        early_stopping_fn=self._early_stopping_fn,
        checkpointer=self._checkpointer,
        job_log_dir=self._job_log_dir,
        eval_prng_seed=self._eval_prng_seed,
        decode_prng_seed=self._decode_prng_seed,
        exit_after_ondemand_checkpoint=self._exit_after_ondemand_checkpoint,
        is_vars_replicated=is_vars_replicated,
        train_prng_seed=self._train_prng_seed,
    )

    # Shutdown the programs and run necessary cleanup.
    logging.info('[PAX STATUS]: Shutting down executor.')
    self._train_program.shutdown()
    for program in self._eval_programs:
      program.shutdown()
    for program in self._decode_programs:
      program.shutdown()
    logging.info('[PAX STATUS]: Executor shutdown complete.')


def _train_and_evaluate_common(
    *,
    task: tasks_lib.SingleTask,
    partitioner: partitioning.Partitioner,
    train_program: programs.BaseTrainProgram,
    train_input: base_input.BaseInput,
    partitioned_train_state: TrainState,
    train_state_provenance: TrainStateProvenance,
    # TODO(hthu): Take a more generalized form of EvalProgram interface.
    eval_programs: Sequence[programs.BaseEvalProgram],
    decode_programs: Sequence[decode_programs_lib.SingleTaskDecodeProgram],
    total_num_params,
    early_stopping_fn,
    checkpointer,
    job_log_dir,
    eval_prng_seed,
    decode_prng_seed,
    is_vars_replicated,
    train_prng_seed,
    exit_after_ondemand_checkpoint,
):
  """Training loop code common to both pmap and spmd."""
  train_p = task.train
  train_state_metadata = partitioner.get_train_state_metadata()
  train_input_for_checkpoint = (
      train_input if train_p.enable_input_checkpointing else None
  )
  initial_global_step = int(
      py_utils.maybe_unreplicate_for_fully_replicated(
          partitioned_train_state.step
      )
  )
  logging.info('Model initial global_step=%d', initial_global_step)
  if checkpointer.step_to_restore is not None:
    assert checkpointer.step_to_restore == initial_global_step, (
        f'Checkpoint number {checkpointer.step_to_restore} and restored step'
        f' number {initial_global_step} mismatch.'
    )

  logging.info('[PAX STATUS]: Starting training loop.')
  step_i = initial_global_step

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
  for program in decode_programs:
    program.setup(task, partitioner, job_log_dir, decode_prng_seed)
  trainer_lib.check_unique_names([p.eval_input for p in eval_programs])
  trainer_lib.check_unique_names([p.decode_input for p in decode_programs])

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
    logging.log_first_n(INFO, '[PAX STATUS]: Beginning step `%d`.', 5, step_i)
    checkpointer.save_if_needed(
        step_i,
        partitioned_train_state,
        train_state_metadata.unpadded_global_shapes,
        train_state_metadata.partition_specs,
        train_input_for_checkpoint,
    )
    if exit_after_ondemand_checkpoint and checkpointer.reached_preemption(
        step_i
    ):
      checkpointer.wait_until_finished()
      exit(1)

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
    train_weighted_scalars = program_output.weighted_scalars
    steps_per_sec = program_output.steps_per_sec
    eval_train_metrics = program_output.eval_train_metrics

    # While the eval ones below are post-model weight updates, hence the step
    # counter is incremented in between.
    step_i = program_output.new_train_step

    eval_metrics: Optional[tuning_lib.EvalMetrics] = None
    # Run eval at regular step interval or the final training step.
    if train_p.eval_interval_steps and (
        step_i % train_p.eval_interval_steps == 0
        or step_i >= train_p.num_train_steps
    ):
      logging.log_first_n(INFO, '[PAX STATUS]:  Starting eval_step().', 5)
      eval_partitioned_train_state = programs.get_eval_train_state(
          task, partitioned_train_state, task.train.eval_use_ema_states
      )
      # If we have eval test then also evaluate on test.
      if eval_programs:
        logging.debug('[PAX STATUS]:  Running eval programs.')
        eval_metrics, elapsed_secs = eval_lib.run_eval_programs(
            eval_programs=eval_programs,
            train_state=eval_partitioned_train_state,
            step=step_i,
        )
        jax.monitoring.record_event_duration_secs(
            '/jax/pax/train/interleaved_eval_duration_sec', elapsed_secs
        )
        logging.log_first_n(
            INFO,
            '[PAX STATUS]:  Completed running eval programs in %f seconds.',
            5,
            elapsed_secs,
        )

    decode_metrics: Optional[tuning_lib.DecodeMetrics] = None
    if (
        decode_programs
        and train_p.decode_interval_steps
        and (
            step_i % train_p.decode_interval_steps == 0
            or step_i >= train_p.num_train_steps
        )
    ):
      decode_partitioned_train_state = programs.get_eval_train_state(
          task, partitioned_train_state, task.train.decode_use_ema_states
      )
      logging.debug('[PAX STATUS]:  Running decode programs.')
      decode_metrics, elapsed_secs = eval_lib.run_decode_programs(
          decode_programs=decode_programs,
          train_state=decode_partitioned_train_state,
          step=step_i,
      )
      jax.monitoring.record_event_duration_secs(
          '/jax/pax/train/interleaved_decode_duration_sec',
          elapsed_secs,
      )
      logging.log_first_n(
          INFO,
          '[PAX STATUS]:  Completed running decode programs in %f seconds.',
          5,
          elapsed_secs,
      )
    logging.log_first_n(
        INFO, '[PAX STATUS]: Step `%d` completed.', 5, step_i - 1
    )

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
              task.train.num_train_steps,
              task.train.eval_interval_steps,
              task.train.decode_interval_steps,
              task.train.save_interval_steps,
              train_to_end=getattr(early_stopping_fn, 'train_to_end', False),
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

  logging.info('[PAX STATUS]: Saving checkpoint for final step.')
  checkpointer.save_final(
      step_i,
      partitioned_train_state=partitioned_train_state,
      train_state_unpadded_shape_dtype_struct=train_state_metadata.unpadded_global_shapes,
      train_state_pspecs=train_state_metadata.partition_specs,
      train_input_pipeline=train_input_for_checkpoint,
  )

  checkpointer.wait_until_finished()
  logging.info('[PAX STATUS]: Final checkpoint saved.')
