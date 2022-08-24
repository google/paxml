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

"""Tuning loop for PAX."""

import os
from typing import Callable, Dict, Optional
from absl import logging
from clu import platform
import jax
from paxml import automl
from paxml import base_experiment
from paxml import trainer_lib
from praxis import py_utils

import pyglove as pg  # mapped to internal
import tensorflow.compat.v2 as tf


TrialFn = Callable[[base_experiment.BaseExperiment,  # Experiment config.
                    platform.WorkUnit,               # Platform work unit.
                    str,                             # Job log dir
                    trainer_lib.EarlyStoppingFn],    # Early stopping fn.
                   None]


def write_file_once(file_path, content):
  """Writes debug information to file only once."""
  if not tf.io.gfile.exists(file_path):
    try:
      with tf.io.gfile.GFile(file_path, 'w') as f:
        f.write(content)
    except tf.errors.NotFoundError:
      logging.warn(
          'Cannot write file %r as another process is writing to the same '
          'file. This is not an issue as the file is only created for '
          'debugging purpose and has the same content among all the workers. '
          'So any successful write will achieve this purpose.', file_path)


def get_search_space(
    experiment_config: base_experiment.BaseExperiment
    ) -> pg.hyper.DynamicEvaluationContext:
  """Gets the search space from experiment config."""
  # Inspect the search space by evaluating the hyperparameters.
  # We include tuning parameters from both the `task` and `datasets` in the
  # search space. A caveat is that when multiple datasets have tunable
  # parameters, even one of them is not evaluated, its tunable parameters will
  # be included. We can improve this in the future if this turns out to be an
  # issue.
  def inspect_search_space() -> None:
    _ = experiment_config.task()
    _ = experiment_config.datasets()
    _ = experiment_config.decoder_datasets()

  return pg.hyper.trace(inspect_search_space, require_hyper_name=True)


def tune(trial_fn: TrialFn,
         experiment_config: base_experiment.BaseExperiment,
         work_unit: platform.WorkUnit,
         job_log_dir: str,
         study: Optional[str] = None,
         pythia_port: Optional[int] = None,
         is_metric_reporting_role: bool = True,
         tuner_group: Optional[str] = None,
         max_num_trials: Optional[int] = None) -> None:
  """Tune an experiment.

  An experiment can be tuned by running a tuning loop, with each iteration
  calling `trial_fn` for evaluating a trial sampled by the controller.

  The tuning procedure is set up with the following steps:
  1) It calls the `search` method of the experiment class to get the
     hyperparameters for the search, which contains the definition for
     the search algorithm and reward function.
  2) It inspects the search space by calling the `task` and `datasets` methods
     of the experiment class, thus all PyGlove search primitives (e.g.
     `pg.oneof`) will be collected.
  3) Then it starts a loop with `pg.sample`, based on the search space and
     search algorithm obtained above.
  4) Within the tuning loop, the `example` is provided as a context manager
     to connect the controller decisions with the return value of each search
     primitive called under the context manager. Therefore, we delegate the
     trial evaluation logic to `trial_fn`, which is done by passing
     a per-trial early stopping function for reporting measurements, completing/
     early stopping the trial.

  Args:
    trial_fn: Trial function, which will be called for each trial in the loop.
    experiment_config: The experiment to run.
    work_unit: Work unit for adding experiment artifact and reporting status.
    job_log_dir: The directory used for storing logs and writing checkpoints.
    study: Vizier study name.
    pythia_port: Pythia port for hosting Vizier algorithms.
    is_metric_reporting_role: Whether current process is in the role for
      reporting metrics. Among train/eval/decoder, only one role can report
      metrics to the controller at the moment.
    tuner_group: The identifier for the tuner group that current process belongs
      to. If None, all processes will be working on different trials. When
      specified, paired training, eval and decoder processes should use the
      same tuner group, which will get the same trial during tuning. Only one
      process (with is_metric_reporting_role=True) should report the measurement
      and signal the completion or stopping of the training.
    max_num_trials: An optional max number of trials for current tuning.
      If not None, it will override the default max number of trials specified
      by the `search` method of the experiment.
  """
  # Google-internal tuning infra init.

  search_hparams = experiment_config.search()
  search_algorithm = search_hparams.search_algorithm.Instantiate()()
  reward_fn = search_hparams.search_reward.Instantiate()
  max_num_trials = max_num_trials or search_hparams.max_num_trials

  search_space = get_search_space(experiment_config)
  if search_space.dna_spec.is_constant:
    raise ValueError(f'Aborting tuning: there is no tunable parameters in'
                     f'experiment {experiment_config.__class__.__name__!r}.')

  tf.io.gfile.makedirs(job_log_dir)
  logging.info('Search space: %s', search_space.dna_spec)
  search_space_debug_file = os.path.join(job_log_dir, 'search_space.txt')
  write_file_once(search_space_debug_file, str(search_space.dna_spec))
  work_unit.create_artifact(platform.ArtifactType.FILE, search_space_debug_file,
                            'search_space')

  logging.info('Search algorithm: %s', search_algorithm)
  algorithm_debug_file = os.path.join(job_log_dir, 'search_algorithm.txt')
  write_file_once(algorithm_debug_file, str(search_algorithm))
  work_unit.create_artifact(platform.ArtifactType.FILE, algorithm_debug_file,
                            'search_algorithm')

  # Helper functions for tuning.
  def get_early_stopping_fn(
      feedback: pg.tuning.Feedback) -> trainer_lib.EarlyStoppingFn:
    """Gets early stopping function based on a feedback object."""

    def should_stop_early(metrics: Dict[str, float],
                          running_mode: trainer_lib.RunningMode,
                          global_step: int, is_last_checkpoint: bool) -> bool:
      """Early stopping function."""
      if is_metric_reporting_role:
        # `metrics_by_dataset` could be None for interleaved train/eval
        # when evaluation is not performed at current global step.
        if (jax.process_index() == 0 and
            (running_mode & trainer_lib.RunningMode.EVAL or
             running_mode & trainer_lib.RunningMode.DECODE)):
          # Computing reward and report back to the tuning service.
          reward = reward_fn(metrics, global_step)
          feedback.add_measurement(reward, metrics=metrics, step=global_step)
          logging.info(
              'Measurement is reported to trial %d at step %d '
              'with reward value %f (mode=%s, is_last_checkpoint=%s): %s.',
              feedback.id, global_step, reward, running_mode,
              is_last_checkpoint, metrics)
        if is_last_checkpoint:
          py_utils.sync_global_devices(
              f'Trial termination at step {global_step} started.')
          # `feedback.done` should be called just once per trial.
          if jax.process_index() == 0:
            feedback.done()
          py_utils.sync_global_devices(
              f'Trial termination at step {global_step} completed.')
          logging.info('Trial %d is now completed.', feedback.id)
      return feedback.should_stop_early()
    return should_stop_early

  for example, feedback in pg.sample(
      search_space, search_algorithm, max_num_trials,
      group=tuner_group,
      name='local' if study is None else None):
    logging.info(
        'Start working on trial %d (group=%r)...', feedback.id, tuner_group)
    # Context manager to deliver different program hyperparameters
    # in each trial.
    with example():
      # Mark trial as infeasible on NaN. PAX user can add more error types here.
      with feedback.skip_on_exceptions((FloatingPointError,)):
        try:
          trial_fn(experiment_config, work_unit,
                   os.path.join(job_log_dir, str(feedback.id)),
                   get_early_stopping_fn(feedback))
        except automl.EarlyStoppingError as e:
          if jax.process_index() == 0:
            if e.skip:
              feedback.skip(e.skip_reason or 'Unknown.')
              logging.info(
                  'Trial %d is early stopped at step %d and will be skipped '
                  'by controller. Reason: %s.',
                  feedback.id, e.step, e.skip_reason)
            else:
              reward = e.reward
              if reward is None:
                reward = reward_fn(e.metrics, e.step)
              feedback.add_measurement(
                  reward=reward,
                  step=e.step,
                  metrics=e.metrics,
                  checkpoint_path=e.checkpoint)
              feedback.done()
              logging.info(
                  'Trial %d is early stopped at step %d with reward %f which '
                  'will be fed back to the controller. Metrics: %s.',
                  feedback.id, e.step, reward, e.metrics)
  logging.info('Completed with all trials for study %r', study)
