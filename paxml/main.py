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

r"""Main file for running a PAX training and evaluation loop.

Example usage:
$ bazel run -c opt \
    third_party/py/paxml/tasks/lm/params:main -- \
    --exp=bert.BertAdamL4H128 \
    --job_log_dir=/tmp/jax_log_dir/exp01 --alsologtostderr
"""

import importlib
import os
import random
import re
import time
from typing import Dict, Optional, Sequence

from absl import app
from absl import flags
from absl import logging
from clu import platform
import jax
from jax.experimental.gda_serialization import serialization as gda_serialization
import numpy as np
from paxml import automl
from paxml import base_experiment
from paxml import eval_lib
from paxml import experiment_registry
from paxml import setup_jax
from paxml import train
from paxml import trainer_lib
from praxis import py_utils
import pyglove as pg  # mapped to internal
import tensorflow.compat.v2 as tf
persistence_gda_serialization = gda_serialization  # mapped to internal


# Required import to setup work units when running through XManager.

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'exp',
    None, 'Experiment configuration identifier name. This name typically '
    'has a format like "<task>.<module>.<experiment>", which should have been '
    'already registered in the global experiment registry with '
    '@experiment_registry.register.',
    required=True)
flags.DEFINE_string('job_log_dir', None,
                    'Directory where all experiment assets will be stored.')
flags.DEFINE_enum('mode', 'train',
                  ['train', 'eval', 'decode', 'decode_once', 'infer'],
                  'Flag to control which job is called.')
flags.DEFINE_bool(
    'eval_on_test', False, 'If True, then the training loop '
    'includes a full evaluation on all the test set splits. '
    'This can be set to True if we do not want an additional job '
    'to run continuous eval.')
flags.DEFINE_bool(
    'decode_during_train', False, 'If True, then the training loop '
    'includes running decode over decoder_datasets(). This can be set to True '
    'if we do not want an additional job to run continuous decode.')
flags.DEFINE_bool(
    'eval_during_decode', False, 'If True, then the decoder run will '
    'include running eval over the non-training data in datasets(). This is '
    'ignored if --mode is not decode or decode_once.')
flags.DEFINE_bool(
    'maybe_use_persistence_checkpointing', False,
    'If suitable, will try to rely on persistence-based checkpointing rather '
    'than Flax-based checkpointing for SPMD models.')
flags.DEFINE_bool(
    'jax_fully_async_checkpoint', False,
    'Enables fully asynchronous checkpointing via GDA and TensorStore. This '
    'means that the training can continue ahead when checkpointing is '
    'happening.')
flags.DEFINE_string(
    'checkpoint_todelete_subdir', None,
    'If set, checkpoints to be deleted will be only renamed into a '
    'subdirectory with the provided string. Otherwise, they will be directly '
    'deleted from the file system. Useful if checkpoint deletion is time '
    'consuming. By default, delete the checkpoint assets.')
flags.DEFINE_string(
    'restore_checkpoint_dir', None,
    'If set, the directory from which to restore checkpoint. '
    'Only supported when --mode=decode_once.')
flags.DEFINE_integer(
    'restore_checkpoint_step', None,
    'If set, the checkpoint step to restore. Only supported when '
    '--mode=decode_once.')
flags.DEFINE_bool(
    'globally_use_hardware_rng', True,
    'Whether to globally use fast hardware RNG. Deterministic only at the '
    'same compiler version and with the same sharding')
flags.DEFINE_integer(
    'jax_profiler_port', None,
    'If set, the jax.profiler port to use. Only needed for profiling in open source.'
)
flags.DEFINE_bool('enable_auto_sharding', False,
                  'Enable the XLA Auto SPMD partitioner.')
# Flags for automatic tuning.
flags.DEFINE_string(
    'study', None,
    'Study name for current tuning. If None, the program will be running in '
    'standard training/evaluation mode. Otherwise, it will run in tuning mode.')
flags.DEFINE_string(
    'tuner_group', None,
    'The identifier for the tuner group that current process belongs to. '
    'If None, all processes will be working on different trials. '
    'When specified, paired training, eval and decoder processes should use '
    'the same tuner group, which will get the same trial during tuning. Only '
    'one process should report the measurement and signal the completion or '
    'stopping of the training. See flag `metrics_from` for details.')
flags.DEFINE_enum(
    'metrics_from', 'eval', ['train', 'eval'],
    'This flag specifies from which process the measurements should be '
    'used for tuning. By default it is set to eval.')
flags.DEFINE_integer(
    'num_trials', None,
    'Max number of trials for tuning. If None, there will be no limit.')
flags.DEFINE_integer(
    'pythia_port', None,
    'Port for hosting Pythia service when non-Vizier built-in algorithms '
    'is used')
# Flags --jax_parallel_functions_output_gda, --jax_backend_target,
# --jax_xla_backend, --jax_enable_checks are available through JAX.


def _get_experiment(experiment_name: str) -> base_experiment.BaseExperimentT:
  """Retrieves an experiment config from the global registry."""
  experiment_class = experiment_registry.get(experiment_name)
  if experiment_class is not None:
    return experiment_class
  # Try to import the module that registers the experiment, assuming the
  # experiment name contains the full path.
  module_name = experiment_name.rsplit('.', 1)[0]
  # internal experiment module import code
  try:
    importlib.import_module(module_name)
  except ModuleNotFoundError as e:
    raise ValueError(f'Could not find experiment `{experiment_name}`: {e}.')
  experiment_class = experiment_registry.get(experiment_name)
  if experiment_class is not None:
    return experiment_class
  raise ValueError(f'Could not find experiment `{experiment_name}`.')


def wait_with_random_jitter(min_secs: int, max_secs: int) -> None:
  """Sleeps for a random short interval to avoid thundering herd RPC calls."""
  time.sleep(random.randint(min_secs, max_secs))


def run_experiment(
    work_unit: platform.WorkUnit,
    job_log_dir: str,
    early_stopping_fn: Optional[trainer_lib.EarlyStoppingFn] = None,
) -> None:
  """Run an experiment.

  Args:
    work_unit: Work unit for adding experiment artifact and reporting status.
    job_log_dir: The directory for storing logs and writing checkpoints.
    early_stopping_fn: The early stopping function for training and evaluation.
      If None, the training and evaluation will train to requested steps.
  """
  experiment_config = _get_experiment(FLAGS.exp)()
  if FLAGS.mode == 'train':
    work_unit.set_task_status(f'Train experiment {FLAGS.exp} at'
                              f' {job_log_dir}')
    if FLAGS.jax_fully_async_checkpoint:
      async_ckpt_manager_cls = gda_serialization.GlobalAsyncCheckpointManager
      if FLAGS.maybe_use_persistence_checkpointing:
        async_ckpt_manager_cls = (
            persistence_gda_serialization.GlobalAsyncCheckpointManager)
      async_ckpt_manager = async_ckpt_manager_cls(timeout_secs=600)
    else:
      async_ckpt_manager = None

    train.train_and_evaluate(
        experiment_config=experiment_config,
        job_log_dir=job_log_dir,
        maybe_use_persistence_checkpointing=FLAGS
        .maybe_use_persistence_checkpointing,
        eval_on_test=FLAGS.eval_on_test,
        checkpoint_todelete_subdir=FLAGS.checkpoint_todelete_subdir,
        early_stopping_fn=early_stopping_fn,
        async_ckpt_manager=async_ckpt_manager,
        run_decode=FLAGS.decode_during_train,
        enable_auto_sharding=FLAGS.enable_auto_sharding)
  elif FLAGS.mode == 'eval':
    work_unit.set_task_status(f'Eval experiment {FLAGS.exp} at'
                              f' {job_log_dir}')
    eval_lib.evaluate(
        experiment_config=experiment_config,
        job_log_dir=job_log_dir,
        maybe_use_persistence_checkpointing=FLAGS
        .maybe_use_persistence_checkpointing,
        early_stopping_fn=early_stopping_fn)
  elif FLAGS.mode == 'decode':
    work_unit.set_task_status(f'Decode experiment {FLAGS.exp} at'
                              f' {job_log_dir}')
    eval_lib.decode(
        experiment_config=experiment_config,
        job_log_dir=job_log_dir,
        maybe_use_persistence_checkpointing=FLAGS
        .maybe_use_persistence_checkpointing,
        restore_checkpoint_dir=None,
        restore_checkpoint_step=None,
        continuous_decode=True,
        run_eval=FLAGS.eval_during_decode,
    )
  elif FLAGS.mode == 'decode_once':
    work_unit.set_task_status(f'Decode-once experiment {FLAGS.exp} at'
                              f' {job_log_dir}')
    eval_lib.decode(
        experiment_config=experiment_config,
        job_log_dir=job_log_dir,
        maybe_use_persistence_checkpointing=FLAGS
        .maybe_use_persistence_checkpointing,
        restore_checkpoint_dir=FLAGS.restore_checkpoint_dir,
        restore_checkpoint_step=FLAGS.restore_checkpoint_step,
        continuous_decode=False,
        run_eval=FLAGS.eval_during_decode,
    )
  elif FLAGS.mode == 'infer':
    work_unit.set_task_status(f'infer experiment {FLAGS.exp} at {job_log_dir}')
    eval_lib.infer_and_write(
        experiment_config=experiment_config,
        job_log_dir=job_log_dir)

  # Wait for all processes to exit at the same time because if some tasks
  # finish early and exited, when a preemption event comes, only a
  # subset of tasks are restarted. Without all tasks being present, the job
  # will hang on startup.
  py_utils.sync_global_devices('All tasks finish.')


def tune_experiment(work_unit: platform.WorkUnit, job_log_dir: str) -> None:
  """Tune an experiment.

  An experiment can be tuned by running a tuning loop, with each iteration
  calling `run_experiment` for evaluating a trial sampled by the controller.

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
     trial evaluation logic to `run_experiment`, which is done by passing
     a per-trial early stopping function for reporting measurements, completing/
     early stopping the trial.

  Args:
    work_unit: Work unit for adding experiment artifact and reporting status.
    job_log_dir: The directory used for storing logs and writing checkpoints.
  """
  assert FLAGS.study is not None
  assert FLAGS.pythia_port is not None
  # Google-internal tuning infra init.

  experiment_config = _get_experiment(FLAGS.exp)()
  search_hparams = experiment_config.search()
  search_algorithm = search_hparams.search_algorithm.Instantiate()()
  reward_fn = search_hparams.search_reward.Instantiate()
  max_num_trials = FLAGS.num_trials or search_hparams.max_num_trials

  # Inspect the search space by evaluating the hyperparameters.
  # We include tuning parameters from both the `task` and `datasets` in the
  # search space. A caveat is that when multiple datasets have tunable
  # parameters, even one of them is not evaluated, its tunable parameters will
  # be included. We can improve this in the future if this turns out to be an
  # issue.
  def inspect_search_space() -> None:
    _ = experiment_config.task()
    _ = experiment_config.datasets()

  search_space = pg.hyper.trace(inspect_search_space, require_hyper_name=True)
  if search_space.dna_spec.is_constant:
    raise ValueError(f'Aborting tuning: there is no tunable parameters in'
                     f'experiment {FLAGS.exp!r}.')

  # Write debug information to files.
  def write_once(file_path, content):
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

  tf.io.gfile.makedirs(job_log_dir)

  logging.info('Search space: %s', search_space.dna_spec)
  search_space_debug_file = os.path.join(job_log_dir, 'search_space.txt')
  write_once(search_space_debug_file, str(search_space.dna_spec))
  work_unit.create_artifact(platform.ArtifactType.FILE, search_space_debug_file,
                            'search_space')

  logging.info('Search algorithm: %s', search_algorithm)
  algorithm_debug_file = os.path.join(job_log_dir, 'search_algorithm.txt')
  write_once(algorithm_debug_file, str(search_algorithm))
  work_unit.create_artifact(platform.ArtifactType.FILE, algorithm_debug_file,
                            'search_algorithm')

  def extract_metrics(
      metrics_by_dataset: Dict[str,
                               trainer_lib.MetricDict]) -> Dict[str, float]:
    """Extract a flattened metric dict from a metric dict per dataset."""
    metrics = {}

    def add_metric(dataset_name, key, value, weight):
      if dataset_name is not None:
        key = f'{dataset_name}/{key}'
      metrics[key] = np.sum(value * weight) / np.sum(weight)

    # When there is only one dataset, we allow the metric key to be directly
    # used in reward_fn without providing the dataset name.
    if len(metrics_by_dataset) == 1:
      for k, (v, w) in list(metrics_by_dataset.values())[0].items():
        add_metric(None, k, v, w)

    for dataset_name, dataset_metrics in metrics_by_dataset.items():
      for k, (v, w) in dataset_metrics.items():
        add_metric(dataset_name, k, v, w)
    return metrics

  # Helper functions for tuning.
  def get_early_stopping_fn(
      feedback: pg.tuning.Feedback) -> trainer_lib.EarlyStoppingFn:
    """Gets early stopping function based on a feedback object."""

    def should_stop_early(metrics_by_dataset: Optional[Dict[
        str, trainer_lib.MetricDict]], global_step: int,
                          is_last_checkpoint: bool) -> bool:
      """Early stopping function."""
      if FLAGS.metrics_from == FLAGS.mode:
        if metrics_by_dataset is None:
          raise ValueError('There is no metrics reported for tuning.')
        # Computing reward and report back to the tuning service.
        metrics = extract_metrics(metrics_by_dataset)
        reward = reward_fn(metrics, global_step)
        feedback.add_measurement(reward, metrics=metrics, step=global_step)
        if is_last_checkpoint:
          feedback.done()
      return feedback.should_stop_early()

    return should_stop_early

  for example, feedback in pg.sample(
      search_space,
      search_algorithm,
      num_examples=max_num_trials,
      group=FLAGS.tuner_group):
    logging.info('Start working on trial %d (group=%r)...', feedback.id,
                 FLAGS.tuner_group)
    # Context manager to deliver different program hyperparameters
    # in each trial.
    with example():
      # Mark trial as infeasible on NaN. PAX user can add more error types here.
      with feedback.skip_on_exceptions((FloatingPointError,)):
        try:
          run_experiment(work_unit, os.path.join(job_log_dir, str(feedback.id)),
                         get_early_stopping_fn(feedback))
        except automl.EarlyStoppingError as e:
          if e.skip:
            feedback.skip(e.skip_reason or 'Unknown.')
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
  logging.info('Completed with all trials for study %r', FLAGS.study)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  setup_jax.setup_jax(FLAGS.globally_use_hardware_rng, FLAGS.jax_backend_target,
                      FLAGS.jax_xla_backend, FLAGS.jax_enable_checks)

  # Add a note so that we can tell which Borg task is which JAX host.
  # (Borg task 0 is not guaranteed to be host 0)
  if jax.process_count() > 128:
    wait_with_random_jitter(min_secs=0, max_secs=60)
  work_unit = platform.work_unit()
  work_unit.set_task_status(f'process_index: {jax.process_index()}, '
                            f'process_count: {jax.process_count()}')
  work_unit.create_artifact(platform.ArtifactType.DIRECTORY, FLAGS.job_log_dir,
                            'job_log_dir')

  # Start jax.profiler for TensorBoard and profiling in open source.
  if FLAGS.jax_profiler_port is not None:
    server = jax.profiler.start_server(FLAGS.jax_profiler_port)  # pylint:disable=unused-variable

  if (FLAGS.restore_checkpoint_dir or
      FLAGS.restore_checkpoint_step) and FLAGS.mode != 'decode_once':
    raise ValueError('--restore_checkpoint_dir and --restore_checkpoint_step '
                     'only supported with --mode=decode_once.')

  if FLAGS.study is None:
    run_experiment(work_unit, job_log_dir=FLAGS.job_log_dir)
  else:
    tune_experiment(work_unit, job_log_dir=FLAGS.job_log_dir)


_TASK_HANDLE_RE = re.compile(r'(?:logs\.)?(\d+)\.(.*)\.([^.]+)\.\d+')

if __name__ == '__main__':
  # Only dump from Borg task 0.
  if 'BORG_TASK_HANDLE' in os.environ:
    handle = os.getenv('BORG_TASK_HANDLE')
    task_id, _, _ = _TASK_HANDLE_RE.match(handle).groups()
    if int(task_id) == 0:
      dump_dir = os.getenv('XLA_DUMP_TO')
      if dump_dir:
        os.environ['XLA_FLAGS'] = f'--xla_dump_to={dump_dir}'

  # Provide access to --jax_backend_target and --jax_xla_backend flags.
  jax.config.config_with_absl()

  # TODO(shafey): Make `job_log_dir` mandatory?
  app.run(main)
