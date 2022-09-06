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
python paxml/main.py \
    --exp=tasks.lm.params.lm_cloud.LmCloudTransformerAdamTest \
    --job_log_dir=/tmp/jax_log_dir/exp01
"""

import importlib
import os
import random
import re
import time
from typing import Dict, Optional, Sequence

from absl import app
from absl import flags
# Required import to setup work units when running through XManager.
from clu import platform
import fiddle as fdl
from fiddle import absl_flags
import jax
from jax.experimental.gda_serialization import serialization as gda_serialization
from paxml import base_experiment
from paxml import checkpoints
from paxml import eval_lib
from paxml import experiment_registry
from paxml import setup_jax
from paxml import train
from paxml import trainer_lib
from paxml import tuning_lib
from praxis import py_utils
import pyglove as pg
import tensorflow.compat.v2 as tf

AsyncPersistenceCheckpointer = checkpoints.AsyncCheckpointer  # mapped to internal
persistence_gda_serialization = gda_serialization  # mapped to internal

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'exp', None,
    'Experiment configuration identifier name. This name typically '
    'has a format like "<task>.<module>.<experiment>", which should have been '
    'already registered in the global experiment registry with '
    '@experiment_registry.register.')
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
flags.DEFINE_bool('use_orbax', False, 'Enables Orbax for checkpointing.')
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
    'metrics_from', 'eval', ['train', 'eval', 'decode'],
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
    raise ValueError(f'Could not find experiment `{experiment_name}`.') from e
  experiment_class = experiment_registry.get(experiment_name)
  if experiment_class is not None:
    return experiment_class
  raise ValueError(f'Could not find experiment `{experiment_name}`.')


def wait_with_random_jitter(min_secs: int, max_secs: int) -> None:
  """Sleeps for a random short interval to avoid thundering herd RPC calls."""
  time.sleep(random.randint(min_secs, max_secs))


def _default_early_stopping_fn(metrics: Dict[str, float],
                               running_mode: trainer_lib.RunningMode,
                               step_i: int, unused_arg: bool) -> bool:
  """Dumping metrics into JSON file for debugging and other consumptions."""
  if jax.process_index() == 0:
    metric_dir = os.path.join(FLAGS.job_log_dir, 'metrics')
    if not tf.io.gfile.exists(metric_dir):
      tf.io.gfile.makedirs(metric_dir)
    if not tf.io.gfile.isdir(metric_dir):
      raise ValueError(f'{metric_dir} should be a directory.')
    metric_file_name = os.path.join(metric_dir, f'step-{step_i:06d}.json')
    # Update and re-save the metrics.
    if (running_mode & trainer_lib.RunningMode.EVAL or
        running_mode & trainer_lib.RunningMode.DECODE):
      if tf.io.gfile.exists(metric_file_name):
        # NOTE(daiyip): converting pg.Dict to dict which allows updates
        # with dot ('.') separated keys. (dot can be a part of dataset name)
        existing_metrics = dict(pg.load(metric_file_name))
      else:
        existing_metrics = {}
      metrics.update(existing_metrics)
      pg.save(metrics, metric_file_name)
  return False


def run_experiment(
    experiment_config: base_experiment.BaseExperiment,
    work_unit: platform.WorkUnit,
    job_log_dir: str,
    early_stopping_fn: Optional[trainer_lib.EarlyStoppingFn] = None,
) -> None:
  """Run an experiment.

  Args:
    experiment_config: The experiment to run.
    work_unit: Work unit for adding experiment artifact and reporting status.
    job_log_dir: The directory for storing logs and writing checkpoints.
    early_stopping_fn: The early stopping function for training, evaluation
      and decoding. If None, the training will train to requested steps.
  """
  train.write_hparams_file(experiment_config, job_log_dir,
                           '' if FLAGS.mode == 'train' else f'{FLAGS.mode}_')
  if FLAGS.mode == 'train':
    work_unit.set_task_status(f'Train experiment {FLAGS.exp} at'
                              f' {job_log_dir}')
    async_checkpointer = None
    async_ckpt_manager = None
    if FLAGS.jax_fully_async_checkpoint:
      if FLAGS.use_orbax:
        if FLAGS.maybe_use_persistence_checkpointing:
          async_checkpointer = AsyncPersistenceCheckpointer(timeout_secs=600)
        else:
          async_checkpointer = checkpoints.AsyncCheckpointer(
              checkpoints.PaxCheckpointHandler(enable_flax=False))
      else:
        if FLAGS.maybe_use_persistence_checkpointing:
          async_ckpt_manager = persistence_gda_serialization.GlobalAsyncCheckpointManager(
              timeout_secs=600)
        else:
          async_ckpt_manager = gda_serialization.GlobalAsyncCheckpointManager(
              timeout_secs=600)

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
        enable_auto_sharding=FLAGS.enable_auto_sharding,
        use_orbax=FLAGS.use_orbax,
        async_checkpointer=async_checkpointer)
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
        early_stopping_fn=early_stopping_fn)
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
        early_stopping_fn=early_stopping_fn)
  elif FLAGS.mode == 'infer':
    work_unit.set_task_status(f'infer experiment {FLAGS.exp} at {job_log_dir}')
    eval_lib.infer_and_write(
        experiment_config=experiment_config, job_log_dir=job_log_dir)

  # Wait for all processes to exit at the same time because if some tasks
  # finish early and exited, when a preemption event comes, only a
  # subset of tasks are restarted. Without all tasks being present, the job
  # will hang on startup.
  py_utils.sync_global_devices('All tasks finish.')


def run(experiment_config: base_experiment.BaseExperiment):
  """Run an experiment.

  This function exists to provide a clear injection seam for a given run.
  Anything that needs to be configured via Fiddle should be injected here
  and passed down to the runner code. Right now, the only thing to be
  injected is the experiment_config.

  Args:
    experiment_config: The experiment to run.
  """

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
    # TODO(b/241666951): disable default_early_stopping_fn since this
    # breaks when training LaMDA models.
    run_experiment(
        experiment_config,
        work_unit,
        job_log_dir=FLAGS.job_log_dir,
        early_stopping_fn=None)
  else:
    tuning_lib.tune(
        trial_fn=run_experiment,
        experiment_config=experiment_config,
        work_unit=work_unit,
        job_log_dir=FLAGS.job_log_dir,
        study=FLAGS.study,
        pythia_port=FLAGS.pythia_port,
        is_metric_reporting_role=(FLAGS.metrics_from == FLAGS.mode),
        tuner_group=FLAGS.tuner_group,
        max_num_trials=FLAGS.num_trials)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  setup_jax.setup_jax(FLAGS.globally_use_hardware_rng, FLAGS.jax_backend_target,
                      FLAGS.jax_xla_backend, FLAGS.jax_enable_checks,
                      FLAGS.jax_fully_async_checkpoint)

  if FLAGS.exp is not None:
    experiment_config = _get_experiment(FLAGS.exp)()
  else:
    cfg = absl_flags.create_buildable_from_flags(
        module=None, allow_imports=True)
    experiment_config = fdl.build(cfg)

  experiment_config.validate()
  run(experiment_config=experiment_config)


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
  app.run(main, flags_parser=absl_flags.flags_parser)
