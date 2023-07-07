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

r"""Main file for running a PAX training and evaluation loop.

Example usage:
python paxml/main.py \
    --exp=tasks.lm.params.lm_cloud.LmCloudTransformerAdamTest \
    --job_log_dir=/tmp/jax_log_dir/exp01
"""
# Internal import for aiding module import speed

import contextlib
import importlib
import os
import pprint
import random
import re
import time
import typing
from typing import Optional, Sequence

from absl import app
from absl import flags
from absl import logging
# Required import to setup work units when running through XManager.
from clu import platform
from etils import epath
from fiddle import absl_flags
import jax
from paxml import base_experiment
from paxml import eval_lib
from paxml import experiment_registry
from paxml import setup_jax
from paxml import tasks_lib
from paxml import tf_data_service_lib
from paxml import train
from paxml import trainer_lib
from paxml import tuning_lib
from praxis import pax_fiddle
from praxis import py_utils

# internal debugging module import
# internal experiment module import


FLAGS = flags.FLAGS

flags.DEFINE_string(
    'exp', None,
    'Experiment configuration identifier name. This name typically '
    'has a format like "<task>.<module>.<experiment>", which should have been '
    'already registered in the global experiment registry with '
    '@experiment_registry.register.')
_JOB_LOGDIR = epath.DEFINE_path(
    'job_log_dir',
    None,
    'Directory where all experiment assets will be stored.',
    # Marked as required below.
)
flags.DEFINE_enum(
    'mode',
    'train',
    [
        'train',
        'eval',
        'decode',
        'decode_once',
        'infer',
        'tf_data_service_dispatcher',
        'tf_data_service_worker',
    ],
    'Flag to control which job is called.',
)
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
flags.DEFINE_bool(
    'exit_after_ondemand_checkpoint',
    False,
    (
        'If True, exits immediately after finishing saving on-demand checkpoint'
        ' due to preemption.'
    ),
)
flags.DEFINE_bool(
    'tensorstore_use_ocdbt',
    False,
    'If True, uses OCDBT format when saving with Tensorstore.',
)
flags.DEFINE_string(
    'jax_traceback_filtering_option', 'auto',
    'Controls how JAX filters internal frames out of tracebacks: '
    'off, auto, tracebackhide, remove_frames. '
    'See https://github.com/google/jax/blob/main/jax/_src/config.py')
flags.DEFINE_bool(
    'decode_output_pickle', True,
    'Output the .pickle file alongside the .jsonl file when decoding, this '
    'can take a lot of memory with large decodings so can be disabled here.')
flags.DEFINE_string(
    'checkpoint_todelete_subdir', None,
    'If set, checkpoints to be deleted will be only renamed into a '
    'subdirectory with the provided string. Otherwise, they will be directly '
    'deleted from the file system. Useful if checkpoint deletion is time '
    'consuming. By default, delete the checkpoint assets.')
epath.DEFINE_path(
    'restore_checkpoint_dir', None,
    'If set, the directory from which to restore checkpoint. Only supported '
    'for --mode=decode_once and --mode=decode.')
flags.DEFINE_multi_integer(
    'restore_checkpoint_step', None,
    ('If set, the checkpoint step to restore. Only supported when '
     '--mode=decode_once.'))
flags.DEFINE_bool(
    'globally_use_hardware_rng', True,
    'Whether to globally use fast hardware RNG. Deterministic only at the '
    'same compiler version and with the same sharding')
flags.DEFINE_integer(
    'jax_profiler_port', None,
    ('If set, the jax.profiler port to use. Only needed for profiling in open'
     ' source.')
)
flags.DEFINE_bool('enable_auto_sharding', False,
                  'Enable the XLA Auto SPMD partitioner.')
flags.DEFINE_bool(
    'enable_checkpoint_saving', True,
    'Enable checkpoint saving. Useful to disable for test- or debug-like runs.')
flags.DEFINE_bool(
    'enforce_restore_shape_check',
    False,
    (
        'If True, raise an error when the requested restore shape of an array'
        ' does not match the shape in the checkpoint.'
    ),
)
# Flags for automatic tuning.
flags.DEFINE_string(
    'study', None,
    'Study name for current tuning. If None, the program will be running in '
    'standard training/evaluation mode. Otherwise, it will run in tuning mode.')
flags.DEFINE_enum(
    'controller_mode', 'auto', ['primary', 'secondary', 'auto'],
    'Mode for tuning controller. If primary, current processs will only work '
    'as the controller, without running tuning workload. If secondary, current '
    'process will only run tuning workload. Otherwise, current process may '
    'elect controller role in a background thread, and run the tuning workload '
    'in the main thread.')
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
flags.DEFINE_string(
    'tfds_data_dir', None,
    'If set, directory used to store datasets prepared by '
    'TensorFlow Datasets that are not available in the public TFDS GCS '
    'bucket.')
  # Google-internal TFDS override-like flag definition.

## multiprocessing GPU flags
flags.DEFINE_bool(
    'multiprocess_gpu', False,
    'Whether to initialize JAX distributed for multi-host GPU')
flags.DEFINE_string(
    'server_addr', None, help='server ip addr')
flags.DEFINE_integer(
    'num_hosts', None, help='num of hosts' )
flags.DEFINE_integer(
    'host_idx', None, help='index of current host' )

# Flags --jax_backend_target, --jax_xla_backend, --jax_enable_checks are
# available through JAX.

# Debugging flag


@py_utils.benchmark('[PAX STATUS]: ')
def get_experiment(experiment_name: str) -> base_experiment.BaseExperimentT:
  """Retrieves an experiment config from the global registry."""
  experiment_class = experiment_registry.get(experiment_name)
  if experiment_class is not None:
    return experiment_class
  # Try to import the module that registers the experiment, assuming the
  # experiment name contains the full path.
  module_name = experiment_name.rsplit('.', 1)[0]
  # Google-internal experiment module import code
  try:
    importlib.import_module(module_name)
  except ModuleNotFoundError as e:
    raise ValueError(
        f'Could not find experiment `{experiment_name}` because could not '
        f'import module `{module_name}`.'
    ) from e
  # Google-internal experiment module import cleanup
  experiment_class = experiment_registry.get(experiment_name)
  if experiment_class is not None:
    return experiment_class
  raise ValueError(
      f'Could not find experiment `{experiment_name}`.\nRegistered experiments '
      f'are: {pprint.pformat(experiment_registry.get_all())}'
  )


@py_utils.benchmark('[PAX STATUS]: ')
def wait_with_random_jitter(min_secs: int, max_secs: int) -> None:
  """Sleeps for a random short interval to avoid thundering herd RPC calls."""
  time.sleep(random.randint(min_secs, max_secs))


@py_utils.benchmark('[PAX STATUS]: ')
def run_experiment(
    experiment_config: base_experiment.BaseExperiment,
    work_unit: platform.WorkUnit,
    job_log_dir: epath.Path,
    early_stopping_fn: Optional[trainer_lib.EarlyStoppingFn] = None,
    enable_checkpoint_saving: bool = True,
) -> None:
  """Run an experiment.

  Args:
    experiment_config: The experiment to run.
    work_unit: Work unit for adding experiment artifact and reporting status.
    job_log_dir: The directory for storing logs and writing checkpoints.
    early_stopping_fn: The early stopping function for training, evaluation
      and decoding. If None, the training will train to requested steps.
    enable_checkpoint_saving: Whether to perform checkpoint saving or not.
  """
  train.write_experiment_class_vars_file(
      experiment_config.__class__, job_log_dir,
      '' if FLAGS.mode == 'train' else f'{FLAGS.mode}_')
  train.write_hparams_file(experiment_config, job_log_dir,
                           '' if FLAGS.mode == 'train' else f'{FLAGS.mode}_')

  task_p = experiment_config.task()
  task_p = typing.cast(pax_fiddle.Config[tasks_lib.SingleTask], task_p)

  if FLAGS.mode == 'train':
    work_unit.set_task_status(f'Train experiment {FLAGS.exp} at'
                              f' {job_log_dir}')
    train.train_and_evaluate(
        experiment_config=experiment_config,
        job_log_dir=job_log_dir,
        maybe_use_persistence_checkpointing=FLAGS.maybe_use_persistence_checkpointing,
        eval_on_test=FLAGS.eval_on_test,
        checkpoint_todelete_subdir=FLAGS.checkpoint_todelete_subdir,
        early_stopping_fn=early_stopping_fn,
        run_decode=FLAGS.decode_during_train,
        enable_auto_sharding=FLAGS.enable_auto_sharding,
        enable_async_checkpointing=FLAGS.jax_fully_async_checkpoint,
        enable_checkpoint_saving=enable_checkpoint_saving,
        enforce_restore_shape_check=FLAGS.enforce_restore_shape_check,
        tensorstore_use_ocdbt=FLAGS.tensorstore_use_ocdbt,
        exit_after_ondemand_checkpoint=FLAGS.exit_after_ondemand_checkpoint,
    )

  elif FLAGS.mode == 'eval':
    work_unit.set_task_status(f'Eval experiment {FLAGS.exp} at'
                              f' {job_log_dir}')
    eval_lib.evaluate(
        experiment_config=experiment_config,
        job_log_dir=job_log_dir,
        maybe_use_persistence_checkpointing=FLAGS.maybe_use_persistence_checkpointing,
        early_stopping_fn=early_stopping_fn,
        enable_auto_sharding=FLAGS.enable_auto_sharding,
        enforce_restore_shape_check=FLAGS.enforce_restore_shape_check,
        tensorstore_use_ocdbt=FLAGS.tensorstore_use_ocdbt,
    )
  elif FLAGS.mode == 'decode':
    work_unit.set_task_status(f'Decode experiment {FLAGS.exp} at'
                              f' {job_log_dir}')
    eval_lib.decode(
        experiment_config=experiment_config,
        job_log_dir=job_log_dir,
        maybe_use_persistence_checkpointing=FLAGS.maybe_use_persistence_checkpointing,
        restore_checkpoint_dir=FLAGS.restore_checkpoint_dir,
        restore_checkpoint_step=None,
        continuous_decode=True,
        run_eval=FLAGS.eval_during_decode,
        early_stopping_fn=early_stopping_fn,
        enable_auto_sharding=FLAGS.enable_auto_sharding,
        enforce_restore_shape_check=FLAGS.enforce_restore_shape_check,
        tensorstore_use_ocdbt=FLAGS.tensorstore_use_ocdbt,
    )
  elif FLAGS.mode == 'decode_once':
    if (restore_checkpoint_steps := FLAGS.restore_checkpoint_step) is None:
      restore_checkpoint_steps = [None]

    for restore_step in restore_checkpoint_steps:
      work_unit.set_task_status(f'Decode-once experiment {FLAGS.exp} at'
                                f' {job_log_dir} for step={restore_step}')
      restore_step = int(restore_step) if restore_step is not None else None
      logging.info('Decode-once on step: %s', restore_step)
      eval_lib.decode(
          experiment_config=experiment_config,
          job_log_dir=job_log_dir,
          maybe_use_persistence_checkpointing=FLAGS.maybe_use_persistence_checkpointing,
          restore_checkpoint_dir=FLAGS.restore_checkpoint_dir,
          restore_checkpoint_step=restore_step,
          continuous_decode=False,
          run_eval=FLAGS.eval_during_decode,
          early_stopping_fn=early_stopping_fn,
          enable_auto_sharding=FLAGS.enable_auto_sharding,
          output_pickle=FLAGS.decode_output_pickle,
          enforce_restore_shape_check=FLAGS.enforce_restore_shape_check,
          tensorstore_use_ocdbt=FLAGS.tensorstore_use_ocdbt,
      )
  elif FLAGS.mode == 'infer':
    work_unit.set_task_status(f'infer experiment {FLAGS.exp} at {job_log_dir}')
    eval_lib.infer_and_write(
        experiment_config=experiment_config,
        job_log_dir=job_log_dir,
        enforce_restore_shape_check=FLAGS.enforce_restore_shape_check,
        tensorstore_use_ocdbt=FLAGS.tensorstore_use_ocdbt,
    )

  # Wait for all processes to exit at the same time because if some tasks
  # finish early and exited, when a preemption event comes, only a
  # subset of tasks are restarted. Without all tasks being present, the job
  # will hang on startup.
  py_utils.sync_global_devices('All tasks finish.')


@py_utils.benchmark('[PAX STATUS]: ')
def _setup_xm_work_unit():
  """Setup the global work unit for XM."""
  work_unit = platform.work_unit()
  work_unit.set_task_status(
      f'process_index: {jax.process_index()}, '
      f'process_count: {jax.process_count()}'
  )
  if jax.process_index() == 0:
    work_unit.create_artifact(
        platform.ArtifactType.DIRECTORY, str(_JOB_LOGDIR.value), 'job_log_dir'
    )
  return work_unit


@py_utils.benchmark('[PAX STATUS]: ')
def run(
    experiment_config: base_experiment.BaseExperiment,
    enable_checkpoint_saving: bool = True,
):
  """Run an experiment.

  This function exists to provide a clear injection seam for a given run.
  Anything that needs to be configured via Fiddle should be injected here
  and passed down to the runner code. Right now, the only thing to be
  injected is the experiment_config.

  Args:
    experiment_config: The experiment to run.
    enable_checkpoint_saving: Whether to perform checkpoint saving or not.
  """

  # Add a note so that we can tell which Borg task is which JAX host.
  # (Borg task 0 is not guaranteed to be host 0)
  if jax.process_count() > 128:
    wait_with_random_jitter(min_secs=0, max_secs=60)
  work_unit = _setup_xm_work_unit()

  # Start jax.profiler for TensorBoard and profiling in open source.
  if FLAGS.jax_profiler_port is not None:
    server = jax.profiler.start_server(FLAGS.jax_profiler_port)  # pylint:disable=unused-variable

  if ((FLAGS.restore_checkpoint_dir or FLAGS.restore_checkpoint_step) and
      FLAGS.mode not in {'decode_once', 'decode'}):
    raise ValueError(
        '--restore_checkpoint_dir and --restore_checkpoint_step only supported '
        'with --mode=decode_once or --mode=decode.')

  search_space = tuning_lib.get_search_space(experiment_config)
  if search_space.dna_spec.is_constant:
    # TODO(b/241666951): disable default_early_stopping_fn since this
    # breaks when training internal models.
    run_experiment(
        experiment_config,
        work_unit,
        job_log_dir=_JOB_LOGDIR.value,
        early_stopping_fn=None,
        enable_checkpoint_saving=enable_checkpoint_saving)
  else:
    if not enable_checkpoint_saving:
      logging.warning(
          'Ignoring flag `--enable_checkpoint_saving` for tuning experiment.')
    tuning_lib.tune(
        trial_fn=run_experiment,
        experiment_config=experiment_config,
        work_unit=work_unit,
        job_log_dir=_JOB_LOGDIR.value,
        study=FLAGS.study,
        pythia_port=FLAGS.pythia_port,
        is_metric_reporting_role=(FLAGS.metrics_from == FLAGS.mode),
        tuner_group=FLAGS.tuner_group,
        max_num_trials=FLAGS.num_trials,
        controller_mode=FLAGS.controller_mode,
        running_mode=FLAGS.mode)


def main(argv: Sequence[str]) -> None:
  _main(argv)


@py_utils.benchmark(prefix='[PAX STATUS]: E2E time: ')
def _main(argv: Sequence[str]) -> None:
  logging.info('[PAX STATUS]: Program start.')
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  if tf_data_service_lib.run_tf_data_service(FLAGS.mode):
    return

  if FLAGS.tfds_data_dir is not None:
    # seqio import is slow so avoid module-level import
    import seqio
    seqio.set_tfds_data_dir_override(FLAGS.tfds_data_dir)
  # Google-internal setting of TFDS override-like param.

  should_initialize_jax_distributed = (
      FLAGS.jax_fully_async_checkpoint or FLAGS.multiprocess_gpu)
  setup_jax.setup_jax(FLAGS.globally_use_hardware_rng, FLAGS.jax_backend_target,
                      FLAGS.jax_xla_backend, FLAGS.jax_enable_checks,
                      FLAGS.jax_traceback_filtering_option,
                      should_initialize_jax_distributed,
                      setup_jax.JaxDistributedOptions(FLAGS.server_addr,
                                                      FLAGS.num_hosts,
                                                      FLAGS.host_idx)
                     )

  if FLAGS.exp is not None:
    experiment_config = get_experiment(FLAGS.exp)()
  elif absl_flags.fdl_flags_supplied():
    cfg = absl_flags.create_buildable_from_flags(
        module=None, allow_imports=True)
    experiment_config = pax_fiddle.build(cfg)
  else:
    raise app.UsageError(
        'No experiment provided. '
        'At least one of --exp, --fdl_config, or --fdl_config_file is required.'
    )

  experiment_config.validate()
  run(experiment_config=experiment_config,
      enable_checkpoint_saving=FLAGS.enable_checkpoint_saving)


_TASK_HANDLE_RE = re.compile(r'(?:logs\.)?(\d+)\.(.*)\.([^.]+)\.\d+')

if __name__ == '__main__':
  # Only dump from Borg task 0.
  if handle := os.getenv('BORG_TASK_HANDLE'):
    if (task_id := _TASK_HANDLE_RE.match(handle).group(1)) == '0':  # pytype: disable=attribute-error  # re-none
      if dump_dir := os.getenv('XLA_DUMP_TO'):
        if existing := os.getenv('XLA_FLAGS'):
          os.environ['XLA_FLAGS'] = f'{existing} --xla_dump_to={dump_dir}'
        else:
          os.environ['XLA_FLAGS'] = f'--xla_dump_to={dump_dir}'

  # Log XLA_FLAGS for easy debugging.
  logging.info("os.environ['XLA_FLAGS']=%s", os.getenv('XLA_FLAGS'))

  # Provide access to --jax_backend_target and --jax_xla_backend flags.
  jax.config.config_with_absl()

  flags.mark_flag_as_required('job_log_dir')
  app.run(main, flags_parser=absl_flags.flags_parser)
