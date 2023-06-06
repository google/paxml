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

"""Tuning loop for PAX."""

import inspect
import math
import re
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Text, Tuple, Type, Union
from absl import logging
from clu import platform
from etils import epath
import jax
from paxml import automl
from paxml import base_experiment
from paxml import metric_utils
from paxml import trainer_lib
from praxis import base_hyperparams
from praxis import pax_fiddle
from praxis import py_utils
from praxis import pytypes
import pyglove as pg  # mapped to internal
import tensorflow.compat.v2 as tf


instantiate = base_hyperparams.instantiate
TrialFn = Callable[[
    base_experiment.BaseExperiment,  # Experiment config.
    platform.WorkUnit,  # Platform work unit.
    epath.Path,  # Job log dir
    trainer_lib.EarlyStoppingFn  # Early stopping fn.
], None]


# Step interval for sub-experiments to report their metrics, which is necessary
# to map progresses of individual sub-experiments to the progress of the trial.
# This means that when there is 3 sub-experiments in a tuning experiment:
# the 1st sub-experiment starts its step 0 at step 0;
# the 2nd sub-experiment starts its step 0 at step 1000_000_000;
# the 3rd sub-experiment starts its step 0 at step 2000_000_000.
SUB_EXPERIMENT_STEP_INTERVAL = 1000_000_000


def _enable_dataset_tuning(
    experiment_config: base_experiment.BaseExperiment,
) -> bool:
  try:
    search_hparams = experiment_config.search()
    return search_hparams.enable_dataset_tuning
  except NotImplementedError:
    return False


@py_utils.benchmark('[PAX STATUS]: ')
def get_search_space(
    experiment_config: base_experiment.BaseExperiment,
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

    # For SeqIO data mixtures, we use a lambda function to create a search space
    # for mixture weights, however, the function will not be called until
    # the input is instantiated. Therefore we instantiate them when inspecting
    # the search space. Currently we only inspect datasets for training.
    if _enable_dataset_tuning(experiment_config):
      for d in experiment_config.datasets():
        if d.is_training:
          _ = instantiate(d)

      _ = experiment_config.decoder_datasets()

  search_space = pg.hyper.trace(inspect_search_space, require_hyper_name=True)
  if (automl.COMBINED_DECISION_ATTR in search_space.hyper_dict
      and len(search_space.hyper_dict) != 1):
    parameter_sweep_attr_names = getattr(
        experiment_config, automl.COMBINED_DECISION_POINT_NAMES)
    extra_parameters = list(search_space.hyper_dict)
    extra_parameters.remove(automl.COMBINED_DECISION_ATTR)
    raise ValueError(
        f'Found extra tuning parameters ({extra_parameters}) beyond the '
        f'parameters ({parameter_sweep_attr_names}) specified in '
        f'`automl.parameter_sweep. Experiment class: '
        f'{experiment_config.__class__}.')
  return search_space


def tune(trial_fn: TrialFn,
         experiment_config: base_experiment.BaseExperiment,
         work_unit: platform.WorkUnit,
         job_log_dir: epath.Path,
         study: Optional[str] = None,
         pythia_port: Optional[int] = None,
         is_metric_reporting_role: bool = True,
         tuner_group: Optional[str] = None,
         max_num_trials: Optional[int] = None,
         controller_mode: str = 'auto',
         running_mode: str = 'train') -> None:
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
    controller_mode: One of 'primary', 'secondary', 'auto'.
      If primary, current processs will only work as the controller, without
      running tuning workload. If secondary, current process will only run
      tuning workload. Otherwise, current process may elect controller role
      in a background thread, and run the tuning workload in the main thread.
    running_mode: One of 'train', 'eval', 'decode', 'decode_once' and 'infer',
      Indicating the running mode that the worker is in.
  """
  search_hparams = experiment_config.search()
  reward_params = search_hparams.search_reward
  reward_fn = instantiate(reward_params) if reward_params else None

  # Make sure tuning is launched with the right running mode.
  _verify_running_mode(reward_fn, running_mode, is_metric_reporting_role)

  search_algorithm = instantiate(search_hparams.search_algorithm)()
  max_num_trials = max_num_trials or search_hparams.max_num_trials
  errors_to_skip = search_hparams.errors_to_skip or []
  cross_step_metric_aggregator = instantiate(
      search_hparams.cross_step_metric_aggregator
      or pax_fiddle.Config(automl.LastReportedMetricValues)
  )
  early_stopping_policy = None
  if search_hparams.early_stopping is not None:
    early_stopping_policy = instantiate(search_hparams.early_stopping)()

  search_space = get_search_space(experiment_config)
  if search_space.dna_spec.is_constant:
    raise ValueError(f'Aborting tuning: there is no tunable parameters in'
                     f'experiment {experiment_config.__class__.__name__!r}.')

  # NOTE(daiyip): The size of a search space will be infinite (-1) when
  # `pg.floatv` or `pg.hyper.CustomHyper` are used. Therefore, we only set
  # the max number of trials when the search space is finite.
  if search_space.dna_spec.space_size != -1:
    max_num_trials = min(max_num_trials, search_space.dna_spec.space_size)

  job_log_dir.mkdir(parents=True, exist_ok=True)
  logging.info('Search space: %s', search_space.dna_spec)
  search_space_debug_file = job_log_dir / 'search_space.txt'
  _write_file_once(search_space_debug_file, str(search_space.dna_spec))
  work_unit.create_artifact(
      platform.ArtifactType.FILE, str(search_space_debug_file), 'search_space')

  logging.info('Search algorithm: %s', search_algorithm)
  algorithm_debug_file = job_log_dir / 'search_algorithm.txt'
  _write_file_once(algorithm_debug_file, str(search_algorithm))
  work_unit.create_artifact(
      platform.ArtifactType.FILE, str(algorithm_debug_file), 'search_algorithm')

  logging.info('Early stopping policy: %s', early_stopping_policy)
  if early_stopping_policy is not None:
    early_stopping_policy_debug_file = (
        job_log_dir / 'early_stopping_policy_debug_file.txt')
    _write_file_once(early_stopping_policy_debug_file,
                     str(early_stopping_policy))
    work_unit.create_artifact(platform.ArtifactType.FILE,
                              str(early_stopping_policy_debug_file),
                              'early_stopping_policy')

  sample_kwargs = {}
  # Google-internal tuning infra init.

  if controller_mode == 'primary':
    _run_dedicated_controller(
        study, search_space.dna_spec,
        search_algorithm, early_stopping_policy, max_num_trials,
        search_hparams.prior_study_ids, search_hparams.add_prior_trials)
    return

  sub_experiments = experiment_config.sub_experiments()
  combined_decision_point_names = getattr(
      experiment_config, automl.COMBINED_DECISION_POINT_NAMES, None)

  trial_dirname_generator = TrialDirectoryNameGenerator(
      job_log_dir, search_space, combined_decision_point_names)
  published_study_link = False
  for example, feedback in pg.sample(
      search_space, search_algorithm, max_num_trials,
      early_stopping_policy=early_stopping_policy,
      group=tuner_group,
      name=study if study and study.startswith('local') else None,
      is_chief=None if controller_mode == 'auto' else False,
      **sample_kwargs):

  # Google-internal tuning infra logging.

    logging.info(
        'Start working on trial %d (group=%r)...', feedback.id, tuner_group)
    # Context manager to deliver different program hyperparameters
    # in each trial.
    with example():
      trial_dirname = trial_dirname_generator.dirname(feedback.id)
      if (search_hparams.add_experiment_config_to_metadata
          and is_metric_reporting_role
          and jax.process_index() == 0):
        _record_experiment_config(sub_experiments, feedback)

      for i, (sub_experiment_id, sub_experiment_cls) in enumerate(
          sub_experiments.items()):
        early_stopping_fn = EarlyStoppingFn(
            feedback=feedback,
            sub_experiment_id=sub_experiment_id,
            reward_fn=reward_fn,
            cross_step_metric_aggregator=cross_step_metric_aggregator,
            is_metric_reporting_role=is_metric_reporting_role,
            is_last_experiment=(i == len(sub_experiments) - 1),
            tuning_step_start=i * automl.SUB_EXPERIMENT_STEP_OFFSET,
            treats_early_stopped_trials_as_done=(
                search_hparams.treats_early_stopped_trials_as_done),
            train_to_end=search_hparams.train_to_end)

        # Mark trial as infeasible on NaN. PAX user can add more error
        # through `SearchHParams.errors_to_skip`.
        with feedback.skip_on_exceptions([FloatingPointError] + errors_to_skip):
          trial_fn(sub_experiment_cls(),  # pytype: disable=wrong-arg-types  # re-none
                   work_unit,
                   trial_dirname,
                   early_stopping_fn)  # pytype: disable=not-instantiable

        # We shortcircuit remaining sub-experiments if current trial is either
        # done or skipped.
        if feedback.get_trial().status == 'COMPLETED':
          break
      logging.info('Trial %d is now completed.', feedback.id)
  logging.info('Completed with all trials for study %r', study)


def _record_experiment_config(
    sub_experiments: Dict[str, Type[base_experiment.BaseExperiment]],
    feedback: pg.tuning.Feedback) -> None:
  """Record experiment config as trial metadata."""
  exp_configs = {}
  for subexp_id, subexp_cls in sub_experiments.items():
    subexp = subexp_cls()  # pytype: disable=not-instantiable
    exp_configs[subexp_id] = {
        # TODO(daiyip): We shall have more machine parse-able format such as
        # JSON or Fiddle config. But we start with raw texts to collect data
        # as early as possible.
        'datasets': [
            base_hyperparams.nested_struct_to_text(ds)
            for ds in subexp.datasets()
        ],
        'decoder_datasets': [
            base_hyperparams.nested_struct_to_text(ds)
            for ds in subexp.decoder_datasets()
        ],
        'task': base_hyperparams.nested_struct_to_text(subexp.task()),
    }

  feedback.set_metadata(
      'experiment_config', {
          # We might change the format of serialized configurations in future.
          # It will be easy for us to filter out metadata of old format based
          # on this format version.
          'format_version': 1.0,
          'source': 'pax',
          'config': exp_configs
      }, per_trial=True)


def _run_dedicated_controller(
    study_name: str,
    search_space: pg.DNASpec,
    search_algorithm: pg.DNAGenerator,
    early_stopping_policy: pg.tuning.EarlyStoppingPolicy,
    max_num_trials: Optional[int] = None,
    prior_study_ids: Optional[List[int]] = None,
    add_prior_trials: bool = False
    ) -> None:
  """Runs dedicated controller and waits for its completion."""
  raise NotImplementedError('Dedicated controller is not supported in OSS paxml.')


def _verify_running_mode(
    reward_fn: Optional[automl.BaseReward],
    running_mode: str,
    is_metric_reporting_role: bool) -> None:
  """Makes sure tuning is running in the right mode and config."""
  if reward_fn is None:
    return

  if is_metric_reporting_role:
    if reward_fn.needs_train and running_mode != 'train':
      raise ValueError(
          f'Tuning uses training metrics but the reporting role is '
          f'{running_mode!r}')
    if reward_fn.needs_decode and running_mode not in ['train', 'decode']:
      raise ValueError(
          f'Tuning uses decode metrics but the reporting role is '
          f'{running_mode!r}')

  if reward_fn.needs_eval or reward_fn.needs_decode:
    metric_type = ' and '.join(
        x[0] for x in zip(
            ['eval', 'decode'], [reward_fn.needs_eval, reward_fn.needs_decode])
        if x[1])
    logging.info(
        'Tuning will be performed based on the %s metrics', metric_type)
  else:
    logging.info(
        'Tuning will be performed based on the training metrics from the last '
        'training step.')


class EarlyStoppingFn:
  """Early stopping function for a sub-experiment."""

  def __init__(
      self,
      feedback: pg.tuning.Feedback,
      sub_experiment_id: str,
      reward_fn: Optional[automl.BaseReward],
      cross_step_metric_aggregator: automl.CrossStepMetricAggregator,
      is_metric_reporting_role: bool,
      is_last_experiment: bool,
      tuning_step_start: int,
      treats_early_stopped_trials_as_done: bool,
      train_to_end: bool):
    self._feedback = feedback
    self._sub_experiment_id = sub_experiment_id
    self._reward_fn = reward_fn
    self._cross_step_metric_aggregator = cross_step_metric_aggregator
    self._is_metric_reporting_role = is_metric_reporting_role
    self._is_last_experiment = is_last_experiment
    self._tuning_step_start = tuning_step_start
    self._treats_early_stopped_trials_as_done = (
        treats_early_stopped_trials_as_done)
    self._train_to_end = train_to_end
    if reward_fn is None:
      self._needs_train = False
      self._needs_eval = False
      self._needs_decode = False
    else:
      self._needs_train = reward_fn.needs_train
      self._needs_eval = reward_fn.needs_eval
      self._needs_decode = reward_fn.needs_decode

  @property
  def train_to_end(self) -> bool:
    return self._train_to_end

  def __call__(self,
               metrics: Dict[str, float],
               running_mode: trainer_lib.RunningMode,
               global_step: int,
               is_last_ckpt: bool) -> bool:
    """Returns True if trial should be stopped early."""
    tuning_step = self._tuning_step_start + global_step
    if self._is_metric_reporting_role:
      # For metric reporting role, when there is no eval and decode, early
      # stopping should always return False.
      if running_mode.has_eval or running_mode.has_decode:

        # NOTE(daiyip): the process of updating metrics may raises errors
        # (e.g. FloatPointError) which will be caught at higher level and
        # treats current trial as infeasible. Nevertheless, we want to always
        # trigger the `sync_global_devices` on all replicas uniformly here.
        try:
          # We only handle metric updates on main host.
          if jax.process_index() == 0:
            self._update_metrics(
                metrics, running_mode, tuning_step, is_last_ckpt)
        finally:
          # NOTE(daiyip): we synchronize all hosts after each eval/decode step
          # so all of them can move to the next train step or stop uniformly.
          # This is necessary since `sync_global_devices` will be called during
          # checkpointing, which requires all hosts to arrive at that point with
          # the same sequence of `sync_global_devices` calls.
          action_str = ' and '.join(
              [s for s, mode in zip(
                  ['evaluation', 'decoding'],
                  [running_mode.has_eval, running_mode.has_decode])])
          py_utils.sync_global_devices(
              f'Sync on trial {self._feedback.id} after {action_str} '
              f'at tuning step {tuning_step}.')
      elif is_last_ckpt:
        if self._needs_eval or self._needs_decode:
          if jax.process_index() == 0:
            trial = self._feedback.get_trial()
            if not trial.measurements:
              self._feedback.skip(
                  f'No eval/decode has taken place before training ended. '
                  f'(trial={self._feedback.id}, step={global_step})')
            elif self._is_last_experiment:
              self._complete_trial(
                  aggregate_metrics=True, global_step=tuning_step + 1)
        else:
          try:
            if jax.process_index() == 0:
              self._update_metrics(
                  metrics, running_mode, tuning_step, is_last_ckpt)
          finally:
            py_utils.sync_global_devices(
                f'Sync on trial {self._feedback.id} with training metrics at '
                f'the last step {tuning_step}.')
      else:
        return False

    # Poll completion or early stopping decision, and sync on trial completion.
    should_stop = False
    if self._feedback.get_trial().status == 'COMPLETED':
      should_stop = True
    elif self._feedback.should_stop_early():
      # `feedback.skip` is preferably called just once, so we always call
      # it on the main host.
      if jax.process_index() == 0:
        if (self._treats_early_stopped_trials_as_done
            and self._feedback.get_trial().measurements):
          self._feedback.done()
        else:
          self._feedback.skip()
      should_stop = True
    if should_stop:
      # NOTE(daiyip): at the end of each trial, we sync all hosts to make sure
      # they advance to the next trial at the same time.
      py_utils.sync_global_devices(
          f'Sync on trial {self._feedback.id} upon completion.')
    return should_stop

  def _compute_reward(
      self, metrics: Dict[str, float], tuning_step: int) -> float:
    if self._reward_fn is None:
      return 0.
    return self._reward_fn(metrics, tuning_step)

  def _update_metrics(
      self,
      metrics: Dict[str, float],
      running_mode: trainer_lib.RunningMode,
      tuning_step: int,
      is_last_ckpt: bool):
    """Handle metric update."""
    assert jax.process_index() == 0

    # Append sub_experiment_id as the suffix.
    if self._sub_experiment_id:
      metrics = {f'{k}:{self._sub_experiment_id}': v
                 for k, v in metrics.items()}

    try:
      # Computing reward and used metrics for reporting back to the
      # tuning service.
      reward, used_metrics = self._reward_and_used_metrics(metrics, tuning_step)
      if math.isnan(reward):
        raise FloatingPointError('Reward is NaN.')
      self._feedback.add_measurement(
          reward, metrics=used_metrics, step=tuning_step)

      logging.info(
          'Measurement is reported to trial %d (sub-experiment=%s) at step '
          '%d with reward value %f (mode=%s, is_last_checkpoint=%s): %s.',
          self._feedback.id, self._sub_experiment_id,
          tuning_step, reward, running_mode,
          is_last_ckpt, used_metrics)

      if is_last_ckpt:
        # `feedback.done` should be called just once per trial.
        if self._is_last_experiment:
          self._complete_trial(
              aggregate_metrics=True, global_step=tuning_step + 1)
    except automl.EarlyStoppingError as e:
      # Calling the reward_fn triggers `EarlyStoppingError`, which indicates
      # user signaled early stopping.
      if e.skip:
        if (self._treats_early_stopped_trials_as_done
            and self._feedback.get_trial().measurements):
          self._feedback.done()
          logging.info(
              'Trial %d is early stopped at step %s and will be treated as '
              'done. Reason: %s.',
              self._feedback.id, e.step, e.skip_reason)
        else:
          self._feedback.skip(e.skip_reason or 'Unknown.')
          logging.info(
              'Trial %d is early stopped at step %s and will be skipped '
              'by controller. Reason: %s.',
              self._feedback.id, e.step, e.skip_reason)
      else:
        reward = e.reward
        if reward is None:
          reward = self._compute_reward(e.metrics, e.step)
        self._feedback.add_measurement(
            reward=reward,
            step=e.step,
            metrics=e.metrics,
            checkpoint_path=e.checkpoint)
        self._complete_trial()
        logging.info(
            'Trial %d is early stopped at step %s with reward %f which '
            'will be fed back to the controller. Metrics: %s.',
            self._feedback.id, e.step, reward, e.metrics)

  def _reward_and_used_metrics(
      self,
      all_metrics: Dict[str, float],
      tuning_step: int) -> Tuple[float, Dict[str, float]]:
    """Returns computed reward and used metrics."""
    reward = self._compute_reward(all_metrics, tuning_step)
    if self._reward_fn is None:
      used_metrics = all_metrics
    else:
      used_metrics = {}
      for metric in self._reward_fn.used_metrics:
        used_metrics.update(dict(metric.match_items(all_metrics)))
    return reward, used_metrics

  def _complete_trial(
      self,
      aggregate_metrics: bool = False,
      global_step: Optional[int] = None):
    """Adds final measurement to trial based on metric aggregator."""
    # Poll the metrics across steps for aggregation.
    if aggregate_metrics:
      assert global_step is not None
      metrics_across_steps = []
      for m in self._feedback.get_trial().measurements:
        metrics = dict(m.metrics)
        metrics['reward'] = m.reward
        metrics_across_steps.append((m.step, metrics))

      final_metrics = self._cross_step_metric_aggregator(metrics_across_steps)
      final_metrics.pop('reward', None)
      final_reward, used_metrics = self._reward_and_used_metrics(
          final_metrics, global_step)
      self._feedback.add_measurement(
          final_reward, used_metrics, step=global_step)
      logging.info(
          'Final measurement is reported to trial %d at step %d '
          'with reward value %f and metrics %s.',
          self._feedback.id, global_step, final_reward, used_metrics)
    self._feedback.done()


def _write_file_once(file_path: epath.Path, content: Text):
  """Writes debug information to file only once."""
  if not file_path.exists():
    try:
      file_path.write_text(content)
    except (tf.errors.NotFoundError, IOError):
      logging.warn(
          'Cannot write file %r as another process is writing to the same '
          'file. This is not an issue as the file is only created for '
          'debugging purpose and has the same content among all the workers. '
          'So any successful write will achieve this purpose.', file_path)


class EvalMetrics(NamedTuple):
  metrics_list: Optional[Sequence[Optional[Dict[str, float]]]] = None
  scoring_metrics_list: Optional[Sequence[Optional[Dict[str, float]]]] = None
  steps_per_sec: Optional[float] = None
  input_names: Optional[Sequence[str]] = None


class DecodeMetrics(NamedTuple):
  metrics_list: Optional[Sequence[Optional[Dict[str, float]]]] = None
  processed_metrics_list: Optional[Sequence[Optional[Dict[str, float]]]] = None
  seqio_metrics_list: Optional[Sequence[Optional[Dict[str, float]]]] = None
  steps_per_sec: Optional[float] = None
  input_names: Optional[Sequence[str]] = None


def should_early_stop(early_stop_fn: trainer_lib.EarlyStoppingFn,
                      global_step: int,
                      is_last_ckpt: bool,
                      train_weighted_scalars: Optional[
                          Union[pytypes.WeightedScalars,
                                pytypes.WeightedScalarsList]] = None,
                      eval_train_metrics: Optional[Dict[str, float]] = None,
                      eval_metrics: Optional[EvalMetrics] = None,
                      decode_metrics: Optional[DecodeMetrics] = None,
                      num_params: Optional[float] = None,
                      train_steps_per_sec: Optional[float] = None) -> bool:
  """Returns True if the training process should stop early."""
  if early_stop_fn is None:
    return False

  # Detect running mode.
  running_mode = trainer_lib.RunningMode.detect(
      has_train_metrics=train_steps_per_sec is not None,
      has_eval_metrics=bool(eval_metrics),
      has_decode_metrics=bool(decode_metrics))

  # Since train metrics will be produced at each step, for performance reasons,
  # we only aggregate the metrics at the last checkpoint or at the step when
  # evaluation or decoding takes place.
  train_metrics = None
  if train_weighted_scalars is not None:
    if is_last_ckpt or running_mode.has_eval or running_mode.has_decode:
      train_weighted_scalars = py_utils.maybe_unreplicate_for_fully_replicated(
          train_weighted_scalars)
      train_metrics = metric_utils.as_float_dict(train_weighted_scalars)
      logging.info(
          ('Aggregate train weighted scalars as tuning metrics. '
           'Metrics=%s, WeightedScalars=%s'),
          train_metrics, train_weighted_scalars)

  # Aggregate metrics for tuning.
  tuning_metrics = _aggregate_metrics(train_metrics, eval_train_metrics,
                                      eval_metrics, decode_metrics, num_params,
                                      train_steps_per_sec)
  return early_stop_fn(tuning_metrics, running_mode, global_step, is_last_ckpt)


def _aggregate_metrics(
    train_metrics: Optional[Dict[str, float]] = None,
    eval_train_metrics: Optional[Dict[str, float]] = None,
    eval_metrics: Optional[EvalMetrics] = None,
    decode_metrics: Optional[DecodeMetrics] = None,
    num_params: Optional[float] = None,
    train_steps_per_sec: Optional[float] = None,
) -> Dict[str, Union[float, None]]:
  """Aggregate metrics from training, evaluation and decoding for tuning."""
  metrics = {}
  if train_metrics is not None:
    metric_utils.update_float_dict(metrics, train_metrics, 'train')

  if eval_train_metrics is not None:
    metric_utils.update_float_dict(
        metrics, eval_train_metrics, 'eval_train/metrics')

  def _add_input_based_metrics(
      input_names: Optional[List[str]],
      metrics_list: Optional[List[Optional[Dict[str, float]]]],
      dataset_type: Optional[str] = None,
      category: Optional[str] = None):
    if input_names is None or metrics_list is None:
      return
    assert len(input_names) == len(metrics_list), (input_names, metrics_list)
    merged = {}
    for input_name, m in zip(input_names, metrics_list):
      if m is not None:
        prefix = input_name
        if dataset_type is not None:
          prefix = f'{dataset_type}_{prefix}'
        if category is not None:
          prefix = f'{prefix}/{category}'
        metric_utils.update_float_dict(merged, m, prefix)
    metric_utils.update_float_dict(metrics, merged)

  if eval_metrics:
    eval_input_names = eval_metrics.input_names
    _add_input_based_metrics(eval_input_names, eval_metrics.metrics_list,
                             'eval_test', 'metrics')
    _add_input_based_metrics(eval_input_names,
                             eval_metrics.scoring_metrics_list,
                             'eval_test', 'scoring_eval')
  if decode_metrics:
    decode_input_names = decode_metrics.input_names
    _add_input_based_metrics(decode_input_names, decode_metrics.metrics_list,
                             'decode_test')
    _add_input_based_metrics(decode_input_names,
                             decode_metrics.processed_metrics_list,
                             'decode_test')
    _add_input_based_metrics(decode_input_names,
                             decode_metrics.seqio_metrics_list,
                             'decode_test')

  # Add training metrics.
  def _add_metric_if_not_none(name: str, value: Optional[float]):
    if value is not None:
      metrics[name] = value

  _add_metric_if_not_none('train_steps_per_sec', train_steps_per_sec)
  if eval_metrics is not None:
    metrics['eval_steps_per_sec'] = eval_metrics.steps_per_sec
  if decode_metrics is not None:
    metrics['decode_steps_per_sec'] = decode_metrics.steps_per_sec
  _add_metric_if_not_none('num_params', num_params)
  return metrics


def is_last_checkpoint(
    running_mode: trainer_lib.RunningMode,
    global_step: int,
    num_train_steps: int,
    eval_interval_steps: int,
    decode_interval_steps: int,
    save_interval_steps: int,
    train_to_end: bool = False) -> bool:
  """Returns True if current step should be treated as last evaluation."""
  if running_mode.has_train:
    # When training and evaluation/decoding are done within the same process,
    # evaluation/decoding are based on current weights stored in memory.
    # Therefore, evaluation/decoding can always been performed regardless
    # whether checkpoints are published at a certain step. So we set
    # `save_interval_steps` to zero in such case.
    save_interval_steps = 0
  remaining = num_train_steps - global_step
  is_last = remaining == 0
  if not train_to_end and not is_last:
    last_eval = False
    if running_mode.has_eval:
      last_eval = remaining < max(eval_interval_steps, save_interval_steps)
    last_decode = False
    if running_mode.has_decode:
      last_decode = remaining < max(decode_interval_steps, save_interval_steps)
    is_last = last_eval or last_decode
  return is_last


class TrialDirectoryNameGenerator:
  """Trial directory name generator.

  Each trial will be creating a sub-directory under the root experiment
  directory. To make it easy to compare trials in TensorBoard, we include
  the search decision point names and values in the directory name. This class
  is introduced for deciding the directory name with human readability.

  By default, it will include both decision names and values with '|' delimited
  string. For example: 'my_experiment/123/x=1|y=abc|z=(CUSTOM)', where 123 is
  the trial ID, which contains 3 decisions points named 'x', 'y', and 'z'.
  '(CUSTOM)' means a value of a custom decision point, which is usually not
  path friendly.

  When there are lots of decision points, usually for NAS, including decision
  names will make the directory name very long (determined by
  `total_name_length_threshold`). In such case, we only include the decision
  values in the path. For example: 'my_experiment/123/0|0.1|abc|ReLU'.
  """

  _NON_PATH_FRIENDLY_CHAR_SET = re.compile(r'[^\w\d=_-{}\(\).,\[\]]+')

  def __init__(self,
               root_dir: epath.Path,
               search_space: pg.hyper.DynamicEvaluationContext,
               combined_decision_point_names: Optional[List[str]] = None,
               total_name_length_threshold: int = 64):
    decision_point_names = list(search_space.hyper_dict.keys())  # pytype: disable=attribute-error
    if combined_decision_point_names:
      assert len(decision_point_names) == 1, decision_point_names
      assert automl.COMBINED_DECISION_ATTR in decision_point_names, (
          decision_point_names)
      decision_point_names = combined_decision_point_names

    self._root_dir = root_dir
    self._search_space = search_space
    self._decision_point_names = decision_point_names
    self._use_combined_decision_point = bool(combined_decision_point_names)
    self._include_decision_names = sum(
        len(n) for n in decision_point_names) < total_name_length_threshold

  def parameter_values(self) -> List[Tuple[str, Any]]:
    """Return the current parameter values and its choice indices.

    NOTE(daiyip): this function is intended to be called under the tuning loop.

    Returns:
      A list of tuple (decision name, decision value, choice index).
    """
    params = []
    for k, hyper in self._search_space.hyper_dict.items():  # pytype: disable=attribute-error
      v = self._search_space.evaluate(hyper)
      if isinstance(hyper, pg.hyper.CustomHyper):
        v = '(CUSTOM)'
      params.append((k, v))

    if self._use_combined_decision_point:
      assert len(params) == 1, params
      assert params[0][0] == automl.COMBINED_DECISION_ATTR, params
      combined_values = params[0][1]
      assert isinstance(combined_values, tuple), combined_values
      assert len(combined_values) == len(self._decision_point_names), (
          self._decision_point_names, combined_values)
      params = list(zip(self._decision_point_names, combined_values))
    return params

  def format_value(self, value: Any) -> str:
    """Formats a parameter value into path-friendly string."""
    if isinstance(value, float):
      return f'{value:.3e}'
    if isinstance(value, (bool, int)):
      return str(value)
    if inspect.isclass(value):
      return value.__name__
    return self._make_path_friendly(str(value))

  def _make_path_friendly(self, s: str) -> str:
    replacements = [(':', '='), ('[', '{'), (']', '}')]
    for src, replacement in replacements:
      s = s.replace(src, replacement)

    return self._NON_PATH_FRIENDLY_CHAR_SET.sub('', s)

  def dirname(self, trial_id: int) -> epath.Path:
    """Gets the directory name for a trial."""
    params = self.parameter_values()
    kv_pairs = [(k, self.format_value(v)) for k, v in params]
    if self._include_decision_names:
      items = [f'{k}={v}' for k, v in kv_pairs]
    else:
      items = [v for _, v in kv_pairs]
    return self._root_dir / str(trial_id) / '|'.join(items)
