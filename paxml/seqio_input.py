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

"""SeqIO input for Pax."""

from __future__ import annotations

import collections
import copy
import dataclasses
import enum
import io
import os
import typing
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, TextIO, Tuple, Union, cast

from absl import logging
from etils import epath
import jax
from jax.experimental import multihost_utils
import jax.numpy as jnp
import numpy as np
from paxml import metric_utils
from praxis import base_hyperparams
from praxis import base_input
from praxis import pax_fiddle
from praxis import py_utils
from praxis import pytypes
import seqio
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

NestedMap = py_utils.NestedMap
NestedNpTensor = pytypes.NestedNpTensor
SummaryWriter = tf.summary.SummaryWriter

MixtureRegistry = seqio.MixtureRegistry
SHARD_INDEX_KEY = py_utils.SHARD_INDEX_KEY
NUM_SHARDS_KEY = py_utils.NUM_SHARDS_KEY
INDEX_WITHIN_SHARD_KEY = py_utils.INDEX_WITHIN_SHARD_KEY
EVAL_METRICS_PREFIX = 'scoring_eval'
DECODE_METRICS_PREFIX = 'decoder'

# TODO(b/244434890): enable computing SeqIO task-defined metrics on model
# outputs other than models.LanguageModel.
_LM_DECODER_OUT_KEY = 'decoded_substr'
_LM_SCORE_KEY = 'scores'
_LM_LABEL_KEY = 'labels'


def _update_keys(answers: Dict[str, Any], targets: Mapping[str, Any],
                 task_name: str) -> None:
  """Insert into answers the keys from targets that only partially match."""
  # We update the dict `answers` in place, but inserting the keys found in
  # `targets` only. Those keys are matched by finding an existing key in answers
  # such that the existing key is a prefix. This can sometimes be needed due to
  # truncation of the original key during input processing.
  for k in targets:
    if k not in answers:
      new_keys = []
      for ak in answers:
        if k.startswith(ak):
          new_keys.append(ak)
      # As a sanity check, we require that there is a unique key that partially
      # matches.
      if len(new_keys) != 1:
        raise ValueError(f'key="{k}" in targets matches to {len(new_keys)} '
                         'entries in answers. This should not happen: please '
                         f'file a bug: task name "{task_name}".')
      answers[k] = answers[new_keys[0]]


def _get_targets_str(example: Mapping[str, Any], task: seqio.Task) -> str:
  """Gets pretokenized target str if available, otherwise reconstructs."""
  pretokenized_target_field_name = 'targets_pretokenized'
  target_field_name = 'targets'
  if pretokenized_target_field_name in example:
    target = example[pretokenized_target_field_name]
  elif target_field_name in example:
    target = example[target_field_name]
    try:
      if np.issubdtype(target[0], np.integer):
        target = [int(x) for x in target]
    except TypeError:
      logging.warning('Could not check if the data type is integer.')
    target = task.output_features[target_field_name].vocabulary.decode(target)
  else:
    target = ''
    logging.info('targets and targets_pretokenized not found in the example.')
  if isinstance(target, bytes):
    target = target.decode('utf-8')
  return target


def _log_plain_text_output(
    answers: Mapping[str, NestedMap], plain_text_output: TextIO) -> None:
  """Logs all examples for inspection in plain text format."""
  for _, ans in answers.items():
    print('---', file=plain_text_output)
    print(ans.get('prefix', ''), file=plain_text_output)
    print('>>>', file=plain_text_output)
    print(ans.get(_LM_DECODER_OUT_KEY, ''), file=plain_text_output)
    if 'seqio_targets' in ans:
      print('REF', file=plain_text_output)
      print(ans['seqio_targets'], file=plain_text_output)


def _convert_bytes_to_str(tree: Any) -> Any:
  """Converts any bytes leafs to strings in a pytree."""
  def _convert_fn(leaf: Any) -> Any:
    if not isinstance(leaf, bytes):
      return leaf

    return leaf.decode('utf-8')

  return jax.tree_map(_convert_fn, tree)


def select_split(
    task: str,
    split_name: Union[str, Callable[[str], str]],
) -> str:
  """Returns a split name given a split selector (Callable) or str literal."""
  if callable(split_name):
    return split_name(task)
  return split_name


def _add_fake_enumeration(ex: Dict[str, Any]) -> Dict[str, Any]:
  ex[SHARD_INDEX_KEY] = tf.cast(-1, tf.int32)
  ex[NUM_SHARDS_KEY] = tf.cast(-1, tf.int32)
  ex[INDEX_WITHIN_SHARD_KEY] = tf.cast(-1, tf.int64)

  return ex


def _is_padding(ex: Dict[str, Any]) -> bool:
  return (ex[INDEX_WITHIN_SHARD_KEY] == -1 and ex[SHARD_INDEX_KEY] == -1
          and ex[NUM_SHARDS_KEY] == -1)


def _enumerate_dataset(
    ds: tf.data.Dataset, is_training: bool,
    shard_info: Optional[seqio.ShardInfo]) -> tf.data.Dataset:
  """Add enumeration fields, only meaningful when is_training=False."""
  if is_training:
    return ds.map(_add_fake_enumeration, num_parallel_calls=tf.data.AUTOTUNE)

  def _add_shard_enumeration(ex: Dict[str, Any]) -> Dict[str, Any]:
    shard_index, num_shards = 0, 1
    if shard_info:
      shard_index, num_shards = shard_info.index, shard_info.num_shards

    ex[SHARD_INDEX_KEY] = tf.cast(shard_index, tf.int32)
    ex[NUM_SHARDS_KEY] = tf.cast(num_shards, tf.int32)

    return ex

  def _fold_in_local_enumeration(index_within_shard: tf.Tensor,
                                 ex: Dict[str, Any]) -> Dict[str, Any]:
    ex[INDEX_WITHIN_SHARD_KEY] = tf.cast(index_within_shard, tf.int64)
    return ex

  ds = ds.map(_add_shard_enumeration, num_parallel_calls=tf.data.AUTOTUNE)
  ds = ds.enumerate()
  ds = ds.map(_fold_in_local_enumeration,
              num_parallel_calls=tf.data.AUTOTUNE)

  return ds


def _get_num_examples(ds: tf.data.Dataset) -> int:
  # Iterate one-by-one instead of len(list(...)) to reduce peak memory.
  num_examples = 0
  for _ in ds:
    num_examples += 1

  return num_examples


def is_packing_on(fc: seqio.FeatureConverter) -> bool:
  """Safely checks whether a given feature converter has packing turned on."""
  return hasattr(fc, '_pack') and fc.pack


def maybe_update_decode_output_keys(
    process_decode_output: Sequence[Tuple[str, Any]],
    decode_out: NestedMap) -> Sequence[Tuple[str, Any]]:
  """Return outputs with updated keys if using SeqIO enum-based keys.

  This method assumes that the order of the per-example kv-pairs returned by
  `model.process_decode_out` is consistent with that of the `decode_out` fed
  into it.

  Args:
    process_decode_output: the per-example kv-pairs output returned by the call
      to `model.process_decode_out(...)`. We assume that the call to
      `process_decode_out` returns per-example kv-pairs that exclude padded
      examples (`b.eval_sample_weights=0`).
    decode_out: the output returned by `model.decode`, which may include padded
      examples.

  Returns:
    The same outputs as that of the call to `model.process_decode_out` but with
    the keys mapped by the enum keys generated by SeqIOInput, if using seqio
    inputs with enumeration IDs.

  Raises:
    RuntimeError: if `model.process_decode_out` doesn't exclude padded examples.
  """
  if not (enum_key_fields := py_utils.filter_by_matching_keys(
      decode_out, [py_utils.PROVENANCE_PREFIX])[0]):
    # not using seqio enum ids for matching examples
    return process_decode_output

  enum_keys = []
  for ex in py_utils.tree_unstack(enum_key_fields, 0):
    if not (key := py_utils.get_enumeration_id(ex)):
      raise ValueError(f'Not able to construct enum-id with {ex}.')
    if not _is_padding(ex):
      enum_keys.append(key)

  if len(enum_keys) != len(process_decode_output):
    raise RuntimeError(
        'The length of enum keys != num kv-pairs returned by '
        f'model.process_decode_out ({len(enum_keys)} != '
        f'{len(process_decode_output)}). Please file a bug as this should not '
        'happen.')

  return [(k, v) for k, (_, v) in zip(enum_keys, process_decode_output)]


def should_process_outputs(inp: base_input.BaseInput) -> bool:
  """Whether the current (input, process_index) pair should process outputs."""
  return (isinstance(inp, SeqIOInput) and jax.process_index() == 0)


def process_outputs(
    inp: base_input.BaseInput,
    model_outputs: Union[List[Dict[str, Any]], List[Tuple[str, Any]]],
    summary_writer: SummaryWriter,
    metric_type: MetricType,
    step: int,
    output_dir: epath.Path,
    verbose_entries: int = 1,
    plain_text_output_fname: Optional[str] = None) -> Dict[str, float]:
  """Computes SeqIO task-defined metric, write to TB, and returns mapping."""
  inp = typing.cast(SeqIOInput, inp)
  logging.info('Computing %s metrics', metric_type.name)

  if metric_type is MetricType.SCORE:
    metric_name_prefix = EVAL_METRICS_PREFIX
    seqio_metrics = inp.compute_metrics_eval(
        model_outputs, verbose_entries=verbose_entries)
    logging.info('Eval metrics from seqio: %s.', seqio_metrics)

  elif metric_type is MetricType.PREDICT:
    metric_name_prefix = DECODE_METRICS_PREFIX
    plain_text_output = io.StringIO()
    seqio_metrics = inp.compute_metrics(
        model_outputs, verbose_entries=verbose_entries,
        plain_text_output=plain_text_output)

    logging.info('Writing plain decoder output to %s', plain_text_output_fname)
    dirname = os.path.dirname(plain_text_output_fname)
    if not tf.io.gfile.exists(dirname):
      tf.io.gfile.makedirs(dirname)
    with tf.io.gfile.GFile(plain_text_output_fname, 'w') as f:
      f.write(plain_text_output.getvalue())

  else:
    raise ValueError(f'unsupported metric type: {metric_type}')

  # Write out seqio metrics with JSON logger to JSONL file.
  output_dir.mkdir(parents=True, exist_ok=True)
  logger = seqio.loggers.JSONLogger(output_dir.as_posix())
  merged_seqio_metrics = {}
  for sm in seqio_metrics:
    merged_seqio_metrics.update(sm)

  # JSON logger only takes MetricValue instances whereas some might be floats.
  merged_seqio_metrics = {
      k: seqio.metrics.Scalar(v)
         if not isinstance(v, seqio.metrics.MetricValue) else v
      for k, v in merged_seqio_metrics.items()
  }
  logger(task_name=inp.mixture_or_task_inst.name, step=step,
         metrics=merged_seqio_metrics, dataset=None, inferences=None,
         targets=None)

  # write metrics to tensorboard
  with summary_writer.as_default():
    metric_utils.write_seqio_metric_summaries(
        seqio_metrics, metric_name_prefix, step)

  # convert metrics to {string: float} mapping
  metrics = {}
  for sm in seqio_metrics:
    # TODO(b/244579359): add `metric_name_prefix` to key names when
    # consolidating seqio vs non-seqio metrics in AutoML.
    metrics = metric_utils.update_float_dict(
        metrics, metric_utils.as_float_dict(sm))

  return metrics


class SeqIOInput(base_input.BaseInput):
  """An adaptor for getting SeqIO data.

  Attributes:
    mixture_name: Optional string. The name for a SeqIO task or mixture. User
      must import the module that defines this task/mixture in order to register
      the task/mixture.
    mixture_or_task: Optional SeqIO task object. The user must specify either
      mixture_name or mixture_or_task params.
    split_name: Required string. The name for the split of data to get. Usually
      "train" or "validation" or "test".
    deterministic_input: If deterministic input is intended, users should set
      this to enable internal validations to ensure that deterministic input is
      indeed used.
    task_feature_lengths: Required. Of type Mapping[str, int]. The keys are the
      features on the original SeqIO task/mixture, typically "inputs" and
      "targets". The values are corresponding sequence lengths. Examples
      exceeding the sequence lengths are truncated.
    feature_converter: An instance of a seqio.FeatureConverter subclass. This is
      used to convert the data from its original format to the format expected
      by the model, e.g. instead of "targets" we have "ids" or "labels" or
      "paddings". This also implements any necessary padding or packing on the
      data.
    shuffle: Whether to shuffle the data. Note that None means this feature is
      decided automatically: True for and only for non-deterministic training
      data, otherwise False. Users can override this by setting this explicitly.
    repeat: Whether to repeat the data. Note that None means this feature is
      decided automatically: True only for non-deterministic training data,
      otherwise False. Users can override this by setting this field explicitly.
    use_cached: Whether to read from the cached directory, if supported by the
      underlying SeqIO task/mixture. Users can set to False to test out data
      changes before the cache is applied.
    trim_output_features: If True, it trims output features to be less than the
      length given by `sequence_length`.
    try_in_mem_cache: If True, caches sufficiently small datasets in memory for
      efficiency.
    eval_auto_pad: Only used when p.is_training=False. Automatically pad the
      data to multiples of global batch size, using the first example in the
      data. Padded entries will have batch_input.eval_sample_weight == 0.0.
    drop_remainder: Whether to drop remaining examples from the last partial
      batch.
    num_batches_to_skip: If not None, skip given number of batches in the
      dataset, to avoid reusing data after restaring the training process. Only
      affects training.
    deterministic_input_start_index: Params to compute the starting example
      index. Used only if the data is a deterministic input, otherwise ignored.
    eval_metrics_targets_length: typically when used in eval, the data returned
      by get_next() would not contain any targets. eval_metrics_targets_length
      overrides the task feature lengths for targets when processing the targets
      as ground truths to compute eval metrics. It has no effect on get_next(),
      but only affects compute_metrics(). If set to None, won't truncate.
    use_enumeration: whether to use enumeration in both batch generation
      (get_next()) and metrics computation. When this param is set to True,
      we'll return a NestedMap including enumeration related provenance fields,
      which will assign each example a globally-unique ID within a given
      dataset. In `__call__` of the model, the user is then expected to return a
      NestedMap including '.enumerated_index' and for `process_decode_out` the
      key in the sequence of tuples should be the enumerated index. At metrics
      computation time, we'll join the enumerated index.
    annotate_padding_fields: whether to manually update the `.weights` and
      `.padding` fields for examples. It is preferable that users use
      `.eval_sample_weights` field that gets set to `=0` for padded eval
      examples.
    overridden_vocab: the vocab overridden. If not set, it would be derived from
      `output_features` of underlining task_or_mixture.
    enable_symbolic_checkpointing: Whether to use symbolic checkpointing instead
      of explicit checkpointing. Note that task.train.enable_input_checkpointing
      must be enabled to use this feature.
    experimental_enable_index_shuffle: Enables index shuffle, which is required
      for symbolic checkpointing of training inputs (otherwise defaults to
      explicit checkpointing). Applicable only for tf.data inputs that have
      checkpointing enabled.
    log_preprocessed_targets: Whether to write preprocessed_targets to log_dir.
      Note that doing so will expose raw data inputs, and user should ensure
      that data are not sensitive and/or the log_dir has the appropriate
      permissions to prevent this from happening.
    eval_enable_cache: If true, caching will be enabled on the eval dataset.
      Note that if caching is disabled, the evals may be non-reproducible if
      there are random ops.
    eval_num_examples: The number of examples in the dataset. Setting this field
      can short-circuit iterating over dataset to get count of examples. If
      unspecified, number of examples are counted from the dataset.
    warm_start: Whether to start background threads of asynchronous
      transformations upon iterator creation, as opposed to during the first
      call to `next()`. This improves the latency of the initial
      'get_next_padded()' calls at the expense of requiring more memory to hold
      prefetched elements between the time of iterator construction and usage.
    dataset: Set to the underlying tf.data.Dataset. This field is inherited from
      `BaseInput`, used for creating iterators and input specs.
  """

  @dataclasses.dataclass(frozen=True)
  class DeterministicInput:
    """Parameters to adjust the starting example index of deterministic input.

    Attributes:
      start_step: This value is subtracted from the latest model train step.
        Used when e.g. finetuning. In other words, treat as if training starts
        from `_latest_model_step` - `start_step`.
      example_index_offset: An offset to be added to `start_example_index`. This
        can be used e.g. when user changes batch size.
      _latest_model_step: Internal param that users do not need to set. The
        latest train step from the latest checkpoint in the model dir,
        accounting for train_p.eval_interval_steps.
    """
    # Params to adjust the starting example index for deterministic input.
    start_step: int = 0
    example_index_offset: int = 0
    # Internal params set by Pax internally.
    _latest_model_step: int = 0

  # Required params.
  mixture_name: Optional[str] = None
  mixture_or_task: Optional[Union[seqio.Task, seqio.Mixture]] = None
  split_name: Optional[str] = None
  deterministic_input: bool = False
  task_feature_lengths: Optional[Mapping[str, int]] = None
  feature_converter: Optional[seqio.FeatureConverter] = None
  # Optional params.
  shuffle: Optional[bool] = None
  repeat: Optional[bool] = None
  use_cached: bool = False
  eval_auto_pad: bool = True
  drop_remainder: bool = True
  num_batches_to_skip: Optional[int] = None
  # trim_output_features flag allow passing this arg to seqio.get_dataset
  # the default value is True so this change will not affect any current
  # behaviour. the main purpose is for prefixlm to not problematically
  # pack on the inputs.
  trim_output_features: bool = True
  # Set `try_in_mem_cache` to maintain backward-compatibility since this is also
  # enabled by default in SeqIO. This flag has side effects to increase host
  # memory usage and may lead to OOMs in some senarios.
  try_in_mem_cache: bool = True
  # Params to adjust the starting example index for deterministic input.
  # Implementation note: `SingleTask` is not defined in the interpreter
  # context here, so we need to wrap it in a lambda which will look it up from
  # the global scope later.
  deterministic_input_start_index: pax_fiddle.Config[
      SeqIOInput.DeterministicInput
  ] = pax_fiddle.template_field(DeterministicInput)
  eval_metrics_targets_length: Optional[int] = None
  use_enumeration: bool = True
  annotate_padding_fields: bool = False
  overridden_vocab: Optional[seqio.Vocabulary] = None
  dataset: tf.data.Dataset = dataclasses.field(init=False, repr=False)
  _iter: Any = dataclasses.field(init=False, repr=False)
  _cached_targets_with_enum_key: Optional[Mapping[str, NestedMap]] = (
      dataclasses.field(init=False, repr=False)
  )
  is_targets_init: Any = dataclasses.field(init=False, repr=False)
  _mixture_or_task_inst: Any = dataclasses.field(init=False, repr=False)
  _shard_info: Any = dataclasses.field(init=False, repr=False)
  _len_full_ds: Any = dataclasses.field(init=False, repr=False)
  targets_ds: Any = dataclasses.field(init=False, repr=False)
  targets_ds_converted: Any = dataclasses.field(init=False, repr=False)
  targets_ds_ori: Any = dataclasses.field(init=False, repr=False)
  ds_non_ragged_tensor_keys: Any = dataclasses.field(init=False, repr=False)
  ds_ragged_tensor_keys: Any = dataclasses.field(init=False, repr=False)
  targets_iter: Any = dataclasses.field(init=False, repr=False)
  targets_iter_converted: Any = dataclasses.field(init=False, repr=False)
  targets_iter_ori: Any = dataclasses.field(init=False, repr=False)
  _num_eval_examples: Any = dataclasses.field(init=False, repr=False)
  enable_symbolic_checkpointing: Optional[bool] = True
  experimental_enable_index_shuffle: Optional[bool] = True
  log_preprocessed_targets: Optional[bool] = False
  eval_enable_cache: bool = True
  eval_num_examples: Optional[int] = None
  warm_start: bool = True

  def __post_init__(self):
    # Modify hparams in-place before freezing hparams
    if not self.name:
      mixture_name = self.mixture_name or self.mixture_or_task.name  # pytype: disable=attribute-error
      self.name = f'{mixture_name}_{self.split_name}'
    if (
        not self.is_training
        and self.input_random_seed is None
        and self.use_enumeration
    ):
      # Since we want the enumeration to be deterministic, in the case that
      # there's no explicit seed set, we default to a fixed seed for evals.
      self.input_random_seed = 42
    self.configure_tf_data_options()
    super().__post_init__()
    self._validate_hparams()

    with py_utils.timeit() as get_dataset_timer:
      self.dataset = self._get_dataset()
    logging.info(
        '[PAX STATUS]: SeqIO init took %d seconds (mixture=%s, split=%s)',
        get_dataset_timer.elapsed,
        self.mixture_name,
        self.split_name,
    )

    with py_utils.timeit() as np_iterator_timer:
      self._iter = self.dataset.as_numpy_iterator()
    logging.info(
        '[PAX STATUS]: SeqIO dataset `as_numpy_iterator` call took %d seconds'
        ' (mixture=%s, split=%s)',
        np_iterator_timer.elapsed,
        self.mixture_name,
        self.split_name,
    )

    # Populated by first call to `self._get_targets_with_enum_key`. Subsequent
    # calls to it short circuit by returning the cached values.
    self._cached_targets_with_enum_key: Optional[Mapping[str, NestedMap]] = None

    self.is_targets_init = False

  def _validate_deterministic(self):
    """Validates deterministic input settings and creates the shard info."""
    if self.deterministic_input:
      raise ValueError('deterministic_input is not supported')
    shard_info = seqio.ShardInfo(
        index=self.infeed_host_index, num_shards=self.num_infeed_hosts
    )
    logging.info(
        'ShardInfo for %s: shard_id: %d, num_shards: %d, ',
        self._mixture_or_task_inst.name,
        shard_info.index,
        shard_info.num_shards,
    )
    return shard_info

  def _validate_eval_task(self):
    assert isinstance(self.mixture_or_task_inst, seqio.Task)
    # weights_on_targets_only must be true if computing scoring metric fns and
    # using LanguageModelFeatures as feature converter.
    if (
        self.mixture_or_task_inst.score_metric_fns
        and isinstance(self.feature_converter, LanguageModelFeatures)
        and not self.feature_converter.weights_on_targets_only
    ):
      raise ValueError(
          'All language modeling scoring evals must set '
          'LanguageModelFeatures.weights_on_targets_only=True')

  def _validate_hparams(self):
    if not self.mixture_name and not self.mixture_or_task:
      raise ValueError("One of 'mixture_name' and 'task' must be set.")
    if self.mixture_name and self.mixture_or_task:
      raise ValueError(
          "Only one of 'mixture_name' and 'mixture_or_task' can be set."
          ' Got %s and %s.' % (self.mixture_name, self.mixture_or_task)
      )
    if self.is_training and self.split_name != 'train':
      logging.warning(
          (
              'SeqIO input hparams p.is_training=True but p.split_name is '
              'not "train" but p.split_name=%s'
          ),
          self.split_name,
      )

    self._mixture_or_task_inst = (
        self.mixture_or_task or seqio.get_mixture_or_task(self.mixture_name)
    )
    self._shard_info = self._validate_deterministic()

    if not self.is_training and isinstance(
        self.mixture_or_task_inst, seqio.Task
    ):
      self._validate_eval_task()

    if self.eval_loop_num_batches is None and self.repeat:
      raise ValueError(
          'Dataset has eval_loop_num_batches set to None while repeat is True. '
          'This would lead to an endless eval.'
      )

  @property
  def is_deterministic(self) -> bool:
    """Indicates whether this SeqIOInput is deterministic or not."""
    return False

  @property
  def should_shuffle(self) -> bool:
    """Indicates whether this SeqIOInput shuffles the data or not."""
    if self.shuffle is None:
      return self.is_training and not self.is_deterministic
    if self.shuffle and not self.is_training:
      logging.warning(
          (
              'Disabling shuffle for decode/eval input; otherwise input'
              ' and target enum fields will be mismatched.'
          ),
      )
    return self.shuffle and self.is_training

  @property
  def should_repeat(self) -> bool:
    """Indicates whether this SeqIOInput repeats the data or not."""
    if self.repeat is None:
      return self.is_training and not self.is_deterministic
    return self.repeat

  @property
  def mixture_or_task_inst(self) -> Union[seqio.Task, seqio.Mixture]:
    return self._mixture_or_task_inst

  @property
  def is_mixture(self) -> bool:
    return self.mixture_or_task_inst.name in MixtureRegistry.names()

  @property
  def task_inst(self) -> seqio.Task:
    return cast(seqio.Task, self.mixture_or_task_inst)

  def configure_tf_data_options(self):
    read_config = tfds.ReadConfig()
    options = tf.data.Options()

    # tf.data `warm_start` feature is newly added and only available on TF
    # nightly, but not the latest stable version(2.12).
    # TODO(rxsang): Remove checking `hasattr` once TF2.13 is released.
    if self.warm_start and (
        hasattr(options.experimental_optimization, 'warm_start')
    ):
      options.experimental_optimization.warm_start = True

    if self.input_checkpointing_enabled and self.enable_symbolic_checkpointing:
      options.experimental_symbolic_checkpoint = True
      read_config.experimental_index_shuffle = (
          self.experimental_enable_index_shuffle
      )
      # Disable readahead for random access used by index shuffle.
      if self.experimental_enable_index_shuffle:
        read_config.override_readahead = tfds.Readahead.DISABLE
    read_config.options = options
    seqio.set_tfds_read_config_override(read_config)

  def _get_num_eval_examples(self) -> int:
    """Returns the number of eval examples.

    Corresponds to`eval_loop_num_batches` when reset_for_eval=False;
    otherwise eval runs on full dataset.
    """
    if (
        not self.is_training
        and self.batch_size
        and self.eval_loop_num_batches
        and not self.reset_for_eval
    ):
      return (
          self.eval_loop_num_batches * self.batch_size * self.num_infeed_hosts
      )
    return -1

  def _get_dataset(self) -> tf.data.Dataset:
    logging.info(
        (
            "Initializing dataset for task '%s' with a per host batch size of"
            ' %d and a seed of %s'
        ),
        self.mixture_or_task_inst.name,
        self.batch_size,
        self.input_random_seed,
    )
    ds = self._get_backing_ds(
        shuffle=self.should_shuffle,
        num_epochs=-1 if self.should_repeat else 1,
        shard_info=self._shard_info)
    ds = self._pad_to_batch_size(ds)
    ds = ds.batch(
        self.batch_size,
        drop_remainder=self.drop_remainder,
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    if self.num_batches_to_skip:
      if self.is_training:
        ds = ds.skip(self.num_batches_to_skip)
      else:
        logging.warning(
            (
                "num_batches_to_skip set tp '%d' but has no effect because "
                'is_training is false'
            ),
            self.num_batches_to_skip,
        )

    return ds

  def _gen_targets_dataset(self):
    self._len_full_ds = 0
    sequence_length = dict(self.task_feature_lengths)
    # if set, p.eval_metrics_targets_length
    # overrides p.task_feature_lengths['targets']
    if self.eval_metrics_targets_length:
      sequence_length['targets'] = self.eval_metrics_targets_length
    sharded_datasets = []
    sharded_datasets_converted = []

    largest_shard = 0
    for host_idx in range(self.num_infeed_hosts):
      shard_info = seqio.ShardInfo(
          index=host_idx, num_shards=self.num_infeed_hosts
      )
      ds_shard = self.mixture_or_task_inst.get_dataset(
          sequence_length=sequence_length,
          split=self.split_name,
          shuffle=False,
          num_epochs=1,
          shard_info=shard_info,
          seed=self.input_random_seed,
          use_cached=self.use_cached,
          trim_output_features=self.trim_output_features,
          try_in_mem_cache=self.try_in_mem_cache,
      )
      if self.should_repeat:
        # repeats the sharded dataset before adding enum fields
        num_epochs = -1
        # repeats the interleaving shards after adding enum fields
        largest_shard = -1
      else:
        num_epochs = 1
        # Interleave sharded dataset so it can be read sequentially on single
        # host. The shards may be uneven so we need to repeat based on the
        # largest shard to avoid missing any data.
        shard_num_examples = _get_num_examples(ds_shard)
        if shard_num_examples > largest_shard:
          largest_shard = shard_num_examples
        logging.info(
            'Targets shard %d/%d is not repeated; has %d examples',
            host_idx,
            self.num_infeed_hosts,
            shard_num_examples,
        )
        self._len_full_ds += shard_num_examples
      ds_shard = ds_shard.repeat(num_epochs)
      if self.use_enumeration:
        ds_shard = _enumerate_dataset(ds_shard, self.is_training, shard_info)
      else:
        sharded_datasets_converted.append(
            self.feature_converter(ds_shard, self.task_feature_lengths)
        )
      sharded_datasets.append(ds_shard)

    choice_dataset = tf.data.Dataset.range(self.num_infeed_hosts).repeat(
        largest_shard
    )
    self.targets_ds = tf.data.experimental.choose_from_datasets(
        sharded_datasets, choice_dataset
    )
    if not self.use_enumeration:
      self.targets_ds_converted = tf.data.experimental.choose_from_datasets(
          sharded_datasets_converted, choice_dataset
      )

  def _gen_filtered_artifacts(self) -> None:
    """Create filtered targets artifacts."""
    (
        self.targets_ds,
        self.targets_ds_ori,
        self.ds_non_ragged_tensor_keys,
        self.ds_ragged_tensor_keys,
    ) = self._filter_ragged_tensor(self.targets_ds)

  def _gen_targets_iter(self) -> None:
    """Generate targets iterator."""
    self.targets_iter = self.targets_ds.as_numpy_iterator()
    if not self.use_enumeration:
      self.targets_iter_converted = (
          self.targets_ds_converted.as_numpy_iterator()
      )
    # Create iterator for ori_targets_ds, which may have ragged tensors and thus
    # can't safely be called with `as_numpy_iterator()`
    # (see `_filter_ragged_tensor()`)
    self.targets_iter_ori = iter(self.targets_ds_ori)

  def _gen_targets_artifacts(self):
    self._num_eval_examples = self._get_num_eval_examples()
    self._gen_targets_dataset()
    self._gen_filtered_artifacts()
    self._gen_targets_iter()
    self.is_targets_init = True

  def save(self, checkpoint_path: epath.PathLike):
    try:
      self._ckpt = tf.train.Checkpoint(it=self._iter)
      self._ckpt.write(checkpoint_path)
    except tf.errors.UnimplementedError as e:
      raise NotImplementedError(
          'Checkpointing is not supported for this SeqIO input') from e

  def restore(self, checkpoint_path: epath.PathLike):
    self._peek = None
    self._ckpt = tf.train.Checkpoint(it=self._iter)
    self._ckpt.read(checkpoint_path).assert_consumed()

  def get_state(self) -> bytes:
    return self._iter._save().numpy()  # pylint: disable=protected-access

  def set_state(self, state: bytes) -> None:
    self._peek = None
    self._iter._restore(state)  # pylint: disable=protected-access

  def get_next(self) -> NestedNpTensor:  # pytype: disable=signature-mismatch  # jax-ndarray
    element = next(self._iter)
    # For non-training single-host infeed, the xla_passthrough will deal with
    # the unsupported types.
    if not self.is_training and self.num_infeed_hosts == 1:
      return element

    # Remove unsupported string (byte) array from input if training.
    # Also remove unsupported string (byte) array from input if there are
    # multiple hosts for eval, since xla passthrough does not support multihost
    # eval b/279795947.
    unsupported_key_dtype_name = {}

    def is_dtype_supported(key, value) -> bool:
      is_supported = np.issubdtype(value.dtype, np.number) or np.issubdtype(
          value.dtype, np.bool_
      )
      if not is_supported:
        unsupported_key_dtype_name[key] = value.dtype.name
      return is_supported

    filtered_element = element.FilterKeyVal(is_dtype_supported)

    if unsupported_key_dtype_name:
      logging.log_first_n(
          logging.INFO,
          'The following keys are filtered for unsupported dtypes: %s',
          1,
          unsupported_key_dtype_name,
      )
    return filtered_element

  def reset(self) -> None:
    self._iter = self.dataset.as_numpy_iterator()

  def _get_vocab(self, key) -> seqio.Vocabulary:
    if self.overridden_vocab is not None:
      return self.overridden_vocab

    features = self.mixture_or_task_inst.output_features
    if key not in ('src', 'tgt', None):
      raise ValueError(
          f"arg 'key' must be one of [None, 'src', 'tgt'], got key={key}."
      )

    real_key = 'targets' if key in (None, 'tgt') else 'inputs'
    return features[real_key].vocabulary

  def ids_to_strings(
      self, ids: pytypes.NpTensor,
      lengths: Union[pytypes.NpTensor, Sequence[pytypes.NpTensor]],
      key: Optional[str] = None) -> Sequence[str]:
    vocab = self._get_vocab(key)

    if lengths is None:
      lengths = [ids.shape[1]] * ids.shape[0]
    ret = []
    for i in range(ids.shape[0]):
      length = int(lengths[i])
      row = ids[i, :length].tolist()
      ret.append(vocab.decode(row))
    return ret

  def _get_backing_ds(self,
                      shuffle: bool,
                      num_epochs: int,
                      shard_info: Optional[seqio.ShardInfo]) -> tf.data.Dataset:
    kwargs = dict(
        sequence_length=self.task_feature_lengths,
        split=self.split_name,
        shuffle=shuffle,
        num_epochs=num_epochs,
        shard_info=shard_info,
        use_cached=self.use_cached,
        seed=self.input_random_seed,
        trim_output_features=self.trim_output_features,
        try_in_mem_cache=self.try_in_mem_cache,
    )
    ds = self.mixture_or_task_inst.get_dataset(**kwargs)

    ds = self.feature_converter(
        ds, task_feature_lengths=self.task_feature_lengths
    )

    if self.use_enumeration:
      # We want to add enumeration provenance fields *after* applying all
      # feature converters since feature converters don't pass through
      # unrecognized fields by default
      ds = _enumerate_dataset(ds, self.is_training, shard_info)

    return ds

  def _get_global_ds(self) -> tf.data.Dataset:
    return self._get_backing_ds(shuffle=False, num_epochs=1, shard_info=None)

  def _get_one_example_ds(self, ds: tf.data.Dataset) -> tf.data.Dataset:
    """Gets a dataset with just one example."""
    ret = ds.take(1)
    if list(ret.as_numpy_iterator()):
      return ret
    # The local shard is empty. This might happen if the global dataset size
    # is smaller than the number of global batch size.
    return self._get_global_ds().take(1)

  def _pad_to_batch_size(self, ds: tf.data.Dataset) -> tf.data.Dataset:
    """Pad the data with new entries to multiples of global batch size."""

    def _add_weight(b):
      if not isinstance(b, py_utils.NestedMap):
        b = py_utils.NestedMap.FromNestedDict(b)
      b.eval_sample_weights = 1.0
      return b

    def _add_pad(b):
      if not isinstance(b, py_utils.NestedMap):
        b = py_utils.NestedMap.FromNestedDict(b)
      b.eval_sample_weights = 0.0
      if self.use_enumeration:
        b = _add_fake_enumeration(b)
      if self.annotate_padding_fields and hasattr(b, 'weights'):
        b.weights *= 0
        if hasattr(b, 'paddings'):
          b.paddings = 1 - b.weights
      return b

    ds = ds.map(_add_weight)
    if (
        self.is_training
        or not self.reset_for_eval
        or not self.eval_auto_pad
        or self.should_repeat
    ):
      return ds

    # self.reset_for_eval=True: We are running eval over exactly one epoch.
    # We explicitly cache the entire epoch (in memory) to ensure that it is the
    # same across different iterations. Note that this is needed not only
    # because of ordering, but for data contents as well. For instance, with
    # seqio's FewshotDataSource preprocessing, some training data is part of the
    # prompt. These training data may be shuffled with
    # `reshuffle_each_iteration=True`. In general, there is no guarantee that
    # the underlying eval dataset stays unchanged across different iterations
    # of epochs.
    if self.eval_enable_cache:
      ds = ds.cache()

    # local_num represents the total number of examples in eval dataset,
    # computed as:
    #   - eval_num_examples, if provided by the user.
    #   - Otherwise, we iterate over dataset to compute
    #     number of examples.
    # Note: Global cached stats are available for Task, when cache_dir is set.
    # It needs to be investigated if local number of examples can be computed
    # from cached_stats using:
    # `self.task_inst.get_cached_stats(split=self.split_name)['examples']`
    # divided by `self.num_infeed_hosts`
    if self.eval_num_examples:
      local_num = self.eval_num_examples
    else:
      local_num = _get_num_examples(ds)
    local_num_batches = (local_num + self.batch_size - 1) // self.batch_size
    # Find the max number of batches required across all Jax processes.
    num_batches_all = multihost_utils.process_allgather(
        jnp.array([local_num_batches]), tiled=False)
    num_batches = np.max(num_batches_all)
    pad_num = num_batches * self.batch_size - local_num
    assert pad_num >= 0
    logging.info(
        'Eval data has %d local entries, padding now with '
        '%d extra entries to get %d batches.', local_num, pad_num, num_batches)
    # Repeat a random example to make the last batch full.
    pad_ds = self._get_one_example_ds(ds).map(_add_pad).repeat(pad_num)
    return ds.concatenate(pad_ds)

  def _filter_ragged_tensor(self, targets_ds: tf.data.Dataset) -> Tuple[
      tf.data.Dataset, tf.data.Dataset, List[str], List[str]]:
    # customized input may contain ragged tensor, which may cause errors in
    # decoding when calling 'as_numpy_iterator()' below. We filter out
    # RaggedTensor here.
    ds_ragged_tensor_keys = []
    ds_non_ragged_tensor_keys = []
    for ds_element_key, ds_element_value in targets_ds.element_spec.items():
      if isinstance(ds_element_value, tf.RaggedTensorSpec):
        ds_ragged_tensor_keys.append(ds_element_key)
      else:
        ds_non_ragged_tensor_keys.append(ds_element_key)
    # keep the original target ds before filtering because if inputs is a
    # ragged tensor, we need both the inputs and targets_pretokenized.
    ori_targets_ds = targets_ds
    targets_ds = targets_ds.map(
        lambda x: {i: x[i] for i in ds_non_ragged_tensor_keys}
    )
    return (
        targets_ds,
        ori_targets_ds,
        ds_non_ragged_tensor_keys,
        ds_ragged_tensor_keys,
    )

  def _build_predict_metric_inputs_with_prefix(
      self, answers: Dict[str, NestedMap], verbose_entries: int,
      plain_text_output: Optional[TextIO] = None) -> Tuple[
          Sequence[str], Sequence[str]]:
    """Builds 1-to-1 mapped predictions and targets lists via prefix matches."""
    assert not self.use_enumeration
    assert self.targets_ds
    # Prepare ground truth label data by dumping out seqio eval dataset and
    # get a dict key-ed by detokenized inputs (tokenized inputs are truncated
    # to inputs_length).
    inputs_length = self.task_feature_lengths['inputs']
    assert inputs_length is not None

    # Note that lists are used per prefix since there may be duplicates
    targets = collections.defaultdict(list)
    examples = collections.defaultdict(list)
    num_examples = 0
    while (
        self._num_eval_examples < 0
        or num_examples < self._num_eval_examples
    ):
      num_examples += 1
      try:
        example = next(self.targets_iter)
        if 'inputs' not in self.ds_non_ragged_tensor_keys:
          example_orig = next(self.targets_iter_ori)
      except (tf.errors.OutOfRangeError, StopIteration) as exc:
        if self._num_eval_examples > 0:
          raise StopIteration(
              'Exhausted eval data with reset_for_eval=False after'
              f' {num_examples-1} examples (batch_size={self.batch_size})'
          ) from exc
        logging.info('Exhausted eval data after %d steps', num_examples - 1)
        self._gen_targets_iter()
        break

      # Note that we intentionally do not use 'inputs_pretokenized' here because
      # it might be very different from the round-trip results below, which
      # wouldn't match with the keys we get from the model inference path.
      if 'inputs' in self.ds_non_ragged_tensor_keys:
        inputs = example['inputs'][np.newaxis, :]
      else:
        # ragged tensor may have multiple elements of various lengths. We
        # linearize all inputs into one array.
        assert (
            'inputs' in example_orig
        ), '"inputs" field is required but not found'
        inputs = example_orig['inputs'].flat_values.numpy()[np.newaxis, :]
      key = self.ids_to_strings(inputs, lengths=[inputs_length], key='src')[0]  # pytype: disable=wrong-arg-types
      t = _get_targets_str(example, self.mixture_or_task_inst)
      targets[key].append(self.task_inst.postprocess_fn(
          t, example=example, is_target=True))
      examples[key].append(example)

    # In case the prefix returned by the model are prefixes of the keys
    # re-constructed here. This can sometimes be needed due to truncation of
    # the original key during input processing.
    _update_keys(answers, targets, self.mixture_or_task_inst.name)

    # Construct (decode output, seqio target) lists by joining on seqio's
    # detok(tok(features['inputs'])[:task_feature_lengths['inputs']])).
    predictions_list = []
    targets_list = []
    for k in targets:
      if k not in answers:
        raise ValueError(
            f'Example not found in decoder output (key={k}): {targets[k]}')
      ans = answers[k]
      answer = ans[_LM_DECODER_OUT_KEY]
      seqio_postprocessed_predictions = []
      for target, e in zip(targets[k], examples[k]):
        targets_list.append(target)
        prediction = self.task_inst.postprocess_fn(
            answer, example=e, is_target=False)
        predictions_list.append(prediction)
        seqio_postprocessed_predictions.append(prediction)

      # Mutate 'ans' dictionary which is written to disk afterwards
      ans['seqio_targets'] = targets[k]
      ans['seqio_postprocessed_predictions'] = _convert_bytes_to_str(
          seqio_postprocessed_predictions
      )
    eval_data_size = (
        self._num_eval_examples
        if self._num_eval_examples > 0
        else self._len_full_ds
    )
    logging.info(
        'Data %s has %s examples for computing eval metrics.',
        self.name,
        eval_data_size,
    )
    if eval_data_size != len(predictions_list):
      raise ValueError(
          f'Data {self.name} expects {eval_data_size} examples for computing'
          f' eval metrics, got {len(predictions_list)}.'
      )

    # Log a few examples for inspection and sanity check.
    it = iter(targets)
    for i in range(verbose_entries):
      k = next(it)
      ans = answers[k]
      e = examples[k][0]
      answer = ans[_LM_DECODER_OUT_KEY]
      answer_processed = self.task_inst.postprocess_fn(
          answer, example=e, is_target=False)
      target = _get_targets_str(e, self.mixture_or_task_inst)
      target_processed = self.task_inst.postprocess_fn(
          target, example=e, is_target=True)
      logging.info(
          'Example %d:\nPROMPT=%s\nMODEL=%s\nFROM %s\nLABEL=%s FROM %s.', i, k,
          answer_processed, answer, target_processed, target)

    # Optionally log all examples for inspection in text format
    if plain_text_output is not None:
      _log_plain_text_output(answers, plain_text_output)

    return predictions_list, targets_list

  def _get_targets_with_enum_key(self) -> Mapping[str, NestedMap]:
    assert self.targets_ds
    if self._cached_targets_with_enum_key is not None:
      return self._cached_targets_with_enum_key
    targets = {}
    num_examples = 0
    while (
        self._num_eval_examples < 0
        or num_examples < self._num_eval_examples
    ):
      num_examples += 1
      try:
        example = next(self.targets_iter)
        if self.ds_ragged_tensor_keys:
          example_orig = next(self.targets_iter_ori)
      except (tf.errors.OutOfRangeError, StopIteration) as exc:
        if self._num_eval_examples > 0:
          raise StopIteration(
              'Exhausted eval data with reset_for_eval=False after'
              f' {num_examples-1} examples (batch_size={self.batch_size})'
          ) from exc
        logging.info(
            'Exhausted target eval data after %d examples', num_examples - 1
        )
        self._gen_targets_iter()
        break

      # remove enum related fields from example as seqio metric_fns API
      # expects the output from the task dataset directly.
      key = py_utils.get_enumeration_id(example, pop=True)
      assert key is not None
      targets[key] = example

      # we keep the ragged tensors and add them into the targets.
      # ragged tensor may have multiple elements of various lengths. We
      # linearize all inputs into one array.
      for field in self.ds_ragged_tensor_keys:
        field_value = example_orig[field]
        linearized_field_value = field_value.flat_values.numpy()
        linearized_field_value = linearized_field_value[np.newaxis, :]
        targets[key][field] = linearized_field_value

    if self.reset_for_eval:
      self._cached_targets_with_enum_key = targets  # populate cache
    return targets

  def _build_predict_metric_inputs_with_enum(
      self, answers: Mapping[str, NestedMap], verbose_entries: int,
      plain_text_output: Optional[TextIO] = None) -> Tuple[
          Sequence[Any], Sequence[Any]]:
    """Builds 1-to-1 mapped predictions and targets lists using enum fields."""
    assert self.use_enumeration

    targets = self._get_targets_with_enum_key()

    predictions_list = []
    targets_list = []
    # Construct (decode output, seqio target) lists by joining on enum IDs.
    # "Left-join" using targets constructed since outputs may have been padded.
    for k in targets:
      if k not in answers:
        raise ValueError(
            f'Example not found in decoder output (key={k}): {targets[k]}')
      ans = answers[k]
      answer = ans[_LM_DECODER_OUT_KEY]

      # postprocess model's decoder output
      prediction = self.task_inst.postprocess_fn(
          answer, example=targets[k], is_target=False)
      predictions_list.append(prediction)

      # postprocess target example for target decoder output str
      t = _get_targets_str(targets[k], self.mixture_or_task_inst)
      seqio_target = self.task_inst.postprocess_fn(
          t, example=targets[k], is_target=True)
      targets_list.append(seqio_target)

      # Mutate 'ans' dictionary which is written to disk afterwards
      ans['seqio_targets'] = seqio_target
      ans['seqio_postprocessed_predictions'] = (
          _convert_bytes_to_str(prediction))

    # Log a few examples for inspection and sanity check.
    it = iter(targets)
    for i in range(verbose_entries):
      k = next(it)
      ans = answers[k]
      e = targets[k]
      answer = ans[_LM_DECODER_OUT_KEY]
      answer_processed = self.task_inst.postprocess_fn(
          answer, example=e, is_target=False)
      target = _get_targets_str(e, self.mixture_or_task_inst)
      target_processed = self.task_inst.postprocess_fn(
          target, example=e, is_target=True)
      logging.info(
          'Example %d:\nPROMPT=%s\nMODEL=%s\nFROM %s\nLABEL=%s FROM %s.',
          i, ans['prefix'], answer_processed, answer, target_processed, target)

    # Optionally log all examples for inspection in text format
    if plain_text_output is not None:
      _log_plain_text_output(answers, plain_text_output)

    return predictions_list, targets_list

  def _build_scoring_metric_inputs_with_labels(
      self, answers: Dict[Tuple[int], NestedMap],
      verbose_entries: int) -> Tuple[Sequence[Any], Sequence[Any]]:
    """Build 1-to-1 mapped scores and targets for metrics via label matching."""
    # TODO(b/241386390): deprecate label-based matching for metrics computation.
    assert self.targets_ds
    # Prepare ground truth label data by dumping out seqio eval dataset and
    # produce a dict key-ed by tuple of `labels` token ids.
    targets = collections.defaultdict(list)
    num_examples = 0
    while (
        self._num_eval_examples < 0
        or num_examples < self._num_eval_examples
    ):
      num_examples += 1
      try:
        example = next(self.targets_iter)
        example_convert = next(self.targets_iter_converted)
      except (tf.errors.OutOfRangeError, StopIteration) as exc:
        if self._num_eval_examples > 0:
          raise StopIteration(
              'Exhausted eval data with reset_for_eval=False after'
              f' {num_examples-1} examples (batch_size={self.batch_size})'
          ) from exc
        logging.info(
            'Exhausted target eval data after %d examples', num_examples - 1
        )
        self._gen_targets_iter()
        break

      key = tuple(example_convert[_LM_LABEL_KEY])
      targets[key].append(example)

    # Construct (scoring output, seqio target) lists by joining on label tokens
    targets_list = []
    scores_list = []
    verbose_entries_idx = 0
    for k in targets:
      if k not in answers:
        raise ValueError(
            f'Example not found in eval output (key={k}): {targets[k][0]}')
      ans = answers[k]
      target = targets[k]
      score = ans[_LM_SCORE_KEY]
      prefix_targets_list = []
      for e in targets[k]:
        target_post = self.task_inst.postprocess_fn(
            target, example=e, is_target=True)
        targets_list.append(target_post)
        prefix_targets_list.append(target_post)
        scores_list.append(score)
        if verbose_entries_idx < verbose_entries:
          logging.info(
              'inputs_pretokenized=%s\ntargets_pretokenized=%s\n'
              'is_correct=%s\ntarget=%s\nscore=%s\n\n',
              e.get('inputs_pretokenized', 'None'),
              e.get('targets_pretokenized', 'None'), e.get('is_correct', 'N/A'),
              target_post, score)
          verbose_entries_idx += 1
      ans['seqio_postprocessed_targets'] = _convert_bytes_to_str(
          prefix_targets_list
      )
    eval_data_size = (
        self._num_eval_examples
        if self._num_eval_examples > 0
        else self._len_full_ds
    )
    logging.info(
        'Data %s has %s examples for computing eval metrics.',
        self.name,
        eval_data_size,
    )
    if eval_data_size != len(scores_list):
      raise ValueError(
          f'Data {self.name} expects {eval_data_size} examples for computing'
          f' eval metrics, got {len(scores_list)}.'
      )

    return scores_list, targets_list

  def _build_scoring_metric_inputs_with_enum(
      self, answers: Dict[str, NestedMap],
      verbose_entries: int) -> Tuple[Sequence[Any], Sequence[Any]]:
    assert self.use_enumeration
    targets = self._get_targets_with_enum_key()

    # Construct (scoring output, seqio target) lists by joining on enum ID
    targets_list = []
    scores_list = []
    verbose_entries_idx = 0
    for k in targets:
      if k not in answers:
        raise ValueError(
            f'Example not found in eval output (key={k}): {targets[k]}')
      ans = answers[k]
      score = ans[_LM_SCORE_KEY]
      example = targets[k]
      target_post = self.task_inst.postprocess_fn(
          targets[k]['targets'], example=example, is_target=True)
      targets_list.append(target_post)
      scores_list.append(score)
      if verbose_entries_idx < verbose_entries:
        logging.info(
            'inputs_pretokenized=%s\ntargets_pretokenized=%s\n'
            'is_correct=%s\ntarget=%s\nscore=%s\n\n',
            example.get('inputs_pretokenized', 'None'),
            example.get('targets_pretokenized', 'None'),
            example.get('is_correct', 'N/A'), target_post, score)
        verbose_entries_idx += 1
      if self.log_preprocessed_targets:
        ans['seqio_preprocessed_targets'] = _convert_bytes_to_str(example)
      ans['seqio_postprocessed_targets'] = _convert_bytes_to_str(target_post)

    return scores_list, targets_list

  def compute_metrics(
      self,
      decoder_outputs: Sequence[Tuple[str, NestedMap]],
      verbose_entries: Optional[int] = 0,
      plain_text_output: Optional[TextIO] = None,
  ) -> Sequence[Mapping[str, Union[seqio.metrics.MetricValue, float]]]:
    """Computes metrics from the given decoder outputs using predict_metric_fns.

    This method is called only on process=0 after aggregating all outputs as
    seqio task's metric_fns take in a global view of examples.

    This function basically does the following (for self.use_enumeration=False):
      1. Iterate through SeqIO task's dataset to construct both (a) the
        input prefix-based key, and (b) the task.postprocess_fn(ex['targets']),
        which is the target that is used to compute metrics.
      2. Iterate through the keys generated in (1) to "left-join" with the
        decoder_outputs mapping.
      3. Feed the prefix-key mapped list of decoder_outputs and targets through
        all task.predict_metric_fns.
      4. Optionally log a couple entries for inspection.
      5. Optionally log all entries in text format for inspection.

    When self.use_enumeration=True, we'll match based on enumeration IDs.

    For tasks with score_metric_fns, use compute_metrics_eval() below.

    Arguments:
      decoder_outputs: a sequence of (str, dict) where the 0-th arg of the tuple
        is the "prefix key", which is what is used to map between decoder
        outputs and seqio's target sequences when computing metrics. This is
        typically just the output of a model's `process_decode_out` method, but
        can also be read from disk using `io_utils.load_outputs()`.
      verbose_entries: int, how many entries to log for inspection and sanity
        checking.
      plain_text_output: optional output file to write decoder outputs in plain
        text format for easier inspection

    Returns:
      The results of predict_metric_fns computed on the decoder outputs.
      Typically a list of metrics, each element being a mapping from a string
      metric name to a float.
    """
    task = self.mixture_or_task_inst
    if not isinstance(task, seqio.Task):
      logging.warning(
          'compute_metrics is only supported for seqio.Tasks, got %s for %s.',
          type(task),
          self.name,
      )
      return []
    # If there are no seqio decode/predict metrics to compute return empty list
    if not task.predict_metric_fns:
      logging.info('no predict_metric_fns defined on task: %s', task.name)
      return []
    if is_packing_on(self.feature_converter):
      logging.error('Will not compute metrics on %s since using a '
                    'FeatureConverter with pack=True.', task.name)
      return []

    if not decoder_outputs:
      return []
    if _LM_DECODER_OUT_KEY not in decoder_outputs[0][1]:
      logging.warning(
          ('LanguageModel output format with "%s" key is expected, but '
           'the key was not found in decoder_outputs (b/244434890)'),
          _LM_DECODER_OUT_KEY)
      return []

    # Create targets artifacts if they don't exist yet
    if not self.is_targets_init:
      self._gen_targets_artifacts()

    answers = dict(decoder_outputs)
    if self.use_enumeration:
      (predictions_list, targets_list) = (
          self._build_predict_metric_inputs_with_enum(
              answers, verbose_entries, plain_text_output
          )
      )
    else:
      (predictions_list, targets_list) = (
          self._build_predict_metric_inputs_with_prefix(
              answers, verbose_entries, plain_text_output
          )
      )

    metrics = []
    for fn in task.predict_metric_fns:
      m = fn(targets_list, predictions_list)
      logging.info('Metrics: %s', m)
      metrics.append(m)

    return metrics

  def compute_metrics_eval(
      self,
      eval_outputs: Sequence[Tuple[Optional[str], NestedMap]],
      verbose_entries: int = 0
  ) -> Sequence[Mapping[str, Union[seqio.metrics.MetricValue, float]]]:
    """Computes metrics from the given eval outputs using score_metric_fns.

    This method is called only on process=0 after aggregating all outputs as
    seqio task's metric_fns take in a global view of examples.

    This function basically does the following (for use_enumeration=False):
      1. Iterate through SeqIO task's dataset to construct both (a) the 'labels'
        tokens based key, and (b) the task.postprocess_fn(ex['targets']),
        which is the target that is used to compute metrics.
      2. Iterate through the keys generated in (1) to "left-join" with the
        decoder_outputs mapping.
      3. Feed the prefix-key mapped list of decoder_outputs and targets through
        all task.predict_metric_fns.
      4. Optionally log a couple entries for inspection.

    When self.use_enumeration=True, we'll match based on enumeration IDs.

    For tasks with predict_metric_fns, use compute_metrics() above.

    Args:
      eval_outputs: A list of flattened scoring outputs. Each element is a map
        from string to NestedMap, and is expected to have keys `labels` and
        `scores`. `labels` is int32 token ids of length [T], where T is the
        sequence length.
      verbose_entries: int, how many entries to log for inspection and sanity
        checking.

    Returns:
      The results of score_metric_fns computed on the eval outputs.
      Typically a list of metrics, each element being a mapping from a string
      metric name to a float.
    """
    task = self.mixture_or_task_inst
    if not isinstance(task, seqio.Task):
      logging.warning(
          (
              'compute_metrics_eval() is only supported for seqio.Tasks, '
              'got %s for %s.'
          ),
          type(task),
          self.name,
      )
      return []
    if not task.score_metric_fns:
      logging.info('no score_metric_fns defined on task: %s', task.name)
      return []
    if is_packing_on(self.feature_converter):
      logging.error('Will not compute metrics on %s since using a '
                    'FeatureConverter with pack=True.', task.name)
      return []

    if not eval_outputs:
      return []
    if _LM_SCORE_KEY not in eval_outputs[0][1]:
      logging.warning(
          ('LanguageModel output format with "%s" key is expected, but '
           'the key was not found in eval_outputs (b/244434890)'),
          _LM_SCORE_KEY)
      return []

    # Create targets artifacts if they don't exist yet
    if not self.is_targets_init:
      self._gen_targets_artifacts()

    if self.use_enumeration:
      answers = dict([(k, v) for k, v in eval_outputs if not _is_padding(v)])
      scores_list, targets_list = self._build_scoring_metric_inputs_with_enum(
          answers, verbose_entries)
    else:
      answers = {}
      for _, ex_d in eval_outputs:
        key = tuple(ex_d[_LM_LABEL_KEY])
        answers[key] = ex_d
      scores_list, targets_list = self._build_scoring_metric_inputs_with_labels(
          answers, verbose_entries)

    metrics = []
    for fn in task.score_metric_fns:
      m = fn(targets_list, scores_list)
      logging.info('Metrics: %s', m)
      metrics.append(m)

    return metrics


###############################################################################
# Pre-canned feature converters
#
# Example usage:
#   seqio_input_params.feature_converter = LanguageModelFeatures(pack=True)
###############################################################################


class LanguageModelFeatures(seqio.DecoderFeatureConverter,
                            base_hyperparams.StrOverride):
  """A feature converter for a language model.

  The output nested map for each map contains:
    - `.ids` and `.labels`: int32, typically ids begin with BOS and labels
      ends with EOS.
    - `.paddings`: float32, 1.0 for padded positions and 0.0 for non-padded.
    - `.segment_ids` and `.segment_pos`: the ids are 1-based indices for each
      segment, pos are 0-based. They are always populated regardless of whether
      packing is enabled or not.
    - `.weights`: non-padded tokens with 0.0 weights will not be included when
      computing the loss.
    - `.inputs_indicator`: 0 for token positions that are in the target
      (for prefixLM) and 1 for token positions in the input (for
      prefixLM). This primarily affects bidirectional models where input tokens
      can attend to later input tokens.
    Example (from seqio): a packed dataset
    ```
    ds = [{"inputs": [7, 8, 5, 1], "targets": [3, 9, 1]},
          {"inputs": [8, 4, 9, 3, 1], "targets": [4, 1]}]

    task_feature_lengths = {"inputs": 7, "targets": 8}

    converted_ds = {
                   "labels": [7, 8, 5, 1, 3, 9, 1, 8, 4, 9, 3, 1, 4, 1, 0],
                      "ids": [0, 7, 8, 5, 1, 3, 9, 0, 8, 4, 9, 3, 1, 4, 0],
                  "weights": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
              "segment_pos": [0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0],
              "segment_ids": [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 0],
         "inputs_indicator": [1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0]
    }
    ```
  """

  def __init__(
      self,
      pack: bool = False,
      use_custom_packing_ops: bool = False,
      weights_on_targets_only: Optional[bool] = None,
      apply_length_check: bool = True,
      bos_id: int = 0,
      reverse_bos_padding: bool = False,
      eos_id: int = 1,
      target_has_suffix: bool = False,
      passthrough_features: Optional[
          Mapping[str, seqio.FeatureConverter.FeatureSpec]
      ] = None,
  ) -> None:
    """Args to construct a language model feature converter.

    Args:
      pack: whether to pack multiple examples into one row.
      use_custom_packing_ops: whether to use tensor2tensor custom ops for more
        efficient packing.
      weights_on_targets_only: when data has both 'inputs' and 'targets'
        features, whether to have weights on targets only. When 'None', defer to
        the underlying data, where the convention is that negative integer ids
        have weights of 0.0, otherwise all non-padding tokens have weights of
        1.0. When set to 'True', only tokens with positive integer ids AND are
        from 'targets' have weights of 1.0. When set to 'False', all non-padded
        tokens have weights of 1.0, even those with negative ids.
      apply_length_check: if True, it checks whether output feature lengths are
        less than the lengths given by `sequence_length` in the get_dataset
        function.
      bos_id: bos id for decoder inputs.
      reverse_bos_padding: Whether to reverse the bos_id padding, and pad to the
        end of labels with eos_id instead.
      eos_id: eos id for decoder inputs, only in effect if reverse_bos_padding
        true.
      target_has_suffix: targets followed by a suffix.
      passthrough_features: a mapping of features that will pass through without
        any processing.
    """
    self._pack = pack
    self._use_custom_packing_ops = use_custom_packing_ops
    self._weights_on_targets_only = weights_on_targets_only
    self._apply_length_check = apply_length_check
    self._bos_id = bos_id
    self._reverse_bos_padding = reverse_bos_padding
    self._eos_id = eos_id
    self._target_has_suffix = target_has_suffix
    super().__init__(
        loss_on_targets_only=True,
        pack=pack,
        use_custom_packing_ops=use_custom_packing_ops,
        apply_length_check=apply_length_check,
        bos_id=bos_id,
        passthrough_features=passthrough_features,
    )

  def __str__(self) -> str:
    return (
        f'{self.__class__.__name__}(pack={self._pack}, '
        f'use_custom_packing_ops={self._use_custom_packing_ops}, '
        f'weights_on_targets_only={self._weights_on_targets_only}, '
        f'apply_length_check={self._apply_length_check}, '
        f'bos_id={self._bos_id},'
        f'reverse_bos_padding={self._reverse_bos_padding},'
        f'eos_id={self._eos_id})'
    )

  @property
  def weights_on_targets_only(self) -> bool:
    return self._weights_on_targets_only

  @property
  def target_has_suffix(self) -> bool:
    return self._target_has_suffix

  def _shift_left_and_pad(self, tensor, pad_val):
    # Expand dims here so that the below code can work with 1-d tensors.
    v = tf.expand_dims(tensor, 0)
    # Make sure we keep tensor as ragged to allow for uneven concat.
    if isinstance(v, tf.Tensor):
      v = tf.RaggedTensor.from_tensor(v)

    # Append padding to the last item of every sequence.
    pad_shape = tf.concat([v.bounding_shape()[:-2], [1, 1]], axis=0)
    pad_tensor = tf.broadcast_to(pad_val, pad_shape)
    last_in_sequence = tf.concat([v[..., -1:, 1:], pad_tensor], axis=-1)
    # Concat back the newly modified final sequence item.
    v = tf.concat([v[..., :-1, :], last_in_sequence], axis=-2)
    # Un-expand outer dimension.
    v = v[0]
    return v

  def _to_pax(self, b) -> NestedMap:
    """Change data format for a Pax LanguageModel."""
    b = py_utils.NestedMap.FromNestedDict(b)
    ret = py_utils.NestedMap()
    ret.ids = b.decoder_input_tokens
    ret.labels = b.decoder_target_tokens
    non_padding = (b.decoder_loss_weights > 0)
    if hasattr(b, 'decoder_causal_attention'):
      non_padding = tf.math.logical_or(b.decoder_causal_attention > 0,
                                       non_padding)
      ret.inputs_indicator = b.decoder_causal_attention
    ret.weights = tf.cast(non_padding, dtype=tf.float32)
    ret.paddings = 1.0 - ret.weights
    if hasattr(b, 'decoder_segment_ids'):  # typical case for packed examples.
      ret.segment_ids = b.decoder_segment_ids
      ret.segment_pos = b.decoder_positions
    else:
      if self.pack:
        raise ValueError('pack must be set to False when the packing'
                         ' related tensors are derived from unpacked examples.')
      ret.segment_ids = tf.cast(1.0 - ret.paddings, tf.int32)
      pos = tf.range(ret.segment_ids.shape[0])
      ret.segment_pos = ret.segment_ids * pos
    for key in self._passthrough_features.keys():
      ret[key] = b[key]

    if self.weights_on_targets_only is None or self.weights_on_targets_only:
      # Process negative ids, which some datasets use to denote input positions
      # that ought to be ignored.
      non_negative_positions = tf.cast(ret.labels >= 0, dtype=tf.float32)
      ret.weights *= non_negative_positions
    if self.weights_on_targets_only:
      non_negative_positions = tf.cast(
          b.decoder_loss_weights > 0, dtype=tf.float32)
      ret.weights *= non_negative_positions

    if self.target_has_suffix:
      ret.weights *= tf.cast(b['target_suffix_weights'], dtype=tf.float32)

    ret.ids = tf.math.abs(ret.ids)
    ret.labels = tf.math.abs(ret.labels)

    if self._reverse_bos_padding:
      ret.ids = ret.labels
      ret.labels = self._shift_left_and_pad(ret.labels, self._eos_id)
      ret.weights = self._shift_left_and_pad(ret.weights, 0.0)
      ret.paddings = self._shift_left_and_pad(ret.paddings, 1.0)

    return ret

  def __call__(self, ds: tf.data.Dataset,
               task_feature_lengths: Mapping[str, int]) -> tf.data.Dataset:
    ds = super().__call__(ds, task_feature_lengths)
    ds = ds.map(self._to_pax)
    return ds


class PackedLanguageModelFeatures(LanguageModelFeatures):
  """A feature converter for a sequence model where examples are already packed.

  This would typically be the case for deterministic datasets where we pack
  examples while writing data to the sstables.
  Output batch has same features as LanguageModelFeatures.
  """

  def __init__(self) -> None:
    super().__init__(pack=False)

  def __call__(
      self, ds: tf.data.Dataset,
      task_feature_lengths: Mapping[str, Union[int, Sequence[int]]]
  ) -> tf.data.Dataset:
    # don't call super(), we want to bypass all validations & packing operation
    # in the base class as examples are already packed.
    ds = ds.map(self._to_pax)
    return ds


class SequenceModelFeatures(seqio.EncDecFeatureConverter):
  """A feature converter for a sequence to sequence model.

  The nested map for each map contains `.src` and `.tgt` for source and target
  respectively. Each field has the same fields as defined in
  LanguageModelFeatures above.
  """

  def __init__(self,
               pack: bool = False,
               use_custom_packing_ops: bool = False,
               bos_id: int = 0) -> None:
    self._pack = pack
    self._use_custom_packing_ops = use_custom_packing_ops
    self._bos_id = bos_id
    super().__init__(pack=pack,
                     use_custom_packing_ops=use_custom_packing_ops,
                     bos_id=bos_id)

  def __str__(self) -> str:
    return (f'{self.__class__.__name__}(pack={self._pack}, '
            f'use_custom_packing_ops={self._use_custom_packing_ops}, '
            f'bos_id={self._bos_id})')

  def _to_pax(self, b) -> NestedMap:
    """Change data format for a Pax SequenceModel."""
    b = py_utils.NestedMap.FromNestedDict(b)
    ret = py_utils.NestedMap()
    ret.src = py_utils.NestedMap()
    ret.tgt = py_utils.NestedMap()
    ret.src.ids = b.encoder_input_tokens
    # Padding positions if and only if `encoder_input_tokens` are zeros.
    # This is also how `decoder_loss_weights` is computed by the
    # EncDecFeatureConverter internally.
    ret.src.paddings = 1.0 - seqio.non_padding_position(
        ret.src.ids, dtype=tf.float32, pad_id=0)
    ret.tgt.ids = b.decoder_input_tokens
    ret.tgt.labels = b.decoder_target_tokens
    ret.tgt.paddings = 1.0 - tf.cast(b.decoder_loss_weights, tf.float32)
    ret.tgt.weights = tf.cast(b.decoder_loss_weights, dtype=tf.float32)
    # typical case where examples are either packed by feature converter,
    # or they were pre-packed in sstables (deterministic data).
    if hasattr(b, 'encoder_segment_ids'):
      ret.src.segment_ids = b.encoder_segment_ids
      ret.src.segment_pos = b.encoder_positions
      ret.tgt.segment_ids = b.decoder_segment_ids
      ret.tgt.segment_pos = b.decoder_positions
    else:
      if self.pack:
        raise ValueError('pack must be set to False when the packing'
                         ' related tensors are derived from unpacked examples.')
      ret.src.segment_ids = tf.cast(1.0 - ret.src.paddings, tf.int32)
      pos = tf.range(ret.src.segment_ids.shape[0])
      ret.src.segment_pos = ret.src.segment_ids * pos
      ret.tgt.segment_ids = tf.cast(1.0 - ret.tgt.paddings, tf.int32)
      pos = tf.range(ret.tgt.segment_ids.shape[0])
      ret.tgt.segment_pos = ret.tgt.segment_ids * pos

    if hasattr(b, 'src_task_ids'):
      ret.src.task_ids = b.src_task_ids
    if hasattr(b, 'tgt_task_ids'):
      ret.tgt.task_ids = b.tgt_task_ids
    if hasattr(b, 'task_ids'):
      ret.src.task_ids = b.task_ids
      ret.tgt.task_ids = b.task_ids
    return ret

  def __call__(self, ds: tf.data.Dataset,
               task_feature_lengths: Mapping[str, int]) -> tf.data.Dataset:
    ds = super().__call__(ds, task_feature_lengths)
    ds = ds.map(self._to_pax)
    return ds


class SequenceModelFeaturesWithTaskInfo(SequenceModelFeatures):
  """A feature converter for a sequence model with custom task level features.

  Typical use case is TaskMoE where we want some task level features to guide
  routing decision.
  """

  def _convert_example(
      self, features: Mapping[str, tf.Tensor]) -> Mapping[str, tf.Tensor]:
    d = super()._convert_example(features)
    d = dict(d)
    if 'src_task_ids' in features:
      d['src_task_ids'] = features['src_task_ids']
    if 'tgt_task_ids' in features:
      d['tgt_task_ids'] = features['tgt_task_ids']
    if 'task_ids' in features:
      d['task_ids'] = features['task_ids']
    return d

  def __call__(self, ds: tf.data.Dataset,
               task_feature_lengths: Mapping[str, int]) -> tf.data.Dataset:
    ds = self._convert_features(ds, task_feature_lengths)
    ds = ds.map(self._to_pax)  # convert features to Pax format.
    return ds


class PackedSequenceModelFeatures(SequenceModelFeatures):
  """A feature converter for a sequence model where examples are already packed.

  This would typically be the case for deterministic datasets where we pack
  examples while writing data to the sstables.
  Output batch has same features as SequenceModelFeatures.
  """

  def __init__(self) -> None:
    super().__init__(pack=False)

  def __call__(self, ds: tf.data.Dataset,
               task_feature_lengths: Mapping[str, int]) -> tf.data.Dataset:
    # don't call super(), we want to bypass all validations & packing operation
    # in the base class as examples are already packed.
    ds = ds.map(self._to_pax)
    return ds


class RemoveProvenance(seqio.PassThroughFeatureConverter):
  """Pass through but removes unused fields added by Deterministic tasks."""

  def __call__(self, ds: tf.data.Dataset,
               task_feature_lengths: Mapping[str, int]) -> tf.data.Dataset:
    del task_feature_lengths

    def filter_prov(b):
      b = py_utils.NestedMap.FromNestedDict(b)
      # Remove all unused keys starting with 'provenance', some are not
      # numerical types.
      return b.FilterKeyVal(lambda k, _: not k.startswith('provenance'))

    ds = ds.map(filter_prov)
    return ds


class UnflattenAndRemoveProvenance(RemoveProvenance):
  """Un-flattens and removes provenance fields added by Deterministic tasks."""

  def __call__(self, ds: tf.data.Dataset,
               task_feature_lengths: Mapping[str, int]) -> tf.data.Dataset:
    ds = ds.map(seqio.unflatten_dict)
    return super().__call__(ds, task_feature_lengths)


class MetricType(enum.Enum):
  """SeqIO metric types."""
  PREDICT = 1  # decoding-based metrics
  SCORE = 2  # eval / target scoring-based metrics


def get_eval_hparams_for_seqio(
    task_or_mixture_name: Union[str, seqio.Task, seqio.Mixture],
    batch_size: int,
    feature_lengths: Mapping[str, int],
    seed: int,
    metric_type: MetricType,
    split_name: Union[str, Callable[[str], str]] = 'validation',
    feature_converter: Optional[
        Union[
            pax_fiddle.Config[seqio.FeatureConverter],
            seqio.FeatureConverter,
        ]
    ] = None,
    num_infeed_hosts: int = 0,
    use_enumeration: bool = True,
    use_cached: bool = False,
    shuffle: bool = None,
    require_metric_fns: bool = True,
    eval_metrics_retain_task_features: bool = False,
    check_split_exists: bool = False,
    eval_loop_num_batches: Optional[int] = None,
    repeat: Optional[bool] = None,
    reset_for_eval: bool = True,
    pass_entire_feature_lengths: bool = False,
) -> list[pax_fiddle.Config[SeqIOInput]]:
  """Returns a list of SeqIOInput configs for SeqIO Task/Mixture for eval.

  This is the easiest way to configure eval hparams in datasets() (for scoring
  metrics) and decoder_datasets() (for prediction metrics) from SeqIO
  Task/Mixture name and a few required params. A SeqIOInput config is
  created for each Task, i.e. Mixtures are split into component Tasks, as each
  Task is evaled separately.

  Example usage:
  >>> def get_ulm_stage_a_eval_bundle():
  >>>   mixture_name = 'ulm_eval:stage_a_few_shot_prompting_bundle:v0.0'
  >>>   batch_size = 32
  >>>   feature_lengths={'inputs': 1024, 'targets': 256}
  >>>   seed = 75303
  >>>   return get_eval_hparams_for_seqio(
  >>>       mixture_name, batch_size, feature_lengths, MetricType.PREDICT, seed)

  Args:
    task_or_mixture_name: SeqIO Task/Mixture instance or name to run eval on.
    batch_size: The global eval batch size.
    feature_lengths: A dict of feature lenghs to trim sequences to, e.g.
      {'inputs': 1024, 'targets': 256}
    seed: A seed to use for loading the dataset from SeqIO. Must be set for
      consistency, especially for few-shot tasks, where the seed affects the
      prompts selected.
    metric_type: The type of metrics to return hparams for. Configure PREDICT
      type in decoder_datasets() and SCORE type in datasets().
    split_name: The split to use for evaluation, defaults to 'validation'. This
      may optionally be a callable that takes a str task name (i.e. a member of
      the provided mixture) and returns the name of the split to use for each
      task.
    feature_converter: The SeqIO FeatureConverter to use to transform data,
      defaults to seqio_input.LanguageModelFeatures with packing disabled
    num_infeed_hosts: Usually set to jax.process_count(). Implementation must
      ensure that the data is sharded into these many shards. If
      num_infeed_hosts is 0, it will be given a default value by the trainer; if
      it is still not set during __init__, a value of 1 will be used.
    use_enumeration: whether to use enumeration in both batch generation
      (get_next()) and metrics computation. For details, see SeqIOInput attrs.
    use_cached: whether to use cached data.
    shuffle: whether to shuffle data.
    require_metric_fns: whether to require that SeqIO tasks have metric_fns.
    eval_metrics_retain_task_features: retain the provided feature lengths.
    check_split_exists: If set, checks for `split_name` existing as a split in
      the SeqIO Task.  Note that for certain TFDS backed tasks, which don't have
      splits specified, this can cause file operations.
    eval_loop_num_batches: Num of batches to process per eval loop. This value
      is ignored if reset_for_eval is set True. If None, eval will run on the
      entire dataset.
    repeat: Whether to repeat the data.
    reset_for_eval: If set, eval will continue until tf.errors.OutOfRange is
      raised, and reset() will called for each eval.
    pass_entire_feature_lengths: If true, updates the task_feature_lengths with
      the remaining feature lengths in feature_lengths.
  """
  if isinstance(task_or_mixture_name, (seqio.Task, seqio.Mixture)):
    task_or_mixture = task_or_mixture_name
    task_or_mixture_name = task_or_mixture.name
  elif isinstance(task_or_mixture_name, str):
    task_or_mixture = seqio.get_mixture_or_task(task_or_mixture_name)
  else:
    raise TypeError('Expected `task_or_mixture_name` to be a string, a Task, '
                    f'or a Mixture; got {task_or_mixture_name!r}')

  if not feature_converter:
    weights_on_targets_only = True if metric_type is MetricType.SCORE else False
    feature_converter = pax_fiddle.Config(
        LanguageModelFeatures,
        pack=False,
        weights_on_targets_only=weights_on_targets_only,
    )
  elif not isinstance(feature_converter, pax_fiddle.Config):
    logging.warning('feature_converter should ideally be a pax_fiddle.Config')
  p = pax_fiddle.Config(
      SeqIOInput,
      name=task_or_mixture_name,
      mixture_or_task=task_or_mixture,
      feature_converter=feature_converter,
      is_training=False,
      reset_for_eval=reset_for_eval,
      shuffle=shuffle,
      batch_size=batch_size,
      num_infeed_hosts=num_infeed_hosts,
      input_random_seed=seed,
      eval_loop_num_batches=eval_loop_num_batches,
      repeat=repeat,
  )

  # Set task_feature_lengths.targets depending on eval vs decode metrics.
  if metric_type is MetricType.PREDICT:
    p.eval_metrics_targets_length = feature_lengths['targets']
    p.task_feature_lengths = {
        'inputs': feature_lengths['inputs'],
        'targets': 1,  # we don't want any targets, except an EOS
    }
  elif metric_type is MetricType.SCORE:
    if eval_metrics_retain_task_features:
      p.task_feature_lengths = feature_lengths
    else:
      p.task_feature_lengths = {
          'inputs': feature_lengths['inputs'],
          'targets': feature_lengths['targets'],
      }
  else:
    raise ValueError(f'unsupported metric type: {metric_type}')

  if pass_entire_feature_lengths:
    remaining_feature_lengths = dict(feature_lengths)
    del remaining_feature_lengths['inputs']
    del remaining_feature_lengths['targets']
    p.task_feature_lengths.update(remaining_feature_lengths)

  # Split hparams per tasks and filter by metric type.
  # First, the mixture_or_task itself may not be deepcopiable, so clear it
  # before calling clone below.
  cloneable_p = copy.copy(p).set(mixture_or_task=None)
  tasks: Sequence[seqio.Task] = seqio.get_subtasks(task_or_mixture)
  hparams = []
  for task in tasks:
    hp = cloneable_p.clone().set(
        mixture_or_task=task,
        name=task.name,
        use_enumeration=use_enumeration,
        use_cached=use_cached,
    )
    # Allow selecting split based on `Callable` `split_name` if mixture contains
    # tasks with varying splits.
    hp.split_name = select_split(task.name, split_name)
    assert isinstance(hp.split_name, str)

    if check_split_exists and hp.split_name not in task.splits:
      logging.warning(
          'task %s does not have split named `%s` and will not be evaluated.',
          task.name,
          hp.split_name,
      )
      continue
    if require_metric_fns:
      if not task.metric_fns:
        logging.warning(
            (
                'task %s is not being added to hparam list as it has no'
                ' metric_fns defined. If you want the task evaluated'
                ' regardless, set require_metric_fns=False'
            ),
            task.name,
        )
      if metric_type is MetricType.PREDICT:
        if task.predict_metric_fns:
          hparams.append(hp)
        else:
          logging.warning(
              (
                  'task %s is not being added to hparam list as it has no'
                  ' predict_metric_fns defined although'
                  ' metric_type=MetricType.Predict. '
              ),
              task.name,
          )
      elif metric_type is MetricType.SCORE:
        if task.score_metric_fns:
          hparams.append(hp)
        else:
          logging.warning(
              (
                  'task %s is not being added to hparam list as it has no'
                  ' score_metric_fns defined although'
                  ' metric_type=MetricType.Score. '
              ),
              task.name,
          )
    else:
      # Show PPLX
      hparams.append(hp)
  return hparams
