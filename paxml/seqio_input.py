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

"""SeqIO input for Pax."""

from __future__ import annotations

import collections
import enum
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

from absl import logging
from jax.experimental import multihost_utils
import jax.numpy as jnp
import numpy as np
from praxis import base_hyperparams
from praxis import base_input
from praxis import py_utils
from praxis import pytypes
import seqio
import tensorflow.compat.v2 as tf

sub_config_field = base_hyperparams.sub_config_field
NestedMap = py_utils.NestedMap
NestedJTensor = pytypes.NestedJTensor
ParamsT = pytypes.HParamsT


def _update_keys(answers: Dict[str, Any], targets: Mapping[str, Any],
                 task_name: str) -> None:
  """Insert into answers the keys that only partially match."""
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
  """Retrieves the targets string from the given example."""
  pretokenized_target_field_name = 'targets_pretokenized'
  target_field_name = 'targets'
  if pretokenized_target_field_name in example:
    target = example[pretokenized_target_field_name]
  else:
    target = task.output_features[target_field_name].vocabulary.decode(
        [int(x) for x in example[target_field_name]])
  if isinstance(target, bytes):
    target = target.decode('utf-8')
  return target


class SeqIOInput(base_input.BaseInput):
  """An adaptor for getting SeqIO data."""

  class DeterministicInputParams(base_hyperparams.BaseHyperParams):
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

  class HParams(base_input.BaseInput.HParams):
    """Hyperparameters for this input class.

    Attributes:
       mixture_name: Required string. The name for a SeqIO task or mixture. User
         must import the module that defines this task/mixture.
       split_name: Required string. The name for the split of data to get.
         Usually "train" or "validation" or "test".
       deterministic_input: If deterministic input is intended, users should set
         this to enable internal validations to ensure that deterministic input
         is indeed used.
       task_feature_lengths: Required. Of type Mapping[str, int]. The keys are
         the features on the original SeqIO task/mixture, typically "inputs" and
         "targets". The values are corresponding sequence lengths. Examples
         exceeding the sequence lengths are truncated.
       feature_converter: An instance of a seqio.FeatureConverter subclass. This
         is used to convert the data from its original format to the format
         expected by the model, e.g. instead of "targets" we have "ids" or
         "labels" or "paddings". This also implements any necessary padding or
         packing on the data.
       shuffle: Whether to shuffle the data. Note that None means this feature
         is decided automatically: True for and only for non-deterministic
         training data, otherwise False. Users can override this by setting this
         explicitly.
       repeat: Whether to repeat the data. Note that None means this feature is
         decided automatically: True for and only for non-deterministic training
         data, otherwise False. Users can override this by setting this field
         explicitly.
       use_cached: Whether to read from the cached directory, if supported by
         the underlying SeqIO task/mixture. Users can set to False to test out
         data changes before the cache is applied.
       eval_auto_pad: Only used when p.is_training=False. Automatically pad the
         data to multiples of global batch size, using the first example in the
         data. Padded entries will have batch_input.eval_sample_weight == 0.0.
       deterministic_input_start_index: Params to compute the starting example
         index. Used only if the data is a deterministic input, otherwise
         ignored.
       eval_metrics_targets_length: typically when used in eval, the data would
         not contain any targets. This value overrides the task feature lengths
         for targets when processing the targets as ground truths to compute
         eval metrics. It has no effect on get_next(), but only affects
         compute_metrics().
    """
    # Required params.
    mixture_name: Optional[str] = None
    split_name: Optional[str] = None
    deterministic_input: bool = False
    task_feature_lengths: Optional[Mapping[str, int]] = None
    feature_converter: seqio.FeatureConverter = None
    # Optional params.
    shuffle: Optional[bool] = None
    repeat: Optional[bool] = None
    use_cached: bool = False
    eval_auto_pad: bool = True
    # Params to adjust the starting example index for deterministic input.
    # Implementation note: `SingleTask` is not defined in the interpreter
    # context here, so we need to wrap it in a lambda which will look it up from
    # the global scope later.
    deterministic_input_start_index: SeqIOInput.DeterministicInputParams = (
        sub_config_field(lazy_ref=lambda: SeqIOInput.DeterministicInputParams))
    eval_metrics_targets_length: Optional[int] = None

  def __init__(self, hparams: ParamsT) -> None:
    if not hparams.name:
      hparams.name = f'{hparams.mixture_name}_{hparams.split_name}'
    super().__init__(hparams)
    self._validate_hparams()
    self._dataset = self._get_dataset()
    self._iter = self._dataset.as_numpy_iterator()

  def _validate_deterministic(self):
    if self.hparams.deterministic_input:
      raise ValueError('deterministic_input is not supported')

  def _validate_hparams(self):
    p = self.hparams
    if p.is_training and p.split_name != 'train':
      logging.warn(
          'SeqIO input hparams p.is_training=True but p.split_name is '
          'not "train" but p.split_name=%s', p.split_name)
    self._mixture_or_task = seqio.get_mixture_or_task(p.mixture_name)
    shard_info = seqio.ShardInfo(
        index=p.infeed_host_index, num_shards=p.num_infeed_hosts)
    logging.info('ShardInfo: shard_id: %d, num_shards: %d, ',
                 shard_info.index, shard_info.num_shards)
    self._shard_info = shard_info
    self._validate_deterministic()

  @property
  def is_deterministic(self) -> bool:
    """Indicates whether this SeqIOInput is deterministic or not."""
    return False

  @property
  def shuffle(self) -> bool:
    """Indicates whether this SeqIOInput shuffles the data or not."""
    p = self.hparams
    if p.shuffle is None:
      return p.is_training and not self.is_deterministic
    return p.shuffle

  @property
  def repeat(self) -> bool:
    """Indicates whether this SeqIOInput repeats the data or not."""
    p = self.hparams
    if p.repeat is None:
      return p.is_training and not self.is_deterministic
    return p.repeat

  def _get_dataset(self):
    p = self.hparams
    logging.info(
        "Initializing dataset for task '%s' with a per host batch size of %d "
        'and a seed of %s', p.mixture_name, p.batch_size, p.input_random_seed)
    ds = seqio.get_dataset(
        mixture_or_task_name=p.mixture_name,
        task_feature_lengths=p.task_feature_lengths,
        dataset_split=p.split_name,
        shuffle=self.shuffle,
        num_epochs=-1 if self.repeat else 1,
        feature_converter=p.feature_converter,
        shard_info=self._shard_info,
        use_cached=p.use_cached,
        seed=p.input_random_seed)
    ds = self._pad_to_batch_size(ds)
    ds = ds.batch(
        p.batch_size, drop_remainder=True, num_parallel_calls=tf.data.AUTOTUNE)
    return ds

  def get_next(self) -> NestedJTensor:
    return next(self._iter)

  def reset(self) -> None:
    self._iter = self._dataset.as_numpy_iterator()

  def ids_to_strings(self,
                     ids: pytypes.NpTensor,
                     lengths: pytypes.NpTensor,
                     key: Optional[str] = None) -> Sequence[str]:
    features = self._mixture_or_task.output_features
    if key is None:
      vocab = features['targets'].vocabulary
    elif key not in ['src', 'tgt']:
      raise ValueError("arg 'key' must be one of [None, 'src', 'tgt'], got "
                       f'key={key}.')
    else:
      vocab = (
          features['targets'].vocabulary
          if key == 'tgt' else features['inputs'].vocabulary)
    if lengths is None:
      lengths = [ids.shape[1]] * ids.shape[0]
    ret = []
    for i in range(ids.shape[0]):
      length = lengths[i]
      row = ids[i, :length].tolist()
      ret.append(vocab.decode(row))
    return ret

  def _get_global_ds(self) -> tf.data.Dataset:
    p = self.hparams
    return seqio.get_dataset(
        mixture_or_task_name=p.mixture_name,
        task_feature_lengths=p.task_feature_lengths,
        dataset_split=p.split_name,
        shuffle=False,
        num_epochs=1,
        feature_converter=p.feature_converter,
        shard_info=None,  # we get the full dataset
        use_cached=p.use_cached,
        seed=p.input_random_seed)

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
      b.eval_sample_weights = 0.0
      return b

    ds = ds.map(_add_weight)
    p = self.hparams
    if p.is_training or not p.reset_for_eval or not p.eval_auto_pad:
      return ds

    # p.reset_for_eval=True: We are running eval over exactly one epoch.
    # We explicitly cache the entire epoch (in memory) to ensure that it is the
    # same across different iterations. Note that this is needed not only
    # because of ordering, but for data contents as well. For instance, with
    # seqio's FewshotDataSource preprocessing, some training data is part of the
    # prompt. These training data may be shuffled with
    # `reshuffle_each_iteration=True`. In general, there is no guarantee that
    # the underlying eval dataset stays unchanged across different iterations
    # of epochs.
    ds = ds.cache()
    local_num = len(list(ds.as_numpy_iterator()))
    local_num_batches = (local_num + p.batch_size - 1) // p.batch_size
    # Find the max number of batches required across all Jax processes.
    num_batches_all = multihost_utils.process_allgather(
        jnp.array([local_num_batches]), tiled=False)
    num_batches = int(jnp.max(num_batches_all))
    pad_num = num_batches * p.batch_size - local_num
    assert pad_num >= 0
    logging.info(
        'Eval data has %d local entries, padding now with '
        '%d extra entries to get %d batches.', local_num, pad_num, num_batches)
    # Repeat a random example to make the last batch full.
    pad_ds = self._get_one_example_ds(ds).map(_add_pad).repeat(pad_num)
    return ds.concatenate(pad_ds)

  def compute_metrics(
      self,
      decoder_outputs: Sequence[Tuple[str, NestedMap]],
      verbose_entries: Optional[int] = 0
  ) -> Sequence[Mapping[str, Union[seqio.metrics.MetricValue, float]]]:
    """Computes metrics from the given decoder outputs using predict_metric_fns.

    For tasks with score_metric_fns, use compute_metrics_eval() below.

    Args:
      decoder_outputs: A list of the entire decoder output on this dataset. This
        is typically obtained by reading them from disk from
        io_utils.load_outputs().
      verbose_entries: int, how many entries to log for inspection and sanity
        checking.

    Returns:
      The results of predict_metric_fns computed on the decoder outputs.
      Typically a list of metrics, each element being a mapping from a string
      metric name to a float.
    """
    # TODO(b/236078932): integrate with seqio evaluator.
    # Current known limitations: assumes LanguageModel decoder output format,
    # specifically the prediction string has key='decoded_substr'.
    # Also assumes inputs and targets field names are 'inputs' and 'targets'.
    p = self.hparams
    inputs_length = p.task_feature_lengths['inputs']
    targets_length = p.eval_metrics_targets_length
    task = self._mixture_or_task
    if not isinstance(task, seqio.Task):
      raise ValueError('compute_metrics() is only supported for seqio.Tasks, '
                       f'got {type(self._mixture_or_task)} for {p.name}.')
    if not task.predict_metric_fns:
      return []
    targets_ds = task.get_dataset(
        sequence_length={
            'inputs': inputs_length,
            'targets': targets_length,
        },
        split=p.split_name,
        shuffle=False,
        num_epochs=1,
        seed=p.input_random_seed,
        use_cached=p.use_cached)
    targets = collections.defaultdict(list)
    examples = collections.defaultdict(list)
    for e in targets_ds.as_numpy_iterator():
      # Note that we intentionally do not use 'inputs_pretokenized' here because
      # it might be very different from the round-trip results below, which
      # wouldn't match with the keys we get from the model inference path.
      key = self.ids_to_strings(
          e['inputs'][None, :],
          lengths=[inputs_length],
          key='src')[0]
      t = _get_targets_str(e, self._mixture_or_task)
      targets[key].append(task.postprocess_fn(t, example=e, is_target=True))
      examples[key].append(e)

    answers = dict(decoder_outputs)
    _update_keys(answers, targets, task.name)

    predictions_list = []
    targets_list = []
    for k in targets:
      ans = answers[k]
      answer = ans['decoded_substr']
      # this line mutates 'decoder_outputs' which is written to disk afterwards
      ans['seqio_targets'] = targets[k]
      for target, e in zip(targets[k], examples[k]):
        targets_list.append(target)
        prediction = task.postprocess_fn(answer, example=e, is_target=False)
        predictions_list.append(prediction)

    eval_data_size = len(list(targets_ds.as_numpy_iterator()))
    logging.info('Data %s has %s examples for computing eval metrics.', p.name,
                 eval_data_size)
    if eval_data_size != len(predictions_list):
      raise ValueError(
          f'Data {p.name} expects {eval_data_size} examples for computing eval'
          f' metrics, got {len(predictions_list)}.')
    metrics = []
    for fn in task.predict_metric_fns:
      m = fn(targets_list, predictions_list)
      logging.info('Metrics: %s', m)
      metrics.append(m)

    # Log a few examples for inspection and sanity check.
    it = iter(targets)
    for i in range(verbose_entries):
      k = next(it)
      ans = answers[k]
      e = examples[k][0]
      answer = ans['decoded_substr']
      answer_processed = task.postprocess_fn(answer, example=e, is_target=False)
      target = _get_targets_str(e, self._mixture_or_task)
      target_processed = task.postprocess_fn(target, example=e, is_target=True)
      logging.info(
          'Example %d:\nPROMPT=%s\nMODEL=%s FROM %s\nLABEL=%s FROM %s.', i, k,
          answer_processed, answer, target_processed, target)
    return metrics

  def compute_metrics_eval(
      self,
      eval_outputs: Sequence[Dict[str, py_utils.JTensor]],
      verbose_entries: int = 0
  ) -> Sequence[Mapping[str, Union[seqio.metrics.MetricValue, float]]]:
    """Computes metrics from the given eval outputs using score_metric_fns.

    For tasks with predict_metric_fns, use compute_metrics() above.

    Args:
      eval_outputs: A list of per_example_outputs. Each element is a nested map
        from string to JTensor, and is expected to have keys `labels` and
        `scores`. `labels` is int32 token ids and should be convertible to shape
        [B, T], and `scores` is float and should be convertible to shape [B],
        where B is batch size and T is sequence length.
      verbose_entries: int, how many entries to log for inspection and sanity
        checking.

    Returns:
      The results of score_metric_fns computed on the eval outputs.
      Typically a list of metrics, each element being a mapping from a string
      metric name to a float.
    """
    p = self.hparams
    task = self._mixture_or_task
    if not isinstance(task, seqio.Task):
      raise ValueError('compute_metrics() is only supported for seqio.Tasks, '
                       f'got {type(self._mixture_or_task)} for {p.name}.')
    if not task.score_metric_fns:
      return []
    targets_ds = task.get_dataset(
        sequence_length=p.task_feature_lengths,
        split=p.split_name,
        shuffle=False,
        num_epochs=1,
        seed=p.input_random_seed,
        use_cached=p.use_cached)
    converted_targets_ds = p.feature_converter(targets_ds,
                                               p.task_feature_lengths)
    targets = collections.defaultdict(list)
    for example, converted_example in zip(
        targets_ds.as_numpy_iterator(),
        converted_targets_ds.as_numpy_iterator()):
      key = tuple(converted_example['labels'])
      targets[key].append(example)

    answers = dict()
    for element in eval_outputs:
      labels = np.asarray(element['labels'])
      scores = np.asarray(element['scores'])
      if len(labels.shape) > 2:
        labels = np.reshape(labels, [-1, labels.shape[-1]])
      if len(scores.shape) > 1:
        scores = np.reshape(scores, [-1])
      for i in range(labels.shape[0]):
        key = tuple(labels[i, :])
        answers[key] = float(scores[i])

    targets_list = []
    scores_list = []
    verbose_entries_idx = 0
    for k in targets:
      if k not in answers:
        raise ValueError(f'Example not found in eval output: {targets[k][0]}')
      target = targets[k]
      score = answers[k]
      for e in targets[k]:
        target_post = task.postprocess_fn(target, example=e, is_target=True)
        targets_list.append(target_post)
        scores_list.append(score)
        if verbose_entries_idx < verbose_entries:
          logging.info(
              'inputs_pretokenized=%s\ntargets_pretokenized=%s\n'
              'is_correct=%s\ntarget=%s\nscore=%s\n\n',
              e['inputs_pretokenized'], e['targets_pretokenized'],
              e['is_correct'], target_post, score)
          verbose_entries_idx += 1

    eval_data_size = len(list(targets_ds.as_numpy_iterator()))
    logging.info('Data %s has %s examples for computing eval metrics.', p.name,
                 eval_data_size)
    if eval_data_size != len(scores_list):
      raise ValueError(
          f'Data {p.name} expects {eval_data_size} examples for computing eval'
          f' metrics, got {len(scores_list)}.')
    metrics = []
    for fn in task.score_metric_fns:
      m = fn(targets_list, scores_list)
      logging.info('Metrics: %s', m)
      metrics.append(m)
    return metrics

# Below we provide some pre-canned feature converters. Example usage:
#   seqio_input_params.feature_converter = LanguageModelFeatures(pack=True)


class LanguageModelFeatures(seqio.DecoderFeatureConverter):
  """A feature converter for a language model.

  The nested map for each map contains:
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

  def __init__(self,
               pack: bool = False,
               use_custom_packing_ops: bool = False,
               weights_on_targets_only: Optional[bool] = None) -> None:
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
    """
    self._weights_on_targets_only = weights_on_targets_only
    super().__init__(
        loss_on_targets_only=True,
        pack=pack,
        use_custom_packing_ops=use_custom_packing_ops,
    )

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

    if self._weights_on_targets_only is None or self._weights_on_targets_only:
      # Process negative ids, which some datasets use to denote input positions
      # that ought to be ignored.
      non_negative_positions = tf.cast(ret.labels >= 0, dtype=tf.float32)
      ret.weights *= non_negative_positions
    if self._weights_on_targets_only:
      non_negative_positions = tf.cast(
          b.decoder_loss_weights > 0, dtype=tf.float32)
      ret.weights *= non_negative_positions

    ret.ids = tf.math.abs(ret.ids)
    ret.labels = tf.math.abs(ret.labels)
    return ret

  def __call__(self, ds: tf.data.Dataset,
               task_feature_lengths: Mapping[str, int]) -> tf.data.Dataset:
    ds = super().__call__(ds, task_feature_lengths)
    ds = ds.map(self._to_pax)
    return ds


class SequenceModelFeatures(seqio.EncDecFeatureConverter):
  """A feature converter for a sequence to sequence model.

  The nested map for each map contains `.src` and `.tgt` for source and target
  respectively. Each field has the same fields as defined in
  LanguageModelFeatures above.
  """

  def __init__(self, pack: bool = False) -> None:
    super().__init__(pack=pack, use_custom_packing_ops=False)

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
    return ret

  def __call__(self, ds: tf.data.Dataset,
               task_feature_lengths: Mapping[str, int]) -> tf.data.Dataset:
    ds = super().__call__(ds, task_feature_lengths)
    ds = ds.map(self._to_pax)
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


class MetricType(enum.Enum):
  """SeqIO metric types."""
  PREDICT = 1
  SCORE = 2


def get_eval_hparams_for_seqio(
    task_or_mixture_name: str,
    batch_size: int,
    feature_lengths: Mapping[str, int],
    seed: int,
    metric_type: MetricType,
    split_name: str = 'validation',
    feature_converter: Optional[seqio.FeatureConverter] = None
) -> Sequence[SeqIOInput.HParams]:
  """Returns a list of `SeqIOInput.HParams` for SeqIO Task/Mixture for eval.

  This is the easiest way to configure eval hparams in datasets() (for scoring
  metrics) and decoder_datasets() (for prediction metrics) from SeqIO
  Task/Mixture name and a few required params. A `SeqIOInput.HParams` obj is
  created for each Task, i.e. Mixtures are split into component Tasks, as each
  Task is evaled separately.

  Example usage:
  def get_ulm_stage_a_eval_bundle():
    mixture_name = 'ulm_eval:stage_a_few_shot_prompting_bundle:v0.0'
    batch_size = 32
    feature_lengths={'inputs': 1024, 'targets': 256}
    seed = 75303
    return create_eval_hparams_for_seqio(
        mixture_name, batch_size, feature_lengths, MetricType.PREDICT, seed)

  Args:
    task_or_mixture_name: SeqIO Task/Mixture name to run eval on.
    batch_size: The global eval batch size.
    feature_lengths: A dict of feature lenghs to trim sequences to, e.g.
      {'inputs': 1024, 'targets': 256}
    seed: A seed to use for loading the dataset from SeqIO. Must be set for
      consistency, especially for few-shot tasks, where the seed affects the
      prompts selected.
    metric_type: The type of metrics to return hparams for. Configure PREDICT
      type in decoder_datasets() and SCORE type in datasets().
    split_name: The split to use for evaluation, defaults to 'validation'.
    feature_converter: The SeqIO FeatureConverter to use to transform data,
      defaults to seqio_input.LanguageModelFeatures with packing disabled
  """
  if not feature_converter:
    weights_on_targets_only = True if metric_type is MetricType.SCORE else False
    feature_converter = LanguageModelFeatures(
        pack=False, weights_on_targets_only=weights_on_targets_only)
  p = SeqIOInput.HParams(
      name=task_or_mixture_name,
      mixture_name=task_or_mixture_name,
      split_name=split_name,
      feature_converter=feature_converter,
      is_training=False,
      eval_loop_num_batches=None,
      reset_for_eval=True,
      batch_size=batch_size,
      input_random_seed=seed)
  if metric_type is MetricType.PREDICT:
    p.eval_metrics_targets_length = feature_lengths['targets']
  if metric_type is MetricType.PREDICT:
    targets_feature_length = 1  # we don't want any targets, except an EOS
  else:
    targets_feature_length = feature_lengths['targets']
  p.task_feature_lengths = {
      'inputs': feature_lengths['inputs'],
      'targets': targets_feature_length
  }
  # Split hparams per taska and filter by metric type.
  tasks = seqio.get_subtasks(seqio.get_mixture_or_task(task_or_mixture_name))
  params = []
  for task in tasks:
    hp = p.clone().set(mixture_name=task.name, name=task.name)
    if task.predict_metric_fns and metric_type is MetricType.PREDICT:
      params.append(hp)
    if task.score_metric_fns and metric_type is MetricType.SCORE:
      params.append(hp)
  return params
