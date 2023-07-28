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

"""Tests for seqio_input."""

from typing import Any, Callable, Mapping, Optional, Sequence, Tuple
from unittest import mock
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from paxml import seqio_input
from praxis import base_hyperparams
from praxis import pax_fiddle
from praxis import py_utils
from praxis import test_utils as flax_test_utils
import seqio
import tensorflow.compat.v2 as tf

instantiate = base_hyperparams.instantiate
NestedMap = py_utils.NestedMap

SHARD_INDEX_KEY = seqio_input.SHARD_INDEX_KEY
NUM_SHARDS_KEY = seqio_input.NUM_SHARDS_KEY
INDEX_WITHIN_SHARD_KEY = seqio_input.INDEX_WITHIN_SHARD_KEY


def _register_task(
    task_name: str,
    ds: tf.data.Dataset,
    output_feature_names: Sequence[str] = ('inputs', 'targets'),
    add_eos: bool = True,
) -> None:
  """Register a dummy task."""
  if add_eos:
    preprocessors = [seqio.preprocessors.append_eos_after_trim]
  else:
    preprocessors = []
  seqio.TaskRegistry.add(
      task_name,
      source=seqio.FunctionDataSource(
          dataset_fn=lambda split, shuffle_files, seed=0: ds,
          splits=['train', 'validation']),
      preprocessors=preprocessors,
      output_features={
          feat: seqio.Feature(
              seqio.test_utils.sentencepiece_vocab(),
              add_eos=add_eos)
          for feat in output_feature_names
      },
      metric_fns=[])


# A score metric function. It must two args: `targets` and `predictions`. See:
# https://github.com/google/seqio/blob/90c76914ed13fcce53f00966b824e45fb266b973/seqio/dataset_providers.py#L817-L821
def _dummy_metric(targets: Sequence[str],
                  predictions: Sequence[str]) -> Mapping[str, float]:
  return {'accuracy': targets + predictions}


# A score metric function. It must two args: `targets` and `scores`. See:
# https://github.com/google/seqio/blob/90c76914ed13fcce53f00966b824e45fb266b973/seqio/dataset_providers.py#L817-L821
def _dummy_score_metric(targets: Sequence[Any],
                        scores: Sequence[float]) -> Mapping[str, float]:
  del targets
  return {'total_score': sum(scores)}


def _register_dummy_task(
    task_name: str, dataset_fn: Callable[[str, bool, Optional[int]],
                                         tf.data.Dataset]
) -> seqio.Task:
  """Register a dummy task for testing eval metrics."""
  output_feature_names = ('inputs', 'targets')
  return seqio.TaskRegistry.add(
      task_name,
      source=seqio.FunctionDataSource(
          dataset_fn=dataset_fn, splits=['train', 'validation']),
      preprocessors=[seqio.preprocessors.append_eos],
      postprocess_fn=None,
      output_features={
          # Mock the sentencepiece vocabulary.
          feat: seqio.Feature(mock.Mock(eos_id=True))
          for feat in output_feature_names
      },
      metric_fns=[_dummy_metric, _dummy_score_metric])


class InputTest(flax_test_utils.TestCase, seqio.test_utils.FakeTaskTest):

  @parameterized.parameters(
      dict(
          inputs1=[7, 8, 5, 6, 9, 4, 3],
          targets1=[13, 19],
          inputs2=[22, 23],
          targets2=[35],
          weights=[
              [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0.],
              [1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0.],
          ],
      ),
      dict(
          inputs1=[7, -8, 5, 6, 9, -4, -3],
          targets1=[-13, 19],
          inputs2=[-22, -23],
          targets2=[-35],
          weights=[
              [1., 0., 1., 1., 1., 0., 1., 0., 1., 1., 0., 0.],
              [0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
          ],
      ),
  )
  def test_input_inputs_target_lm(self, inputs1, targets1, inputs2, targets2,
                                  weights):
    name = 'mixture_name'
    x = [{
        'inputs': inputs1,
        'targets': targets1,
    }, {
        'inputs': inputs2,
        'targets': targets2,
    }]
    ds = seqio.test_utils.create_default_dataset(x)
    _register_task(name, ds)

    p = pax_fiddle.Config(seqio_input.SeqIOInput)
    p.mixture_name = name
    p.split_name = 'validation'
    p.task_feature_lengths = {'inputs': 7, 'targets': 5}
    p.feature_converter = seqio_input.LanguageModelFeatures(pack=False)
    p.batch_size = 2
    p.is_training = False
    inp = instantiate(p)
    batch = inp.get_next()
    # Note how the first example's inputs is truncated.
    self.assertArraysEqual(
        batch.ids,
        np.array([[0, 7, 8, 5, 6, 9, 4, 1, 13, 19, 1, 0],
                  [0, 22, 23, 1, 35, 1, 0, 0, 0, 0, 0, 0]],
                 dtype=np.int32))
    self.assertArraysEqual(
        batch.labels,
        np.array([[7, 8, 5, 6, 9, 4, 1, 13, 19, 1, 0, 0],
                  [22, 23, 1, 35, 1, 0, 0, 0, 0, 0, 0, 0]],
                 dtype=np.int32))
    self.assertArraysEqual(
        batch.paddings,
        np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1.],
                  [0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1.]],
                 dtype=np.float32))
    self.assertArraysEqual(
        batch.segment_ids,
        np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                  [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]],
                 dtype=np.int32))
    self.assertArraysEqual(
        batch.segment_pos,
        np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0],
                  [0, 1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0]],
                 dtype=np.int32))
    self.assertArraysEqual(batch.weights, np.array(weights, dtype=np.float32))
    # decoding to string terminates at EOS, which is 1 here.
    self.assertEqual(
        inp.ids_to_strings(
            batch.ids, lengths=np.array([12, 12], dtype=np.int32)),
        ['oiasle', 'nr'],
    )
    # With eval, the dataset is not repeated.
    with self.assertRaises(StopIteration):
      inp.get_next()

  def test_input_inputs_target_lm_weights(self):
    name = 'test_weights'
    x = [{
        'inputs': [27, 28, 29, 30],
        'targets': [133, 134],
    }, {
        'inputs': [-32, -33, 34, 35],
        'targets': [-145, -146, 147],
    }]
    ds = seqio.test_utils.create_default_dataset(x)
    _register_task(name, ds)
    expected_labels = np.array([[27, 28, 29, 30, 1, 133, 134, 1, 0],
                                [32, 33, 34, 35, 1, 145, 146, 147, 1]],
                               dtype=np.int32)
    # Note how the final EOS on row 2 is truncated in ids.
    expected_ids = np.array([[0, 27, 28, 29, 30, 1, 133, 134, 1],
                             [0, 32, 33, 34, 35, 1, 145, 146, 147]],
                            dtype=np.int32)
    expected_paddings = np.array([[0., 0., 0., 0., 0., 0., 0., 0., 1.],
                                  [0., 0., 0., 0., 0., 0., 0., 0., 0.]],
                                 dtype=np.float32)

    p = pax_fiddle.Config(seqio_input.SeqIOInput)
    p.mixture_name = name
    p.split_name = 'validation'
    p.task_feature_lengths = {'inputs': 5, 'targets': 4}
    p.feature_converter = seqio_input.LanguageModelFeatures(
        pack=False, weights_on_targets_only=None)
    p.batch_size = 2
    p.is_training = False
    inp = instantiate(p)
    batch = inp.get_next()
    self.assertArraysEqual(batch.ids, expected_ids)
    self.assertArraysEqual(batch.labels, expected_labels)
    self.assertArraysEqual(batch.paddings, expected_paddings)
    # weights_on_targets_only=None: negative ids have weights of 0.0
    self.assertArraysEqual(
        batch.weights,
        np.array([[1., 1., 1., 1., 1., 1., 1., 1., 0.],
                  [0., 0., 1., 1., 1., 0., 0., 1., 1.]],
                 dtype=np.float32))
    p.feature_converter = seqio_input.LanguageModelFeatures(
        pack=False, weights_on_targets_only=True)
    inp = instantiate(p)
    batch = inp.get_next()
    self.assertArraysEqual(batch.ids, expected_ids)
    self.assertArraysEqual(batch.labels, expected_labels)
    self.assertArraysEqual(batch.paddings, expected_paddings)
    # weights_on_targets_only=True: negative ids have weights of 0.0, inputs
    # have weights of 0.0.
    self.assertArraysEqual(
        batch.weights,
        np.array([[0., 0., 0., 0., 0., 1., 1., 1., 0.],
                  [0., 0., 0., 0., 0., 0., 0., 1., 1.]],
                 dtype=np.float32))
    p.feature_converter = seqio_input.LanguageModelFeatures(
        pack=False, weights_on_targets_only=False)
    inp = instantiate(p)
    batch = inp.get_next()
    self.assertArraysEqual(batch.ids, expected_ids)
    self.assertArraysEqual(batch.labels, expected_labels)
    self.assertArraysEqual(batch.paddings, expected_paddings)
    # weights_on_targets_only=False, all non-padded tokens have weights of 1.0.
    self.assertArraysEqual(
        batch.weights,
        np.array([[1., 1., 1., 1., 1., 1., 1., 1., 0.],
                  [1., 1., 1., 1., 1., 1., 1., 1., 1.]],
                 dtype=np.float32))

  def test_input_inputs_target_lm_causal_attention(self):
    name = 'test_causal_attention'
    x = [{
        'inputs': [27, 28, 29, 30],
        'targets': [133, 134],
    }, {
        'inputs': [-32, -33, 34, 35],
        'targets': [-145, -146, 147],
    }]
    ds = seqio.test_utils.create_default_dataset(x)
    _register_task(name, ds)
    # Note the inputs indicator mask has one extra 1 compared to the input. This
    # corresponds to the position of the final input token due to the additional
    # BOS token.
    expected_inputs_indicator = np.array(
        [[1, 1, 1, 1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1, 0, 0, 0]],
        dtype=np.int32)

    p = pax_fiddle.Config(seqio_input.SeqIOInput)
    p.mixture_name = name
    p.split_name = 'validation'
    p.task_feature_lengths = {'inputs': 5, 'targets': 4}
    p.feature_converter = seqio_input.LanguageModelFeatures(
        pack=False, weights_on_targets_only=True)
    p.batch_size = 2
    p.is_training = False
    inp = instantiate(p)
    batch = inp.get_next()
    self.assertArraysEqual(batch.inputs_indicator, expected_inputs_indicator)

  def test_input_targets_only(self):
    name = 'target_only'
    x = [{
        'targets': [7, 8, 5, 6, 9],
    }, {
        'targets': [18, 14]
    }]
    ds = seqio.test_utils.create_default_dataset(x, ['targets'])
    _register_task(name, ds, output_feature_names=['targets'])

    p = pax_fiddle.Config(seqio_input.SeqIOInput)
    p.mixture_name = name
    p.split_name = 'train'
    p.task_feature_lengths = {'targets': 6}
    p.feature_converter = seqio_input.LanguageModelFeatures(pack=False)
    p.batch_size = 2
    p.is_training = True
    p.input_random_seed = 137
    inp = instantiate(p)
    batch = inp.get_next()
    self.assertArraysEqual(
        batch.ids,
        np.array([[0, 7, 8, 5, 6, 9], [0, 18, 14, 1, 0, 0]], dtype=np.int32))
    self.assertArraysEqual(
        batch.labels,
        np.array([[7, 8, 5, 6, 9, 1], [18, 14, 1, 0, 0, 0]], dtype=np.int32))
    self.assertArraysEqual(
        batch.paddings,
        np.array([[0., 0., 0., 0., 0., 0.], [0., 0., 0., 1., 1., 1.]],
                 dtype=np.float32))
    # training data is repeated.
    for _ in range(5):
      inp.get_next()

  def test_passthrough_features(self):
    name = 'passthrough_features'
    passthrough_features = {
        'passthrough': seqio.feature_converters.FeatureConverter.FeatureSpec(
            dtype=tf.int32
        ),
    }
    x = [{
        'inputs': [1, 2],
        'targets': [3, 4],
        'passthrough': [5, 6],
    }]
    ds = seqio.test_utils.create_default_dataset(
        x, feature_names=['inputs', 'targets', 'passthrough']
    )
    _register_task(
        name, ds, output_feature_names=['inputs', 'targets', 'passthrough']
    )
    p = pax_fiddle.Config(seqio_input.SeqIOInput)
    p.mixture_name = name
    p.split_name = 'train'
    p.task_feature_lengths = {'inputs': 3, 'targets': 3, 'passthrough': 3}
    p.feature_converter = seqio_input.LanguageModelFeatures(
        pack=False, passthrough_features=passthrough_features
    )
    p.batch_size = 1
    p.is_training = True
    p.input_random_seed = 137
    inp = instantiate(p)
    batch = inp.get_next()
    self.assertArraysEqual(
        batch.passthrough,
        np.array([[5, 6, 1]], dtype=np.int32),
    )

  # TODO(b/272314337): enable after the next TF OSS release.
  def test_file_based_checkpointing(self):
    it = tf.data.Dataset.range(1).as_numpy_iterator()
    if not isinstance(it, tf.__internal__.tracking.Trackable):
      # TODO(b/272314337): enable after the next TF OSS release.
      self.skipTest('file-based iterator checkpointing is not supported')

    ckpt_dir = self.create_tempdir(name='checkpointing_test').full_path
    ckpt_path = ckpt_dir + '/checkpoint'

    name = 'checkpointing_files'
    x = [{
        'targets': [7, 8, 5, 6, 9],
    }, {
        'targets': [18, 14]
    }, {
        'targets': [21, 22, 23]
    }]
    ds = seqio.test_utils.create_default_dataset(x, ['targets'])
    _register_task(name, ds, output_feature_names=['targets'])

    p = pax_fiddle.Config(seqio_input.SeqIOInput)
    p.mixture_name = name
    p.split_name = 'train'
    p.task_feature_lengths = {'targets': 6}
    p.feature_converter = seqio_input.LanguageModelFeatures(pack=False)
    p.batch_size = 1
    inp = instantiate(p)
    batch = inp.get_next()
    self.assertArraysEqual(
        batch.ids,
        np.array([[0, 7, 8, 5, 6, 9]], dtype=np.int32))
    batch = inp.get_next()
    self.assertArraysEqual(
        batch.ids,
        np.array([[0, 18, 14, 1, 0, 0]], dtype=np.int32))
    inp.save(ckpt_path)
    batch = inp.get_next()
    self.assertArraysEqual(
        batch.ids,
        np.array([[0, 21, 22, 23, 1, 0]], dtype=np.int32))
    inp.restore(ckpt_path)
    batch = inp.get_next()
    self.assertArraysEqual(
        batch.ids,
        np.array([[0, 21, 22, 23, 1, 0]], dtype=np.int32))

  def test_byte_array_based_checkpointing(self):
    it = tf.data.Dataset.range(1).as_numpy_iterator()
    if not hasattr(it, '_save'):
      # TODO(b/272314337): enable after the next TF OSS release.
      self.skipTest('byte-based iterator checkpointing is not supported')
    name = 'checkpointing_bytes'
    x = [{
        'targets': [7, 8, 5, 6, 9],
    }, {
        'targets': [18, 14]
    }, {
        'targets': [21, 22, 23]
    }]
    ds = seqio.test_utils.create_default_dataset(x, ['targets'])
    _register_task(name, ds, output_feature_names=['targets'])

    p = pax_fiddle.Config(seqio_input.SeqIOInput)
    p.mixture_name = name
    p.split_name = 'train'
    p.task_feature_lengths = {'targets': 6}
    p.feature_converter = seqio_input.LanguageModelFeatures(pack=False)
    p.batch_size = 1
    inp = instantiate(p)
    batch = inp.get_next()
    self.assertArraysEqual(
        batch.ids,
        np.array([[0, 7, 8, 5, 6, 9]], dtype=np.int32))
    batch = inp.get_next()
    self.assertArraysEqual(
        batch.ids,
        np.array([[0, 18, 14, 1, 0, 0]], dtype=np.int32))
    state = inp.get_state()
    batch = inp.get_next()
    self.assertArraysEqual(
        batch.ids,
        np.array([[0, 21, 22, 23, 1, 0]], dtype=np.int32))
    inp.set_state(state)
    batch = inp.get_next()
    self.assertArraysEqual(
        batch.ids,
        np.array([[0, 21, 22, 23, 1, 0]], dtype=np.int32))

  def test_input_targets_only_pack(self):
    name = 'target_only_pack'
    x = [{
        'targets': [7, 6, 9],
    }, {
        'targets': [18, 14]
    }, {
        'targets': [23, 25]
    }]
    ds = seqio.test_utils.create_default_dataset(x, ['targets'])
    _register_task(name, ds, output_feature_names=['targets'])

    p = pax_fiddle.Config(seqio_input.SeqIOInput)
    p.mixture_name = name
    p.split_name = 'train'
    p.task_feature_lengths = {'targets': 12}
    p.feature_converter = seqio_input.LanguageModelFeatures(pack=True)
    p.batch_size = 1
    p.shuffle = False
    p.is_training = True
    inp = instantiate(p)
    batch = inp.get_next()
    self.assertArraysEqual(
        batch.ids,
        np.array([[0, 7, 6, 9, 0, 18, 14, 0, 23, 25, 0, 0]], dtype=np.int32))
    self.assertArraysEqual(
        batch.labels,
        np.array([[7, 6, 9, 1, 18, 14, 1, 23, 25, 1, 0, 0]], dtype=np.int32))
    self.assertArraysEqual(
        batch.paddings,
        np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1.]],
                 dtype=np.float32))
    self.assertArraysEqual(
        batch.segment_ids,
        np.array([[1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 0, 0]], dtype=np.int32))
    self.assertArraysEqual(
        batch.segment_pos,
        np.array([[0, 1, 2, 3, 0, 1, 2, 0, 1, 2, 0, 0]], dtype=np.int32))

  def test_deterministic_fails(self):
    name = 'deterministic'
    x = [{'targets': [5, 6, 7]}]
    ds = seqio.test_utils.create_default_dataset(x, ['targets'])
    _register_task(name, ds, output_feature_names=['targets'])
    p = pax_fiddle.Config(seqio_input.SeqIOInput)
    p.mixture_name = name
    p.deterministic_input = True
    p.use_cached = True
    p.split_name = 'train'
    p.task_feature_lengths = {'inputs': 100, 'targets': 100}
    p.feature_converter = seqio_input.LanguageModelFeatures(pack=False)
    p.batch_size = 4
    p.is_training = True

    with self.assertRaisesRegex(ValueError,
                                'deterministic_input is not supported'):
      _ = instantiate(p)

  def test_input_inputs_target_seq(self):
    name = 'seq'
    x = [{
        'inputs': [7, 8, 5, 9, 4],
        'targets': [13, 19]
    }, {
        'inputs': [28, 24],
        'targets': [34]
    }]
    ds = seqio.test_utils.create_default_dataset(x)
    _register_task(name, ds)

    p = pax_fiddle.Config(seqio_input.SeqIOInput)
    p.mixture_name = name
    p.split_name = 'validation'
    p.task_feature_lengths = {'inputs': 6, 'targets': 3}
    p.feature_converter = seqio_input.SequenceModelFeatures(pack=False)
    p.batch_size = 2
    p.shuffle = False
    p.is_training = True
    inp = instantiate(p)
    batch = inp.get_next()
    self.assertSameElements(batch.src.keys(),
                            ['ids', 'paddings', 'segment_ids', 'segment_pos'])
    self.assertSameElements(
        batch.tgt.keys(),
        ['ids', 'labels', 'paddings', 'weights', 'segment_ids', 'segment_pos'])
    self.assertArraysEqual(
        batch.src.ids,
        np.array([[7, 8, 5, 9, 4, 1], [28, 24, 1, 0, 0, 0]], dtype=np.int32))
    self.assertArraysEqual(
        batch.src.paddings,
        np.array([[0., 0., 0., 0., 0., 0.], [0., 0., 0., 1., 1., 1.]],
                 dtype=np.float32))
    self.assertArraysEqual(
        batch.src.segment_ids,
        np.array([[1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0]], dtype=np.int32))
    self.assertArraysEqual(
        batch.src.segment_pos,
        np.array([[0, 1, 2, 3, 4, 5], [0, 1, 2, 0, 0, 0]], dtype=np.int32))
    self.assertArraysEqual(batch.tgt.ids,
                           np.array([[0, 13, 19], [0, 34, 1]], dtype=np.int32))
    self.assertArraysEqual(batch.tgt.labels,
                           np.array([[13, 19, 1], [34, 1, 0]], dtype=np.int32))
    self.assertArraysEqual(
        batch.tgt.paddings,
        np.array([[0., 0., 0.], [0., 0., 1.]], dtype=np.float32))
    self.assertArraysEqual(batch.tgt.segment_ids,
                           np.array([[1, 1, 1], [1, 1, 0]], dtype=np.int32))
    self.assertArraysEqual(batch.tgt.segment_pos,
                           np.array([[0, 1, 2], [0, 1, 0]], dtype=np.int32))

    p.feature_converter = seqio_input.SequenceModelFeatures(pack=True)
    p.task_feature_lengths = {'inputs': 10, 'targets': 5}
    p.batch_size = 1
    inp2 = instantiate(p)
    batch_pack = inp2.get_next()
    self.assertSameElements(batch_pack.src.keys(),
                            ['ids', 'paddings', 'segment_ids', 'segment_pos'])
    self.assertSameElements(
        batch_pack.tgt.keys(),
        ['ids', 'labels', 'paddings', 'weights', 'segment_ids', 'segment_pos'])
    self.assertArraysEqual(
        batch_pack.src.ids,
        np.array([[7, 8, 5, 9, 4, 1, 28, 24, 1, 0]], dtype=np.int32))
    self.assertArraysEqual(
        batch_pack.src.paddings,
        np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]], dtype=np.float32))
    self.assertArraysEqual(
        batch_pack.src.segment_ids,
        np.array([[1, 1, 1, 1, 1, 1, 2, 2, 2, 0]], dtype=np.int32))
    self.assertArraysEqual(batch_pack.tgt.labels,
                           np.array([[13, 19, 1, 34, 1]], dtype=np.int32))
    self.assertArraysEqual(batch_pack.tgt.paddings,
                           np.array([[0., 0., 0., 0., 0.]], dtype=np.float32))
    self.assertArraysEqual(batch_pack.tgt.segment_pos,
                           np.array([[0, 1, 2, 0, 1]], dtype=np.int32))

  def test_sequence_model_features_with_task_info(self):
    name = 'seq_with_task_info'
    x = [{
        'inputs': [7, 8, 5, 9, 4],
        'targets': [13, 19],
        'task_ids': [3, 17, 9]
    }, {
        'inputs': [28, 24],
        'targets': [34],
        'task_ids': [13, 7]
    }]
    feature_names = ['inputs', 'targets', 'task_ids']
    ds = seqio.test_utils.create_default_dataset(x, feature_names=feature_names)
    _register_task(name, ds, output_feature_names=feature_names)

    p = pax_fiddle.Config(seqio_input.SeqIOInput)
    p.mixture_name = name
    p.split_name = 'validation'
    task_feature_lengths = {'inputs': 6, 'targets': 3, 'task_ids': 6}
    p.task_feature_lengths = task_feature_lengths
    p.feature_converter = seqio_input.SequenceModelFeaturesWithTaskInfo(
        pack=False)
    p.batch_size = 2
    p.shuffle = False
    p.is_training = True
    inp = instantiate(p)
    batch = inp.get_next()
    self.assertSameElements(
        batch.src.keys(),
        ['ids', 'paddings', 'segment_ids', 'segment_pos', 'task_ids'])
    self.assertSameElements(batch.tgt.keys(), [
        'ids', 'labels', 'paddings', 'weights', 'segment_ids', 'segment_pos',
        'task_ids'
    ])
    self.assertArraysEqual(
        batch.src.ids,
        np.array([[7, 8, 5, 9, 4, 1], [28, 24, 1, 0, 0, 0]], dtype=np.int32))
    self.assertArraysEqual(
        batch.src.paddings,
        np.array([[0., 0., 0., 0., 0., 0.], [0., 0., 0., 1., 1., 1.]],
                 dtype=np.float32))
    self.assertArraysEqual(
        batch.src.segment_ids,
        np.array([[1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0]], dtype=np.int32))
    self.assertArraysEqual(
        batch.src.segment_pos,
        np.array([[0, 1, 2, 3, 4, 5], [0, 1, 2, 0, 0, 0]], dtype=np.int32))
    self.assertArraysEqual(
        batch.src.segment_pos,
        np.array([[0, 1, 2, 3, 4, 5], [0, 1, 2, 0, 0, 0]], dtype=np.int32))
    self.assertArraysEqual(
        batch.src.task_ids,
        np.array([[3, 17, 9, 1, 0, 0], [13, 7, 1, 0, 0, 0]], dtype=np.int32))
    self.assertArraysEqual(batch.tgt.ids,
                           np.array([[0, 13, 19], [0, 34, 1]], dtype=np.int32))
    self.assertArraysEqual(batch.tgt.labels,
                           np.array([[13, 19, 1], [34, 1, 0]], dtype=np.int32))
    self.assertArraysEqual(
        batch.tgt.paddings,
        np.array([[0., 0., 0.], [0., 0., 1.]], dtype=np.float32))
    self.assertArraysEqual(batch.tgt.segment_ids,
                           np.array([[1, 1, 1], [1, 1, 0]], dtype=np.int32))
    self.assertArraysEqual(batch.tgt.segment_pos,
                           np.array([[0, 1, 2], [0, 1, 0]], dtype=np.int32))
    self.assertArraysEqual(
        batch.tgt.task_ids,
        np.array([[3, 17, 9, 1, 0, 0], [13, 7, 1, 0, 0, 0]], dtype=np.int32))

    # Test the same converter with packing now.
    p.feature_converter = seqio_input.SequenceModelFeaturesWithTaskInfo(
        pack=True)
    p.task_feature_lengths = {'inputs': 10, 'targets': 5, 'task_ids': 8}
    p.batch_size = 1
    inp2 = instantiate(p)
    batch_pack = inp2.get_next()
    self.assertSameElements(
        batch.src.keys(),
        ['ids', 'paddings', 'segment_ids', 'segment_pos', 'task_ids'])
    self.assertSameElements(
        batch.src.keys(),
        ['ids', 'paddings', 'segment_ids', 'segment_pos', 'task_ids'])
    self.assertSameElements(batch.tgt.keys(), [
        'ids', 'labels', 'paddings', 'weights', 'segment_ids', 'segment_pos',
        'task_ids'
    ])
    self.assertArraysEqual(
        batch_pack.src.ids,
        np.array([[7, 8, 5, 9, 4, 1, 28, 24, 1, 0]], dtype=np.int32))
    self.assertArraysEqual(
        batch_pack.src.paddings,
        np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]], dtype=np.float32))
    self.assertArraysEqual(
        batch_pack.src.segment_ids,
        np.array([[1, 1, 1, 1, 1, 1, 2, 2, 2, 0]], dtype=np.int32))
    self.assertArraysEqual(batch_pack.tgt.labels,
                           np.array([[13, 19, 1, 34, 1]], dtype=np.int32))
    self.assertArraysEqual(batch_pack.tgt.paddings,
                           np.array([[0., 0., 0., 0., 0.]], dtype=np.float32))
    self.assertArraysEqual(batch_pack.tgt.segment_pos,
                           np.array([[0, 1, 2, 0, 1]], dtype=np.int32))
    self.assertArraysEqual(
        batch_pack.src.task_ids,
        np.array([[3, 17, 9, 1, 13, 7, 1, 0]], dtype=np.int32))
    self.assertArraysEqual(
        batch_pack.tgt.task_ids,
        np.array([[3, 17, 9, 1, 13, 7, 1, 0]], dtype=np.int32))

  @parameterized.parameters(True, False)
  def test_compute_metrics(self, shuffle):
    task_name = 'compute_metrics'
    targets = [f'ex target{i}' for i in range(5)]
    ds = tf.data.Dataset.from_tensor_slices({
        'inputs': [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]],
        'targets': [[10, 9], [7, 6], [5, 4], [3, 2], [1, 2]],
        'targets_pretokenized': targets,
    })
    dataset_fn = lambda split, shuffle_files, seed=None: ds
    _register_dummy_task(task_name, dataset_fn)
    p = pax_fiddle.Config(seqio_input.SeqIOInput)
    p.mixture_name = task_name
    p.split_name = 'validation'
    p.task_feature_lengths = {'inputs': 4, 'targets': 1}
    p.feature_converter = seqio_input.SequenceModelFeatures(pack=False)
    p.batch_size = 1
    p.is_training = False
    p.eval_metrics_targets_length = 3
    p.reset_for_eval = True
    p.shuffle = shuffle
    inp = instantiate(p)
    # shuffle is disabled when is_training=False.
    self.assertEqual(inp.should_shuffle, False)
    vocab = inp.mixture_or_task_inst.output_features['inputs'].vocabulary
    vocab.decode = mock.Mock(return_value='blahhh')
    decoder_outputs = []
    inp._gen_targets_artifacts()
    for _ in range(len(inp.dataset)):
      ex = next(inp._iter)
      enum_id = py_utils.get_enumeration_id(ex)
      decoder_outputs.append((enum_id, {'decoded_substr': 'ex pred'}))
      ex_targets = next(inp.targets_iter)
      # Ensure that the inputs are the same i.e. not shuffled. Otherwise this
      # assertion will fail with mismatching elements.
      self.assertAllClose(ex_targets['inputs'][:1], ex['src']['ids'][0][:1])
    # Reset targets
    inp._gen_targets_artifacts()
    m = inp.compute_metrics(decoder_outputs)
    self.assertLen(m, 1)
    self.assertEqual(m[0]['accuracy'], targets + ['ex pred'] * len(inp.dataset))

  @parameterized.named_parameters(
      ('repeat', True, 10, 1),
      ('repeat_multihost', True, 10, 4),
      ('no_repeat', False, 2, 1),
      ('no_repeat_multihost', False, 2, 2),
  )
  def test_compute_metrics_eval_num_batches(
      self, is_repeat, eval_loop_num_batches, num_hosts
  ):
    task_name = 'compute_metrics_eval_num_batches'
    # Create dataset with 13 examples
    ds = tf.data.Dataset.from_tensors({
        'inputs': [7, 8],
        'targets': [3, 9],
        'targets_pretokenized': 'ex target',
    }).repeat(13)
    dataset_fn = lambda split, shuffle_files, seed=None: ds
    _register_dummy_task(task_name, dataset_fn)
    decoder_outputs = []
    # simulate multi-host setup by iterating on multiple input generators
    for host_index in range(num_hosts):
      p = pax_fiddle.Config(
          seqio_input.SeqIOInput,
          mixture_name=task_name,
          split_name='validation',
          repeat=is_repeat,
          num_infeed_hosts=num_hosts,
          task_feature_lengths={'inputs': 1, 'targets': 1},
          # pack=False because we skip metrics computation when pack=True
          feature_converter=seqio_input.SequenceModelFeatures(pack=False),
          batch_size=3,
          is_training=False,
          reset_for_eval=False,
          eval_loop_num_batches=eval_loop_num_batches,
          infeed_host_index=host_index,
      )
      inp = instantiate(p)
      inp_iter = inp._iter
      # Create fake decoded outputs
      for _ in range(eval_loop_num_batches):
        batch = next(inp_iter)
        for ex in py_utils.tree_unstack(batch, 0):
          enum_id = py_utils.get_enumeration_id(ex)
          decoder_outputs.append((enum_id, {'decoded_substr': 'ex pred'}))
    # Compute metrics
    m = inp.compute_metrics(decoder_outputs)
    metric_output = m[0]['accuracy']
    # Dummy metric = {'accuracy': targets + predictions}
    num_eval_examples = inp._num_eval_examples
    self.assertLen(metric_output, num_eval_examples * 2)
    expected_output = ['ex target'] * num_eval_examples + [
        'ex pred'
    ] * num_eval_examples
    self.assertEqual(metric_output, expected_output)

  def _construct_scoring_task_enum_fields(
      self,
      p: pax_fiddle.Config[seqio_input.SeqIOInput],
      ds: tf.data.Dataset,
      scores: Sequence[float],
  ) -> Sequence[Tuple[Optional[str], NestedMap]]:
    enumerated_ds = seqio_input._enumerate_dataset(
        ds, p.is_training, shard_info=None
    )
    enumerated_iter = enumerated_ds.as_numpy_iterator()
    eval_output = []
    for i in range(len(list(ds))):
      ex = next(enumerated_iter)
      enum_id = py_utils.get_enumeration_id(ex)
      ex.update({'scores': scores[i]})
      eval_output.append((enum_id, ex))
    return eval_output

  def test_compute_metrics_eval(self):
    task_name = 'compute_metrics_eval'
    x = [{
        'inputs': [7, 8],
        'targets': [3, 9],
    }, {
        'inputs': [15, 16, 17],
        'targets': [29],
    }]
    ds = seqio.test_utils.create_default_dataset(x)
    dataset_fn = lambda split, shuffle_files, seed=None: ds
    _register_dummy_task(task_name, dataset_fn)
    p = pax_fiddle.Config(seqio_input.SeqIOInput)
    p.mixture_name = task_name
    p.split_name = 'validation'
    p.task_feature_lengths = {'inputs': 4, 'targets': 2}
    p.feature_converter = seqio_input.LanguageModelFeatures(
        pack=False, weights_on_targets_only=True)
    p.batch_size = 1
    p.is_training = False
    p.reset_for_eval = True
    inp = instantiate(p)
    scores = np.array([1.0, 2.5], dtype=np.float32)
    eval_output = self._construct_scoring_task_enum_fields(p, ds, scores)
    m = inp.compute_metrics_eval(eval_output)
    self.assertLen(m, 1)
    self.assertEqual(m[0]['total_score'], 3.5)

  def test_instantiatee_with_mixture(self):
    task_name = 'test_task'
    x = [{
        'inputs': [7, 8],
        'targets': [3, 9],
    }, {
        'inputs': [15, 16, 17],
        'targets': [29],
    }]
    ds = seqio.test_utils.create_default_dataset(x)
    dataset_fn = lambda split, shuffle_files, seed=None: ds
    _register_dummy_task(task_name, dataset_fn)
    seqio.MixtureRegistry.add(
        'test_eval_mixture', ['test_task'],
        default_rate=1.0)
    p = pax_fiddle.Config(seqio_input.SeqIOInput)
    p.mixture_name = 'test_eval_mixture'
    p.split_name = 'validation'
    p.task_feature_lengths = {'inputs': 4, 'targets': 2}
    p.feature_converter = seqio_input.LanguageModelFeatures(
        pack=False, weights_on_targets_only=True)
    p.batch_size = 1
    p.is_training = False
    p.reset_for_eval = True
    inp = instantiate(p)
    # Check instantiation succeeds with the mixture for eval.
    assert(inp.mixture_or_task_inst.name == 'test_eval_mixture')

  @parameterized.named_parameters(
      ('log_preprocessed_targets', True),
      ('skip_log_preprocessed_targets', False),
  )
  def test_mutate_outputs_to_write(self, log_preprocessed_targets):
    task_name = 'compute_metrics_eval'
    x = [{
        'inputs': [7, 8],
        'targets': [3, 9],
    }]
    ds = seqio.test_utils.create_default_dataset(x)
    dataset_fn = lambda split, shuffle_files, seed=None: ds
    _register_dummy_task(task_name, dataset_fn)
    p = pax_fiddle.Config(seqio_input.SeqIOInput)
    p.mixture_name = task_name
    p.split_name = 'validation'
    p.task_feature_lengths = {'inputs': 2, 'targets': 2}
    p.feature_converter = seqio_input.LanguageModelFeatures(
        pack=False, weights_on_targets_only=True)
    p.batch_size = 1
    p.is_training = False
    p.log_preprocessed_targets = log_preprocessed_targets
    inp = instantiate(p)
    scores = np.array([1.0, 2.5], dtype=np.float32)
    eval_output = self._construct_scoring_task_enum_fields(p, ds, scores)
    _ = inp.compute_metrics_eval(eval_output)
    self.assertIn('seqio_postprocessed_targets', eval_output[0][1])
    if log_preprocessed_targets:
      self.assertIn('seqio_preprocessed_targets', eval_output[0][1])
    else:
      self.assertNotIn('seqio_preprocessed_targets', eval_output[0][1])

  def _setup_seqio_test_registry(self,
                                 num_examples=10,
                                 task_feature_lengths=None):
    if not task_feature_lengths:
      task_feature_lengths = {'inputs': 1024, 'targets': 256}

    output_features = {
        feature_name: seqio.Feature(seqio.test_utils.sentencepiece_vocab())
        for feature_name in task_feature_lengths.keys()
    }

    def dataset_fn(split, shuffle_files=None, seed=42):
      del split, shuffle_files, seed
      d = {}
      for k in task_feature_lengths:
        d[k] = np.arange(
            num_examples * task_feature_lengths[k], dtype=np.int32
        ).reshape(num_examples, -1)
      return tf.data.Dataset.from_tensor_slices(d)

    def pred_metric(targets, predictions):
      del targets, predictions
      return {'metric': 1.0}

    def score_metric(targets, scores):
      del targets, scores
      return {'metric': 1.0}

    seqio.TaskRegistry.add(
        'pred_task',
        source=seqio.FunctionDataSource(dataset_fn, splits=['validation']),
        output_features=output_features,
        metric_fns=[pred_metric])
    seqio.TaskRegistry.add(
        'score_task',
        source=seqio.FunctionDataSource(dataset_fn, splits=['validation']),
        output_features=output_features,
        metric_fns=[score_metric])
    seqio.TaskRegistry.add(
        'pred_and_score_task',
        source=seqio.FunctionDataSource(dataset_fn, splits=['validation']),
        output_features=output_features,
        metric_fns=[pred_metric, score_metric])
    seqio.MixtureRegistry.add(
        'test_mixture', ['pred_task', 'score_task', 'pred_and_score_task'],
        default_rate=1.0)

  def test_get_eval_hparams_for_seqio(self):
    self._setup_seqio_test_registry()
    mixture_name = 'test_mixture'
    batch_size = 32
    feature_lengths = {'inputs': 1024, 'targets': 256}
    seed = 123
    predict_hparams = seqio_input.get_eval_hparams_for_seqio(
        mixture_name, batch_size, feature_lengths, seed,
        seqio_input.MetricType.PREDICT)
    score_hparams = seqio_input.get_eval_hparams_for_seqio(
        mixture_name, batch_size, feature_lengths, seed,
        seqio_input.MetricType.SCORE)
    self.assertListEqual([p.name for p in predict_hparams], [
        'pred_and_score_task',
        'pred_task',
    ])
    self.assertListEqual([p.name for p in score_hparams], [
        'pred_and_score_task',
        'score_task',
    ])

  def test_get_eval_hparams_for_seqio_missing_split(self):
    self._setup_seqio_test_registry()
    mixture_name = 'test_mixture'
    batch_size = 32
    feature_lengths = {'inputs': 1024, 'targets': 256}
    seed = 123
    predict_hparams = seqio_input.get_eval_hparams_for_seqio(
        mixture_name,
        batch_size,
        feature_lengths,
        seed,
        seqio_input.MetricType.PREDICT,
        split_name='eval',
        check_split_exists=True,
    )
    # None of the tasks have 'eval' split so this should be empty.
    self.assertListEqual([p.name for p in predict_hparams], [])

  def test_get_eval_hparams_for_seqio_scoring_keeps_all_lengths(self):
    feature_lengths = {
        'inputs': 1024,
        'targets': 3,
        'weights': 3,
        'embeddings': 16,
    }
    self._setup_seqio_test_registry(task_feature_lengths=feature_lengths)
    mixture_name = 'test_mixture'
    batch_size = 32
    seed = 123
    score_hparams = seqio_input.get_eval_hparams_for_seqio(
        mixture_name,
        batch_size,
        feature_lengths,
        seed,
        seqio_input.MetricType.SCORE,
        eval_metrics_retain_task_features=True,
        feature_converter=seqio.PassThroughFeatureConverter(),
        pass_entire_feature_lengths=True,
    )
    inp: seqio_input.SeqIOInput = instantiate(score_hparams[0])
    self.assertSameElements(inp.task_feature_lengths.keys(),
                            ['inputs', 'targets', 'weights', 'embeddings'])

    score_hparams = seqio_input.get_eval_hparams_for_seqio(
        mixture_name,
        batch_size,
        feature_lengths,
        seed,
        seqio_input.MetricType.SCORE,
        eval_metrics_retain_task_features=False,
        feature_converter=seqio.PassThroughFeatureConverter(),
        pass_entire_feature_lengths=True,
    )
    inp: seqio_input.SeqIOInput = instantiate(score_hparams[0])
    inp.get_next()
    self.assertSameElements(
        inp.task_feature_lengths.keys(),
        ['inputs', 'targets', 'weights', 'embeddings'],
    )

  def test_get_eval_hparams_for_seqio_scoring_keeps_lengths(self):
    feature_lengths = {'inputs': 1024, 'targets': 3, 'weights': 3}
    self._setup_seqio_test_registry(task_feature_lengths=feature_lengths)
    mixture_name = 'test_mixture'
    batch_size = 32
    seed = 123
    score_hparams = seqio_input.get_eval_hparams_for_seqio(
        mixture_name,
        batch_size,
        feature_lengths,
        seed,
        seqio_input.MetricType.SCORE,
        eval_metrics_retain_task_features=True,
        feature_converter=seqio.PassThroughFeatureConverter(),
    )
    inp: seqio_input.SeqIOInput = instantiate(score_hparams[0])
    self.assertSameElements(inp._hparams.task_feature_lengths.keys(),
                            ['inputs', 'targets', 'weights'])

    score_hparams = seqio_input.get_eval_hparams_for_seqio(
        mixture_name,
        batch_size,
        feature_lengths,
        seed,
        seqio_input.MetricType.SCORE,
        eval_metrics_retain_task_features=False,
        feature_converter=seqio.PassThroughFeatureConverter(),
    )
    inp: seqio_input.SeqIOInput = instantiate(score_hparams[0])
    inp.get_next()
    self.assertSameElements(
        inp._hparams.task_feature_lengths.keys(),
        ['inputs', 'targets'],
    )

  def test_get_eval_hparams_for_seqio_with_task_object(self):
    self._setup_seqio_test_registry()
    mixture = seqio.get_mixture_or_task('test_mixture')
    batch_size = 32
    feature_lengths = {'inputs': 1024, 'targets': 256}
    seed = 123
    predict_hparams = seqio_input.get_eval_hparams_for_seqio(
        mixture, batch_size, feature_lengths, seed,
        seqio_input.MetricType.PREDICT)
    score_hparams = seqio_input.get_eval_hparams_for_seqio(
        mixture, batch_size, feature_lengths, seed,
        seqio_input.MetricType.SCORE)
    self.assertListEqual([p.name for p in predict_hparams], [
        'pred_and_score_task',
        'pred_task',
    ])
    self.assertListEqual([p.name for p in score_hparams], [
        'pred_and_score_task',
        'score_task',
    ])

  def test_repeat_on_full_eval_fails(self):
    self._setup_seqio_test_registry()
    p = pax_fiddle.Config(
        seqio_input.SeqIOInput,
        mixture_name='pred_and_score_task',
        is_training=False,
        split_name='eval',
        task_feature_lengths={'inputs': 1024, 'targets': 256},
        feature_converter=seqio_input.LanguageModelFeatures(
            pack=False, weights_on_targets_only=True
        ),
        batch_size=4,
        reset_for_eval=True,
        eval_loop_num_batches=None,
        eval_auto_pad=False,
        repeat=True,
    )

    err_msg_rgx = (
        'Dataset has eval_loop_num_batches set to None while repeat is True.')
    with self.assertRaisesRegex(ValueError, err_msg_rgx):
      _ = instantiate(p)

  @parameterized.named_parameters(
      ('_no_padding', False, 1),
      ('_with_padding', True, 1),
      ('_multihost_with_padding', True, 4),
  )
  def test_enumerated_dataset(self, pad, num_hosts):
    batch_size = 2
    num_examples_per_host = 10
    num_examples = num_examples_per_host * num_hosts
    if pad:
      # make num_examples % batch_size != 0 s.t. we pad the last batch
      num_examples += 1

    task_feature_lengths = {'inputs': 1024, 'targets': 1}
    self._setup_seqio_test_registry(
        num_examples=num_examples, task_feature_lengths=task_feature_lengths)

    all_ids = []

    # simulate multi-host setup by iterating on multiple input generators
    for host_index in range(num_hosts):
      p = pax_fiddle.Config(
          seqio_input.SeqIOInput,
          mixture_name='pred_and_score_task',
          split_name='validation',
          task_feature_lengths=task_feature_lengths,
          feature_converter=seqio_input.LanguageModelFeatures(
              pack=False, weights_on_targets_only=True
          ),
          is_training=False,
          batch_size=batch_size,
          use_enumeration=True,
          reset_for_eval=True,
          # multi-process configs
          num_infeed_hosts=num_hosts,
          infeed_host_index=host_index,
      )
      inp = instantiate(p)

      while True:
        try:
          batch = inp.get_next()
        except StopIteration:
          break

        for ex in py_utils.tree_unstack(batch, 0):
          if ex.eval_sample_weights == 0:
            continue  # skip padded batches

          example_id = py_utils.get_enumeration_id(ex)
          self.assertIsNotNone(example_id)
          all_ids.append(example_id)

    self.assertEqual(len(all_ids), len(set(all_ids)))
    self.assertLen(all_ids, num_examples)

  def test_no_update_decode_output_keys(self):
    process_decode_output = [('prefix-key-0', NestedMap(eval_sample_weights=1)),
                             ('prefix-key-1', NestedMap(eval_sample_weights=1))]
    decode_out = NestedMap(eval_sample_weights=np.ones(2, dtype=np.float32))

    updated = seqio_input.maybe_update_decode_output_keys(
        process_decode_output, decode_out)

    self.assertEqual(updated, process_decode_output)

  def test_invalid_update_decode_output_keys(self):
    # fails when model.process_decode_out doesn't filter out padded eval samples
    common_kv = {SHARD_INDEX_KEY: 0, NUM_SHARDS_KEY: 1}
    process_decode_output = [
        (
            'prefix-key-0',
            NestedMap.FromNestedDict({INDEX_WITHIN_SHARD_KEY: 0, **common_kv}),
        ),
        (
            'prefix-key-1',
            NestedMap.FromNestedDict({INDEX_WITHIN_SHARD_KEY: 1, **common_kv}),
        ),
        (
            'prefix-key-2',
            NestedMap.FromNestedDict({
                SHARD_INDEX_KEY: -1,
                NUM_SHARDS_KEY: -1,
                INDEX_WITHIN_SHARD_KEY: -1,
            }),
        ),
    ]
    decode_out = NestedMap.FromNestedDict({
        SHARD_INDEX_KEY: np.array([0, 0, -1]),
        NUM_SHARDS_KEY: np.array([1, 1, -1]),
        INDEX_WITHIN_SHARD_KEY: np.array([0, 1, -1]),
    })

    with self.assertRaisesRegex(
        RuntimeError, 'The length of enum keys != num kv-pairs returned by .*'
    ):
      _ = seqio_input.maybe_update_decode_output_keys(
          process_decode_output, decode_out
      )

  def test_update_decode_output_keys(self):
    common_kv = {
        'eval_sample_weights': 1, SHARD_INDEX_KEY: 0, NUM_SHARDS_KEY: 1}
    process_decode_output = [
        ('prefix-key-0', NestedMap.FromNestedDict({
            INDEX_WITHIN_SHARD_KEY: 0, **common_kv})),
        ('prefix-key-1', NestedMap.FromNestedDict({
            INDEX_WITHIN_SHARD_KEY: 1, **common_kv})),
        ('prefix-key-2', NestedMap.FromNestedDict({
            INDEX_WITHIN_SHARD_KEY: 2, **common_kv}))]
    decode_out = NestedMap.FromNestedDict({
        'eval_sample_weights': np.ones(3, dtype=np.float32),
        SHARD_INDEX_KEY: np.zeros(3, dtype=np.float32),
        NUM_SHARDS_KEY: np.ones(3, dtype=np.float32),
        INDEX_WITHIN_SHARD_KEY: np.arange(3, dtype=np.float32)})

    expected = [(py_utils.get_enumeration_id(ex), ex)
                for _, ex in process_decode_output]
    updated = seqio_input.maybe_update_decode_output_keys(
        process_decode_output, decode_out)

    self.assertEqual(updated, expected)

  def test_weights_not_on_targets_only_raises(self):
    self._setup_seqio_test_registry()
    p = pax_fiddle.Config(
        seqio_input.SeqIOInput,
        mixture_name='score_task',
        split_name='validation',
        task_feature_lengths={'inputs': 1024, 'targets': 1},
        feature_converter=seqio_input.LanguageModelFeatures(pack=False),
        is_training=False,
        batch_size=1,
    )

    with self.assertRaisesRegex(
        ValueError,
        '.*must set LanguageModelFeatures.weights_on_targets_only=True'):
      _ = instantiate(p)

  def test_ininputs_target_suffix_lm(self):
    name = 'test_weights'
    x = [{
        'inputs': [2, 8, 9, 3],
        'targets': [2, 4],
        'suffixes': [3, 1],
    }, {
        'inputs': [3, 4, 6, 4],
        'targets': [4, 8, 7],
        'suffixes': []
    }]
    ds = seqio.test_utils.create_default_dataset(
        x, feature_names=('inputs', 'targets', 'suffixes'))
    _register_task(name, ds, add_eos=False)
    expected_labels = np.array(
        [[2, 8, 9, 3, 2, 4, 3, 1, 3, 4, 6, 4, 4, 8, 7, 0, 0]],
        dtype=np.int32)
    expected_ids = np.array(
        [[0, 2, 8, 9, 3, 2, 4, 3, 0, 3, 4, 6, 4, 4, 8, 0, 0]],
        dtype=np.int32)
    expected_paddings = np.array(
        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]],
        dtype=np.float32)
    expected_weights = np.array(
        [[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0]],
        dtype=np.float32)
    expected_inputs_indicator = np.array(
        [[1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0]],
        dtype=np.int32)

    p = pax_fiddle.Config(seqio_input.SeqIOInput)
    p.mixture_name = name
    p.split_name = 'validation'
    p.task_feature_lengths = {'inputs': 10, 'targets': 4, 'suffixes': 3}
    p.feature_converter = seqio_input.LanguageModelFeatures(
        pack=True, weights_on_targets_only=True, target_has_suffix=True)
    p.batch_size = 1
    p.is_training = False
    inp = instantiate(p)
    batch = inp.get_next()
    self.assertArraysEqual(batch.ids, expected_ids)
    self.assertArraysEqual(batch.labels, expected_labels)
    self.assertArraysEqual(batch.paddings, expected_paddings)
    self.assertArraysEqual(batch.weights, expected_weights)
    self.assertArraysEqual(batch.inputs_indicator, expected_inputs_indicator)

    name = 'test_weights_2'
    x = [{
        'inputs': [2, 8, 9, 3],
        'targets': [2, 4],
        'suffixes': [3, 1],
    }, {
        'inputs': [3, 4, 6, 4],
        'targets': [],
        'suffixes': [4, 8, 7]
    }]
    ds = seqio.test_utils.create_default_dataset(
        x, feature_names=('inputs', 'targets', 'suffixes'))
    _register_task(name, ds, add_eos=False)
    expected_labels = np.array(
        [[2, 8, 9, 3, 2, 4, 3, 1, 3, 4, 6, 4, 4, 8, 7, 0, 0]],
        dtype=np.int32)
    expected_ids = np.array(
        [[0, 2, 8, 9, 3, 2, 4, 3, 0, 3, 4, 6, 4, 4, 8, 0, 0]],
        dtype=np.int32)
    expected_paddings = np.array(
        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]],
        dtype=np.float32)
    expected_weights = np.array(
        [[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0]],
        dtype=np.float32)
    expected_inputs_indicator = np.array(
        [[1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0]],
        dtype=np.int32)

    p = pax_fiddle.Config(seqio_input.SeqIOInput)
    p.mixture_name = name
    p.split_name = 'validation'
    p.task_feature_lengths = {'inputs': 10, 'targets': 4, 'suffixes': 3}
    p.feature_converter = seqio_input.LanguageModelFeatures(
        pack=True, weights_on_targets_only=True, target_has_suffix=True)
    p.batch_size = 1
    p.is_training = False
    inp = instantiate(p)
    batch = inp.get_next()
    self.assertArraysEqual(batch.ids, expected_ids)
    self.assertArraysEqual(batch.labels, expected_labels)
    self.assertArraysEqual(batch.paddings, expected_paddings)
    self.assertArraysEqual(batch.weights, expected_weights)
    self.assertArraysEqual(batch.inputs_indicator, expected_inputs_indicator)

if __name__ == '__main__':
  absltest.main()
