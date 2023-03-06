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

"""Tests for io_utils."""

import json
import pathlib
import pickle
import random
import string
from typing import Any, Sequence

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
import numpy
from paxml import io_utils
import tensorflow.compat.v2 as tf


FLAGS = flags.FLAGS


def _read_jsonl_file(filename: epath.Path) -> Sequence[Any]:
  contents = []
  with filename.open('r') as f:
    for line in f:
      contents.append(json.loads(line))

  return contents


class IoUtilsTest(parameterized.TestCase):

  def test_write_key_value_pairs(self):
    filename = epath.Path(FLAGS.test_tmpdir) / 'kv.pickle'
    kv = [('word1', 7), ('word2', 4), ('word3', 5)]
    io_utils.write_key_value_pairs(filename, kv)
    self.assertTrue(filename.exists())

  def test_write_key_value_pairs_with_device_array(self):

    # Class that mocks the interface of xla_extension.DeviceArray for testing,
    # without actually depending on this part of tensorflow library.
    class MockDeviceArray(object):

      def __init__(self, ndarray_value):
        self._ndarray_value = ndarray_value

      @property
      def _value(self):
        return self._ndarray_value

    filename = epath.Path(FLAGS.test_tmpdir) / 'kvd.pickle'
    kv = [
        ('word1', MockDeviceArray(numpy.asarray([7]))),
        ('word2', MockDeviceArray(numpy.asarray([4]))),
        ('word3', MockDeviceArray(numpy.asarray([5]))),
    ]
    io_utils.write_key_value_pairs(filename, kv, cast_to_ndarray=True)
    with filename.open('rb') as f:
      kv_reload = pickle.load(f)
    self.assertEqual(numpy.ndarray, type(kv_reload[0][1]))
    self.assertEqual(numpy.ndarray, type(kv_reload[1][1]))
    self.assertEqual(numpy.ndarray, type(kv_reload[2][1]))

  def test_validate_none_step_invalid(self):
    fnames = [f'decoder_out_200_shard_{x}.pickle' for x in range(3)]
    fnames.append('decoder_out_300_shard_0.pickle')
    with self.assertRaises(ValueError):
      io_utils._validate_filenames(fnames, step=None)

  def test_validate_none_step(self):
    fnames = [f'decoder_out_200_shard_{x}.pickle' for x in range(3)]
    step, num_shards = io_utils._validate_filenames(fnames, step=None)
    self.assertEqual(step, 200)
    self.assertEqual(num_shards, 3)

  def test_validate_step_not_found(self):
    fnames = [f'decoder_out_200_shard_{x}.pickle' for x in range(3)]
    with self.assertRaises(ValueError):
      io_utils._validate_filenames(fnames, step=300)

  def test_validate_filename_invalid(self):
    fnames = [f'decoder_out_200_shards_{x}.pickle' for x in range(3)]
    with self.assertRaises(ValueError):
      io_utils._validate_filenames(fnames, step=None)

  def test_validate_missing_shards(self):
    fnames = [f'decoder_out_200_shard_{x}.pickle' for x in range(4)]
    del fnames[1]
    with self.assertRaises(ValueError):
      io_utils._validate_filenames(fnames)

  def test_write_key_value_pairs_jsonl(self):
    filename = epath.Path(FLAGS.test_tmpdir) / 'kv.jsonl'
    kv = [('key1', {'out1': 1}), ('key2', {'out2': 2}), ('key3', {'out3': 3})]
    io_utils.write_key_value_pairs(filename, kv)
    self.assertTrue(pathlib.Path(filename).exists())
    self.assertEqual(_read_jsonl_file(filename), [v for (_, v) in kv])

  @parameterized.named_parameters(
      ('_eval', io_utils.EvaluationMode.EVAL),
      ('_decode', io_utils.EvaluationMode.DECODE),
  )
  def test_eval_resume_from_same_step(self, mode):
    job_log_dir = epath.Path(FLAGS.test_tmpdir)
    checkpoint_dir = job_log_dir / 'checkpoints'
    checkpoint_step = 1234

    with io_utils.checkpoint_progress(job_log_dir, checkpoint_step, mode):
      written_checkpoint_step = io_utils.get_checkpoint_step(
          job_log_dir, checkpoint_dir, mode)
      self.assertEqual(written_checkpoint_step, checkpoint_step)


class ShardedParallelWriterTest(absltest.TestCase):

  @staticmethod
  def _get_random_bytes(chunk_size: int = 10) -> Sequence[bytes]:
    rand_bytes = []
    for _ in range(chunk_size):
      rand_string = ''.join(
          random.choice(string.ascii_lowercase) for i in range(50))
      rand_bytes.append(rand_string.encode('utf-8'))

    return rand_bytes

  def test_tfrecord_write(self):
    num_shards, num_chunks = 4, 10
    tmpdir = epath.Path(FLAGS.test_tmpdir)
    writer = io_utils.ShardedParallelWriter(
        tmpdir / 'output',
        num_shards,
        output_format=io_utils.OutputFormatType.TFRECORD)

    expected = set()
    for _ in range(num_chunks):
      chunk = self._get_random_bytes()
      writer.write(chunk)
      expected |= set(chunk)

    writer.close()

    # Read and check contents are the same
    ds = tf.data.TFRecordDataset(list(tmpdir.glob('output*')))
    for tf_ex in ds:
      ex_bytes = tf_ex.numpy()
      self.assertIn(ex_bytes, expected)
      expected.remove(ex_bytes)

    self.assertEmpty(expected)


if __name__ == '__main__':
  absltest.main()
