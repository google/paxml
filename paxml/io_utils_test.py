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

"""Tests for io_utils."""

import os
import pathlib
import random
import string
from typing import Sequence

from absl import flags
from absl.testing import absltest
from paxml import io_utils
import tensorflow.compat.v2 as tf

FLAGS = flags.FLAGS


class IoUtilsTest(absltest.TestCase):

  def test_write_key_value_pairs(self):
    filename = os.path.join(FLAGS.test_tmpdir, 'kv.pickle')
    kv = {'word1': 7, 'word2': 4, 'word3': 5}
    io_utils.write_key_value_pairs(filename, kv)
    self.assertTrue(pathlib.Path(filename).exists())

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
    filename = os.path.join(FLAGS.test_tmpdir, 'output')
    writer = io_utils.ShardedParallelWriter(
        filename, num_shards, output_format=io_utils.OutputFormatType.TFRECORD)

    expected = set()
    for _ in range(num_chunks):
      chunk = self._get_random_bytes()
      writer.write(chunk)
      expected = expected.union(chunk)

    writer.close()

    # Read and check contents are the same
    ds = tf.data.TFRecordDataset(tf.io.gfile.glob(filename + '*'))
    for tf_ex in ds:
      ex_bytes = tf_ex.numpy()
      self.assertIn(ex_bytes, expected)
      expected.remove(ex_bytes)

    self.assertEmpty(expected)


if __name__ == '__main__':
  absltest.main()
