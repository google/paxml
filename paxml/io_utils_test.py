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

from absl import flags
from absl.testing import absltest
from paxml import io_utils

FLAGS = flags.FLAGS


class IoUtilsTest(absltest.TestCase):

  def test_write_key_value_pairs(self):
    filename = os.path.join(FLAGS.test_tmpdir, 'kv.pickle')
    kv = {'word1': 7, 'word2': 4, 'word3': 5}
    io_utils.write_key_value_pairs(filename, kv)
    self.assertTrue(pathlib.Path(filename).exists())


if __name__ == '__main__':
  absltest.main()
