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

"""Tests for xla_passthrough."""

from absl.testing import absltest
import numpy as np
from paxml import xla_passthrough


class InputUtilsTest(absltest.TestCase):

  def test_split_out_xla_unsupported_batch_noop(self):
    batch = {'a': np.array([1, 2, 3, 4]), 'b': np.array([5, 6, 7, 8])}
    partitioning_spec = {'a': 'fake_spec', 'b': 'fake_spec'}
    out_batch, out_unsupported, partitioning_spec = (
        xla_passthrough.split_out_xla_unsupported_batch(
            batch, partitioning_spec=partitioning_spec
        )
    )
    # In no-op cases, the exact same object should be returned.
    self.assertIs(batch, out_batch)
    self.assertEmpty(out_unsupported)
    xla_passthrough.merge_back_xla_unsupported_batch(out_batch, out_unsupported)
    self.assertEqual(out_batch, batch)
    # Partitioning spec should not be modified.
    self.assertIsNotNone(partitioning_spec)
    self.assertCountEqual(partitioning_spec.keys(), {'a', 'b'})

  def test_split_out_xla_unsupported_batch_singly_nested(self):
    batch = {
        'a': np.array([1, 2, 3, 4]),
        'b': np.array([5, 6, 7, 8]),
        'c': np.array(['a', 'b', 'c', 'd']),
    }
    partitioning_spec = {'a': 'fake_spec', 'b': 'fake_spec', 'c': 'fake_spec'}
    out_batch, out_unsupported, new_partitioning_spec = (
        xla_passthrough.split_out_xla_unsupported_batch(
            batch, partitioning_spec=partitioning_spec
        )
    )
    self.assertCountEqual(out_batch.keys(), {'a', 'b'})
    self.assertCountEqual(out_unsupported.keys(), {'c'})
    # Verify that the unsupported parts were flattened.
    self.assertEqual(list(out_unsupported['c']), ['a', 'b', 'c', 'd'])
    xla_passthrough.merge_back_xla_unsupported_batch(out_batch, out_unsupported)
    self.assertCountEqual(out_batch.keys(), {'a', 'b', 'c'})
    # The original partitioning_spec should not be modified.
    self.assertCountEqual(partitioning_spec.keys(), {'a', 'b', 'c'})
    # The unsupported key should have been deleted from the new partitioning
    # spec.
    self.assertIsNotNone(new_partitioning_spec)
    self.assertCountEqual(new_partitioning_spec.keys(), {'a', 'b'})

  def test_split_out_xla_unsupported_batch_multi_nested(self):
    batch = {
        'a': np.array([1, 2, 3, 4]),
        'b': np.array([5, 6, 7, 8]),
        'c': {
            'd': np.array(['a', 'b', 'c', 'd']),
            'e': np.array([1, 2, 3, 4]),
        },
    }
    partitioning_spec = {
        'a': 'fake_spec',
        'b': 'fake_spec',
        'c': {'d': 'fake_spec', 'e': 'fake_spec'},
    }
    out_batch, out_unsupported, new_partitioning_spec = (
        xla_passthrough.split_out_xla_unsupported_batch(
            batch, partitioning_spec=partitioning_spec
        )
    )
    self.assertCountEqual(out_batch.keys(), {'a', 'b'})
    self.assertCountEqual(out_unsupported.keys(), {'c'})
    self.assertCountEqual(out_unsupported['c'].keys(), {'d'})
    # Verify that the unsupported parts were flattened.
    self.assertEqual(list(out_unsupported['c']['d']), ['a', 'b', 'c', 'd'])
    xla_passthrough.merge_back_xla_unsupported_batch(out_batch, out_unsupported)
    self.assertCountEqual(out_batch.keys(), {'a', 'b', 'c'})
    self.assertCountEqual(out_batch['c'].keys(), {'d'})
    # The original partitioning_spec should not be modified.
    self.assertCountEqual(partitioning_spec.keys(), {'a', 'b', 'c'})
    self.assertCountEqual(partitioning_spec['c'].keys(), {'d', 'e'})
    # The unsupported key should have been deleted from the new partitioning
    # spec.
    self.assertIsNotNone(new_partitioning_spec)
    self.assertCountEqual(new_partitioning_spec.keys(), {'a', 'b', 'c'})
    self.assertCountEqual(new_partitioning_spec['c'].keys(), {'e'})


if __name__ == '__main__':
  absltest.main()
