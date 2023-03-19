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

"""Tests for partitioning."""
from absl.testing import absltest
from absl.testing import parameterized
import jax
from paxml import partitioning
from praxis import py_utils
from praxis import test_utils

NestedMap = py_utils.NestedMap
PartitionSpec = jax.sharding.PartitionSpec


class PartitioningTest(test_utils.TestCase):

  @parameterized.parameters([(NestedMap, dict), (dict, NestedMap)])
  def test_filter_nested_map_basics(self, src_type, filter_type):
    full_set = src_type(a=1, b=src_type(c=2, d=[3, src_type(e=6, f=7)]))
    partial_set = filter_type(a=0, b=filter_type(d=[0, filter_type(e=0)]))

    expected = src_type(a=1, b=src_type(d=[3, src_type(e=6)]))
    actual = partitioning.filter_nestedmap(full_set, partial_set)

    self.assertIsInstance(actual, src_type)
    self.assertEqual(expected, actual)

  def test_filter_nested_map_with_partition_spec(self):
    full_set = dict(a=[PartitionSpec(None), dict(b=2, c=PartitionSpec(None))])
    partial_set = dict(a=[0, dict(c=0)])

    expected = dict(a=[PartitionSpec(None), dict(c=PartitionSpec(None))])
    actual = partitioning.filter_nestedmap(full_set, partial_set)

    self.assertIsInstance(actual, dict)
    self.assertEqual(expected, actual)


if __name__ == '__main__':
  absltest.main()
