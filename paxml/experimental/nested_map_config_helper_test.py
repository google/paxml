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

"""Tests for nested_map_config_helper."""

from absl.testing import absltest
from paxml.experimental import nested_map_config_helper


class NestedMapConfigHelperTest(absltest.TestCase):

  def test_make_nested_map(self):
    result = nested_map_config_helper.make_nested_map(base={"foo": 1, "bar": 2})
    self.assertEqual(result.foo, 1)
    self.assertEqual(result.bar, 2)


if __name__ == "__main__":
  absltest.main()
