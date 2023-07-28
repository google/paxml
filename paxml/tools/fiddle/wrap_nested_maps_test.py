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

"""Tests for wrap_nested_maps."""

from absl.testing import absltest
import fiddle.testing
from paxml.experimental import nested_map_config_helper
from paxml.tools.fiddle import wrap_nested_maps
from praxis import pax_fiddle
from praxis import pytypes


class WrapNestedMapsTest(fiddle.testing.TestCase):

  def test_wrap_nested_maps(self):
    shared = {"value": 1}
    config = [
        pytypes.NestedMap(
            foo={
                "subdict": pytypes.NestedMap(bar=shared),
                "another_value": (shared,),
            }
        )
    ]
    result = wrap_nested_maps.wrap_nested_maps(config=config)
    shared2 = {"value": 1}
    expected = [
        pax_fiddle.Config(
            nested_map_config_helper.make_nested_map,
            base={
                "foo": {
                    "subdict": pax_fiddle.Config(
                        nested_map_config_helper.make_nested_map,
                        base={"bar": shared2},
                    ),
                    "another_value": (shared2,),
                }
            },
        )
    ]
    self.assertDagEqual(result, expected)


if __name__ == "__main__":
  absltest.main()
