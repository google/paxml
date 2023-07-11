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

"""Tests for remove_sharding."""

import dataclasses
import typing

from absl.testing import absltest
import fiddle as fdl
from fiddle import daglish
from paxml.tools.fiddle import remove_sharding
from praxis import base_layer
from praxis import pax_fiddle


class TestLayer(base_layer.BaseLayer):
  x: int = 12


def fake_config():
  config = pax_fiddle.Config(TestLayer, x=14)
  config.weight_split_dims_mapping.wt = ("foo", "bar")
  return config


def _is_sharding_config(typ):
  origin = typing.get_origin(typ)
  if not isinstance(origin, type) or not issubclass(origin, pax_fiddle.Config):
    return False
  args = typing.get_args(typ)
  if len(args) != 1:
    return False
  return args[0] in remove_sharding.SHARDING_TYPES


class RemoveShardingTest(absltest.TestCase):

  def test_base_layer_sharding_fields(self):
    detected = {
        name
        for name, typ in typing.get_type_hints(base_layer.BaseLayer).items()
        if _is_sharding_config(typ)
    }
    self.assertEqual(detected, remove_sharding.BASE_LAYER_SHARDING_FIELDS)

  def test_is_sharding_annotation(self):
    self.assertTrue(
        remove_sharding._is_sharding_annotation(
            value=pax_fiddle.Config(base_layer.BaseLayer.WeightSharding),
            path=(),
        )
    )

  def test_is_sharding_annotation_works_with_function(self):
    self.assertFalse(
        remove_sharding._is_sharding_annotation(
            value=pax_fiddle.Config(fake_config),
            path=(),
        )
    )

  def test_is_sharding_annotation_by_path(self):
    # This is just to catch if the user forgets to inherit from WeightSharding /
    # ActivationSharding.
    class Foo(base_layer.BaseLayer):

      @dataclasses.dataclass
      class MyActivationSplitDimsMapping:
        bar_sharding: list[str] = dataclasses.field(default_factory=list)

      activation_split_dims_mapping: MyActivationSplitDimsMapping = (
          dataclasses.field(default_factory=MyActivationSplitDimsMapping)
      )

    config = pax_fiddle.Config(
        Foo,
        activation_split_dims_mapping=pax_fiddle.Config(
            Foo.MyActivationSplitDimsMapping, bar_sharding=["baz", "qux"]
        ),
    )
    self.assertTrue(
        remove_sharding._is_sharding_annotation(
            value=config.activation_split_dims_mapping,
            path=(daglish.Attr("activation_split_dims_mapping"),),
        )
    )

  def test_remove_sharding(self):
    config = fake_config()
    without_sharding = remove_sharding.remove_sharding(config=config)
    self.assertIn("weight_split_dims_mapping", fdl.ordered_arguments(config))
    self.assertNotIn(
        "weight_split_dims_mapping", fdl.ordered_arguments(without_sharding)
    )

    # Ensure other attributes were not deleted.
    self.assertEqual(config.x, 14)
    self.assertEqual(without_sharding.x, 14)

  def test_replace_with_defaults(self):
    config = fake_config()
    without_sharding = remove_sharding.remove_sharding(
        config=config, replace_with_default=True
    )
    without_sharding.weight_split_dims_mapping.wt = ()


if __name__ == "__main__":
  absltest.main()
