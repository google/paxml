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

"""Tests for unshare_sharding."""

import dataclasses

from absl.testing import absltest
from paxml.tools.fiddle import unshare_sharding
from praxis import base_layer
from praxis import pax_fiddle


class TestLayer(base_layer.BaseLayer):
  x: int = 12


class TestWrapperLayer(base_layer.BaseLayer):
  sublayers: list[base_layer.BaseLayer] = dataclasses.field(
      default_factory=list
  )


def layer_config(sharding):
  config = pax_fiddle.Config(TestLayer, x=14)
  config.weight_split_dims_mapping.wt = sharding
  return config


def fake_config():
  shared = ["foo", "bar"]
  return pax_fiddle.Config(
      TestWrapperLayer, sublayers=[layer_config(shared), layer_config(shared)]
  )


def fake_config_split_dims_cls_shared():
  sharding = pax_fiddle.Config(
      base_layer.BaseLayer.WeightSharding, wt=["foo", "bar"]
  )
  return pax_fiddle.Config(
      TestWrapperLayer,
      sublayers=[
          pax_fiddle.Config(TestLayer, weight_split_dims_mapping=sharding),
          pax_fiddle.Config(TestLayer, weight_split_dims_mapping=sharding),
      ],
  )


class UnshareShardingTest(absltest.TestCase):

  def test_deep_copy(self):
    shared = ["foo", "bar"]
    value = [shared, shared]
    self.assertIs(value[0], value[1])
    copied = unshare_sharding._deep_copy(value)
    self.assertEqual(copied, value)
    self.assertEqual(copied[0], copied[1])
    self.assertIsNot(copied, value)
    self.assertIsNot(copied[0], copied[1])

  def test_unshare_sharding(self):
    config = fake_config()

    # Ensure the fixture is set up correctly. Here, the `wt` list is shared, but
    # the enclosing BaseLayer.WeightSharding config is not.
    self.assertIsNot(
        config.sublayers[0].weight_split_dims_mapping,
        config.sublayers[1].weight_split_dims_mapping,
    )
    self.assertIs(
        config.sublayers[0].weight_split_dims_mapping.wt,
        config.sublayers[1].weight_split_dims_mapping.wt,
    )

    unshared = unshare_sharding.unshare_sharding(config=config)

    # Ensure that the value is correct. Fiddle's equality method now tests
    # object sharing, so let's just be explicit and things should generalize.
    self.assertLen(unshared.sublayers, 2)
    self.assertEqual(
        unshared.sublayers[0].weight_split_dims_mapping.wt, ["foo", "bar"]
    )
    self.assertEqual(
        unshared.sublayers[1].weight_split_dims_mapping.wt, ["foo", "bar"]
    )

    # Check that the sharding annotation was successfully unshared.
    self.assertIsNot(
        unshared.sublayers[0].weight_split_dims_mapping.wt,
        unshared.sublayers[1].weight_split_dims_mapping.wt,
    )

  def test_unshare_toplevel_sharding(self):
    config = fake_config_split_dims_cls_shared()
    self.assertIs(
        config.sublayers[0].weight_split_dims_mapping,
        config.sublayers[1].weight_split_dims_mapping,
    )  # ensure the fixture is set up correctly
    unshared = unshare_sharding.unshare_sharding(config=config)

    # Ensure that the value is correct. Fiddle's equality method now tests
    # object sharing, so let's just be explicit and things should generalize.
    self.assertLen(unshared.sublayers, 2)
    self.assertEqual(
        unshared.sublayers[0].weight_split_dims_mapping.wt, ["foo", "bar"]
    )
    self.assertEqual(
        unshared.sublayers[1].weight_split_dims_mapping.wt, ["foo", "bar"]
    )

    # Check that the sharding annotation was successfully unshared.
    self.assertIsNot(
        unshared.sublayers[0].weight_split_dims_mapping,
        unshared.sublayers[1].weight_split_dims_mapping,
    )
    self.assertIsNot(
        unshared.sublayers[0].weight_split_dims_mapping.wt,
        unshared.sublayers[1].weight_split_dims_mapping.wt,
    )

  def test_doesnt_unshare_random_other_objects(self):
    shared = {"hi": 1, "there": 2}
    config = [shared, (shared, shared)]
    transformed = unshare_sharding.unshare_sharding(config=config)
    self.assertIsNot(transformed, config)
    self.assertIs(transformed[0], transformed[1][0])
    self.assertIs(transformed[1][0], transformed[1][1])


if __name__ == "__main__":
  absltest.main()
