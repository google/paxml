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

"""Tests for base_inference_runner."""

from __future__ import annotations

from typing import Any, List

from absl.testing import absltest
import jax
import numpy as np
from paxml import base_inference_runner
from praxis import base_hyperparams
from praxis import base_layer
from praxis import base_model
from praxis import py_utils
from praxis import pytypes
from praxis import test_utils
from praxis import train_states
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

instantiate = base_hyperparams.instantiate
NestedMap = py_utils.NestedMap
NestedWeightHParams = base_layer.NestedWeightHParams
PRNGKey = pytypes.PRNGKey
TrainState = train_states.TrainState


class DummyInference(base_inference_runner.BaseInferenceRunner):

  class HParams(base_inference_runner.BaseInferenceRunner.HParams):
    output: Any = None
    output_schema: Any = None

  def infer(self, train_state: TrainState, prng_key: PRNGKey,
            var_weight_hparams: NestedWeightHParams,
            input_batch: NestedMap) -> NestedMap:
    return self.hparams.output

  @property
  def output_schema(self) -> NestedMap:
    return self.hparams.output_schema


class BaseInferenceRunnerTest(test_utils.TestCase):

  def test_infer(self):
    dummy_output = NestedMap(
        tensor=np.arange(64, dtype=np.float32).reshape(8, 8),
        nested=NestedMap(
            text=np.array([f'{i}'.encode('utf-8') for i in range(8)],
                          dtype=object)))
    dummy_schema = NestedMap(
        tensor=tfds.features.Tensor(shape=(8,), dtype=tf.float32),
        nested=NestedMap(text=tfds.features.Text()))

    infer_runner_p = DummyInference.HParams(
        output=dummy_output, output_schema=dummy_schema)
    infer_runner = infer_runner_p.Instantiate(model=None)

    serialized_outputs = infer_runner.serialize_outputs(
        # Pass dummy values to all 4 arguments of infer().
        infer_runner.infer(*([None] * 4)))

    expected_outputs: List[NestedMap] = py_utils.tree_unstack(dummy_output, 0)
    self.assertEqual(len(serialized_outputs), len(expected_outputs))

    features_dict = tfds.features.FeaturesDict(dummy_schema)
    for serialized, expected in zip(serialized_outputs, expected_outputs):
      output = features_dict.deserialize_example(serialized)
      output_np = jax.tree_map(lambda x: x.numpy(), output)

      for output_leaf, expected_leaf in zip(
          jax.tree_util.tree_leaves(output_np),
          jax.tree_util.tree_leaves(expected)):
        self.assertArraysEqual(output_leaf, expected_leaf)


if __name__ == '__main__':
  absltest.main()
