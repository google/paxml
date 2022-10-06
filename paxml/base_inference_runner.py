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

"""Base API for inference runners."""

from __future__ import annotations

import abc
from typing import List

from praxis import base_hyperparams
from praxis import base_layer
from praxis import base_model
from praxis import py_utils
from praxis import pytypes
from praxis import train_states
import tensorflow_datasets as tfds

NestedMap = py_utils.NestedMap
NestedWeightHParams = base_layer.NestedWeightHParams
PRNGKey = pytypes.PRNGKey
TrainState = train_states.TrainState


class BaseInferenceRunner(base_hyperparams.BaseParameterizable, abc.ABC):
  """Abstract base class for users to override.

  This class is essentially a container for a functional infer method and
  output schema definition. It defines (1) how to run inference to generate
  outputs given a model and some inputs, and (2) the corresponding schema for
  the output.

  TODO(b/238220793): Currently we only write Jax native types since we do all
  computations in a jit-ed context. We may eventually want to support non jax-
  native types such as strings.
  """

  def __init__(self, hparams: BaseInferenceRunner.HParams,
               model: base_model.BaseModel) -> None:
    super().__init__(hparams)
    self._model = model

  @abc.abstractmethod
  def infer(self, train_state: TrainState, prng_key: PRNGKey,
            var_weight_hparams: NestedWeightHParams,
            input_batch: NestedMap) -> NestedMap:
    """Generates some output given a model and input. Should be pmap-able."""

  @property
  @abc.abstractmethod
  def output_schema(self) -> NestedMap:
    """Returns the schema for the output to be serialized.

    This must be a nested map of `tfds.features.FeatureConnector` types. See
    https://www.tensorflow.org/datasets/api_docs/python/tfds/features/FeatureConnector
    for more information. The following is an example:

    ```
    return NestedMap(
        bucket_keys=tfds.features.Scalar(dtype=tf.int32),
        nested=NestedMap(
            field=tfds.features.Tensor(shape=(1000,), dtype=tf.int32)
        ),
        logprobs=tfds.features.Tensor(shape=(1, 32,), dtype=tf.float32),
    )
    ```
    """

  def serialize_outputs(self, outputs: NestedMap) -> List[bytes]:
    input_batch_dim = 0
    features_dict = tfds.features.FeaturesDict(self.output_schema)
    examples = py_utils.tree_unstack(outputs, input_batch_dim)

    serialized_examples = []
    for ex in examples:
      serialized_examples.append(features_dict.serialize_example(ex))

    return serialized_examples
