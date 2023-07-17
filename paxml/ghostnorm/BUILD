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

# Description:
#   Library of layers that implement ghost norm protocol
load("//paxml:paxml.bzl", "py_strict_test")
load("//paxml:paxml.bzl", "pytype_strict_library")
load("//praxis:build-visibility.bzl", "JAX_VISIBILITY")

package(default_visibility = JAX_VISIBILITY)

pytype_strict_library(
    name = "base",
    srcs = ["base.py"],
    deps = [
        # Implicit flax.core dependency.
        # Implicit jax dependency.
        "//praxis:pytypes",
    ],
)

pytype_strict_library(
    name = "linears",
    srcs = ["linears.py"],
    deps = [
        ":base",
        # Implicit jax dependency.
        "//praxis:pytypes",
        "//praxis/layers",
        "//praxis/layers:base_ops",
    ],
)

pytype_strict_library(
    name = "embedding",
    srcs = ["embedding.py"],
    deps = [
        ":base",
        ":linears",
        # Implicit jax dependency.
        # Implicit jax.experimental.sparse dependency.
        "//praxis:base_layer",
        "//praxis:pytypes",
        "//praxis/layers",
        "//praxis/layers:base_ops",
    ],
)

pytype_strict_library(
    name = "generic_wrapper",
    srcs = ["generic_wrapper.py"],
    deps = [
        ":base",
        ":embedding",
        ":linears",
        # Implicit jax dependency.
        "//praxis:base_layer",
        "//praxis:pax_fiddle",
        "//praxis:pytypes",
        "//praxis/layers:attentions",
        "//praxis/layers:embedding_softmax",
        "//praxis/layers:linears",
        "//praxis/layers:normalizations",
        "//praxis/layers:transformers",
    ],
)

py_strict_test(
    name = "layers_test",
    srcs = ["layers_test.py"],
    deps = [
        ":base",
        ":embedding",
        ":generic_wrapper",
        ":linears",
        # Implicit absl.testing.absltest.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
        # Implicit optax dependency.
        "//praxis:base_layer",
        "//praxis:pax_fiddle",
        "//praxis:py_utils",
        "//praxis/layers:embedding_softmax",
        "//praxis/layers:linears",
        "//praxis/layers:transformer_models",
        # Implicit tensorflow_no_contrib dependency.
    ],
)
