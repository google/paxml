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
    deps = ["//praxis:pytypes"],
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

py_strict_test(
    name = "layers_test",
    srcs = ["layers_test.py"],
    deps = [
        ":base",
        ":linears",
        # Implicit absl.testing.absltest.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
        # Implicit optax dependency.
        "//praxis:base_layer",
        "//praxis:pax_fiddle",
        "//praxis/layers:linears",
        # Implicit tensorflow_no_contrib dependency.
    ],
)
