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

# Description:
#   Language modeling-specific libraries and model configurations

load("//paxml:paxml.bzl", "pytype_strict_library")
load("//paxml:paxml.bzl", "py_strict_test")
load("//praxis:build-visibility.bzl", "JAX_VISIBILITY")

licenses(["notice"])

package(default_visibility = JAX_VISIBILITY)

pytype_strict_library(
    name = "input_generator",
    srcs = ["input_generator.py"],
    srcs_version = "PY3",
    deps = [
        # Implicit absl.logging dependency.
        # Implicit jax dependency.
        "//praxis:base_input",
        "//praxis:py_utils",
        "//praxis:pytypes",
        # Implicit tensorflow_no_contrib dependency.
    ],
)

pytype_strict_library(
    name = "model_params",
    srcs = ["model_params.py"],
    deps = [
        # Implicit fiddle dependency.
        # Implicit jax dependency.
        "//paxml:base_experiment",
        "//paxml:tasks_lib",
        "//praxis:asserts",
        "//praxis:base_layer",
        "//praxis:base_model",
        "//praxis:optimizers",
        "//praxis:pax_fiddle",
        "//praxis:py_utils",
        "//praxis:schedules",
        "//praxis/layers",
        "//praxis/layers:activations",
        "//praxis/layers:embedding_softmax",
        "//praxis/layers:models",
        "//praxis/layers:transformer_models",
    ],
)

filegroup(
    name = "testdata",
    testonly = 1,
    srcs = glob(["testdata/*"]),
)

py_strict_test(
    name = "input_generator_test",
    srcs = ["input_generator_test.py"],
    data = [":testdata"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":input_generator",
        # Implicit absl.testing.absltest.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit numpy dependency.
        "//paxml:test_helper",
        "//praxis:base_hyperparams",
        "//praxis:test_utils",
        # Implicit tensorflow_no_contrib dependency.
    ],
)
