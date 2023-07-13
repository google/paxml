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

# Fiddle-related tooling for Pax experiments.

load("//paxml:paxml.bzl", "pytype_strict_library", "pytype_strict_test")
load("//praxis:build-visibility.bzl", "JAX_VISIBILITY")

package(default_visibility = JAX_VISIBILITY)

licenses(["notice"])

pytype_strict_library(
    name = "codegen_tracer",
    srcs = ["codegen_tracer.py"],
    deps = [
        # Implicit fiddle.daglish dependency.
        "//paxml:base_experiment",
    ],
)

pytype_strict_library(
    name = "remove_sharding",
    srcs = ["remove_sharding.py"],
    deps = [
        # Implicit fiddle dependency.
        # Implicit fiddle.daglish dependency.
        "//praxis:base_layer",
        "//praxis:pax_fiddle",
    ],
)

pytype_strict_test(
    name = "remove_sharding_test",
    srcs = ["remove_sharding_test.py"],
    deps = [
        ":remove_sharding",
        # Implicit absl.testing.absltest.absltest dependency.
        # Implicit fiddle dependency.
        # Implicit fiddle.daglish dependency.
        "//praxis:base_layer",
        "//praxis:pax_fiddle",
    ],
)

pytype_strict_test(
    name = "codegen_tracer_test",
    srcs = ["codegen_tracer_test.py"],
    deps = [
        ":codegen_tracer",
        # Implicit absl.testing.absltest.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        "//paxml:base_experiment",
        "//paxml:base_task",
        "//paxml:tasks_lib",
        "//praxis:base_input",
        "//praxis:base_model",
        "//praxis:pax_fiddle",
        "//praxis/layers:checkpoint_policy",
    ],
)
