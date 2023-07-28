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

# Experimental packages.

load("//paxml:paxml.bzl", "pytype_strict_library", "pytype_strict_test")
load("//praxis:build-visibility.bzl", "JAX_VISIBILITY")

package(default_visibility = JAX_VISIBILITY)

licenses(["notice"])

pytype_strict_library(
    name = "baseline_experiment",
    srcs = ["baseline_experiment.py"],
    deps = [
        # Implicit fiddle dependency.
        "//paxml:parameterized_experiment",
        "//praxis:pax_fiddle",
    ],
)

pytype_strict_library(
    name = "nested_map_config_helper",
    srcs = ["nested_map_config_helper.py"],
    deps = ["//praxis:pytypes"],
)

pytype_strict_test(
    name = "nested_map_config_helper_test",
    srcs = ["nested_map_config_helper_test.py"],
    deps = [
        ":nested_map_config_helper",
        # Implicit absl.testing.absltest.absltest dependency.
    ],
)

pytype_strict_test(
    name = "baseline_experiment_test",
    srcs = ["baseline_experiment_test.py"],
    deps = [
        ":baseline_experiment",
        # Implicit absl.testing.absltest.absltest dependency.
        "//paxml:parameterized_experiment",
        "//paxml:tasks_lib",
        "//praxis:base_model",
        "//praxis:pax_fiddle",
    ],
)
