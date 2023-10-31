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

# Tools and utilities for Pax users.

load("//paxml:paxml.bzl", "pytype_strict_library")
load("//praxis:build-visibility.bzl", "JAX_VISIBILITY")

package(default_visibility = JAX_VISIBILITY)

licenses(["notice"])

pytype_strict_library(
    name = "dump_input_specs_lib",
    srcs = ["dump_input_specs_lib.py"],
    deps = [
        # Implicit absl.logging dependency.
        # Implicit jax dependency.
        "//paxml:base_experiment",
        "//praxis:base_hyperparams",
        "//praxis:base_input",
        "//praxis:pytypes",
        # Implicit pyglove dependency.
    ],
)

exports_files([
    "dump_hparams.py",
    "dump_input_specs.py",
    "model_analysis.py",
    "validate_config.py",
])

pytype_strict_library(
    name = "validate_config_lib",
    srcs = ["validate_config_lib.py"],
    deps = [
        # Implicit absl.app dependency.
        # Implicit absl.flags dependency.
        # Implicit absl.logging dependency.
        # Implicit fiddle.absl_flags dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
        "//paxml:base_experiment",
        "//paxml:experiment_registry",
        "//praxis:base_layer",
        "//praxis:pax_fiddle",
        "//praxis:py_utils",
    ],
)

pytype_strict_library(
    name = "dump_hparams_lib",
    srcs = ["dump_hparams_lib.py"],
    deps = [
        # Implicit absl.app dependency.
        # Implicit absl.flags dependency.
        # Implicit absl.logging dependency.
        # Implicit etils dependency.
        # Implicit fiddle.absl_flags dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
        "//paxml:base_experiment",
        "//paxml:experiment_registry",
        "//paxml:experiment_utils",
        "//praxis:base_hyperparams",
        "//praxis:base_layer",
        "//praxis:pax_fiddle",
        "//praxis:py_utils",
        "//praxis:pytypes",
        # Implicit pyglove dependency.
        # Implicit tensorflow_no_contrib dependency.
    ],
)
