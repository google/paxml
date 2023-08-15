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
#   Vision modeling model configurations.

load("//paxml:paxml.bzl", "pytype_library")
# Import for Google-internal testing code.
load("//paxml:build_defs.bzl", "pax_targets")
load("//praxis:build-visibility.bzl", "JAX_VISIBILITY")

package(default_visibility = JAX_VISIBILITY)

licenses(["notice"])

pytype_library(
    name = "params",
    srcs = [
        "imagenet_resnets.py",
    ],
    deps = [
        # Implicit absl.flags dependency.
        # Implicit jax dependency.
        "//paxml:base_experiment",
        "//paxml:experiment_registry",
        "//paxml:learners",
        "//paxml:tasks_lib",
        "//paxml/tasks/vision:input_generator",
        "//praxis:base_input",
        "//praxis:base_layer",
        "//praxis:optimizers",
        "//praxis:pax_fiddle",
        "//praxis:py_utils",
        "//praxis:pytypes",
        "//praxis:schedules",
        "//praxis/layers",
    ],
)

pytype_library(
    name = "mnist",
    srcs = ["mnist.py"],
    deps = [
        # Implicit jax dependency.
        "//paxml:base_experiment",
        "//paxml:base_task",
        "//paxml:experiment_registry",
        "//paxml:learners",
        "//paxml:tasks_lib",
        "//praxis:base_input",
        "//praxis:base_layer",
        "//praxis:base_model",
        "//praxis:optimizers",
        "//praxis:pax_fiddle",
        "//praxis:py_utils",
        "//praxis:pytypes",
        "//praxis:schedules",
        "//praxis/layers:activations",
        "//praxis/layers:convolutions",
        "//praxis/layers:linears",
        "//praxis/layers:poolings",
        # Implicit tensorflow_no_contrib dependency.
        # Implicit tensorflow_datasets dependency.
    ],
)

pax_targets(
    experiments = [
        ":mnist",
    ],
    prefix_name = "mnist",
)

pax_targets(
    experiments = [
        ":params",
    ],
)

# Google-internal testing code.
