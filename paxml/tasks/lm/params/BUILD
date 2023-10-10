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
#   Language modeling model configurations.

load("//paxml:paxml.bzl", "pytype_library")
load("//paxml:paxml.bzl", "py_strict_test")
load("//paxml:build_defs.bzl", "pax_targets")
load("//praxis:build-visibility.bzl", "JAX_VISIBILITY")

package(default_visibility = JAX_VISIBILITY)

licenses(["notice"])

pytype_library(
    name = "params",
    srcs = [
        "bert.py",
        "c4.py",
        "c4_multislice.py",
        "lm_cloud.py",
        "optimal_scaling.py",
    ],
    tags = ["keep_dep"],
    deps = [
        ":lm_cloud",
        # Implicit absl.logging dependency.
        # Implicit etils dependency.
        # Implicit fiddle dependency.
        # Implicit jax dependency.
        "//paxml:base_experiment",
        "//paxml:experiment_registry",
        "//paxml:seqio_input",
        "//paxml:tasks_lib",
        "//paxml:trainer_lib",
        "//paxml/tasks/lm:input_generator",
        "//paxml/tasks/lm:model_params",
        "//praxis:base_hyperparams",
        "//praxis:base_input",
        "//praxis:base_layer",
        "//praxis:optimizers",
        "//praxis:pax_fiddle",
        "//praxis:schedules",
        "//praxis/layers",
        "//praxis/layers:transformers",
        # Implicit seqio dependency.
        # Implicit t5.data dependency.
        # Implicit t5.data.preprocessors dependency.
    ],
)

pytype_library(
    name = "nvidia",
    srcs = [
        "nvidia.py",
    ],
    tags = ["keep_dep"],
    deps = [
        ":lm_cloud",
        ":params",
        # Implicit jax dependency.
        "//paxml:experiment_registry",
        "//paxml:tasks_lib",
        "//praxis:base_layer",
        "//praxis:optimizers",
        "//praxis:pax_fiddle",
        "//praxis:schedules",
        "//praxis/layers",
        "//praxis/layers:gpu_fast_attention",
    ],
)

pytype_library(
    name = "lm_cloud",
    srcs = [
        "lm_cloud.py",
    ],
    deps = [
        # Implicit absl.logging dependency.
        # Implicit jax dependency.
        "//paxml:base_experiment",
        "//paxml:experiment_registry",
        "//paxml:tasks_lib",
        "//paxml/tasks/lm:input_generator",
        "//paxml/tasks/lm:model_params",
        "//praxis:base_layer",
        "//praxis:pax_fiddle",
        "//praxis/layers",
    ],
)

py_strict_test(
    name = "c4_test",
    timeout = "long",
    srcs = ["c4_test.py"],
    deps = [
        ":params",
        # Implicit absl.testing.absltest.absltest dependency.
        # Implicit fiddle dependency.
        # Implicit jax dependency.
        "//praxis:optimizers",
        "//praxis:schedules",
        "//praxis:test_utils",
        "//praxis/layers",
    ],
)

pax_targets(
    experiments = [
        ":params",
    ],
)

pax_targets(
    experiments = [
        ":nvidia",
    ],
    prefix_name = "gpu",
)
