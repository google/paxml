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
#   Vision modeling-specific libraries and model configurations

load("//paxml:paxml.bzl", "pytype_strict_library")
load("//paxml:paxml.bzl", "py_strict_test")
load("//praxis:build-visibility.bzl", "JAX_VISIBILITY")

licenses(["notice"])

package(default_visibility = JAX_VISIBILITY)

pytype_strict_library(
    name = "input_generator",
    srcs = [
        "input_generator.py",
        "resnet_preprocessing.py",
    ],
    deps = [
        # Implicit absl.logging dependency.
        # Implicit tensorflow_no_contrib dependency.
    ],
)

py_strict_test(
    name = "input_generator_test",
    srcs = ["input_generator_test.py"],
    tags = [
        "external",
        "notap",
        "requires-net:external",
    ],
    deps = [
        ":input_generator",
        # Implicit absl.testing.absltest.absltest dependency.
    ],
)
