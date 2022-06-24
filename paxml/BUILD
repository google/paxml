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
#   Paxml: a library for training machine learning models in Jax.

load("//paxml:paxml.bzl", "pytype_binary", "pytype_strict_library")
load("//paxml:paxml.bzl", "py_strict_test")
load("//praxis:build-visibility.bzl", "JAX_VISIBILITY")
load("@bazel_skylib//:bzl_library.bzl", "bzl_library")

licenses(["notice"])

package(default_visibility = JAX_VISIBILITY)

exports_files(
    [
        "LICENSE",
        "experiment_imports_all_test.py",
        "main.py",
    ],
)

bzl_library(
    name = "build_defs_bzl",
    srcs = ["build_defs.bzl"],
    # Implicit build flag
    deps = ["//paxml:paxml.bzl"],
)

# Linker option to fit main under small code model 2GB limit. Unfortunately,
# pytype_binary doesn't accept linkopts, so we need to create an empty
# cc_library just for this.
cc_library(
    name = "linker_opts",
    linkopts = [
        "-T",
        "$(location :reorder_sections.lds)",
    ],
    deps = ["reorder_sections.lds"],
)

pytype_strict_library(
    name = "automl_interfaces",
    srcs = ["automl_interfaces.py"],
    deps = [
        "//praxis:base_hyperparams",
        # Implicit pyglove dependency.
    ],
)

pytype_strict_library(
    name = "automl",
    srcs = ["automl.py"],
    deps = [
        ":automl_interfaces",
        "//praxis:base_hyperparams",
        # Implicit pyglove dependency.
    ],
)

pytype_strict_library(
    name = "checkpoint_managers",
    srcs = ["checkpoint_managers.py"],
    deps = [
        ":checkpoint_py_pb2",
        # Implicit absl.logging dependency.
        # Implicit jax dependency.
        "//praxis:py_utils",
        # Implicit tensorflow_no_contrib dependency.
    ],
)

pytype_strict_library(
    name = "base_experiment",
    srcs = ["base_experiment.py"],
    srcs_version = "PY3",
    deps = [
        ":automl",
        ":base_task",
        "//praxis:base_input",
    ],
)

pytype_strict_library(
    name = "base_metrics",
    srcs = ["base_metrics.py"],
    srcs_version = "PY3",
    deps = [
        ":summary_utils",
        # Implicit jax dependency.
        # Implicit numpy dependency.
        "//praxis:base_hyperparams",
        "//praxis:base_layer",
    ],
)

pytype_strict_library(
    name = "base_task",
    srcs = ["base_task.py"],
    srcs_version = "PY3",
    deps = [
        "//praxis:base_hyperparams",
        "//praxis:base_layer",
        "//praxis:py_utils",
    ],
)

pytype_strict_library(
    name = "checkpoints",
    srcs = ["checkpoints.py"],
    srcs_version = "PY3",
    deps = [
        ":base_task",
        ":checkpoint_py_pb2",
        # Implicit absl.logging dependency.
        # Implicit flax.training dependency.
        # Implicit jax dependency.
        # Implicit internal jax.experimental.gda_serialization dependency.
        # Implicit jax.experimental.gda_serialization dependency.
        # Implicit numpy dependency.
        "//praxis:asserts",
        "//praxis:py_utils",
        "//praxis:train_states",
        # Implicit tensorflow_no_contrib dependency.
    ],
)

pytype_strict_library(
    name = "eval_lib",
    srcs = ["eval_lib.py"],
    srcs_version = "PY3",
    deps = [
        ":base_experiment",
        ":base_metrics",
        ":base_task",
        ":checkpoint_py_pb2",
        ":metric_tracker_utils",
        ":summary_utils",
        ":tasks_lib",
        ":trainer_lib",
        # Implicit absl.flags dependency.
        # Implicit absl.logging dependency.
        # Implicit clu.platform dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
        "//paxml:checkpoints",
        "//paxml:io_utils",
        "//praxis:base_hyperparams",
        "//praxis:base_input",
        "//praxis:base_layer",
        "//praxis:base_model",
        "//praxis:py_utils",
        "//praxis:pytypes",
        "//praxis:train_states",
        # Implicit seqio dependency.
        # Implicit tensorflow_no_contrib dependency.
    ],
)

pytype_strict_library(
    name = "experiment_imports",
    srcs = ["experiment_imports.py"],
    srcs_version = "PY3",
    deps = [
        # Implicit lingvo.model_imports_no_params dependency.
        "//paxml/tasks:all_params",
    ],
)

pytype_strict_library(
    name = "experiment_imports_no_params",
    srcs = ["experiment_imports.py"],
    srcs_version = "PY3",
    deps = [
        # Implicit lingvo.model_imports_no_params dependency.
    ],
)

pytype_strict_library(
    name = "experiment_imports_test_helper",
    testonly = True,
    srcs = ["experiment_imports_test_helper.py"],
    deps = [
        ":base_task",
        # Implicit absl.testing.absltest.absltest dependency.
        "//praxis:base_hyperparams",
        "//praxis:base_input",
    ],
)

pytype_strict_library(
    name = "experiment_registry",
    srcs = ["experiment_registry.py"],
    srcs_version = "PY3",
    deps = [
        ":base_experiment",
        # Implicit absl.logging dependency.
    ],
)

pytype_binary(
    name = "main",
    srcs = ["main.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":linker_opts",
        ":main_lib",
        "//paxml:experiment_imports",
    ],
)

pytype_strict_library(
    name = "io_utils",
    srcs = ["io_utils.py"],
    srcs_version = "PY3",
)

pytype_strict_library(
    name = "main_lib",
    srcs = ["main.py"],
    srcs_version = "PY3",
    deps = [
        ":automl",
        ":base_experiment",
        ":eval_lib",
        ":experiment_registry",
        ":setup_jax",
        ":train",
        ":trainer_lib",
        # Implicit internal filesystem #1 dependency.
        # Implicit internal filesystem #2 dependency.
        # Implicit Jax TPU support dependency.
        # Implicit hardware CPU monitoring dependency.
        # Implicit hardware GPU monitoring dependency.
        # Implicit Jax software support dependency.
        # Implicit envelope dependency.
        # Implicit absl.app dependency.
        # Implicit absl.flags dependency.
        # Implicit absl.logging dependency.
        # Implicit clu.platform dependency.
        # Implicit jax dependency.
        # Implicit jax.experimental.gda_serialization dependency.
        # Implicit numpy dependency.
        "//paxml:experiment_imports_no_params",
        "//praxis:py_utils",
        # Implicit internal pyglove dependency.
        # Implicit tensorflow_no_contrib dependency.
    ],
)

pytype_strict_library(
    name = "setup_jax",
    srcs = ["setup_jax.py"],
    deps = [
        # Implicit absl.logging dependency.
        # Implicit jax dependency.
        "//praxis:py_utils",
        # Implicit tensorflow_no_contrib dependency.
    ],
)

pytype_strict_library(
    name = "summary_utils",
    srcs = ["summary_utils.py"],
    srcs_version = "PY3",
    deps = [
        # Implicit absl.flags dependency.
        # Implicit absl.logging dependency.
        # Implicit clu.platform dependency.
        # Implicit flax dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
        "//praxis:base_layer",
        "//praxis:py_utils",
        "//praxis:pytypes",
        "//praxis:train_states",
        # Implicit tensorflow_no_contrib dependency.
    ],
)

pytype_strict_library(
    name = "tasks_lib",
    srcs = ["tasks_lib.py"],
    srcs_version = "PY3",
    deps = [
        ":base_metrics",
        ":base_task",
        # Implicit absl.logging dependency.
        # Implicit jax dependency.
        "//third_party/py/optax",
        "//paxml:checkpoints",
        "//praxis:base_hyperparams",
        "//praxis:base_layer",
        "//praxis:base_model",
        "//praxis:learners",
        "//praxis:optimizers",
        "//praxis:py_utils",
        "//praxis:pytypes",
        "//praxis:train_states",
        # Implicit tensorflow_no_contrib dependency.
    ],
)

pytype_strict_library(
    name = "test_helper",
    srcs = ["test_helper.py"],
    srcs_version = "PY3",
    deps = [
        # Implicit absl.testing.absltest.absltest dependency.
    ],
)

pytype_strict_library(
    name = "train",
    srcs = ["train.py"],
    srcs_version = "PY3",
    deps = [
        ":base_experiment",
        ":base_task",
        ":checkpoint_managers",
        ":checkpoint_py_pb2",
        ":eval_lib",
        ":summary_utils",
        ":trainer_lib",
        # Implicit absl.logging dependency.
        # Implicit jax dependency.
        # Implicit jax.experimental.gda_serialization dependency.
        # Implicit numpy dependency.
        "//paxml:checkpoints",
        "//praxis:base_hyperparams",
        "//praxis:base_input",
        "//praxis:base_layer",
        "//praxis:py_utils",
        "//praxis:train_states",
        # Implicit tensorflow_no_contrib dependency.
    ],
)

pytype_strict_library(
    name = "trainer_lib",
    srcs = ["trainer_lib.py"],
    srcs_version = "PY3",
    deps = [
        ":base_metrics",
        ":summary_utils",
        ":tasks_lib",
        # Implicit absl.logging dependency.
        # Implicit jax dependency.
        "//paxml:checkpoints",
        "//praxis:base_hyperparams",
        "//praxis:base_input",
        "//praxis:base_layer",
        "//praxis:base_model",
        "//praxis:py_utils",
        "//praxis:pytypes",
        "//praxis:train_states",
        # Implicit tensorflow_no_contrib dependency.
    ],
)

py_strict_test(
    name = "automl_test",
    srcs = ["automl_test.py"],
    python_version = "PY3",
    deps = [
        ":automl",
        ":base_experiment",
        ":base_task",
        # Implicit absl.testing.absltest.absltest dependency.
        # Implicit pyglove dependency.
    ],
)

py_strict_test(
    name = "base_metrics_test",
    srcs = ["base_metrics_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":base_metrics",
        # Implicit absl.testing.absltest.absltest dependency.
        # Implicit jax dependency.
        "//praxis:base_layer",
        "//praxis:py_utils",
        "//praxis:test_utils",
    ],
)

py_strict_test(
    name = "checkpoint_managers_test",
    srcs = ["checkpoint_managers_test.py"],
    deps = [
        ":checkpoint_managers",
        ":checkpoint_py_pb2",
        # Implicit absl.flags dependency.
        # Implicit absl.testing.absltest.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit tensorflow_no_contrib dependency.
    ],
)

py_strict_test(
    name = "experiment_imports_test",
    srcs = ["experiment_imports_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":experiment_imports",
        # Implicit absl.testing.absltest.absltest dependency.
    ],
)

py_strict_test(
    name = "experiment_registry_test",
    srcs = ["experiment_registry_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":base_experiment",
        ":experiment_registry",
        # Implicit absl.testing.absltest.absltest dependency.
        "//paxml/tasks/test/params",
        "//praxis/layers",
    ],
)

py_strict_test(
    name = "io_utils_test",
    srcs = ["io_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":io_utils",
        # Implicit absl.flags dependency.
        # Implicit absl.testing.absltest.absltest dependency.
    ],
)

py_strict_test(
    name = "tasks_lib_test",
    srcs = ["tasks_lib_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":base_task",
        ":checkpoints",
        ":tasks_lib",
        ":trainer_lib",
        # Implicit absl.testing.absltest.absltest dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
        "//praxis:base_hyperparams",
        "//praxis:base_layer",
        "//praxis:base_model",
        "//praxis:optimizers",
        "//praxis:py_utils",
        "//praxis:pytypes",
        "//praxis:schedules",
        "//praxis:test_utils",
    ],
)

load("@com_google_protobuf//:protobuf.bzl", "py_proto_library")

py_proto_library(
    name = "checkpoint_py_pb2",
    srcs = ["checkpoint.proto"],
    default_runtime = "@com_google_protobuf//:protobuf_python",
    protoc = "@com_google_protobuf//:protoc",
    srcs_version = "PY3",
    deps = [
        "@com_google_protobuf//:protobuf_python",
    ],
)

pytype_strict_library(
    name = "seqio_input",
    srcs = ["seqio_input.py"],
    srcs_version = "PY3",
    deps = [
        # Implicit absl.logging dependency.
        # Implicit jax dependency.
        # Implicit jax.experimental dependency.
        # Implicit numpy dependency.
        "//praxis:base_hyperparams",
        "//praxis:base_input",
        "//praxis:py_utils",
        "//praxis:pytypes",
        # Implicit seqio dependency.
        # Implicit tensorflow_no_contrib dependency.
    ],
)

pytype_strict_library(
    name = "metric_tracker_utils",
    srcs = ["metric_tracker_utils.py"],
    srcs_version = "PY3",
    deps = [
        # Implicit absl.logging dependency.
        # Implicit tensorflow_no_contrib dependency.
    ],
)

py_strict_test(
    name = "metric_tracker_utils_test",
    srcs = ["metric_tracker_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":metric_tracker_utils",
        # Implicit absl.flags dependency.
        # Implicit absl.testing.absltest.absltest dependency.
    ],
)
