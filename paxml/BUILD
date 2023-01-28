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

load("//paxml:paxml.bzl", "pytype_binary", "pytype_strict_library", "pytype_strict_test")
load("//paxml:paxml.bzl", "py_strict_test")
load("//praxis:build-visibility.bzl", "JAX_VISIBILITY")
load("@bazel_skylib//:bzl_library.bzl", "bzl_library")

licenses(["notice"])

package(default_visibility = JAX_VISIBILITY)

exports_files(
    [
        "LICENSE",
        "build_defs.bzl",
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

bzl_library(
    name = "paxml_bzl",
    srcs = ["paxml.bzl"],
    # Implicit build flag
    visibility = ["//visibility:private"],
)

# Internal linker rule.

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
        ":checkpoints",
        # Implicit etils dependency.
        # Implicit jax dependency.
        # Implicit orbax.checkpoint dependency.
        "//paxml:preemption",
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
    name = "base_inference_runner",
    srcs = ["base_inference_runner.py"],
    srcs_version = "PY3",
    deps = [
        ":train_states",
        "//praxis:base_hyperparams",
        "//praxis:base_layer",
        "//praxis:base_model",
        "//praxis:py_utils",
        "//praxis:pytypes",
        # Implicit tensorflow_datasets dependency.
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
    deps = ["//praxis:base_hyperparams"],
)

pytype_strict_library(
    name = "checkpoints",
    srcs = ["checkpoints.py"],
    srcs_version = "PY3",
    deps = [
        ":base_task",
        ":checkpoint_py_pb2",
        ":checkpoint_version",
        ":train_states",
        # Implicit absl.logging dependency.
        # Implicit etils dependency.
        # Implicit flax.core dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
        # Implicit optax dependency.
        # Implicit orbax.checkpoint dependency.
        "//praxis:py_utils",
        "//praxis:pytypes",
    ],
)

pytype_strict_library(
    name = "checkpoint_version",
    srcs = ["checkpoint_version.py"],
)

pytype_strict_library(
    name = "eval_lib",
    srcs = ["eval_lib.py"],
    srcs_version = "PY3",
    deps = [
        ":base_experiment",
        ":base_metrics",
        ":checkpoint_py_pb2",
        ":io_utils",
        ":metric_tracker_utils",
        ":metric_utils",
        ":seqio_input",
        ":summary_utils",
        ":tasks_lib",
        ":train_states",
        ":trainer_lib",
        ":tuning_lib",
        # Implicit absl.flags dependency.
        # Implicit absl.logging dependency.
        # Implicit clu.platform dependency.
        # Implicit etils dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
        "//paxml:checkpoints",
        "//praxis:base_hyperparams",
        "//praxis:base_input",
        "//praxis:base_layer",
        "//praxis:optimizer_prefix_vectorization",
        "//praxis:py_utils",
        "//praxis:pytypes",
        # Implicit tensorflow_no_contrib dependency.
        # Implicit tensorflow_datasets dependency.
    ],
)

pytype_strict_library(
    name = "experiment_imports_test_helper",
    testonly = True,
    srcs = ["experiment_imports_test_helper.py"],
    deps = [
        ":base_task",
        # Implicit absl.testing.absltest.absltest dependency.
        # Implicit jax dependency.
        "//praxis:base_hyperparams",
        "//praxis:base_input",
        "//praxis:base_model",
        "//praxis:py_utils",
        # Implicit pyglove dependency.
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

pytype_strict_library(
    name = "experiment_utils",
    srcs = ["experiment_utils.py"],
    srcs_version = "PY3",
    deps = [
        ":base_experiment",
    ],
)

pytype_strict_library(
    name = "io_utils",
    srcs = ["io_utils.py"],
    srcs_version = "PY3",
    deps = [
        # Implicit absl.flags dependency.
        # Implicit absl.logging dependency.
        # Implicit etils dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
        "//paxml:checkpoints",
        "//praxis:py_utils",
        # Implicit tensorflow_no_contrib dependency.
    ],
)

pytype_strict_library(
    name = "learners",
    srcs = ["learners.py"],
    deps = [
        ":sgf",
        # Implicit jax dependency.
        # Implicit optax dependency.
        "//praxis:asserts",
        "//praxis:base_hyperparams",
        "//praxis:base_layer",
        "//praxis:optimizer_prefix_vectorization",
        "//praxis:optimizers",
        "//praxis:py_utils",
    ],
)

pytype_binary(
    name = "main",
    srcs = ["main.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        # Please add additional dependencies to the target "main_lib" below.
        # so that they are also included in the BUILD macro pax_targets().
        ":main_lib",
    ],
)

# Internal GPU binary rule

pytype_strict_library(
    name = "main_lib",
    srcs = ["main.py"],
    srcs_version = "PY3",
    deps = [
        ":base_experiment",
        ":checkpoints",
        ":eval_lib",
        ":experiment_registry",
        ":setup_jax",
        ":tasks_lib",
        ":train",
        ":trainer_lib",
        ":tuning_lib",
        # Implicit internal Jax runtime dependency.  # BC internal suffix.
        # Implicit Jax software support dependency.  # BC internal suffix.
        # Implicit absl.app dependency.
        # Implicit absl.flags dependency.
        # Implicit absl.logging dependency.
        # Implicit clu.platform dependency.
        # Implicit etils dependency.
        # Implicit fiddle dependency.
        # Implicit fiddle.absl_flags dependency.
        # Implicit jax dependency.
        "//praxis:pax_fiddle",
        "//praxis:py_utils",
        # Implicit pyglove dependency.
    ],
)

pytype_strict_library(
    name = "metric_tracker_utils",
    srcs = ["metric_tracker_utils.py"],
    srcs_version = "PY3",
    deps = [
        # Implicit absl.logging dependency.
        # Implicit etils dependency.
    ],
)

pytype_strict_library(
    name = "metric_utils",
    srcs = ["metric_utils.py"],
    srcs_version = "PY3",
    deps = [
        ":summary_utils",
        # Implicit absl.logging dependency.
        # Implicit clu.values dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
        # Implicit internal platform check dependency.
        "//praxis:base_input",
        "//praxis:py_utils",
        "//praxis:pytypes",
        # Implicit seqio dependency.
        # Implicit tensorflow_no_contrib dependency.
    ] + select({
        # Internal empty GCE platform
        "//conditions:default": [
            # Implicit internal Jax runtime dependency.
        ],
    }),
)

pytype_strict_library(
    name = "preemption",
    srcs = ["preemption.py"],
    deps = [
        # Implicit jax dependency.
    ],
)

pytype_strict_library(
    name = "profiling",
    srcs = ["profiling.py"],
    deps = [
        # Implicit absl.logging dependency.
    ],
)

pytype_strict_library(
    name = "seqio_input",
    srcs = ["seqio_input.py"],
    srcs_version = "PY3",
    deps = [
        ":metric_utils",
        # Implicit absl.logging dependency.
        # Implicit etils dependency.
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
    name = "sgf",
    srcs = ["sgf.py"],
    deps = [
        # Implicit flax.core dependency.
        # Implicit jax dependency.
        # Implicit optax dependency.
        "//praxis:base_hyperparams",
        "//praxis:py_utils",
        "//praxis:pytypes",
    ],
)

pytype_strict_library(
    name = "summary_utils",
    srcs = ["summary_utils.py"],
    srcs_version = "PY3",
    deps = [
        ":train_states",
        # Implicit absl.flags dependency.
        # Implicit absl.logging dependency.
        # Implicit clu.platform dependency.
        # Implicit etils dependency.
        # Implicit flax dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
        "//praxis:base_layer",
        "//praxis:py_utils",
        "//praxis:pytypes",
        # Implicit tensorflow_no_contrib dependency.
    ],
)

pytype_strict_library(
    name = "tasks_lib",
    srcs = ["tasks_lib.py"],
    srcs_version = "PY3",
    deps = [
        ":base_inference_runner",
        ":base_metrics",
        ":base_task",
        ":io_utils",
        ":learners",
        ":train_states",
        # Implicit absl.logging dependency.
        # Implicit etils dependency.
        # Implicit flax.core dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
        # Implicit optax dependency.
        "//paxml:checkpoints",
        "//praxis:asserts",
        "//praxis:base_hyperparams",
        "//praxis:base_input",
        "//praxis:base_layer",
        "//praxis:base_model",
        "//praxis:optimizers",
        "//praxis:pax_fiddle",
        "//praxis:py_utils",
        "//praxis:pytypes",
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
        ":checkpoint_managers",
        ":checkpoint_py_pb2",
        ":eval_lib",
        ":experiment_utils",
        ":metric_utils",
        ":summary_utils",
        ":tasks_lib",
        ":train_states",
        ":trainer_lib",
        ":tuning_lib",
        # Implicit absl.logging dependency.
        # Implicit etils dependency.
        # Implicit jax dependency.
        "//paxml:checkpoints",
        "//paxml:profiling",
        "//praxis:base_hyperparams",
        "//praxis:base_input",
        "//praxis:base_layer",
        "//praxis:py_utils",
        "//praxis:pytypes",
        # Implicit tensorflow_no_contrib dependency.
    ],
)

pytype_strict_library(
    name = "trainer_lib",
    srcs = ["trainer_lib.py"],
    srcs_version = "PY3",
    deps = [
        ":base_metrics",
        ":sgf",
        ":summary_utils",
        ":tasks_lib",
        ":train_states",
        # Implicit absl.logging dependency.
        # Implicit clu.platform dependency.
        # Implicit etils dependency.
        # Implicit flax.core dependency.
        # Implicit jax dependency.
        "//paxml:checkpoints",
        "//praxis:asserts",
        "//praxis:base_hyperparams",
        "//praxis:base_input",
        "//praxis:base_layer",
        "//praxis:base_model",
        "//praxis:py_utils",
        "//praxis:pytypes",
    ],
)

pytype_strict_library(
    name = "train_states",
    srcs = ["train_states.py"],
    deps = [
        # Implicit flax.core dependency.
        # Implicit jax dependency.
        # Implicit optax dependency.
        "//praxis:base_layer",
    ],
)

pytype_strict_library(
    name = "tuning_lib",
    srcs = ["tuning_lib.py"],
    deps = [
        ":automl",
        ":base_experiment",
        ":metric_utils",
        ":trainer_lib",
        # Implicit absl.logging dependency.
        # Implicit clu.platform dependency.
        # Implicit etils dependency.
        # Implicit jax dependency.
        "//praxis:base_hyperparams",
        "//praxis:base_input",
        "//praxis:py_utils",
        "//praxis:pytypes",
        # Implicit internal pyglove dependency.
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
        "//praxis:base_hyperparams",
        # Implicit pyglove dependency.
    ],
)

py_strict_test(
    name = "base_inference_runner_test",
    srcs = ["base_inference_runner_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":base_inference_runner",
        ":train_states",
        # Implicit absl.testing.absltest.absltest dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
        "//praxis:base_hyperparams",
        "//praxis:base_layer",
        "//praxis:base_model",
        "//praxis:py_utils",
        "//praxis:pytypes",
        "//praxis:test_utils",
        # Implicit tensorflow_no_contrib dependency.
        # Implicit tensorflow_datasets dependency.
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
        ":checkpoints",
        ":train_states",
        # Implicit absl.flags dependency.
        # Implicit absl.testing.absltest.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
        # Implicit orbax.checkpoint dependency.
        # Implicit tensorflow_no_contrib dependency.
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
        "//paxml/tasks/test:synthetic",
        "//praxis:pax_fiddle",
        "//praxis/layers",
    ],
)

pytype_strict_test(
    name = "io_utils_test",
    srcs = ["io_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":io_utils",
        # Implicit absl.flags dependency.
        # Implicit absl.testing.absltest.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit etils dependency.
        # Implicit numpy dependency.
        # Implicit tensorflow_no_contrib dependency.
    ],
)

py_strict_test(
    name = "learners_test",
    srcs = ["learners_test.py"],
    deps = [
        ":learners",
        # Implicit absl.logging dependency.
        # Implicit absl.testing.absltest.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
        # Implicit optax dependency.
        "//praxis:asserts",
        "//praxis:base_hyperparams",
        "//praxis:base_layer",
        "//praxis:optimizer_prefix_vectorization",
        "//praxis:optimizers",
        "//praxis:py_utils",
        "//praxis:schedules",
        "//praxis:test_utils",
    ],
)

pytype_strict_test(
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

py_strict_test(
    name = "metric_utils_test",
    srcs = ["metric_utils_test.py"],
    deps = [
        ":metric_utils",
        ":summary_utils",
        # Implicit absl.testing.absltest.absltest dependency.
        # Implicit clu.metrics dependency.
        # Implicit clu.values dependency.
        # Implicit flax dependency.
        # Implicit numpy dependency.
        # Implicit internal platform check dependency.
        # Implicit seqio dependency.
    ] + select({
        # Internal empty GCE platform
        "//conditions:default": [
            # Implicit internal Jax runtime dependency.
        ],
    }),
)

py_strict_test(
    name = "seqio_input_test",
    srcs = ["seqio_input_test.py"],
    shard_count = 3,
    deps = [
        ":seqio_input",
        # Implicit absl.testing.absltest.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit numpy dependency.
        "//praxis:base_hyperparams",
        "//praxis:py_utils",
        "//praxis:test_utils",
        # Implicit seqio dependency.
        # Implicit tensorflow_no_contrib dependency.
    ],
)

py_strict_test(
    name = "summary_utils_test",
    srcs = ["summary_utils_test.py"],
    deps = [
        ":summary_utils",
        # Implicit absl.testing.absltest.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
        # Implicit tensorflow_no_contrib dependency.
    ],
)

py_strict_test(
    name = "tasks_lib_test",
    srcs = ["tasks_lib_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":checkpoints",
        ":tasks_lib",
        ":trainer_lib",
        # Implicit absl.testing.absltest.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit flax.core dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
        "//praxis:base_hyperparams",
        "//praxis:base_input",
        "//praxis:base_layer",
        "//praxis:base_model",
        "//praxis:optimizers",
        "//praxis:pax_fiddle",
        "//praxis:py_utils",
        "//praxis:pytypes",
        "//praxis:schedules",
        "//praxis:test_utils",
        "//praxis/layers",
    ],
)

pytype_strict_test(
    name = "trainer_lib_test",
    srcs = ["trainer_lib_test.py"],
    deps = [
        ":tasks_lib",
        ":trainer_lib",
        # Implicit absl.testing.absltest.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
        "//praxis:base_input",
        "//praxis:base_layer",
        "//praxis:base_model",
        "//praxis:optimizers",
        "//praxis:pax_fiddle",
        "//praxis:py_utils",
        "//praxis:pytypes",
        "//praxis:schedules",
        "//praxis/layers",
    ],
)

py_strict_test(
    name = "tuning_lib_test",
    srcs = ["tuning_lib_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":automl",
        ":base_experiment",
        ":trainer_lib",
        ":tuning_lib",
        # Implicit absl.testing.absltest.absltest dependency.
        # Implicit clu.platform dependency.
        # Implicit etils dependency.
        "//praxis:base_hyperparams",
        # Implicit pyglove dependency.
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
