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
    name = "codegen",
    srcs = ["codegen.py"],
    deps = [
        ":codegen_external_init_checkpoint_fns",
        ":codegen_highlevel_parameterization",
        ":codegen_pax_code_ir",
        ":codegen_sharding",
        ":codegen_tracer",
        ":config_normalization",
        ":make_parameterized_experiment",
        # Implicit fiddle dependency.
        # Implicit fiddle.daglish dependency.
        # Implicit fiddle.diffing dependency.
        # Implicit fiddle.selectors dependency.
        # Implicit fiddle.validation dependency.
        # Implicit fiddle._src.codegen.auto_config.code_ir dependency.
        # Implicit fiddle._src.codegen.auto_config.ir_to_cst dependency.
        # Implicit fiddle.codegen dependency.
        # Implicit fiddle.codegen.codegen_diff dependency.
        # Implicit fiddle.codegen.auto_config.experimental_top_level_api dependency.
        # Implicit fiddle.experimental.visualize dependency.
        # Implicit fiddle.extensions.jax dependency.
        # Implicit fiddle.extensions.seqio dependency.
        # Implicit libcst dependency.
        "//paxml:base_experiment",
        "//paxml:parameterized_experiment",
        "//paxml/experimental:baseline_experiment",
        "//praxis:pax_fiddle",
    ],
)

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

pytype_strict_library(
    name = "test_fixtures",
    srcs = ["test_fixtures.py"],
    deps = [
        # Implicit jax dependency.
        "//paxml:base_experiment",
        "//paxml:base_task",
        "//paxml:parameterized_experiment",
        "//paxml:tasks_lib",
        "//paxml/experimental:baseline_experiment",
        "//praxis:base_input",
        "//praxis:base_layer",
        "//praxis:base_model",
        "//praxis:pax_fiddle",
    ],
)

pytype_strict_library(
    name = "codegen_pax_code_ir",
    srcs = ["codegen_pax_code_ir.py"],
    deps = [
        # Implicit fiddle._src.codegen.auto_config.code_ir dependency.
        # Implicit libcst dependency.
    ],
)

pytype_strict_library(
    name = "codegen_external_init_checkpoint_fns",
    srcs = ["codegen_external_init_checkpoint_fns.py"],
    deps = [
        ":codegen_pax_code_ir",
        # Implicit fiddle dependency.
        # Implicit fiddle.daglish dependency.
        # Implicit fiddle._src.codegen.auto_config.code_ir dependency.
        # Implicit fiddle._src.codegen.auto_config.import_manager_wrapper dependency.
        # Implicit fiddle.codegen.auto_config.experimental_top_level_api dependency.
        "//paxml:base_experiment",
        "//paxml:tasks_lib",
    ],
)

pytype_strict_library(
    name = "unshare_sharding",
    srcs = ["unshare_sharding.py"],
    deps = [
        ":remove_sharding",
        # Implicit fiddle.daglish dependency.
    ],
)

pytype_strict_library(
    name = "convert_seqio_task_objects",
    srcs = ["convert_seqio_task_objects.py"],
    deps = [
        # Implicit fiddle.daglish dependency.
        "//praxis:pax_fiddle",
        # Implicit seqio dependency.
    ],
)

pytype_strict_library(
    name = "config_normalization",
    srcs = ["config_normalization.py"],
    deps = [
        ":convert_seqio_task_objects",
        ":remove_sharding",
        ":unshare_sharding",
        ":wrap_nested_maps",
        # Implicit fiddle dependency.
        # Implicit fiddle.experimental.auto_config dependency.
        # Implicit fiddle.experimental.dataclasses dependency.
        # Implicit fiddle.experimental.visualize dependency.
        "//paxml:base_task",
        "//paxml:parameterized_experiment",
        "//praxis:pax_fiddle",
    ],
)

pytype_strict_library(
    name = "make_parameterized_experiment",
    srcs = ["make_parameterized_experiment.py"],
    deps = [
        ":config_normalization",
        "//paxml:base_experiment",
        "//paxml:parameterized_experiment",
        "//praxis:pax_fiddle",
    ],
)

pytype_strict_library(
    name = "codegen_highlevel_parameterization",
    srcs = ["codegen_highlevel_parameterization.py"],
    deps = [
        ":codegen_pax_code_ir",
        # Implicit fiddle.daglish dependency.
        # Implicit fiddle._src.codegen.auto_config.code_ir dependency.
        # Implicit fiddle.codegen.auto_config.experimental_top_level_api dependency.
    ],
)

pytype_strict_test(
    name = "codegen_highlevel_parameterization_test",
    srcs = ["codegen_highlevel_parameterization_test.py"],
    deps = [
        ":codegen",
        ":codegen_highlevel_parameterization",
        ":codegen_tracer",
        # Implicit absl.testing.absltest.absltest dependency.
        # Implicit fiddle._src.codegen.auto_config.ir_printer dependency.
    ],
)

pytype_strict_library(
    name = "codegen_sharding",
    srcs = ["codegen_sharding.py"],
    deps = [
        ":codegen_pax_code_ir",
        ":remove_sharding",
        ":unshare_sharding",
        # Implicit fiddle.diffing dependency.
        # Implicit fiddle.codegen.codegen_diff dependency.
        # Implicit fiddle.codegen.auto_config.experimental_top_level_api dependency.
        # Implicit libcst dependency.
        "//paxml:base_experiment",
        "//praxis:pax_fiddle",
    ],
)

pytype_strict_library(
    name = "graphviz_utils",
    srcs = ["graphviz_utils.py"],
    deps = [
        ":config_normalization",
        # Implicit fiddle.graphviz dependency.
        # Implicit graphviz dependency.
    ],
)

pytype_strict_test(
    name = "graphviz_utils_test",
    srcs = ["graphviz_utils_test.py"],
    deps = [
        ":graphviz_utils",
        ":test_fixtures",
        # Implicit absl.testing.absltest.absltest dependency.
    ],
)

pytype_strict_library(
    name = "wrap_nested_maps",
    srcs = ["wrap_nested_maps.py"],
    deps = [
        # Implicit fiddle.daglish dependency.
        "//paxml/experimental:nested_map_config_helper",
        "//praxis:pax_fiddle",
        "//praxis:pytypes",
    ],
)

pytype_strict_test(
    name = "wrap_nested_maps_test",
    srcs = ["wrap_nested_maps_test.py"],
    deps = [
        ":wrap_nested_maps",
        # Implicit absl.testing.absltest.absltest dependency.
        # Implicit fiddle.testing dependency.
        "//paxml/experimental:nested_map_config_helper",
        "//praxis:pax_fiddle",
        "//praxis:pytypes",
    ],
)

pytype_strict_test(
    name = "config_normalization_test",
    srcs = ["config_normalization_test.py"],
    deps = [
        ":config_normalization",
        ":make_parameterized_experiment",
        ":test_fixtures",
        # Implicit absl.testing.absltest.absltest dependency.
        # Implicit fiddle dependency.
        "//paxml:parameterized_experiment",
        "//praxis:pax_fiddle",
        "//praxis:pytypes",
    ],
)

pytype_strict_test(
    name = "convert_seqio_task_objects_test",
    srcs = ["convert_seqio_task_objects_test.py"],
    deps = [
        ":convert_seqio_task_objects",
        # Implicit absl.testing.absltest.absltest dependency.
        # Implicit fiddle dependency.
        "//praxis:pax_fiddle",
        # Implicit seqio dependency.
        # Implicit seqio.test_utils dependency.
    ],
)

pytype_strict_test(
    name = "unshare_sharding_test",
    srcs = ["unshare_sharding_test.py"],
    deps = [
        ":unshare_sharding",
        # Implicit absl.testing.absltest.absltest dependency.
        "//praxis:base_layer",
        "//praxis:pax_fiddle",
    ],
)

pytype_strict_test(
    name = "codegen_external_init_checkpoint_fns_test",
    srcs = ["codegen_external_init_checkpoint_fns_test.py"],
    deps = [
        ":codegen_external_init_checkpoint_fns",
        ":codegen_pax_code_ir",
        ":test_fixtures",
        # Implicit absl.testing.absltest.absltest dependency.
        # Implicit fiddle._src.codegen.auto_config.init_task dependency.
        # Implicit fiddle._src.codegen.auto_config.ir_printer dependency.
        # Implicit fiddle.experimental.visualize dependency.
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

pytype_strict_test(
    name = "codegen_test",
    srcs = ["codegen_test.py"],
    tags = ["fast_and_reliable_fiddle_integration_test"],
    deps = [
        ":codegen",
        ":codegen_tracer",
        ":test_fixtures",
        # Implicit absl.testing.absltest.absltest dependency.
        # Implicit fiddle dependency.
        # Implicit fiddle._src.codegen.auto_config.ir_printer dependency.
        # Implicit seqio dependency.
    ],
)
