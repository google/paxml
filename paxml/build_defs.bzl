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

"""Starlark macros for Pax users."""

load("//paxml:paxml.bzl", "py_strict_test")
load("//paxml:paxml.bzl", "pytype_binary", "pytype_strict_binary")
# Internal fragmented binary bazel rule.

def _shell_quote(s):
    """Copy of bazel-skylib's shell.quote.

    Quotes the given string for use in a shell command.

    This function quotes the given string (in case it contains spaces or other
    shell metacharacters.)

    Args:
      s: The string to quote.

    Returns:
      A quoted version of the string that can be passed to a shell command.
    """
    return "'" + s.replace("'", "'\\''") + "'"

def _export_sources_impl(ctx):
    files = []
    for dep in ctx.attr.deps:
        files += dep[DefaultInfo].files.to_list()

    srcs = depset(direct = files)
    return [
        DefaultInfo(files = srcs),
    ]

# This rule defines a target which contains the list of source files
# (`srcs`) from a list of Python library dependencies in `deps`.
#
# For example, if we have:
#   py_library(name = "my_lib", srcs = ["my_lib.py"], ...)
# , then
#   export_sources(name = "lib_files", deps = [":my_lib"])
# defines a target ":lib_files" which is [":my_lib.py"].
export_sources = rule(
    implementation = _export_sources_impl,
    attrs = {
        "deps": attr.label_list(
            providers = [PyInfo],
            allow_empty = True,
            mandatory = True,
            cfg = "exec",
        ),
    },
)

def pax_targets(
        experiments = None,
        extra_deps = None,
        prefix_name = "",
        name = "",
        add_main_gpu_target = True,
        add_main_mpm_target = True,
        main_kwargs = None,
        add_smoke_test = True,
        smoke_test_exclude_regexes = "",
        smoke_test_include_only_regexes = "",
        smoke_test_py_test_rule = py_strict_test,
        smoke_test_args = None,
        smoke_test_kwargs = None,
        dump_input_specs_kwargs = None,
        # Internal enable fragmented build argument, toggled to True.
        # Internal tooling mock backend attribute
        main_src = "//paxml:main.py",
        model_analysis_kwargs = None):
    """Macro to define a collection of Pax targets with custom dependencies.

    It currently defines the following targets:

    ":main", a Python binary that can be passed to the xm launcher to run
        the experiments.
    ":main_mpm", an MPM target, which contains main.par, that can be used
        in the xm launcher with --binary_type=mpm_target. Only available if
        `add_main_mpm_target` is True.
    # Internal mock backend target docstring
    ":all_experiments_smoke_test", a Python test that runs a sanity check
        on all registered experiments.
    ":dump_hparams", a Python util binary that writes experiment hparams to
        file.
    ":validate_config", a Python binary that validates an experiment config.

    Args:
      experiments: a list of py_library targets that defines and registers all
          experiments for this collection. Experiments should be registered
          when the `srcs` files are imported.
      extra_deps: a list of extra dependencies not already included.
      prefix_name: string, a common prefix for the generated targets.
          An underscore is added, e.g. if prefix_name="test", the defined
          main target is ":test_main".
      name: unused.
      add_main_gpu_target: Build with jax GPU dependency.
      add_main_mpm_target: Add a ':main_mpm' target.
      main_kwargs: dict of args to provide when building main binary.
      add_smoke_test: Whether to add the :all_experiments_smoke_test target.
      smoke_test_py_test_rule: Test rule use to build the smoke test.
      smoke_test_args: The list of command line arguments that can be passed to the
       :all_experiments_smoke_test target.
      smoke_test_exclude_regexes: Exclusion regexes of experiment configurations to be
          passed to the smoke test. The matching experiment configurations will
          be disabled from the smoke test.
      smoke_test_include_only_regexes: If provided, then any experiment name must
          match one of these regexes in order to be smoke tested.
      smoke_test_kwargs: Additional kwargs that are passed to the
          :all_experiments_smoke_test target.
      dump_input_specs_kwargs: Additional kwargs that are passed to the
          :dump_input_specs target.
      # Internal mock backend docstrings
      main_src: The src file for the ":main" target created.
      model_analysis_kwargs: Additional kwargs that are passed to the
          :model_analysis target.
    """
    if not experiments:
        fail("pax_targets() expects a non-empty list of deps that defines " +
             "and registers experiments.")
    if name:
        fail("name is not used and has no effect. Specify prefix_name instead.")

    exp_sources = ("_exp_sources" if not prefix_name else "_%s_exp_sources" % prefix_name)
    export_sources(
        name = exp_sources,
        deps = experiments,
    )
    extra_deps = experiments + (extra_deps or [])

    if not main_kwargs:
        main_kwargs = {}
    main_name = "main"
    main_name = main_name if not prefix_name else "%s_%s" % (prefix_name, main_name)
    export_binary(
        name = main_name,
        main = main_src,
        # Internal enable fragmented build argument.
        py_binary_rule = pytype_binary,
        deps = [
            "//paxml:main_lib",
            # Implicit tpu dependency.
        ] + extra_deps,
        exp_sources = exp_sources,
        # Implicit py_binary flag
        **main_kwargs
    )
    if add_main_mpm_target and hasattr(native, "genmpm"):
        main_name = "main_mpm"
        main_name = main_name if not prefix_name else "%s_%s" % (prefix_name, main_name)
        main_dep = "main"
        main_dep = main_dep if not prefix_name else "%s_%s" % (prefix_name, main_dep)
        native.genmpm(
            name = main_name,
            # Make package temporal since most pax users cannot build under
            # learning/multipod/pax. Otherwise we could set the package_name to
            # package_name = "%s/%s" % (native.package_name().removesuffix('/params'), main_dep),
            temporal = 1,
            srcs = [":%s.par" % main_dep],
        )

    if add_main_gpu_target:
        main_name = "main_gpu"
        main_name = main_name if not prefix_name else "%s_%s" % (prefix_name, main_name)
        export_binary(
            name = main_name,
            main = main_src,
            # Internal enable fragmented build argument.
            py_binary_rule = pytype_binary,
            deps = [
                "//paxml:main_lib",
                # Implicit gpu dependency.
            ] + extra_deps,
            exp_sources = exp_sources,
            # # Implicit py_binary flag
            # PAR reticulation OOMs for gpu_main.
            exec_properties = {"mem": "24g"},
        )

        # Add a test to check that the experiments are importable in a GPU build.
        # This prevents libraries that are incompatible with GPU (e.g. those that
        # runs TensorFlow ops during import).
        test_name = "gpu_import_test"
        test_name = test_name if not prefix_name else "%s_%s" % (prefix_name, test_name)

        # Google-internal tests.

    if add_smoke_test:
        test_name = "all_experiments_smoke_test"
        test_name = test_name if not prefix_name else "%s_%s" % (prefix_name, test_name)

        smoke_test_args = smoke_test_args or []
        if smoke_test_exclude_regexes:
            smoke_test_args.append("--exclude_regexes=" + _shell_quote(smoke_test_exclude_regexes))
        if smoke_test_include_only_regexes:
            smoke_test_args.append("--include_only_regexes=" + _shell_quote(smoke_test_include_only_regexes))

        smoke_test_kwargs = smoke_test_kwargs or {}
        export_test(
            name = test_name,
            test_src = "//paxml:experiment_imports_all_test.py",
            exp_sources = exp_sources,
            deps = [
                # Implicit absl.app dependency.
                # Implicit absl.flags dependency.
                # Implicit absl.testing.absltest.absltest dependency.
                "//paxml:experiment_imports_test_helper",
                "//paxml:experiment_registry",
            ] + extra_deps,
            timeout = "long",
            py_test_rule = smoke_test_py_test_rule,
            args = smoke_test_args,
            **smoke_test_kwargs
        )

    dump_hparams_name = "dump_hparams"
    dump_hparams_name = dump_hparams_name if not prefix_name else "%s_%s" % (
        prefix_name,
        dump_hparams_name,
    )
    export_binary(
        name = dump_hparams_name,
        main = "//paxml/tools:dump_hparams.py",
        py_binary_rule = pytype_strict_binary,
        deps = [
            "//paxml/tools:dump_hparams_lib",
        ] + extra_deps,
        exp_sources = exp_sources,
        exec_properties = {"mem": "20g"},  # dump_hparams can be a very large executable.
        # Implicit py_binary flag
    )

    dump_input_specs_name = "dump_input_specs"
    dump_input_specs_name = dump_input_specs_name if not prefix_name else "%s_%s" % (
        prefix_name,
        dump_input_specs_name,
    )
    dump_input_specs_kwargs = dump_input_specs_kwargs or {}
    export_binary(
        name = dump_input_specs_name,
        main = "//paxml/tools:dump_input_specs.py",
        py_binary_rule = pytype_strict_binary,
        deps = [
            # Implicit absl.app dependency.
            # Implicit absl.flags dependency.
            "//paxml:experiment_registry",
            "//paxml/tools:dump_input_specs_lib",
            # Implicit tensorflow_no_contrib dependency.
        ] + extra_deps,
        exp_sources = exp_sources,
        **dump_input_specs_kwargs
    )

    model_analysis_name = "model_analysis"
    model_analysis_name = model_analysis_name if not prefix_name else "%s_%s" % (
        prefix_name,
        model_analysis_name,
    )
    model_analysis_kwargs = model_analysis_kwargs or {}
    export_binary(
        name = model_analysis_name,
        main = "//paxml/tools:model_analysis.py",
        py_binary_rule = pytype_strict_binary,
        deps = [
            # Implicit absl.app dependency.
            # Implicit absl.flags dependency.
            # Implicit jax dependency.
            # Implicit numpy dependency.
            "//paxml:experiment_registry",
            "//paxml:tasks_lib",
            "//paxml:trainer_lib",
            "//praxis:base_layer",
            "//praxis:py_utils",
        ] + extra_deps,
        exp_sources = exp_sources,
        **model_analysis_kwargs
    )

    validate_config_name = "validate_config"
    validate_config_name = validate_config_name if not prefix_name else "%s_%s" % (
        prefix_name,
        validate_config_name,
    )
    export_binary(
        name = validate_config_name,
        main = "//paxml/tools:validate_config.py",
        py_binary_rule = pytype_strict_binary,
        deps = [
            "//paxml/tools:validate_config_lib",
        ] + extra_deps,
        exp_sources = exp_sources,
        exec_properties = {"mem": "20g"},  # validate_config is a very large executable.
        # Implicit py_binary flag
    )

    # Internal mock backend target.

def export_test(
        name,
        test_src,
        deps,
        exp_sources,
        py_test_rule,
        args,
        **kwargs):
    """Define a `py_test()` at the current package.

    Args:
      name: name of the generated test.
      test_src: target of the original source of the py_test.
      deps: Dependencies of the py_test.
      exp_sources: target of experiment source files.
      args: arguments/flags to be passed to the test rule.
      **kwargs: all remaining arguments are passed through.
    """
    test_copied = "%s.py" % name
    _copy_src(output_name = test_copied, source_target = test_src, exp_sources = exp_sources)

    py_test_rule(
        name = name,
        srcs = [test_copied],
        args = args,
        deps = deps,
        **kwargs
    )

def export_binary(
        name,
        main,
        deps,
        py_binary_rule,
        exp_sources,
        # Internal arguments for fragmented build.
        **kwargs):
    """Define an existing `py_binary()` at the current package.

    Args:
      name: name of the generated rule.
      main: Binary src.
      deps: Dependencies required by binary src.
      py_binary_rule: the Bazel rule to use to create the final binary.
      exp_sources: target of experiment source files.
      **kwargs: all remaining arguments are passed through.
    """
    main_copied = "%s.py" % name
    _copy_src(output_name = main_copied, source_target = main, exp_sources = exp_sources)

    # Internal implementation for fragmented build.

    # Main script.
    py_binary_rule(
        name = name,
        main = main_copied,
        srcs = [main_copied],
        deps = deps,
        **kwargs
    )

def _copy_src(output_name, source_target, exp_sources):
    # To avoid build warning when using `srcs` on a `py_binary()` outside the
    # current package, copy the file locally with a new rule.
    # We also prepend the source file with imports that registers all
    # experiments.
    native.genrule(
        name = output_name + ".copy",
        outs = [output_name],
        srcs = [source_target, exp_sources],
        cmd = """cat <<EOF > $@ && cat $(location %s) >> $@
# Auto-generated code to import and register all experiments.
# See Pax's pax_targets() Starlark macro.
import importlib
import_str = '$(locations %s)'
for d in import_str.split(' '):
  assert d.endswith('.py'), d
  d = d.replace('/', '.')[:-len('.py')]
  # internal build_defs.bzl imports code
  importlib.import_module(d)
# End of auto-generated code.

EOF
        """ % (source_target, exp_sources),
    )
