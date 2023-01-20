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

"""Starlark macros for Pax users."""

load("//paxml:paxml.bzl", "py_strict_test")
load("//paxml:paxml.bzl", "pytype_binary", "pytype_strict_binary")

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
        smoke_test_exclude_regexes = "",
        smoke_test_include_only_regexes = "",
        smoke_test_args = None,
        smoke_test_kwargs = None,
        main_src = "//paxml:main.py"):
    """Macro to define a collection of Pax targets with custom dependencies.

    It currently defines the following targets:

    ":main", a Python binary that can be passed to the xm launcher to run
        the experiments.
    ":main_mpm", an MPM target, which contains main.par, that can be used
        in the xm launcher with --binary_type=mpm_target. Only available if
        `add_main_mpm_target` is True.
    ":all_experiments_smoke_test", a Python test that runs a sanity check
        on all registered experiments.
    ":dump_hparams", a Python util binary that writes experiment hparams to
        file.

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
      smoke_test_args: The list of command line arguments that can be passed to the
       :all_experiments_smoke_test target.
      smoke_test_exclude_regexes: Exclusion regexes of experiment configurations to be
          passed to the smoke test. The matching experiment configurations will
          be disabled from the smoke test.
      smoke_test_include_only_regexes: If provided, then any experiment name must
          match one of these regexes in order to be smoke tested.
      smoke_test_kwargs: Additional kwargs that are passed to the
          :all_experiments_smoke_test target.
      main_src: The src file for the ":main" target created.
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

    main_name = "main"
    main_name = main_name if not prefix_name else "%s_%s" % (prefix_name, main_name)
    export_binary(
        name = main_name,
        main = main_src,
        py_binary_rule = pytype_binary,
        deps = [
            "//paxml:main_lib",
            # Implicit tpu dependency.
        ] + extra_deps,
        exp_sources = exp_sources,
        # Implicit py_binary flag
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

    test_name = "all_experiments_smoke_test"
    test_name = test_name if not prefix_name else "%s_%s" % (prefix_name, test_name)

    smoke_test_args = smoke_test_args or []
    if smoke_test_exclude_regexes:
        smoke_test_args.append("--exclude_regexes=" + _shell_quote(smoke_test_exclude_regexes))
    if smoke_test_include_only_regexes:
        smoke_test_args.append("--include_only_regexes=" + _shell_quote(smoke_test_include_only_regexes))

    smoke_test_kwargs = smoke_test_kwargs or {}
    _export_test(
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
            # Implicit absl.app dependency.
            # Implicit absl.flags dependency.
            # Implicit absl.logging dependency.
            # Implicit jax dependency.
            # Implicit numpy dependency.
            "//paxml:base_experiment",
            "//paxml:experiment_registry",
            "//praxis:base_hyperparams",
            "//praxis:base_layer",
            "//praxis:py_utils",
            # Implicit tensorflow_no_contrib dependency.
        ] + extra_deps,
        exp_sources = exp_sources,
    )

    dump_input_specs_name = "dump_input_specs"
    dump_input_specs_name = dump_input_specs_name if not prefix_name else "%s_%s" % (
        prefix_name,
        dump_input_specs_name,
    )
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
    )

def _export_test(
        name,
        test_src,
        deps,
        exp_sources,
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

    py_strict_test(
        name = name,
        python_version = "PY3",
        srcs_version = "PY3",
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

    # Main script.
    py_binary_rule(
        name = name,
        python_version = "PY3",
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
