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

r"""A simple utility to dump hparams.

See dump_hparams.py for usage.
"""

import contextlib
import os

from absl import app
from absl import flags
from absl import logging
from etils import epath
from fiddle import absl_flags
import jax
import numpy as np
from paxml import base_experiment
from paxml import experiment_registry
from paxml import experiment_utils
from praxis import base_hyperparams
from praxis import base_layer
from praxis import pax_fiddle as fdl
from praxis import py_utils
from praxis import pytypes
import pyglove as pg


_EXP = flags.DEFINE_string('exp', None, 'A registered experiment name.')
_OUTFILE = epath.DEFINE_path(
    'params_ofile', '/dev/stdout', 'Dump pre-init params to this.'
)
_POST_INIT_OUTFILE = epath.DEFINE_path(
    'post_init_params_ofile',
    None,
    'If not None, Dump post-init params to this file.',
)
_CLS_VARS_OUTFILE = epath.DEFINE_path(
    'cls_vars_ofile',
    None,
    'If not None, Dump experiment_cls_vars this file.',
)

instantiate = base_layer.instantiate


def _get_experiment(experiment_name: str) -> base_experiment.BaseExperimentT:
  """Retrieves a experiment config from the global registry."""
  experiment_class = experiment_registry.get(experiment_name)
  if experiment_class is None:
    all_experiments = list(experiment_registry.get_all().keys())
    all_names = '\n'.join(all_experiments)
    msg = (
        f'Could not find experiment {experiment_name}. Available experiments '
        f'are:\n{all_names}.'
    )
    raise ValueError(msg)
  return experiment_class


def _write_post_init_model_hparams_file(
    model_param: fdl.Config[base_layer.BaseLayer],
    filepath: epath.Path,
    input_specs: pytypes.NestedShapeDtypeStruct,
) -> None:
  """Dumps post init model hparams file."""
  model = instantiate(model_param)

  # TODO(pax-dev): Add better/cleaner API to identify pmap vs. pjit models
  # (and check for dcn_mesh_shape too).
  if hasattr(model, 'ici_mesh_shape') and model.ici_mesh_shape is not None:
    input_specs = jax.tree_map(
        py_utils.get_global_input_shape_dtype, input_specs
    )

  hyper_params = model.abstract_init_with_mdl_config(input_specs)
  params_inits = model.abstract_init_with_metadata(input_specs)

  with filepath.open('w') as fout:
    hyper_params_dump = base_hyperparams.nested_struct_to_text(hyper_params)
    fout.write(hyper_params_dump)
    fout.write('\n\n')

    params_inits_text = base_hyperparams.nested_struct_to_text(params_inits)
    fout.write(params_inits_text)


def _write_cls_vars_file(
    exp_config: base_experiment.BaseExperiment,
    filepath: epath.Path,
) -> None:
  """Dumps experiment_cls_vars.txt of a model's type to file."""
  filepath.parent.mkdir(parents=True, exist_ok=True)
  with filepath.open('w') as fout:
    cls_vars_summary = experiment_utils.get_cls_vars_summary(type(exp_config))
    fout.write(cls_vars_summary)


def _extract_num_cores(model_p) -> int | None:
  """Extracts the number of cores used by the experiment.

  Args:
    model_p: The model config.

  Returns:
    The number of cores across all TPU slices for pjit experiments or None for
    pmap ones.
  """

  if model_p.ici_mesh_shape is None:
    return None

  def _compute_num_cores(mesh_shape) -> int:
    if mesh_shape is None:
      # Default to 1 if unset.
      return 1
    return np.prod(mesh_shape).item()

  ici_num_cores = _compute_num_cores(model_p.ici_mesh_shape)
  dcn_num_cores = _compute_num_cores(model_p.dcn_mesh_shape)
  return ici_num_cores * dcn_num_cores


def _main(argv) -> None:
  del argv  # unused.

  if _EXP.value is not None:
    logging.info('Dumping out params for experiment: %s', _EXP.value)
    experiment_config = _get_experiment(_EXP.value)()
  else:
    logging.info('Dumping out params from fiddle configuration')
    cfg = absl_flags.create_buildable_from_flags(
        module=None, allow_imports=True
    )
    experiment_config = fdl.build(cfg)

  # NOTE(daiyip): putting `task()`, `datasets()` and `decode_datasets()` under
  # an AutoML context allows dynamic evaluation of hyperparameters that is to
  # be swept. The first values of all `pg.oneof` will be used.
  automl_context = pg.hyper.DynamicEvaluationContext(require_hyper_name=True)
  with automl_context.collect():
    task_p = experiment_config.task()

  num_cores = _extract_num_cores(task_p.model)

  prev_xla_flags = os.getenv('XLA_FLAGS')
  flags_str = prev_xla_flags or ''
  # Don't override user-specified device count, or other XLA flags.
  if (
      num_cores is not None
      and 'xla_force_host_platform_device_count' not in flags_str
  ):
    os.environ['XLA_FLAGS'] = (
        flags_str + f'--xla_force_host_platform_device_count={num_cores}'
    )

  # Note datasets and decode_datasets must be called after setting XLA_FLAGS
  # because it may run JAX and initialized XLA backend without XLA_FLAGS.
  with automl_context.collect():
    datasets = experiment_config.datasets()
    dec_datasets = experiment_config.decode_datasets()

  with _OUTFILE.value.open('w') as params_file:
    params_file.write('============= Trainer / Evaler datasets.\n\n')
    for dataset in datasets:
      params_file.write(base_hyperparams.nested_struct_to_text(dataset))
      params_file.write('\n\n')

    params_file.write('============= Decode datasets.\n\n')
    for dataset in dec_datasets:
      params_file.write(base_hyperparams.nested_struct_to_text(dataset))
      params_file.write('\n\n')

    params_file.write(base_hyperparams.nested_struct_to_text(task_p))
    # TODO(b/236417790): Dump input specs for model weight initialization.

  if _POST_INIT_OUTFILE.value:
    input_specs_provider = instantiate(
        experiment_config.get_input_specs_provider_params()
    )
    input_specs = input_specs_provider.get_input_specs()

    if task_p.model.dcn_mesh_shape is not None:
      context_manager = jax.sharding.Mesh(
          py_utils.create_device_mesh(
              task_p.model.ici_mesh_shape,
              task_p.model.dcn_mesh_shape,
              contiguous_submeshes=task_p.model.contiguous_submeshes,
          ),
          task_p.model.mesh_axis_names,
      )
    else:
      context_manager = contextlib.nullcontext()

    with context_manager:
      _write_post_init_model_hparams_file(
          task_p.model, _POST_INIT_OUTFILE.value, input_specs
      )

  if _CLS_VARS_OUTFILE.value:
    _write_cls_vars_file(experiment_config, _CLS_VARS_OUTFILE.value)

def main():
  app.run(_main, flags_parser=absl_flags.flags_parser)
