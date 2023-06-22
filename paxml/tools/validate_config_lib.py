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

r"""A simple utility to validate an experiment config.

See validate_config.py for usage.
"""

import contextlib
import os
from typing import Optional

from absl import app
from absl import flags
from absl import logging
from fiddle import absl_flags
import jax
import numpy as np
from paxml import base_experiment
from paxml import experiment_registry
from praxis import base_layer
from praxis import pax_fiddle
from praxis import py_utils


_EXP = flags.DEFINE_string('exp', None, 'A registered experiment name.')
_CHECK_LEVEL = flags.DEFINE_integer(
    'check_level',
    0,
    (
        'Level of completeness of the checks as a scale from 0 (minimum number'
        ' of checks) to 10.'
    ),
)

FLAGS = flags.FLAGS
BaseExperimentT = base_experiment.BaseExperimentT
instantiate = base_layer.instantiate


def _get_experiment(experiment_name: str) -> BaseExperimentT:
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


def _hparams_post_init(model_param, input_specs) -> None:
  """Calls post-init of model hparams."""
  model = instantiate(model_param)

  # TODO(pax-dev): Add better/cleaner API to identify pmap vs. pjit models
  # (and check for dcn_mesh_shape too).
  if hasattr(model, 'ici_mesh_shape') and model.ici_mesh_shape is not None:
    input_specs = jax.tree_map(
        py_utils.get_global_input_shape_dtype, input_specs
    )

  _ = model.abstract_init_with_mdl_config(input_specs)
  _ = model.abstract_init_with_metadata(input_specs)


def _extract_num_cores(model_p) -> Optional[int]:
  """Extracts the number of cores used by the experiment.

  Args:
    model_p: The model hparams.

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

  logging.info('Retrieving experiment `%s` from the registry.', _EXP.value)
  if _EXP.value is not None:
    experiment_config = _get_experiment(_EXP.value)()
  else:
    cfg = absl_flags.create_buildable_from_flags(
        module=None, allow_imports=True)
    experiment_config = pax_fiddle.build(cfg)

  if _CHECK_LEVEL.value <= 0:
    return

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

  _ = instantiate(task_p)
  if _CHECK_LEVEL.value <= 2:
    return

  input_specs_provider = instantiate(
      experiment_config.get_input_specs_provider_params()
  )
  input_specs = input_specs_provider.get_input_specs()
  if task_p.model.dcn_mesh_shape is not None:
    device_mesh = py_utils.create_device_mesh(
        task_p.model.ici_mesh_shape,
        task_p.model.dcn_mesh_shape,
        contiguous_submeshes=task_p.model.contiguous_submeshes,
    )
    context_manager = jax.sharding.Mesh(
        device_mesh, task_p.model.mesh_axis_names
    )
  else:
    context_manager = contextlib.nullcontext()
  with context_manager:
    _hparams_post_init(task_p.model, input_specs)
  if _CHECK_LEVEL.value <= 4:
    return

  datasets = experiment_config.datasets()
  for dataset in datasets:
    if _CHECK_LEVEL.value >= 8:
      _ = instantiate(dataset)

  dec_datasets = experiment_config.decoder_datasets()
  for dataset in dec_datasets:
    if _CHECK_LEVEL.value >= 8:
      _ = instantiate(dataset)


def main():
  app.run(_main, flags_parser=absl_flags.flags_parser)
