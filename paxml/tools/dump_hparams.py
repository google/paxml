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

r"""A simple utility to dump experiment hparams to a txt file.

The binary target `:dump_hparams` is defined by `pax_targets()` in the `BUILD`
file.

Example commandline:
python paxml/tools/dump_hparams.py \
    --exp=tasks.lm.params.lm_cloud.LmCloudTransformerAdamTest \
    --params_ofile=/tmp/bert.txt

To examine post-init model params, specify one more parameter:
  --post_init_params_ofile=/tmp/lm_post.txt
"""

import contextlib
import os
from typing import Optional

from absl import app
from absl import flags
from absl import logging
import jax
from jax.experimental import maps
import numpy as np
from paxml import base_experiment
from paxml import experiment_registry
from praxis import base_hyperparams
from praxis import base_layer
from praxis import py_utils
import tensorflow.compat.v2 as tf


flags.DEFINE_string('exp', None, 'A registered experiment name.')
flags.DEFINE_string('params_ofile', None, 'Dump pre-init params to this file.')
flags.DEFINE_string('post_init_params_ofile', None,
                    'If not None, Dump post-init params to this file.')

FLAGS = flags.FLAGS
BaseExperimentT = base_experiment.BaseExperimentT
instantiate = base_layer.instantiate


def _get_experiment(experiment_name: str) -> BaseExperimentT:
  """Retrieves a experiment config from the global registry."""
  experiment_class = experiment_registry.get(experiment_name)
  if experiment_class is None:
    raise ValueError(f'Could not find experiment `{experiment_name}`.')
  return experiment_class


def _write_post_init_model_hparams_file(model_param, filepath,
                                        input_specs) -> None:
  """Dumps post init model hparams file."""
  model = instantiate(model_param)

  prng_key = jax.random.PRNGKey(seed=123)

  def gen_post_init_hparams(prng_key):
    return model.apply({},
                       rngs={base_layer.PARAMS: prng_key},
                       method=model.post_init_hparams,
                       mutable=True)[1]

  variables_abstract = jax.eval_shape(gen_post_init_hparams, prng_key)
  assert base_layer.HYPER_PARAMS in variables_abstract

  hyper_params = jax.tree_map(
      lambda x: x.meta,
      variables_abstract[base_layer.HYPER_PARAMS],
      is_leaf=lambda x: isinstance(x, base_layer.WrappedHParams))

  with tf.io.gfile.GFile(filepath, 'w') as fout:

    hyper_params_dump = base_hyperparams.nested_struct_to_text(hyper_params)
    fout.write(hyper_params_dump)
    fout.write('\n\n')

    # TODO(pax-dev): Add better/cleaner API to identify pmap vs. pjit models
    # (and check for dcn_mesh_shape too).
    if hasattr(model, 'ici_mesh_shape') and model.ici_mesh_shape is not None:
      input_specs = jax.tree_map(py_utils.get_global_input_shape_dtype,
                                 input_specs)
    params_inits = model.abstract_init_with_metadata(input_specs)
    params_inits_text = base_hyperparams.nested_struct_to_text(params_inits)
    fout.write(params_inits_text)


def _extract_num_cores(model_p) -> Optional[int]:
  """Extracts the number of cores used by the experiment.

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


def main(argv) -> None:
  del argv  # unused.

  # We use xmap only with SPMD.
  jax.config.update('experimental_xmap_spmd_lowering', True)
  # Use the manual partitioning lowering of xmap to avoid vectorization.
  jax.config.update('experimental_xmap_spmd_lowering_manual', True)

  logging.info('Dumping out params for experiment: %s', FLAGS.exp)
  experiment_config = _get_experiment(FLAGS.exp)()
  task_p = experiment_config.task()

  num_cores = _extract_num_cores(task_p.model)

  prev_xla_flags = os.getenv('XLA_FLAGS')
  flags_str = prev_xla_flags or ''
  # Don't override user-specified device count, or other XLA flags.
  if (num_cores is not None and
      'xla_force_host_platform_device_count' not in flags_str):
    os.environ['XLA_FLAGS'] = (
        flags_str + f'--xla_force_host_platform_device_count={num_cores}')

  with tf.io.gfile.GFile(FLAGS.params_ofile, 'w') as params_file:
    params_file.write('============= Trainer / Evaler datasets.\n\n')
    datasets = experiment_config.datasets()
    for dataset in datasets:
      params_file.write(dataset.to_text())
      params_file.write('\n\n')

    params_file.write('============= Decoder datasets.\n\n')
    dec_datasets = experiment_config.decoder_datasets()
    for dataset in dec_datasets:
      params_file.write(dataset.to_text())
      params_file.write('\n\n')

    params_file.write(task_p.to_text())
    # TODO(b/236417790): Dump input specs for model weight initialization.

  if FLAGS.post_init_params_ofile:
    input_specs_provider = instantiate(
        experiment_config.get_input_specs_provider_params())
    input_specs = input_specs_provider.get_input_specs()

    if task_p.model.dcn_mesh_shape is not None:
      device_mesh = py_utils.create_device_mesh(task_p.model.ici_mesh_shape,
                                                task_p.model.dcn_mesh_shape)
      context_manager = maps.Mesh(device_mesh, task_p.model.mesh_axis_names)
    else:
      context_manager = contextlib.nullcontext()

    with context_manager:
      _write_post_init_model_hparams_file(task_p.model,
                                          FLAGS.post_init_params_ofile,
                                          input_specs)


if __name__ == '__main__':
  app.run(main)
