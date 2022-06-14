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
bazel run //PATH/TO/PAX/TARGETS:dump_hparams -- \
    --exp=lm1b.Lm1bTransformerL32H8kSPMD8x8Repeat \
    --params_ofile=/tmp/lm.txt

To examine post-init model params, specify one more parameter:
  --post_init_params_ofile=/tmp/lm_post.txt
"""

from absl import app
from absl import flags
from absl import logging
import jax
from paxml import base_experiment
from paxml import experiment_registry
from praxis import base_hyperparams
from praxis import base_layer
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


def _write_post_init_model_hparams_file(model_param, filepath) -> None:
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

    params_inits = model.abstract_init_with_metadata(prng_key)
    params_inits_text = base_hyperparams.nested_struct_to_text(params_inits)
    fout.write(params_inits_text)


def main(argv) -> None:
  del argv  # unused.

  logging.info('Dumping out params for experiment: %s', FLAGS.exp)
  experiment_config = _get_experiment(FLAGS.exp)()

  with tf.io.gfile.GFile(FLAGS.params_ofile, 'w') as params_file:
    datasets = experiment_config.datasets()
    for dataset in datasets:
      params_file.write(dataset.to_text())
      params_file.write('\n\n')
    params_file.write(experiment_config.task().to_text())

  if FLAGS.post_init_params_ofile:
    _write_post_init_model_hparams_file(experiment_config.task().model,
                                        FLAGS.post_init_params_ofile)


if __name__ == '__main__':
  app.run(main)
