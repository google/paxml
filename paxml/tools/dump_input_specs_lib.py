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

"""Input specification retrieval from either provider or input pipeline."""

import pprint
from typing import Optional, Tuple

from absl import logging
import jax
from paxml import base_experiment
from praxis import base_hyperparams
from praxis import base_input
from praxis import pytypes
import pyglove as pg

NestedShapeDtypeStruct = pytypes.NestedShapeDtypeStruct
instantiate = base_hyperparams.instantiate


def extract_input_specs(
    experiment_config: base_experiment.BaseExperiment
) -> Tuple[Optional[NestedShapeDtypeStruct], Optional[NestedShapeDtypeStruct]]:
  """Extracts the input specs for a given experiment config."""
  logging.info('Starting extraction of input specs info.')

  # Input specs from input_specs_provider.
  input_specs_provider_p = experiment_config.get_input_specs_provider_params()
  input_specs_from_provider = None
  if not isinstance(input_specs_provider_p,
                    base_input.DatasetInputSpecsProvider):
    logging.info('Extracting input specs info from provider.')
    specs_provider = instantiate(input_specs_provider_p)
    input_specs_from_provider = specs_provider.get_input_specs()

  # NOTE(daiyip): putting `training_dataset()` and `instantiate(train_input_p)`
  # under an AutoML context allows dynamic evaluation of hyperparameters that is
  # to be swept. The first values of all `pg.oneof` will be used.
  with pg.hyper.DynamicEvaluationContext(require_hyper_name=True).collect():
    # Input specs from experiment config
    logging.info('Extracting input specs info from input pipeline.')
    try:
      # Clone it since we may mutate a few attributes below.
      train_input_p = experiment_config.training_dataset().clone()
    except ValueError:
      logging.info('Could not find a training input pipeline for %s',
                   experiment_config)
      train_input_p = None

    if train_input_p is None:
      return input_specs_from_provider, None

    # Attempt at reducing loading time when using Lingvo input.
    if isinstance(train_input_p, base_input.LingvoInputAdaptor):
      train_input_p.input.num_batcher_threads = 1
      train_input_p.input.file_parallelism = 1
      train_input_p.input.file_buffer_size = 32

    logging.info('Instantiating input pipeline...')
    input_pipeline = instantiate(train_input_p)
    logging.info('Retrieving specs from input pipeline...')
    input_specs_from_input_pipeline = jax.tree_util.tree_map(
        lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype),
        input_pipeline.get_next_padded())

    return input_specs_from_provider, input_specs_from_input_pipeline


def specs_to_string(
    experiment_name: str, specs: Tuple[Optional[NestedShapeDtypeStruct],
                                       Optional[NestedShapeDtypeStruct]]
) -> str:
  """Converts input specs into a readable string."""
  pp = pprint.PrettyPrinter(indent=2)
  specs_provider, specs_pipeline = specs
  out_lst = []
  out_lst.append(experiment_name)
  out_lst.append('From InputSpecsProvider:')
  out_lst.append(pp.pformat(specs_provider))
  out_lst.append('From training input pipeline:')
  out_lst.append(pp.pformat(specs_pipeline))
  out_lst.append('\n\n')
  out_str = '\n'.join(out_lst)
  logging.info(out_str)
  return out_str
