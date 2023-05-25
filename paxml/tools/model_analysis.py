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

r"""Compute FLOPS or Debug locally in Pax.

**************
Example usage:
**************

python paxml/tools/model_analysis.py -- \
  --usage=flops \
  --exp=tasks.lm.params.bert.BertAdamL4H128 \
  --fprop_func=compute_predictions

**************
Example output 0 (--usage=flops):
**************

##### bert.BertAdamL4H128 #####

GFLOPS = 47.58


**************
Example output 1 (--usage=params):
**************

params/lm/final_ln/bias                                                              (128,)                   128             trainable
params/lm/final_ln/scale                                                             (128,)                   128             trainable
params/lm/softmax/logits_ffn/bias/b                                                  (32000,)                 32000           trainable
params/lm/softmax/logits_ffn/linear/w                                                (128, 32000)             4096000         trainable
params/lm/transformer/x_layers_0/ff_layer/ffn_layer1/bias/b                          (512,)                   512             trainable
params/lm/transformer/x_layers_0/ff_layer/ffn_layer1/linear/w                        (128, 512)               65536           trainable

...

params/lm/transformer/x_layers_3/self_attention/key/w                                (128, 8, 16)             16384           trainable
params/lm/transformer/x_layers_3/self_attention/per_dim_scale/per_dim_scale          (16,)                    16              trainable
params/lm/transformer/x_layers_3/self_attention/post/w                               (128, 8, 16)             16384           trainable
params/lm/transformer/x_layers_3/self_attention/query/w                              (128, 8, 16)             16384           trainable
params/lm/transformer/x_layers_3/self_attention/value/w                              (128, 8, 16)             16384           trainable
===========================================================================
Total #params (all): 4919360
Total #params (trainable): 4919360

"""

from absl import app
from absl import flags
import jax
import jax.numpy as jnp
import numpy as np
from paxml import experiment_registry
from paxml import tasks_lib
from paxml import trainer_lib
from praxis import base_layer
from praxis import py_utils


_USAGE = flags.DEFINE_string(
    'usage',
    'flops',
    'The purpose of using this script. Pls choose from [flops, fprop, params].',
)
_EXP = flags.DEFINE_string(
    'exp',
    None,
    'The exp to debug or to compute flops upon.',
)
_FPROP_FUNC = flags.DEFINE_string(
    'fprop_func', 'compute_predictions', 'The function used in fprop.'
)


class ExperimentParser:
  """The class which helps analyze the experiment setups.

  3 Options:
  flops: compute the total FLOPS of the experiment.
  params: print all the model parameters and their shapes, sizes, etc.
  fprop: a forward pass, mainly for the debugging purpose.
  """

  def __init__(self, exp_name: str, seed: int = 1202):
    self.exp_name = exp_name
    exp_class = self._get_experiment(exp_name)
    exp = exp_class()
    self.exp = exp

    self.seed = seed
    self.prng_key = jax.random.PRNGKey(1202)
    _, self.init_key = jax.random.split(self.prng_key)

  def _get_experiment(self, experiment_name: str):
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

  def _extract_task(self):
    assert self.exp is not None
    exp = self.exp
    task_p = exp.task()
    task = task_p.Instantiate()
    return task

  def _extract_input_specs(self):
    assert self.exp is not None
    input_specs_provider_p = self.exp.get_input_specs_provider_params()
    input_specs_provider = input_specs_provider_p.Instantiate()
    input_specs = input_specs_provider.get_input_specs()
    return input_specs

  def _generate_datum(self):
    input_specs = self._extract_input_specs()
    datum = jax.tree_map(
        lambda x: jnp.zeros(shape=x.shape, dtype=x.dtype), input_specs
    )
    return datum

  def _extract_model(self, datum):
    task = self._extract_task()
    model = task.model
    model_states, _ = trainer_lib.initialize_model_state(
        task,
        self.init_key,
        datum,
        discard_opt_states=True,
        do_init_checkpoint_rules=False,
        is_eval=True,
    )
    return model, model_states

  def _create_fprop_func(self, fprop_func: str = 'compute_predictions'):
    datum = self._generate_datum()
    model, model_states = self._extract_model(datum)
    model_vars = model_states.mdl_vars

    if not hasattr(model, fprop_func):
      raise ValueError(f'{fprop_func} does not exist in {self.exp_name}.')
    model_fprop = lambda input_batch: model.apply(
        model_vars,
        input_batch,
        method=getattr(model, fprop_func),
        rngs={base_layer.RANDOM: self.init_key},
        mutable=True,
    )
    return model_fprop, datum

  def get_flops(self, fprop_func: str = 'compute_predictions'):
    model_fprop, datum = self._create_fprop_func(fprop_func)
    with base_layer.JaxContext.new_context(
        hparams=base_layer.JaxContext.HParams(do_eval=True)
    ):
      analysis = jax.jit(model_fprop).lower(datum).cost_analysis()
      flops = analysis['flops']
      gflops = flops / 1e9

      print('\n' + '#' * 5 + ' ' + self.exp_name + ' ' + '#' * 5)
      print(f'\nGFLOPS = {gflops:.2f}')

  def fprop(self, fprop_func: str = 'compute_predictions'):
    model_fprop, datum = self._create_fprop_func(fprop_func)
    with base_layer.JaxContext.new_context(
        hparams=base_layer.JaxContext.HParams(do_eval=True)
    ):
      model_fprop(datum)
      print('Run finishes successfully.')

  def print_variables_info(self):
    task_p = self.exp.task()
    task = task_p.Instantiate()
    datum = self._generate_datum()
    model, model_states = self._extract_model(datum)
    model_vars = model_states.mdl_vars

    with base_layer.JaxContext.new_context(
        hparams=base_layer.JaxContext.HParams(do_eval=False)
    ):
      var_weight_hparams = model.abstract_init_with_metadata(datum)

    learner = task.learners[0]
    excluded_for_grad = tasks_lib.get_excluded_var_mask_for_grad(
        var_weight_hparams, learner
    )
    included_for_grad = jax.tree_map(lambda x: not x, excluded_for_grad)
    trainable_variables = py_utils.NestedMap.FromNestedDict(included_for_grad)

    prefixes = py_utils.extract_prefixed_keys_from_nested_map(
        trainable_variables
    )
    max_param_len = max(
        [len(prefix) for prefix in jax.tree_util.tree_leaves(prefixes)]
    )
    params_count = {
        'non_trainable': 0,
        'trainable': 0,
        'frozen': 0,
        'others': 0,
    }
    params_list = []

    def collect_params(param_name, included, param_weight):
      param_shape = tuple(np.array(param_weight.shape))
      param_size = np.prod(param_shape)
      if param_name.startswith('non_trainable'):
        trainable_type = 'non_trainable'
      elif included:
        trainable_type = 'trainable'
      elif param_name.startswith('params'):
        trainable_type = 'frozen'
      else:
        trainable_type = 'others'
      params_count[trainable_type] += param_size
      leaf = param_name, param_shape, param_size, trainable_type
      params_list.append(leaf)
      return leaf

    jax.tree_map(
        collect_params,
        prefixes,
        trainable_variables,
        py_utils.NestedMap.FromNestedDict(model_vars),
    )
    for param_name, param_shape, param_size, trainable_type in params_list:
      output_line = output_line = (
          f'{param_name:<85}{str(param_shape):<25}{str(int(param_size)):16}'
          f'{trainable_type}'
      )
      print(output_line)
    print('=' * max_param_len)
    total_params_count = sum(params_count.values())
    print(f'Total #params (all): {total_params_count}')
    for key, value in params_count.items():
      if value > 0:
        print(f'Total #params ({key}): {value}')


def main(unused_argv):
  if _USAGE.value not in ['flops', 'fprop', 'params']:
    raise ValueError(f'Usage {_USAGE.value} is not supported yet.')

  exp_parser = ExperimentParser(_EXP.value)

  if _USAGE.value == 'params':
    exp_parser.print_variables_info()
  elif _USAGE.value == 'flops':
    exp_parser.get_flops(_FPROP_FUNC.value)
  else:
    exp_parser.fprop(_FPROP_FUNC.value)


if __name__ == '__main__':
  app.run(main)
