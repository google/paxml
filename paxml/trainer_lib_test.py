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

"""Tests for trainer_lib."""

import itertools
from typing import Tuple

from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
import jax.numpy as jnp
from paxml import tasks_lib
from paxml import trainer_lib
from praxis import base_hyperparams
from praxis import base_layer
from praxis import base_model
from praxis import optimizers
from praxis import pax_fiddle
from praxis import py_utils
from praxis import pytypes
from praxis import schedules


NestedMap = py_utils.NestedMap
JTensor = pytypes.JTensor


class RunningModeTest(parameterized.TestCase):

  def test_unknown_mode(self):
    self.assertEqual(
        trainer_lib.RunningMode.detect(False, False, False),
        trainer_lib.RunningMode.UNKNOWN,
    )

  @parameterized.parameters(
      ('has_train', True, False, False),
      ('has_train', True, True, False),
      ('has_train', True, False, True),
      ('has_train', True, True, True),
      ('has_eval', False, True, False),
      ('has_eval', True, True, False),
      ('has_eval', False, True, True),
      ('has_eval', True, True, True),
      ('has_decode', False, False, True),
      ('has_decode', False, True, True),
      ('has_decode', True, False, True),
      ('has_decode', True, True, True),
  )
  def test_valid_modes(
      self, running_mode, has_train_metrics, has_eval_metrics, has_test_metrics
  ):
    self.assertTrue(
        getattr(
            trainer_lib.RunningMode.detect(
                has_train_metrics, has_eval_metrics, has_test_metrics
            ),
            running_mode,
        )
    )


class TestModel(base_model.BaseModel):
  """Simple model for testing.

  Attributes:
    input_dims: Depth of the input.
    output_dims: Depth of the output.
  """

  input_dims: int = 0
  output_dims: int = 0

  def setup(self) -> None:
    self.create_variable(
        'weights',
        base_layer.WeightHParams(shape=[self.input_dims, self.output_dims]),
    )

  def compute_predictions(self, input_batch: NestedMap) -> JTensor:
    ret = jnp.einsum('bi,io->bo', input_batch.inputs, self.theta.weights)
    self.add_summary('debug', ret, verbosity=4)
    self.add_summary('info', ret, verbosity=3)
    return ret

  def compute_loss(
      self, predictions: JTensor, input_batch: NestedMap
  ) -> Tuple[NestedMap, NestedMap]:
    del input_batch
    prediction_loss = jnp.sum(predictions)
    theta_loss = jnp.max(jnp.abs(self.theta.weights))
    # Here loss is the main loss to back-prop into, and loss02 is an eval
    # metric.
    per_example_out = NestedMap()
    return (
        NestedMap(
            prediction_loss=(
                prediction_loss,
                jnp.array(1.0, prediction_loss.dtype),
            ),
            theta_loss=(theta_loss, jnp.array(1.0, theta_loss.dtype)),
        ),
        per_example_out,
    )


class TrainerLibTest(parameterized.TestCase):

  def _create_jax_task(self, input_dims: int, output_dims: int):
    config = pax_fiddle.Config(tasks_lib.SingleTask, name='task')
    config.model = pax_fiddle.Config(
        TestModel,
        name='test_model',
        input_dims=input_dims,
        output_dims=output_dims,
    )
    learner = config.train.learner
    learner.loss_name = 'loss'
    learner.optimizer = pax_fiddle.Config(optimizers.Adam)
    learner.optimizer.lr_schedule = pax_fiddle.Config(schedules.Constant)

    return base_hyperparams.instantiate(config)

  @parameterized.parameters(itertools.product((True, False), (True, False)))
  def test_create_train_state_metadata(self, discard_opt_states, do_eval):
    input_dims = 3
    output_dims = 5
    inputs = jnp.ones((1, input_dims), dtype=jnp.float32)
    task = self._create_jax_task(input_dims, output_dims)
    train_shape_dtype = NestedMap(inputs=inputs)

    metadata = trainer_lib.create_train_state_metadata(
        task, train_shape_dtype, discard_opt_states, do_eval
    )
    self.assertTrue((metadata.input_shape_dtype['inputs'] == inputs).all())

    var_weight_hparams = task.model.abstract_init_with_metadata(
        train_shape_dtype, do_eval=do_eval
    )
    self.assertEqual(metadata.var_weight_hparams, var_weight_hparams)

    padded_global_shapes = task.create_train_state_padded_shapes(
        var_weight_hparams, discard_opt_states=discard_opt_states
    )
    self.assertEqual(metadata.padded_global_shapes, padded_global_shapes)

    unpadded_global_shapes = task.create_train_state_unpadded_shapes(
        var_weight_hparams, discard_opt_states=discard_opt_states
    )
    self.assertEqual(metadata.unpadded_global_shapes, unpadded_global_shapes)

    partition_specs = task.create_train_state_partition_specs(
        var_weight_hparams, discard_opt_states=discard_opt_states
    )
    self.assertEqual(metadata.partition_specs, partition_specs)

  @parameterized.parameters(itertools.product((True, False), (True, False)))
  def test_write_post_init_model_hparams_file(
      self, discard_opt_states, do_eval
  ):
    input_dims = 3
    output_dims = 5
    layer_cfg = pax_fiddle.Config(
        TestModel,
        name='test_model',
        input_dims=input_dims,
        output_dims=output_dims,
    )
    model = pax_fiddle.build(layer_cfg)
    inputs = jnp.ones((1, input_dims), dtype=jnp.float32)
    task = self._create_jax_task(input_dims, output_dims)
    train_shape_dtype = NestedMap(inputs=inputs)
    train_state_metadata = trainer_lib.create_train_state_metadata(
        task, train_shape_dtype, discard_opt_states, do_eval
    )
    job_log_dir = (
        epath.Path(absltest.get_default_test_tmpdir())
        / f'model_hparams_{discard_opt_states}_{do_eval}'
    )
    trainer_lib.write_post_init_model_hparams_file(
        model, train_state_metadata, job_log_dir, do_eval
    )

    params_fpath = job_log_dir / 'post_init_model_params.txt'
    with params_fpath.open() as params_file:
      hyper_params, param_weights = params_file.read().split('\n\n')

    hyper_params_config = model.abstract_init_with_mdl_config(
        train_state_metadata.input_shape_dtype, do_eval=do_eval
    )
    hyper_params_expected = base_hyperparams.nested_struct_to_text(
        hyper_params_config
    )
    param_weights_expected = base_hyperparams.nested_struct_to_text(
        train_state_metadata.var_weight_hparams
    )

    self.assertEqual(hyper_params.strip(), hyper_params_expected.strip())
    self.assertEqual(param_weights.strip(), param_weights_expected.strip())


if __name__ == '__main__':
  absltest.main()
