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

"""Basic, lightweight tests for checkpoints.py."""

import json

from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
import flax
from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from paxml import checkpoint_metadata
from paxml import checkpoint_paths
from paxml import checkpoint_types
from paxml import checkpoints
from paxml import train_states
from praxis import pytypes


orbax_utils = ocp.utils
ArrayMetadata = checkpoint_metadata.ArrayMetadata
TrainState = train_states.TrainState


class CheckpointsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.directory = epath.Path(
        self.create_tempdir(name='checkpointing_test').full_path
    )

  def test_is_tmp_checkpoint_asset(self):
    # TODO(b/267498552) Remove hasattr when possible.
    if hasattr(orbax_utils, 'is_tmp_checkpoint'):
      step_prefix = 'checkpoint'
      step_dir = orbax_utils.get_save_directory(
          5, self.directory, step_prefix=step_prefix
      )
      step_dir.mkdir(parents=True)
      self.assertFalse(orbax_utils.is_tmp_checkpoint(step_dir))
      tmp_step_dir = orbax_utils.create_tmp_directory(step_dir)
      self.assertTrue(orbax_utils.is_tmp_checkpoint(tmp_step_dir))

      item_dir = orbax_utils.get_save_directory(
          10, self.directory, name='state', step_prefix=step_prefix
      )
      item_dir.mkdir(parents=True)
      self.assertFalse(orbax_utils.is_tmp_checkpoint(item_dir))
      tmp_item_dir = orbax_utils.create_tmp_directory(item_dir)
      self.assertTrue(orbax_utils.is_tmp_checkpoint(tmp_item_dir))

  def test_train_state_type_check(self):
    class Model(nn.Module):

      @nn.compact
      def __call__(self, x: jax.Array) -> jax.Array:
        return nn.linear.Dense(128)(x)

    m = Model()

    model_vars = m.init(jax.random.PRNGKey(0), jnp.zeros([4, 256]))
    optimizer = optax.sgd(0.1)
    opt_params = optimizer.init(model_vars['params'])
    extra_state = ()
    train_state = train_states.TrainState(  # pytype: disable=wrong-arg-types  # dataclass_transform
        jnp.asarray([0], jnp.int64), model_vars, opt_params, extra_state
    )
    # Save the "checkpoint".
    tmp_dir = self.create_tempdir('test_train_state_type_check_checkpoint')
    checkpoints.save_checkpoint(train_state, tmp_dir, overwrite=True)
    # This is the correct usage.
    checkpoints.restore_checkpoint(train_state, tmp_dir)
    # This is what happens if you forget to destructure the (state,
    # provenance) tuple that some PAX APIs return. This should fail.
    with self.assertRaisesRegex(ValueError, 'must be a subclass of') as _:
      checkpoints.restore_checkpoint((train_state, []), tmp_dir)  # type: ignore

  @parameterized.parameters(
      ('foobar123', False),
      ('foobar_123', False),
      ('checkpoint1000', False),
      ('tmp_1010101.checkpoint_100', True),
      ('tmp_1010101.checkpoint_000100', True),
  )
  def test_is_tmp_checkpoint_asset_legacy(self, name, expected_result):
    ckpt = self.directory / name
    self.assertEqual(
        checkpoint_paths.is_tmp_checkpoint_asset(ckpt), expected_result
    )

  def test_is_tmp_checkpoint_asset_legacy_flax(self):
    ckpt = self.directory / 'checkpoint'
    ckpt.write_text('some data')
    self.assertFalse(checkpoint_paths.is_tmp_checkpoint_asset(ckpt))

  @parameterized.parameters(
      (
          epath.Path('/tmp/checkpoints/checkpoint_1234'),
          1234,
      ),
      (
          epath.Path('/tmp/checkpoints/1234'),
          1234,
      ),
  )
  def test_get_step_from_checkpoint_asset(self, path, expected_step):
    self.assertEqual(
        checkpoint_paths.get_step_from_checkpoint_asset(path), expected_step
    )

  @parameterized.parameters(
      'checkpoint_1234',
      '1234',
  )
  def test_latest_checkpoint_succeeds_for_finalized_ckpt(self, directory_name):
    ckpt = self.directory / directory_name
    ckpt.write_text('some data')

    self.assertEqual(
        checkpoints.latest_checkpoint_if_exists(self.directory), ckpt
    )

  @parameterized.parameters(
      'checkpoint1234',
      'tmp_1010101.checkpoint_1234',
  )
  def test_latest_checkpoint_returns_none_for_pending_cpkt(
      self, directory_name
  ):
    ckpt = self.directory / directory_name
    ckpt.write_text('some data')

    self.assertIsNone(checkpoints.latest_checkpoint_if_exists(self.directory))

  def test_retrieve_latest_checkpoint_step_no_checkpoint_raises_exception(self):
    checkpoint_dir = epath.Path(self.create_tempdir('random').full_path)
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        f'No checkpoints were found in directory {checkpoint_dir=!r}',
    ):
      checkpoints.retrieve_latest_checkpoint_step(checkpoint_dir)

  def test_retrieve_latest_checkpoint_step_dir_does_not_exist_raises_exception(
      self,
  ):
    checkpoint_dir = epath.Path('does_not_exist')
    with self.assertRaisesWithLiteralMatch(
        ValueError, f'{checkpoint_dir=!r} does not exist'
    ):
      checkpoints.retrieve_latest_checkpoint_step(checkpoint_dir)


class _CustomPyTreeNode(flax.struct.PyTreeNode):
  a: pytypes.NestedShapeDtypeLike


@flax.struct.dataclass
class _CustomFlaxDataclass:
  a: pytypes.NestedShapeDtypeLike


class PaxMetadataTest(parameterized.TestCase):

  def test_from_dict(self):
    d = dict(
        version=1.0,
        train_state_metadata={
            'a': ArrayMetadata(
                unpadded_shape_dtype_struct=jax.ShapeDtypeStruct(
                    shape=(1, 2), dtype=np.float32
                ),
                is_optax_masked_node=False,
            ).to_dict(),
            'b': {
                'b1': ArrayMetadata(
                    unpadded_shape_dtype_struct=jax.ShapeDtypeStruct(
                        shape=(3, 4), dtype=np.int32
                    ),
                    is_optax_masked_node=True,
                ).to_dict(),
                'b2': ArrayMetadata(
                    unpadded_shape_dtype_struct=jax.ShapeDtypeStruct(
                        shape=(5,), dtype=np.float32
                    ),
                    is_optax_masked_node=False,
                ).to_dict(),
            },
        },
    )

    d_restored = checkpoint_metadata.PaxMetadata.from_dict(d).to_dict()
    self.assertTrue(checkpoint_metadata._trees_are_equal(d, d_restored))

  def test_from_padded_and_unpadded(self):
    padded = TrainState(  # pytype: disable=wrong-arg-types  # dataclass_transform
        step=0,
        opt_states=[],
        extra_state=(),
        mdl_vars={
            'a': jnp.ones((2, 3), dtype=jnp.float32),
            'b': {
                'b1': optax.MaskedNode(),
                'b2': jnp.zeros((9,), dtype=jnp.float32),
            },
        },
    )
    unpadded = TrainState(  # pytype: disable=wrong-arg-types  # dataclass_transform
        step=0,
        opt_states=[],
        extra_state=(),
        mdl_vars={
            'a': jax.ShapeDtypeStruct(shape=(1, 2), dtype=np.float32),
            'b': {
                'b1': jax.ShapeDtypeStruct(shape=(3, 4), dtype=np.int32),
                'b2': jax.ShapeDtypeStruct(shape=(5,), dtype=np.float32),
            },
        },
    )

    d_restored = checkpoint_metadata.PaxMetadata.from_padded_and_unpadded(
        padded,
        unpadded,
        version=1.0,
    ).to_dict()

    d_expected = dict(
        version=1.0,
        train_state_metadata=dict(
            mdl_vars={
                'a': ArrayMetadata(
                    unpadded_shape_dtype_struct=jax.ShapeDtypeStruct(
                        shape=(1, 2), dtype=np.float32
                    ),
                    is_optax_masked_node=False,
                ).to_dict(),
                'b': {
                    'b1': ArrayMetadata(
                        unpadded_shape_dtype_struct=jax.ShapeDtypeStruct(
                            shape=(3, 4), dtype=np.int32
                        ),
                        is_optax_masked_node=True,
                    ).to_dict(),
                    'b2': ArrayMetadata(
                        unpadded_shape_dtype_struct=jax.ShapeDtypeStruct(
                            shape=(5,), dtype=np.float32
                        ),
                        is_optax_masked_node=False,
                    ).to_dict(),
                },
            },
        ),
    )
    self.assertTrue(
        checkpoint_metadata._trees_are_equal(d_expected, d_restored)
    )

  def test_equals(self):
    def _get_metadata(version, shape, dtype, is_masked):
      return checkpoint_metadata.PaxMetadata(
          version=version,
          train_state_metadata={
              'a': ArrayMetadata(
                  unpadded_shape_dtype_struct=jax.ShapeDtypeStruct(
                      shape=shape,
                      dtype=dtype,
                  ),
                  is_optax_masked_node=is_masked,
              )
          },
      )

    expected = _get_metadata(1.0, (1, 2), np.float32, True)

    actual1 = _get_metadata(1.0, (1, 2), np.float32, True)
    self.assertTrue(expected.equals(actual1))

    actual2 = _get_metadata(2.0, (1, 2), np.float32, True)
    self.assertFalse(expected.equals(actual2))

    actual3 = _get_metadata(1.0, (2, 2), np.float32, True)
    self.assertFalse(expected.equals(actual3))

    actual4 = _get_metadata(1.0, (1, 2), np.int32, True)
    self.assertFalse(expected.equals(actual4))

    actual5 = _get_metadata(1.0, (1, 2), np.float32, False)
    self.assertFalse(expected.equals(actual5))

  def test_is_compatible(self):
    def _get_metadata(version, shape, dtype, is_masked):
      return checkpoint_metadata.PaxMetadata(
          version=version,
          train_state_metadata={
              'a': ArrayMetadata(
                  unpadded_shape_dtype_struct=jax.ShapeDtypeStruct(
                      shape=shape,
                      dtype=dtype,
                  ),
                  is_optax_masked_node=is_masked,
              )
          },
      )

    expected1 = _get_metadata(1.0, (1, 2), np.float32, True)  # masked
    expected2 = _get_metadata(1.0, (1, 2), np.float32, False)  # not masked

    actual1 = _get_metadata(1.0, (1, 2), np.float32, True)
    self.assertTrue(expected1.is_compatible(actual1))
    self.assertFalse(expected2.is_compatible(actual1))

    actual2 = _get_metadata(1.0, (1, 2), np.float32, False)
    self.assertFalse(expected1.is_compatible(actual2))
    self.assertTrue(expected2.is_compatible(actual2))

  @parameterized.parameters(
      (
          checkpoint_metadata.PaxMetadata(
              version=1.1,
              train_state_metadata={
                  'mdl_vars': {
                      'a': ArrayMetadata(
                          unpadded_shape_dtype_struct=jax.ShapeDtypeStruct(
                              shape=(1, 2),
                              dtype=np.float32,
                          ),
                          is_optax_masked_node=False,
                      )
                  },
              },
          ),
      ),
      (
          checkpoint_metadata.PaxMetadata(
              version=1.1,
              train_state_metadata={
                  'mdl_vars': _CustomPyTreeNode(
                      a=ArrayMetadata(
                          unpadded_shape_dtype_struct=jax.ShapeDtypeStruct(
                              shape=(1, 2),
                              dtype=np.float32,
                          ),
                          is_optax_masked_node=False,
                      ),
                  ),
              },
          ),
      ),
      (
          checkpoint_metadata.PaxMetadata(
              version=1.1,
              train_state_metadata={
                  'mdl_vars': _CustomFlaxDataclass(
                      a=ArrayMetadata(
                          unpadded_shape_dtype_struct=jax.ShapeDtypeStruct(
                              shape=(1, 2),
                              dtype=np.float32,
                          ),
                          is_optax_masked_node=False,
                      ),
                  ),
              },
          ),
      ),
  )
  def test_is_json_serializable(self, pax_metadata):
    d = pax_metadata.to_dict()
    serialized = json.dumps(d)
    expected = (
        '{"version": 1.1, "train_state_metadata": {"mdl_vars": {"a":'
        ' {"_array_metadata_tag": true, "dtype": "float32",'
        ' "is_optax_masked_node": false, "unpadded_shape": [1, 2]}}}}'
    )
    self.assertEqual(expected, serialized)


if __name__ == '__main__':
  absltest.main()
