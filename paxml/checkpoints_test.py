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

from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
from orbax.checkpoint import utils as orbax_utils
from paxml import checkpoints


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

  @parameterized.parameters(
      ('foobar123', False),
      ('foobar_123', False),
      ('checkpoint1000', False),
      ('tmp_1010101.checkpoint_100', True),
      ('tmp_1010101.checkpoint_000100', True),
  )
  def test_is_tmp_checkpoint_asset_legacy(self, name, expected_result):
    ckpt = self.directory / name
    self.assertEqual(checkpoints.is_tmp_checkpoint_asset(ckpt), expected_result)

  def test_is_tmp_checkpoint_asset_legacy_flax(self):
    ckpt = self.directory / 'checkpoint'
    ckpt.write_text('some data')
    self.assertFalse(checkpoints.is_tmp_checkpoint_asset(ckpt))

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
        checkpoints.get_step_from_checkpoint_asset(path), expected_step
    )

  @parameterized.parameters(
      (
          checkpoints.CheckpointType.UNSPECIFIED,
          epath.Path('/tmp/checkpoints/1234'),
          checkpoints.CheckpointType.UNSPECIFIED,
      ),
      (
          checkpoints.CheckpointType.GDA,
          epath.Path('/tmp/checkpoint_1234'),
          checkpoints.CheckpointType.GDA,
      ),
      (
          checkpoints.CheckpointType.GDA,
          epath.Path('/tmp/checkpoints/1234'),
          checkpoints.CheckpointType.GDA_VERSION_SUBDIR,
      ),
  )
  def test_maybe_update_checkpoint_type(
      self, specified_format, path, expected_format
  ):
    self.assertEqual(
        checkpoints.maybe_update_checkpoint_type(specified_format, path),
        expected_format,
    )


if __name__ == '__main__':
  absltest.main()
