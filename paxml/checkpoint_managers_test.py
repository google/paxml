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

"""Tests for Pax checkpoint_managers."""

import datetime
import functools
import os
from typing import Any, List, Optional
from unittest import mock

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax.experimental import multihost_utils
from jax.sharding import Mesh
import numpy as np
import orbax.checkpoint
from paxml import checkpoint_managers
from paxml import checkpoint_pb2
from paxml import checkpoints
from paxml import train_states
import tensorflow.compat.v2 as tf


CheckpointType = checkpoint_pb2.CheckpointType
FLAGS = flags.FLAGS
CHECKPOINT_PREFIX = checkpoint_managers.CHECKPOINT_PREFIX
TrainState = train_states.TrainState


def _expected_checkpoint_filenames(
    steps: List[int],
    checkpoint_type: CheckpointType = CheckpointType.CHECKPOINT_GDA):
  """Returns checkpoint basenames corresponding to all the `steps`."""
  results = []
  for step in steps:
    if checkpoint_type == CheckpointType.CHECKPOINT_FLAX:
      name = f'{CHECKPOINT_PREFIX}{step}'
    else:
      name = f'{CHECKPOINT_PREFIX}{step:08d}'
    results.append(name)
  return results


def _actual_checkpoint_filenames(directory: str) -> List[str]:
  return [
      os.path.basename(v) for v in tf.io.gfile.glob(
          os.path.join(directory, f'{CHECKPOINT_PREFIX}*'))
  ]


def create_train_state(step: int = 0):
  mdl_vars = orbax.checkpoint.test_utils.setup_pytree()
  global_mesh = Mesh(np.asarray(jax.devices()), ('x',))
  axes = jax.sharding.PartitionSpec(
      None,
  )
  mdl_vars = jax.tree_util.tree_map(
      functools.partial(
          orbax.checkpoint.test_utils.create_sharded_array,
          mesh=global_mesh,
          mesh_axes=axes,
      ),
      mdl_vars,
  )
  opt_states = [mdl_vars]
  train_state = TrainState(step=step, mdl_vars=mdl_vars, opt_states=opt_states)

  def _create_sharded_array(x):
    return orbax.checkpoint.test_utils.create_sharded_array(
        x, global_mesh, axes
    )

  train_state = jax.tree_util.tree_map(_create_sharded_array, train_state)
  state_specs = jax.tree_util.tree_map(
      lambda _: axes,
      train_state,
  )
  return global_mesh, state_specs, train_state


class CheckpointManagerTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.directory = self.create_tempdir(name='checkpointing_test').full_path
    self.global_mesh, self.state_specs, self.train_state = create_train_state()

  def create_checkpointer(self, checkpoint_type: CheckpointType):
    if checkpoint_type == CheckpointType.CHECKPOINT_FLAX:
      checkpointer = checkpoints.FlaxCheckpointer(
          checkpoints.FlaxCheckpointHandler()
      )
    elif checkpoint_type == CheckpointType.CHECKPOINT_GDA:
      checkpointer = orbax.checkpoint.Checkpointer(
          checkpoints.PaxCheckpointHandler()
      )
    else:
      raise ValueError('Unsupported CheckpointType.')
    return checkpointer

  def create_checkpoint_manager(
      self,
      options: checkpoint_managers.CheckpointManagerOptions,
      checkpoint_type: CheckpointType = CheckpointType.CHECKPOINT_GDA
  ) -> checkpoint_managers.OrbaxCheckpointManager:
    checkpointer = self.create_checkpointer(checkpoint_type)
    return checkpoint_managers.OrbaxCheckpointManager(
        self.directory,
        checkpointer,
        checkpoint_type=checkpoint_type,
        options=options,
    )

  def save(
      self,
      checkpoint_manager: checkpoint_managers.OrbaxCheckpointManager,
      step: int,
      train_state: Any,
  ) -> bool:
    train_state = train_state.replace(
        step=orbax.checkpoint.test_utils.create_sharded_array(
            step,
            self.global_mesh,
            jax.sharding.PartitionSpec(
                None,
            ),
        )
    )
    return checkpoint_manager.save(step, train_state)

  def restore(
      self,
      checkpoint_manager: checkpoint_managers.OrbaxCheckpointManager,
      step: int,
      train_state: Any,
      state_specs: Any,
      checkpoint_type: CheckpointType,
      global_mesh: Optional[Mesh] = None,
  ) -> Any:
    if global_mesh is None:
      global_mesh = self.global_mesh
    if checkpoint_type == CheckpointType.CHECKPOINT_GDA:
      restore_kwargs = {'specs': state_specs, 'mesh': global_mesh}
    elif checkpoint_type == CheckpointType.CHECKPOINT_FLAX:
      restore_kwargs = None
    return checkpoint_manager.restore(
        step, train_state, restore_kwargs=restore_kwargs
    )

  @parameterized.parameters(
      (CheckpointType.CHECKPOINT_GDA,), (CheckpointType.CHECKPOINT_FLAX,)
  )
  def test_save_restore(self, checkpoint_type):
    checkpoint_manager = self.create_checkpoint_manager(
        checkpoint_managers.CheckpointManagerOptions(),
        checkpoint_type=checkpoint_type,
    )
    self.save(checkpoint_manager, 0, self.train_state)

    expected = self.train_state
    if checkpoint_type == CheckpointType.CHECKPOINT_FLAX:
      expected = jax.tree_util.tree_map(
          lambda x: np.asarray(x.addressable_data(0)),
          expected,
      )
    train_state_global_shapes = jax.eval_shape(lambda x: x, self.train_state)
    restored = self.restore(
        checkpoint_manager,
        0,
        train_state_global_shapes,
        self.state_specs,
        checkpoint_type,
        global_mesh=self.global_mesh,
    )
    orbax.checkpoint.test_utils.assert_tree_equal(self, expected, restored)

  @parameterized.parameters((None, CheckpointType.CHECKPOINT_GDA),
                            (None, CheckpointType.CHECKPOINT_FLAX),
                            (2, CheckpointType.CHECKPOINT_GDA),
                            (2, CheckpointType.CHECKPOINT_FLAX))
  def test_save_max_to_keep(self, max_to_keep, checkpoint_type):
    options = checkpoint_managers.CheckpointManagerOptions(
        save_interval_steps=1000, max_to_keep=max_to_keep)
    checkpoint_manager = self.create_checkpoint_manager(
        options, checkpoint_type=checkpoint_type)
    steps = list(range(0, 10000, 1000))
    for step in steps:
      self.save(checkpoint_manager, step, self.train_state)

    if max_to_keep is None:
      expected_steps = steps
    else:
      expected_steps = steps[-max_to_keep:]

    self.assertSameElements(
        _expected_checkpoint_filenames(
            expected_steps, checkpoint_type=checkpoint_type),
        _actual_checkpoint_filenames(self.directory))
    self.assertSameElements(expected_steps, checkpoint_manager.all_steps())

  @parameterized.parameters((CheckpointType.CHECKPOINT_GDA,),
                            (CheckpointType.CHECKPOINT_FLAX,))
  def test_save_checkpoint_keep_interval_timedelta(self, checkpoint_type):
    tz = datetime.timezone.utc
    current_datetime = datetime.datetime.now(tz=tz)
    zero_datetime = datetime.datetime.fromtimestamp(0, tz=tz)
    with mock.patch('datetime.datetime', autospec=True) as dt:
      dt.now.return_value = current_datetime
      dt.fromtimestamp.return_value = zero_datetime
      options = checkpoint_managers.CheckpointManagerOptions(
          save_interval_steps=1000,
          max_to_keep=2,
          keep_time_interval=datetime.timedelta(hours=2))
      checkpoint_manager = self.create_checkpoint_manager(
          options, checkpoint_type=checkpoint_type
      )

    steps = list(range(0, 10000, 1000))
    checkpoint_datetimes = []
    for step in steps:
      with mock.patch('datetime.datetime', autospec=True) as dt:
        dt.now.return_value = current_datetime
        dt.fromtimestamp.return_value = zero_datetime
        self.save(checkpoint_manager, step, self.train_state)
        checkpoint_datetimes.append(current_datetime)
        current_datetime += datetime.timedelta(hours=1)

    saved_steps = [0, 2000, 4000, 6000, 8000, 9000]

    self.assertSameElements(
        _expected_checkpoint_filenames(
            saved_steps, checkpoint_type=checkpoint_type),
        _actual_checkpoint_filenames(self.directory))
    self.assertSameElements(saved_steps, checkpoint_manager.all_steps())

  @parameterized.parameters((CheckpointType.CHECKPOINT_GDA,),
                            (CheckpointType.CHECKPOINT_FLAX,))
  def test_save_restore_manager_case_1_default(self, checkpoint_type):
    tz = datetime.timezone.utc
    current_datetime = datetime.datetime.now(tz=tz)
    zero_datetime = datetime.datetime.fromtimestamp(0, tz=tz)

    options = checkpoint_managers.CheckpointManagerOptions(
        save_interval_steps=2000, max_to_keep=4)
    checkpoint_manager = self.create_checkpoint_manager(
        options, checkpoint_type=checkpoint_type)

    steps = list(range(0, 10000, 1000))
    for step in steps:
      self.save(checkpoint_manager, step, self.train_state)

    saved_steps = [2000, 4000, 6000, 8000]

    self.assertSameElements(
        _expected_checkpoint_filenames(
            saved_steps, checkpoint_type=checkpoint_type),
        _actual_checkpoint_filenames(self.directory))
    self.assertSameElements(saved_steps, checkpoint_manager.all_steps())

    del checkpoint_manager
    with mock.patch('datetime.datetime', autospec=True) as dt:
      dt.now.return_value = current_datetime
      dt.fromtimestamp.return_value = zero_datetime
      options = checkpoint_managers.CheckpointManagerOptions(
          save_interval_steps=3000,
          max_to_keep=6,
          keep_time_interval=datetime.timedelta(hours=3))
      checkpoint_manager = self.create_checkpoint_manager(
          options, checkpoint_type=checkpoint_type
      )

    saved_steps_2_init = [2000, 4000, 6000, 8000]

    self.assertSameElements(
        _expected_checkpoint_filenames(
            saved_steps_2_init, checkpoint_type=checkpoint_type),
        _actual_checkpoint_filenames(self.directory))
    self.assertSameElements(saved_steps_2_init, checkpoint_manager.all_steps())

    steps_2 = list(range(10000, 20000, 1000))
    for step in steps_2:
      with mock.patch('datetime.datetime', autospec=True) as dt:
        dt.now.return_value = current_datetime
        dt.fromtimestamp.return_value = zero_datetime
        self.save(checkpoint_manager, step, self.train_state)
        current_datetime += datetime.timedelta(hours=1)

    # expect saved steps at multipliers of 3000.
    saved_steps_2 = saved_steps_2_init + [12000, 15000, 18000]

    self.assertSameElements(
        _expected_checkpoint_filenames(
            saved_steps_2, checkpoint_type=checkpoint_type),
        _actual_checkpoint_filenames(self.directory))
    self.assertSameElements(saved_steps_2, checkpoint_manager.all_steps())

  @parameterized.parameters((CheckpointType.CHECKPOINT_GDA,),
                            (CheckpointType.CHECKPOINT_FLAX,))
  def test_save_restore_manager_case_2_mutant(self, checkpoint_type):
    options = checkpoint_managers.CheckpointManagerOptions(
        save_interval_steps=100, max_to_keep=None)
    checkpoint_manager = self.create_checkpoint_manager(
        options, checkpoint_type=checkpoint_type)

    steps = list(range(0, 10000, 1000))
    for step in steps:
      self.save(checkpoint_manager, step, self.train_state)

    saved_steps = steps

    self.assertSameElements(
        _expected_checkpoint_filenames(
            saved_steps, checkpoint_type=checkpoint_type),
        _actual_checkpoint_filenames(self.directory))
    self.assertSameElements(saved_steps, checkpoint_manager.all_steps())

    del checkpoint_manager
    max_to_keep = 5
    options = checkpoint_managers.CheckpointManagerOptions(
        save_interval_steps=1000, max_to_keep=max_to_keep)
    checkpoint_manager = self.create_checkpoint_manager(
        options, checkpoint_type=checkpoint_type)

    step = 10000
    steps.append(step)
    self.save(checkpoint_manager, step, self.train_state)

    saved_steps_2 = steps[-max_to_keep:]

    self.assertSameElements(
        _expected_checkpoint_filenames(
            saved_steps_2, checkpoint_type=checkpoint_type),
        _actual_checkpoint_filenames(self.directory))
    self.assertSameElements(saved_steps_2, checkpoint_manager.all_steps())

  def test_save_on_preemption(self):
    options = checkpoint_managers.CheckpointManagerOptions(
        save_interval_steps=1000, max_to_keep=None)
    checkpoint_manager = self.create_checkpoint_manager(options)

    save_step = 3
    jax.config.update('jax_coordination_service', True)
    multihost_utils.reached_preemption_sync_point = (
        lambda step_id: step_id == save_step)

    for step in range(save_step + 1):
      self.save(checkpoint_manager, step, self.train_state)

    saved_steps = [0, save_step]

    self.assertSameElements(
        _expected_checkpoint_filenames(saved_steps),
        _actual_checkpoint_filenames(self.directory))
    self.assertSameElements(saved_steps, checkpoint_manager.all_steps())

  def test_cleanup(self):
    def _fake_on_commit_callback(*args, **kwargs):
      del args, kwargs
      pass  # Do nothing to simulate failure of finalization.

    options = checkpoint_managers.CheckpointManagerOptions(
        save_interval_steps=1
    )

    with mock.patch.object(
        orbax.checkpoint.utils, 'ensure_atomic_save', autospec=True
    ) as commit_callback:
      commit_callback.side_effect = _fake_on_commit_callback
      checkpoint_manager = self.create_checkpoint_manager(options)
      self.save(checkpoint_manager, 0, self.train_state)
      # Step 0 not finalized.
      tmp_checkpoint_pattern = (
          checkpoint_managers.STATE_ITEM_NAME
          + orbax.checkpoint.utils.TMP_DIR_SUFFIX
          + '*'
      )
      item_dir = checkpoint_manager._manager._get_save_directory(
          0, checkpoint_manager.directory
      )
      self.assertNotEmpty(list(item_dir.glob(tmp_checkpoint_pattern)))

    checkpoint_manager = self.create_checkpoint_manager(options)
    self.assertEmpty(
        list(checkpoint_manager.directory.glob(tmp_checkpoint_pattern))
    )
    self.save(checkpoint_manager, 0, self.train_state)
    self.assertSameElements(
        _expected_checkpoint_filenames([0]),
        _actual_checkpoint_filenames(checkpoint_manager.directory),
    )
    self.assertSameElements([0], checkpoint_manager.all_steps())

  @parameterized.parameters((CheckpointType.CHECKPOINT_GDA,),
                            (CheckpointType.CHECKPOINT_FLAX,))
  def test_todelete_subdir(self, checkpoint_type):
    options = checkpoint_managers.CheckpointManagerOptions(
        max_to_keep=2, todelete_subdir='archive')
    checkpoint_manager = self.create_checkpoint_manager(
        options, checkpoint_type=checkpoint_type)

    for step in range(4):
      self.save(checkpoint_manager, step, self.train_state)

    self.assertSameElements(
        _expected_checkpoint_filenames([0, 1], checkpoint_type=checkpoint_type),
        _actual_checkpoint_filenames(os.path.join(self.directory, 'archive')))
    self.assertSameElements(
        _expected_checkpoint_filenames([2, 3], checkpoint_type=checkpoint_type),
        _actual_checkpoint_filenames(os.path.join(self.directory)))
    self.assertIn('archive', tf.io.gfile.listdir(self.directory))
    self.assertSameElements([2, 3], checkpoint_manager.all_steps())

  @parameterized.parameters((CheckpointType.CHECKPOINT_GDA,),
                            (CheckpointType.CHECKPOINT_FLAX,))
  def test_reinitialize(self, checkpoint_type):
    options = checkpoint_managers.CheckpointManagerOptions(max_to_keep=2)
    checkpoint_manager = self.create_checkpoint_manager(
        options, checkpoint_type=checkpoint_type)

    for step in range(3):
      self.save(checkpoint_manager, step, self.train_state)
    self.assertSameElements([1, 2], checkpoint_manager.all_steps())

    new_checkpoint_manager = self.create_checkpoint_manager(
        options, checkpoint_type=checkpoint_type)
    self.assertSameElements([1, 2], new_checkpoint_manager.all_steps())
    self.save(new_checkpoint_manager, 3, self.train_state)
    self.assertSameElements([2, 3], new_checkpoint_manager.all_steps())

  @parameterized.parameters(
      (CheckpointType.CHECKPOINT_GDA,), (CheckpointType.CHECKPOINT_FLAX,)
  )
  def test_restore_legacy_format(self, checkpoint_type):
    checkpoint_manager = self.create_checkpoint_manager(
        checkpoint_managers.CheckpointManagerOptions(),
        checkpoint_type=checkpoint_type,
    )
    self.save(checkpoint_manager, 0, self.train_state)

    step_dir = checkpoint_manager._manager._get_save_directory(
        0, checkpoint_manager.directory
    )
    self.assertTrue(checkpoints.is_checkpoint_asset(step_dir))
    self.assertTrue((step_dir / 'state').exists())
    self.assertTrue((step_dir / 'metadata').exists())

    # Transform directory to what we would expect in a version 0 checkpoint with
    # no per-item subdirectories.
    (step_dir / 'metadata').rmtree()
    for d in (step_dir / 'state').iterdir():  # parameter directories
      if checkpoint_type == CheckpointType.CHECKPOINT_GDA:
        assert d.is_dir(), d
        (step_dir / d.name).mkdir()
        for f in d.iterdir():
          assert f.is_file(), f
          f.copy(step_dir / d.name / f.name)
      else:
        f = d
        assert f.is_file(), f
        assert f.name == 'checkpoint'
        f.copy(step_dir / f.name)
    (step_dir / 'state').rmtree()
    checkpoint_manager._manager._version = 0.0

    expected = self.train_state
    if checkpoint_type == CheckpointType.CHECKPOINT_FLAX:
      expected = jax.tree_util.tree_map(
          lambda x: np.asarray(x.addressable_data(0)),
          expected,
      )
    train_state_global_shapes = jax.eval_shape(lambda x: x, self.train_state)
    restored = self.restore(
        checkpoint_manager,
        0,
        train_state_global_shapes,
        self.state_specs,
        checkpoint_type,
        global_mesh=self.global_mesh,
    )
    orbax.checkpoint.test_utils.assert_tree_equal(self, expected, restored)

    # Saving again, we expect it to be saved with the old format.
    self.save(checkpoint_manager, 1, self.train_state)
    step_dir = checkpoint_manager._manager._get_save_directory(
        1, checkpoint_manager.directory
    )
    self.assertTrue(checkpoints.is_checkpoint_asset(step_dir))
    self.assertFalse((step_dir / 'state').exists())
    self.assertFalse((step_dir / 'metadata').exists())


if __name__ == '__main__':
  absltest.main()
