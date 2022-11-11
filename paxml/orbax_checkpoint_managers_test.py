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
import os
from typing import List
from unittest import mock

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax.experimental import multihost_utils
import numpy as np
import orbax.checkpoint
from paxml import checkpoint_managers
from paxml import checkpoint_pb2
from paxml import checkpoints
import tensorflow.compat.v2 as tf

CheckpointType = checkpoint_pb2.CheckpointType
FLAGS = flags.FLAGS
CHECKPOINT_PREFIX = checkpoint_managers.CHECKPOINT_PREFIX


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


class CheckpointManagerTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.directory = self.create_tempdir(name='checkpointing_test').full_path
    self.train_state = {
        'a': 42,
        'b': np.arange(42),
    }

  def save_with_args(self,
                     checkpoint_manager,
                     step,
                     train_state,
                     checkpoint_type=CheckpointType.CHECKPOINT_GDA):
    save_args = {}
    if checkpoint_type == CheckpointType.CHECKPOINT_FLAX:
      save_args = {'step': step}
    checkpoint_manager.save(step, train_state, save_kwargs=save_args)

  def create_checkpointer(self, checkpoint_type: CheckpointType):
    if checkpoint_type == CheckpointType.CHECKPOINT_FLAX:
      checkpointer = checkpoints.FlaxCheckpointer()
    elif checkpoint_type == CheckpointType.CHECKPOINT_GDA:
      checkpointer = orbax.checkpoint.Checkpointer(
          orbax.checkpoint.PyTreeCheckpointHandler(enable_aggregation=False))
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
        directory=self.directory,
        checkpointers=checkpointer,
        checkpoint_type=checkpoint_type,
        options=options)

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
      self.save_with_args(
          checkpoint_manager,
          step,
          self.train_state,
          checkpoint_type=checkpoint_type)

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
    current_datetime = datetime.datetime.now()
    zero_datetime = datetime.datetime.fromtimestamp(0)
    with mock.patch('datetime.datetime', autospec=True) as dt:
      dt.now.return_value = current_datetime
      dt.fromtimestamp.return_value = zero_datetime
      options = checkpoint_managers.CheckpointManagerOptions(
          save_interval_steps=1000,
          max_to_keep=2,
          keep_time_interval=datetime.timedelta(hours=2))
      checkpoint_manager = checkpoint_managers.OrbaxCheckpointManager(
          directory=self.directory,
          checkpointers=self.create_checkpointer(checkpoint_type),
          checkpoint_type=checkpoint_type,
          options=options)

    steps = list(range(0, 10000, 1000))
    checkpoint_datetimes = []
    for step in steps:
      with mock.patch('datetime.datetime', autospec=True) as dt:
        dt.now.return_value = current_datetime
        dt.fromtimestamp.return_value = zero_datetime
        self.save_with_args(
            checkpoint_manager,
            step,
            self.train_state,
            checkpoint_type=checkpoint_type)
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
    current_datetime = datetime.datetime.now()
    zero_datetime = datetime.datetime.fromtimestamp(0)

    options = checkpoint_managers.CheckpointManagerOptions(
        save_interval_steps=2000, max_to_keep=4)
    checkpoint_manager = self.create_checkpoint_manager(
        options, checkpoint_type=checkpoint_type)

    steps = list(range(0, 10000, 1000))
    for step in steps:
      self.save_with_args(
          checkpoint_manager,
          step,
          self.train_state,
          checkpoint_type=checkpoint_type)

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
      checkpoint_manager = checkpoint_managers.OrbaxCheckpointManager(
          directory=self.directory,
          checkpointers=self.create_checkpointer(checkpoint_type),
          checkpoint_type=checkpoint_type,
          options=options)

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
        self.save_with_args(
            checkpoint_manager,
            step,
            self.train_state,
            checkpoint_type=checkpoint_type)
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
      self.save_with_args(
          checkpoint_manager,
          step,
          self.train_state,
          checkpoint_type=checkpoint_type)

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
    self.save_with_args(
        checkpoint_manager,
        step,
        self.train_state,
        checkpoint_type=checkpoint_type)

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
      self.save_with_args(checkpoint_manager, step, self.train_state)

    saved_steps = [0, save_step]

    self.assertSameElements(
        _expected_checkpoint_filenames(saved_steps),
        _actual_checkpoint_filenames(self.directory))
    self.assertSameElements(saved_steps, checkpoint_manager.all_steps())

  @parameterized.parameters((CheckpointType.CHECKPOINT_GDA,),
                            (CheckpointType.CHECKPOINT_FLAX,))
  def test_todelete_subdir(self, checkpoint_type):
    options = checkpoint_managers.CheckpointManagerOptions(
        max_to_keep=2, todelete_subdir='archive')
    checkpoint_manager = self.create_checkpoint_manager(
        options, checkpoint_type=checkpoint_type)

    for step in range(4):
      self.save_with_args(
          checkpoint_manager,
          step,
          self.train_state,
          checkpoint_type=checkpoint_type)

    self.assertSameElements(
        _expected_checkpoint_filenames([0, 1], checkpoint_type=checkpoint_type),
        _actual_checkpoint_filenames(os.path.join(self.directory, 'archive')))
    self.assertSameElements(
        _expected_checkpoint_filenames([2, 3], checkpoint_type=checkpoint_type),
        _actual_checkpoint_filenames(os.path.join(self.directory)))
    self.assertIn('archive', tf.io.gfile.listdir(self.directory))
    self.assertSameElements([2, 3], checkpoint_manager.all_steps())

if __name__ == '__main__':
  absltest.main()
