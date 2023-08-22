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

"""Tests for Pax checkpoint_managers."""

import contextlib
import copy
import datetime
import functools
import json
import os
from typing import Any, Dict, List, Optional
from unittest import mock

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
import jax
from jax.experimental import multihost_utils
from jax.sharding import Mesh
import numpy as np
import optax
import orbax.checkpoint as ocp
from paxml import checkpoint_managers
from paxml import checkpoint_paths
from paxml import checkpoint_types
from paxml import checkpoints
from paxml import train_states
from praxis import base_input
from praxis import py_utils
import tensorflow.compat.v2 as tf
import tensorstore as ts


FLAGS = flags.FLAGS
CheckpointType = checkpoint_types.CheckpointType
CHECKPOINT_PREFIX = checkpoint_paths.CHECKPOINT_PREFIX
TrainState = train_states.TrainState


@contextlib.contextmanager
def ocdbt_checkpoint_context(use_ocdbt: bool, ts_context: Any):
  """Use OCDBT driver within context."""
  original_registry = list(
      ocp.type_handlers._TYPE_REGISTRY  # pylint: disable=protected-access
  )
  if use_ocdbt:
    ocp.type_handlers.register_standard_handlers_with_options(
        use_ocdbt=use_ocdbt, ts_context=ts_context
    )
  try:
    yield
  finally:
    ocp.type_handlers._TYPE_REGISTRY = (  # pylint: disable=protected-access
        original_registry
    )


def _expected_checkpoint_filenames(
    steps: List[int], checkpoint_type: CheckpointType = CheckpointType.GDA
):
  """Returns checkpoint basenames corresponding to all the `steps`."""
  results = []
  for step in steps:
    if checkpoint_type == CheckpointType.FLAX:
      name = f'{CHECKPOINT_PREFIX}{step}'
    else:
      name = f'{CHECKPOINT_PREFIX}{step:08d}'
    results.append(name)
  return results


def _actual_checkpoint_filenames(directory: str) -> List[str]:
  return [
      os.path.basename(v)
      for v in tf.io.gfile.glob(
          os.path.join(directory, f'{CHECKPOINT_PREFIX}*')
      )
  ]


def create_train_state(step: int = 0):
  mdl_vars = ocp.test_utils.setup_pytree()
  global_mesh = Mesh(np.asarray(jax.devices()), ('x',))
  axes = jax.sharding.PartitionSpec(
      None,
  )
  mdl_vars = jax.tree_util.tree_map(
      functools.partial(
          ocp.test_utils.create_sharded_array,
          mesh=global_mesh,
          mesh_axes=axes,
      ),
      mdl_vars,
  )
  opt_states = [mdl_vars]
  extra_state = [mdl_vars]
  train_state = TrainState(
      step=step,
      mdl_vars=mdl_vars,
      opt_states=opt_states,
      extra_state=extra_state,
  )

  def _create_sharded_array(x):
    return ocp.test_utils.create_sharded_array(x, global_mesh, axes)

  train_state = jax.tree_util.tree_map(_create_sharded_array, train_state)
  state_specs = jax.tree_util.tree_map(
      lambda _: axes,
      train_state,
  )
  train_state_unpadded_shape_dtype_struct = jax.tree_util.tree_map(
      lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), train_state
  )
  train_state.mdl_vars['b'] = optax.MaskedNode()  # masking a node
  return (
      global_mesh,
      state_specs,
      train_state,
      train_state_unpadded_shape_dtype_struct,
  )


class TestInput(base_input.BaseInput):

  def __post_init__(self):
    super().__post_init__()
    self._dataset = self._get_dataset()
    self._iter = iter(self._dataset)

  def get_next(self) -> py_utils.NestedMap:
    assert tf.compat.v1.executing_eagerly()
    ret = self._iter.get_next()
    return tf.nest.map_structure(lambda x: x.numpy(), ret)

  def reset(self):
    self._iter = iter(self._dataset)

  def save(self, filename: epath.Path):
    ckpt = tf.train.Checkpoint(ds=self._iter)
    ckpt.write(os.fspath(filename))

  def restore(self, filename: epath.Path) -> None:
    ckpt = tf.train.Checkpoint(ds=self._iter)
    ckpt.read(os.fspath(filename)).assert_consumed()

  def _to_nested_map(self, x) -> py_utils.NestedMap:
    t = tf.ones(shape=[4], dtype=tf.int32) * tf.cast(x, dtype=tf.int32)
    return py_utils.NestedMap(data=t)

  def _get_dataset(self):
    d = tf.data.Dataset.range(10)
    d = d.shard(self.num_infeed_hosts, self.infeed_host_index)
    d = d.map(self._to_nested_map)
    d = d.batch(self.batch_size)
    return d


class CheckpointManagerTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.directory = self.create_tempdir(name='checkpointing_test').full_path
    (
        self.global_mesh,
        self.state_specs,
        self.train_state,
        self.train_state_unpadded_shape_dtype_struct,
    ) = create_train_state()

  def create_checkpointer(
      self, checkpoint_type: CheckpointType, tensorstore_use_ocdbt: bool = False
  ):
    if checkpoint_type == CheckpointType.FLAX:
      checkpointer = checkpoints.FlaxCheckpointer(
          checkpoints.FlaxCheckpointHandler()
      )
    elif checkpoint_type == CheckpointType.GDA:
      checkpointer = ocp.Checkpointer(
          checkpoints.PaxCheckpointHandler(use_ocdbt=tensorstore_use_ocdbt)
      )
    else:
      raise ValueError('Unsupported CheckpointType.')
    return checkpointer

  def create_checkpoint_manager(
      self,
      options: checkpoint_managers.CheckpointManagerOptions,
      checkpoint_type: CheckpointType = CheckpointType.GDA,
      train_input_pipeline: Optional[base_input.BaseInput] = None,
      tensorstore_use_ocdbt: bool = False,
  ) -> checkpoint_managers.OrbaxCheckpointManager:
    checkpointer = self.create_checkpointer(
        checkpoint_type, tensorstore_use_ocdbt=tensorstore_use_ocdbt
    )
    train_input_checkpointer = (
        checkpoints.Checkpointer(checkpoints.BaseInputCheckpointHandler())
        if train_input_pipeline
        else None
    )
    return checkpoint_managers.OrbaxCheckpointManager(
        self.directory,
        checkpointer,
        train_input_checkpointer,
        checkpoint_type=checkpoint_type,
        options=options,
        tensorstore_use_ocdbt=tensorstore_use_ocdbt,
    )

  def save(
      self,
      checkpoint_manager: checkpoint_managers.OrbaxCheckpointManager,
      step: int,
      train_state: Any,
      train_input_pipeline: Optional[base_input.BaseInput] = None,
      train_state_unpadded_shape_dtype_struct: Optional[Any] = None,
  ) -> bool:
    train_state = train_state.replace(
        step=ocp.test_utils.create_sharded_array(
            step,
            self.global_mesh,
            jax.sharding.PartitionSpec(
                None,
            ),
        )
    )

    return checkpoint_manager.save(
        step,
        train_state,
        train_state_unpadded_shape_dtype_struct,
        train_input_pipeline,
    )

  def restore(
      self,
      checkpoint_manager: checkpoint_managers.OrbaxCheckpointManager,
      step: int,
      train_state: Any,
      state_specs: Any,
      checkpoint_type: CheckpointType,
      global_mesh: Optional[Mesh] = None,
      train_input_pipeline: Optional[base_input.BaseInput] = None,
      train_state_unpadded_shape_dtype_struct: Optional[Any] = None,
      transforms: Optional[Dict[str, Any]] = None,
  ) -> Any:
    if global_mesh is None:
      global_mesh = self.global_mesh
    if checkpoint_type == CheckpointType.GDA:
      restore_kwargs = {
          'specs': state_specs,
          'mesh': global_mesh,
          'transforms': transforms,
      }
    elif checkpoint_type == CheckpointType.FLAX:
      restore_kwargs = None
    else:
      raise ValueError(f'Unsupported CheckpointType {checkpoint_type}.')
    return checkpoint_manager.restore(
        step,
        train_state,
        train_state_unpadded_shape_dtype_struct,
        train_input_pipeline,
        restore_kwargs=restore_kwargs,
    )

  @parameterized.parameters(
      (CheckpointType.GDA, False),
      (CheckpointType.FLAX, False),
      (CheckpointType.GDA, True),
      (CheckpointType.FLAX, True),
  )
  def test_save_restore(self, checkpoint_type, use_train_input):
    train_input_pipeline = None
    if use_train_input:
      train_input_pipeline = TestInput(
          batch_size=2,
      )
      _ = train_input_pipeline.get_next()
      expected_inputs = train_input_pipeline.get_next()
      train_input_pipeline.reset()
      _ = train_input_pipeline.get_next()

    checkpoint_manager = self.create_checkpoint_manager(
        checkpoint_managers.CheckpointManagerOptions(),
        checkpoint_type=checkpoint_type,
        train_input_pipeline=train_input_pipeline,
    )
    self.save(
        checkpoint_manager,
        0,
        self.train_state,
        train_input_pipeline,
        train_state_unpadded_shape_dtype_struct=(
            self.train_state_unpadded_shape_dtype_struct
        ),
    )
    if use_train_input:
      train_input_pipeline.reset()
    expected = self.train_state
    if checkpoint_type == CheckpointType.FLAX:
      expected = jax.tree_util.tree_map(
          lambda x: np.asarray(x.addressable_data(0)),
          expected,
      )
    train_state_global_shapes = jax.eval_shape(lambda x: x, self.train_state)
# Internal Orbax infra configuration test

    restored = self.restore(
        checkpoint_manager,
        0,
        train_state_global_shapes,
        self.state_specs,
        checkpoint_type,
        global_mesh=self.global_mesh,
        train_input_pipeline=train_input_pipeline,
        train_state_unpadded_shape_dtype_struct=self.train_state_unpadded_shape_dtype_struct,
    )
    if train_input_pipeline:
      # restored inputs should start from the second batch
      restored_inputs = train_input_pipeline.get_next()
      ocp.test_utils.assert_tree_equal(self, expected_inputs, restored_inputs)
    ocp.test_utils.assert_tree_equal(self, expected, restored)

    # incompatible unpadded shape
    wrong_unpadded_shape_dtype_struct = copy.deepcopy(
        self.train_state_unpadded_shape_dtype_struct
    )
    a = self.train_state_unpadded_shape_dtype_struct.mdl_vars['a']
    wrong_a = jax.ShapeDtypeStruct((10,), a.dtype)
    wrong_unpadded_shape_dtype_struct.mdl_vars['a'] = wrong_a
    with self.assertRaises(ValueError):
      restored = self.restore(
          checkpoint_manager,
          0,
          train_state_global_shapes,
          self.state_specs,
          checkpoint_type,
          global_mesh=self.global_mesh,
          train_input_pipeline=train_input_pipeline,
          train_state_unpadded_shape_dtype_struct=wrong_unpadded_shape_dtype_struct,
      )

    # wrong unpadded shape of a masked node (no error)
    wrong_masked_unpadded_shape_dtype_struct = copy.deepcopy(
        self.train_state_unpadded_shape_dtype_struct
    )
    b = self.train_state_unpadded_shape_dtype_struct.mdl_vars['b']
    wrong_b = jax.ShapeDtypeStruct((10,), b.dtype)
    wrong_masked_unpadded_shape_dtype_struct.mdl_vars['b'] = wrong_b
    restored = self.restore(
        checkpoint_manager,
        0,
        train_state_global_shapes,
        self.state_specs,
        checkpoint_type,
        global_mesh=self.global_mesh,
        train_input_pipeline=train_input_pipeline,
        train_state_unpadded_shape_dtype_struct=wrong_masked_unpadded_shape_dtype_struct,
    )

  @parameterized.parameters(
      (CheckpointType.GDA,),
      (CheckpointType.FLAX,),
  )
  def test_save_restore_unpadded(self, checkpoint_type):
    # test restore with unpadded shape saved/not saved
    checkpoint_manager = self.create_checkpoint_manager(
        checkpoint_managers.CheckpointManagerOptions(),
        checkpoint_type=checkpoint_type,
        train_input_pipeline=None,
    )

    def _save(save_unpadded_shape):
      self.save(
          checkpoint_manager,
          0,
          self.train_state,
          train_state_unpadded_shape_dtype_struct=(
              self.train_state_unpadded_shape_dtype_struct
              if save_unpadded_shape
              else None
          ),
      )

    with self.assertRaises(ValueError):
      _save(save_unpadded_shape=False)
    _save(save_unpadded_shape=True)

    expected = self.train_state
    if checkpoint_type == CheckpointType.FLAX:
      expected = jax.tree_util.tree_map(
          lambda x: np.asarray(x.addressable_data(0)),
          expected,
      )
    train_state_global_shapes = jax.eval_shape(lambda x: x, self.train_state)

    def _restore(check_unpadded_shape):
      return self.restore(
          checkpoint_manager,
          0,
          train_state_global_shapes,
          self.state_specs,
          checkpoint_type,
          global_mesh=self.global_mesh,
          train_state_unpadded_shape_dtype_struct=(
              self.train_state_unpadded_shape_dtype_struct
              if check_unpadded_shape
              else None
          ),
      )

    restored = _restore(check_unpadded_shape=True)
    ocp.test_utils.assert_tree_equal(self, expected, restored)

  @parameterized.parameters(
      (CheckpointType.GDA),
      (CheckpointType.FLAX),
  )
  def test_restore_no_inputs(self, checkpoint_type):
    train_input_pipeline = TestInput(
        batch_size=2,
    )
    expected_inputs = train_input_pipeline.get_next()
    train_input_pipeline.reset()

    checkpoint_manager = self.create_checkpoint_manager(
        checkpoint_managers.CheckpointManagerOptions(),
        checkpoint_type=checkpoint_type,
        train_input_pipeline=train_input_pipeline,
    )
    self.save(
        checkpoint_manager,
        0,
        self.train_state,
        None,
        train_state_unpadded_shape_dtype_struct=self.train_state_unpadded_shape_dtype_struct,
    )
    train_state_global_shapes = jax.eval_shape(lambda x: x, self.train_state)
    _ = self.restore(
        checkpoint_manager,
        0,
        train_state_global_shapes,
        self.state_specs,
        checkpoint_type,
        global_mesh=self.global_mesh,
        train_input_pipeline=train_input_pipeline,
        train_state_unpadded_shape_dtype_struct=self.train_state_unpadded_shape_dtype_struct,
    )
    restored_inputs = train_input_pipeline.get_next()
    ocp.test_utils.assert_tree_equal(self, expected_inputs, restored_inputs)

  @parameterized.parameters(
      (None, CheckpointType.GDA),
      (None, CheckpointType.FLAX),
      (2, CheckpointType.GDA),
      (2, CheckpointType.FLAX),
  )
  def test_save_max_to_keep(self, max_to_keep, checkpoint_type):
    options = checkpoint_managers.CheckpointManagerOptions(
        save_interval_steps=1000, max_to_keep=max_to_keep
    )
    checkpoint_manager = self.create_checkpoint_manager(
        options, checkpoint_type=checkpoint_type
    )
    steps = list(range(0, 10000, 1000))
    for step in steps:
      self.save(
          checkpoint_manager,
          step,
          self.train_state,
          train_state_unpadded_shape_dtype_struct=self.train_state_unpadded_shape_dtype_struct,
      )

    if max_to_keep is None:
      expected_steps = steps
    else:
      expected_steps = steps[-max_to_keep:]

    self.assertSameElements(
        _expected_checkpoint_filenames(
            expected_steps, checkpoint_type=checkpoint_type
        ),
        _actual_checkpoint_filenames(self.directory),
    )
    self.assertSameElements(expected_steps, checkpoint_manager.all_steps())

  @parameterized.parameters((CheckpointType.GDA,), (CheckpointType.FLAX,))
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
          keep_time_interval=datetime.timedelta(hours=2),
      )
      checkpoint_manager = self.create_checkpoint_manager(
          options, checkpoint_type=checkpoint_type
      )

    steps = list(range(0, 10000, 1000))
    checkpoint_datetimes = []
    for step in steps:
      with mock.patch('datetime.datetime', autospec=True) as dt:
        dt.now.return_value = current_datetime
        dt.fromtimestamp.return_value = zero_datetime
        self.save(
            checkpoint_manager,
            step,
            self.train_state,
            train_state_unpadded_shape_dtype_struct=self.train_state_unpadded_shape_dtype_struct,
        )
        checkpoint_datetimes.append(current_datetime)
        current_datetime += datetime.timedelta(hours=1)

    saved_steps = [0, 2000, 4000, 6000, 8000, 9000]

    self.assertSameElements(
        _expected_checkpoint_filenames(
            saved_steps, checkpoint_type=checkpoint_type
        ),
        _actual_checkpoint_filenames(self.directory),
    )
    self.assertSameElements(saved_steps, checkpoint_manager.all_steps())

  @parameterized.parameters((CheckpointType.GDA,), (CheckpointType.FLAX,))
  def test_save_restore_manager_case_1_default(self, checkpoint_type):
    tz = datetime.timezone.utc
    current_datetime = datetime.datetime.now(tz=tz)
    zero_datetime = datetime.datetime.fromtimestamp(0, tz=tz)

    options = checkpoint_managers.CheckpointManagerOptions(
        save_interval_steps=2000, max_to_keep=4
    )
    checkpoint_manager = self.create_checkpoint_manager(
        options, checkpoint_type=checkpoint_type
    )

    steps = list(range(0, 10000, 1000))
    for step in steps:
      self.save(
          checkpoint_manager,
          step,
          self.train_state,
          train_state_unpadded_shape_dtype_struct=self.train_state_unpadded_shape_dtype_struct,
      )

    saved_steps = [2000, 4000, 6000, 8000]

    self.assertSameElements(
        _expected_checkpoint_filenames(
            saved_steps, checkpoint_type=checkpoint_type
        ),
        _actual_checkpoint_filenames(self.directory),
    )
    self.assertSameElements(saved_steps, checkpoint_manager.all_steps())

    del checkpoint_manager
    with mock.patch('datetime.datetime', autospec=True) as dt:
      dt.now.return_value = current_datetime
      dt.fromtimestamp.return_value = zero_datetime
      options = checkpoint_managers.CheckpointManagerOptions(
          save_interval_steps=3000,
          max_to_keep=6,
          keep_time_interval=datetime.timedelta(hours=3),
      )
      checkpoint_manager = self.create_checkpoint_manager(
          options, checkpoint_type=checkpoint_type
      )

    saved_steps_2_init = [2000, 4000, 6000, 8000]

    self.assertSameElements(
        _expected_checkpoint_filenames(
            saved_steps_2_init, checkpoint_type=checkpoint_type
        ),
        _actual_checkpoint_filenames(self.directory),
    )
    self.assertSameElements(saved_steps_2_init, checkpoint_manager.all_steps())

    steps_2 = list(range(10000, 20000, 1000))
    for step in steps_2:
      with mock.patch('datetime.datetime', autospec=True) as dt:
        dt.now.return_value = current_datetime
        dt.fromtimestamp.return_value = zero_datetime
        self.save(
            checkpoint_manager,
            step,
            self.train_state,
            train_state_unpadded_shape_dtype_struct=self.train_state_unpadded_shape_dtype_struct,
        )
        current_datetime += datetime.timedelta(hours=1)

    # expect saved steps at multipliers of 3000.
    saved_steps_2 = saved_steps_2_init + [12000, 15000, 18000]

    self.assertSameElements(
        _expected_checkpoint_filenames(
            saved_steps_2, checkpoint_type=checkpoint_type
        ),
        _actual_checkpoint_filenames(self.directory),
    )
    self.assertSameElements(saved_steps_2, checkpoint_manager.all_steps())

  @parameterized.parameters((CheckpointType.GDA,), (CheckpointType.FLAX,))
  def test_save_restore_manager_case_2_mutant(self, checkpoint_type):
    options = checkpoint_managers.CheckpointManagerOptions(
        save_interval_steps=100, max_to_keep=None
    )
    checkpoint_manager = self.create_checkpoint_manager(
        options, checkpoint_type=checkpoint_type
    )

    steps = list(range(0, 10000, 1000))
    for step in steps:
      self.save(
          checkpoint_manager,
          step,
          self.train_state,
          train_state_unpadded_shape_dtype_struct=self.train_state_unpadded_shape_dtype_struct,
      )

    saved_steps = steps

    self.assertSameElements(
        _expected_checkpoint_filenames(
            saved_steps, checkpoint_type=checkpoint_type
        ),
        _actual_checkpoint_filenames(self.directory),
    )
    self.assertSameElements(saved_steps, checkpoint_manager.all_steps())

    del checkpoint_manager
    max_to_keep = 5
    options = checkpoint_managers.CheckpointManagerOptions(
        save_interval_steps=1000, max_to_keep=max_to_keep
    )
    checkpoint_manager = self.create_checkpoint_manager(
        options, checkpoint_type=checkpoint_type
    )

    step = 10000
    steps.append(step)
    self.save(
        checkpoint_manager,
        step,
        self.train_state,
        train_state_unpadded_shape_dtype_struct=self.train_state_unpadded_shape_dtype_struct,
    )

    saved_steps_2 = steps[-max_to_keep:]

    self.assertSameElements(
        _expected_checkpoint_filenames(
            saved_steps_2, checkpoint_type=checkpoint_type
        ),
        _actual_checkpoint_filenames(self.directory),
    )
    self.assertSameElements(saved_steps_2, checkpoint_manager.all_steps())

  def test_save_on_preemption(self):
    options = checkpoint_managers.CheckpointManagerOptions(
        save_interval_steps=1000, max_to_keep=None
    )
    checkpoint_manager = self.create_checkpoint_manager(options)

    save_step = 3
    multihost_utils.reached_preemption_sync_point = (
        lambda step_id: step_id == save_step
    )

    for step in range(save_step + 1):
      self.save(
          checkpoint_manager,
          step,
          self.train_state,
          train_state_unpadded_shape_dtype_struct=self.train_state_unpadded_shape_dtype_struct,
      )

    saved_steps = [0, save_step]

    self.assertSameElements(
        _expected_checkpoint_filenames(saved_steps),
        _actual_checkpoint_filenames(self.directory),
    )
    self.assertSameElements(saved_steps, checkpoint_manager.all_steps())

  def test_cleanup(self):
    def _fake_on_commit_callback(*args, **kwargs):
      del args, kwargs
      pass  # Do nothing to simulate failure of finalization.

    options = checkpoint_managers.CheckpointManagerOptions(
        save_interval_steps=1,
        cleanup_tmp_directories=True,
    )

    with mock.patch.object(
        ocp.utils, 'ensure_atomic_save', autospec=True
    ) as commit_callback:
      commit_callback.side_effect = _fake_on_commit_callback
      checkpoint_manager = self.create_checkpoint_manager(options)
      self.save(
          checkpoint_manager,
          0,
          self.train_state,
          train_state_unpadded_shape_dtype_struct=self.train_state_unpadded_shape_dtype_struct,
      )
      # Step 0 not finalized.
      self.assertLen(
          ocp.utils.tmp_checkpoints(checkpoint_manager.directory),
          1,
      )

    checkpoint_manager = self.create_checkpoint_manager(options)
    self.assertEmpty(ocp.utils.tmp_checkpoints(checkpoint_manager.directory))
    self.save(
        checkpoint_manager,
        0,
        self.train_state,
        train_state_unpadded_shape_dtype_struct=self.train_state_unpadded_shape_dtype_struct,
    )
    self.assertSameElements(
        _expected_checkpoint_filenames([0]),
        _actual_checkpoint_filenames(checkpoint_manager.directory),
    )
    self.assertSameElements([0], checkpoint_manager.all_steps())

  @parameterized.parameters((CheckpointType.GDA,), (CheckpointType.FLAX,))
  def test_todelete_subdir(self, checkpoint_type):
    options = checkpoint_managers.CheckpointManagerOptions(
        max_to_keep=2, todelete_subdir='archive'
    )
    checkpoint_manager = self.create_checkpoint_manager(
        options, checkpoint_type=checkpoint_type
    )

    for step in range(4):
      self.save(
          checkpoint_manager,
          step,
          self.train_state,
          train_state_unpadded_shape_dtype_struct=(
              self.train_state_unpadded_shape_dtype_struct
          ),
      )

    self.assertSameElements(
        _expected_checkpoint_filenames([0, 1], checkpoint_type=checkpoint_type),
        _actual_checkpoint_filenames(os.path.join(self.directory, 'archive')),
    )
    self.assertSameElements(
        _expected_checkpoint_filenames([2, 3], checkpoint_type=checkpoint_type),
        _actual_checkpoint_filenames(os.path.join(self.directory)),
    )
    self.assertIn('archive', tf.io.gfile.listdir(self.directory))
    self.assertSameElements([2, 3], checkpoint_manager.all_steps())

  @parameterized.parameters((CheckpointType.GDA,), (CheckpointType.FLAX,))
  def test_reinitialize(self, checkpoint_type):
    options = checkpoint_managers.CheckpointManagerOptions(max_to_keep=2)
    checkpoint_manager = self.create_checkpoint_manager(
        options, checkpoint_type=checkpoint_type
    )

    for step in range(3):
      self.save(
          checkpoint_manager,
          step,
          self.train_state,
          train_state_unpadded_shape_dtype_struct=self.train_state_unpadded_shape_dtype_struct,
      )
    self.assertSameElements([1, 2], checkpoint_manager.all_steps())

    new_checkpoint_manager = self.create_checkpoint_manager(
        options, checkpoint_type=checkpoint_type
    )
    self.assertSameElements([1, 2], new_checkpoint_manager.all_steps())
    self.save(
        new_checkpoint_manager,
        3,
        self.train_state,
        train_state_unpadded_shape_dtype_struct=self.train_state_unpadded_shape_dtype_struct,
    )
    self.assertSameElements([2, 3], new_checkpoint_manager.all_steps())

  @parameterized.parameters(
      (CheckpointType.GDA, 0.0),
      (CheckpointType.FLAX, 0.0),
      (CheckpointType.GDA, 1.0),
      (CheckpointType.FLAX, 1.0),
  )
  def test_restore_legacy_format(self, checkpoint_type, legacy_version):
    live_version = 1.1
    checkpoint_manager = self.create_checkpoint_manager(
        checkpoint_managers.CheckpointManagerOptions(),
        checkpoint_type=checkpoint_type,
    )
    self.save(
        checkpoint_manager,
        0,
        self.train_state,
        train_state_unpadded_shape_dtype_struct=self.train_state_unpadded_shape_dtype_struct,
    )

    step_dir = checkpoint_manager._manager._get_save_directory(
        0, checkpoint_manager.directory
    )
    self.assertTrue(checkpoint_paths.is_checkpoint_asset(step_dir))
    self.assertTrue((step_dir / 'state').exists())
    self.assertTrue((step_dir / 'metadata').exists())
    fp_metadata = step_dir / 'metadata' / 'metadata'
    metadata = json.loads(fp_metadata.read_text())
    self.assertEqual(live_version, metadata['version'])
    self.assertIn('train_state_metadata', metadata)

    if legacy_version == 0.0:
      # Transform directory to what we would expect in a version 0 checkpoint
      # with no per-item subdirectories.
      (step_dir / 'metadata').rmtree()
      for d in (step_dir / 'state').iterdir():  # parameter directories
        if checkpoint_type == CheckpointType.GDA:
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
      checkpoint_manager._manager._options.enable_descriptor = False
    elif legacy_version == 1.0:
      # Transform metadata to what we would expect in a version 1 checkpoint
      metadata = json.loads(fp_metadata.read_text())
      metadata['version'] = 1.0
      del metadata['train_state_metadata']
      fp_metadata.write_text(json.dumps(metadata))
      checkpoint_manager._manager._version = 1.0

    expected = self.train_state
    if checkpoint_type == CheckpointType.FLAX:
      expected = jax.tree_util.tree_map(
          lambda x: np.asarray(x.addressable_data(0)),
          expected,
      )
    train_state_global_shapes = jax.eval_shape(lambda x: x, self.train_state)
    train_state_unpadded_shape_dtype_struct = jax.tree_util.tree_map(
        lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype),
        train_state_global_shapes,
    )
    # restoring old checkpoint using old checkpoint_manager
    restored = self.restore(
        checkpoint_manager,
        0,
        train_state_global_shapes,
        self.state_specs,
        checkpoint_type,
        global_mesh=self.global_mesh,
        train_state_unpadded_shape_dtype_struct=train_state_unpadded_shape_dtype_struct,
    )
    ocp.test_utils.assert_tree_equal(self, expected, restored)

    # Saving again, we expect it to be saved with the old format.
    self.save(checkpoint_manager, 1, self.train_state)
    step_dir = checkpoint_manager._manager._get_save_directory(
        1, checkpoint_manager.directory
    )
    if legacy_version == 0.0:
      # assertions against version 0 checkpoint
      self.assertTrue(checkpoint_paths.is_checkpoint_asset(step_dir))
      self.assertFalse((step_dir / 'state').exists())
      self.assertFalse((step_dir / 'metadata').exists())
    elif legacy_version == 1.0:
      # assertions against version 1 checkpoint
      self.assertTrue(checkpoint_paths.is_checkpoint_asset(step_dir))
      self.assertTrue((step_dir / 'state').exists())
      self.assertTrue((step_dir / 'metadata').exists())
      metadata = json.loads(fp_metadata.read_text())
      self.assertNotIn('train_state_metadata', metadata)
    else:
      raise ValueError(f'Unknown legacy version {legacy_version}')

    # Restoring again the old format.
    # We construct checkpoint_manager again to make sure it reads the correct
    # version from the checkpoint dir.
    checkpoint_manager = self.create_checkpoint_manager(
        checkpoint_managers.CheckpointManagerOptions(),
        checkpoint_type=checkpoint_type,
    )
    self.assertEqual(legacy_version, checkpoint_manager._manager._version)

    restored = self.restore(
        checkpoint_manager,
        1,
        train_state_global_shapes,
        self.state_specs,
        checkpoint_type,
        global_mesh=self.global_mesh,
        train_state_unpadded_shape_dtype_struct=train_state_unpadded_shape_dtype_struct,
    )
    expected = self.train_state
    expected = expected.replace(step=expected.step + 1)  # increment the step
    if checkpoint_type == CheckpointType.FLAX:
      expected = jax.tree_util.tree_map(
          lambda x: np.asarray(x.addressable_data(0)),
          expected,
      )
    ocp.test_utils.assert_tree_equal(self, expected, restored)

  def test_transforms(self):
    ts_context = ts.Context({
        'cache_pool#ocdbt': {'total_bytes_limit': 100000000},
        'file_io_concurrency': {'limit': 128},
    })
    # OCDBT required to use transforms.
    with ocdbt_checkpoint_context(True, ts_context):
      checkpoint_type = CheckpointType.GDA
      checkpoint_manager = self.create_checkpoint_manager(
          checkpoint_managers.CheckpointManagerOptions(),
          checkpoint_type=checkpoint_type,
          tensorstore_use_ocdbt=True,
      )

      train_state_global_shapes = jax.eval_shape(lambda x: x, self.train_state)
      train_state_unpadded_shape_dtype_struct = jax.tree_util.tree_map(
          lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype),
          train_state_global_shapes,
      )
      self.save(
          checkpoint_manager,
          0,
          self.train_state,
          train_state_unpadded_shape_dtype_struct=train_state_unpadded_shape_dtype_struct,
      )

      mdl_vars = self.train_state.mdl_vars
      mdl_vars['x'] = mdl_vars['a']
      mdl_vars['y'] = {'z': mdl_vars['b']}
      del mdl_vars['a'], mdl_vars['b']
      expected = TrainState(
          self.train_state.step,
          mdl_vars,
          ocp.test_utils.apply_function(
              self.train_state.opt_states, lambda x: x * 2
          ),
          self.train_state.extra_state,
      )
      axes = jax.sharding.PartitionSpec(
          None,
      )
      state_specs = jax.tree_util.tree_map(
          lambda _: axes,
          expected,
      )
      transforms = {
          r'(.*)opt_states(.*)': ocp.Transform(value_fn=lambda x: x * 2),
          'mdl_vars': {
              'x': ocp.Transform(original_key='mdl_vars/a'),
              'y': {
                  'z': ocp.Transform(original_key='mdl_vars/b'),
              },
          },
      }

      train_state_global_shapes = jax.eval_shape(lambda x: x, expected)
      train_state_unpadded_shape_dtype_struct = jax.tree_util.tree_map(
          lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype),
          train_state_global_shapes,
      )
      restored = self.restore(
          checkpoint_manager,
          0,
          train_state_global_shapes,
          state_specs,
          checkpoint_type,
          global_mesh=self.global_mesh,
          train_state_unpadded_shape_dtype_struct=train_state_unpadded_shape_dtype_struct,
          transforms=transforms,
      )
      ocp.test_utils.assert_tree_equal(self, expected, restored)

  @parameterized.parameters((True,), (False,))
  def test_restoring_gda_independent_of_prefix(self, remove_checkpoint_prefix):
    # Test that GDA checkpoint formats can be restored, whether the
    # "checkpoint_" prefix is there or not.
    train_input_pipeline = None
    checkpoint_manager = self.create_checkpoint_manager(
        checkpoint_managers.CheckpointManagerOptions(),
        checkpoint_type=CheckpointType.GDA,
        train_input_pipeline=train_input_pipeline,
    )
    self.save(
        checkpoint_manager,
        0,
        self.train_state,
        train_input_pipeline,
        train_state_unpadded_shape_dtype_struct=(
            self.train_state_unpadded_shape_dtype_struct
        ),
    )
    if remove_checkpoint_prefix:
      os.rename(
          os.path.join(self.directory, 'checkpoint_00000000'),
          os.path.join(self.directory, '0'),
      )
    expected = self.train_state
    train_state_global_shapes = jax.eval_shape(lambda x: x, self.train_state)
    with mock.patch.object(
        checkpoint_managers,
        '_has_digit_step_subdirectory',
        return_value=remove_checkpoint_prefix,
    ) as mock_has_digit:
      checkpoint_manager = self.create_checkpoint_manager(
          checkpoint_managers.CheckpointManagerOptions(),
          checkpoint_type=CheckpointType.GDA,
          train_input_pipeline=train_input_pipeline,
      )
      mock_has_digit.assert_called_once()
    self.assertEqual(checkpoint_manager.all_steps(), [0])

    restored = checkpoint_manager.restore(
        0,
        train_state_global_shapes,
        train_input_pipeline=train_input_pipeline,
        train_state_unpadded_shape_dtype_struct=self.train_state_unpadded_shape_dtype_struct,
        restore_kwargs={
            'specs': self.state_specs,
            'mesh': self.global_mesh,
            'transforms': None,
        },
    )

    ocp.test_utils.assert_tree_equal(self, expected, restored)


if __name__ == '__main__':
  absltest.main()
