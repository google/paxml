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

"""Checkpointing-related utilities to handle TrainState instances."""

import os
import re
from typing import Any, Optional, Sequence, Tuple, Union

from absl import logging
from etils import epath
import flax.serialization
from flax.training import checkpoints as flax_checkpoints
import jax
from jax.experimental import maps
from jax.experimental import multihost_utils
import numpy as np
import optax
import orbax.checkpoint
from paxml import base_task
from paxml import checkpoint_pb2
from praxis import py_utils
from praxis import pytypes
from praxis import train_states

CHECKPOINT_PREFIX = 'checkpoint_'
TMP_PREFIX = 'tmp_'
CHECKPOINT_PATTERN_RE = re.compile(rf'{CHECKPOINT_PREFIX}[\d]+$')
TMP_CHECKPOINT_PATTERN_RE = re.compile(
    rf'{TMP_PREFIX}[\d]+.{CHECKPOINT_PREFIX}[\d]+$')
# Large value to disable flax-specific checkpoint management.
_MAX_CHECKPOINT_FLAX = 1000000

CheckpointType = checkpoint_pb2.CheckpointType
JTensorOrPartitionSpec = pytypes.JTensorOrPartitionSpec
PyTreeDef = pytypes.PyTreeDef
AsyncCheckpointer = orbax.checkpoint.AsyncCheckpointer
Checkpointer = orbax.checkpoint.Checkpointer
COMMIT_SUCCESS_FILE = 'commit_success.txt'


def is_checkpoint_asset(x: epath.Path) -> bool:
  return bool(CHECKPOINT_PATTERN_RE.match(os.path.basename(x)))


def _is_tmp_checkpoint_asset(x: epath.Path) -> bool:
  return bool(TMP_CHECKPOINT_PATTERN_RE.match(os.path.basename(x)))


def checkpoint_name(step: int) -> str:
  return f'{CHECKPOINT_PREFIX}{step:08d}'


def _make_checkpoint_step_dir(checkpoint_dir: epath.Path,
                              step: int) -> epath.Path:
  return checkpoint_dir / checkpoint_name(step)


def get_step_from_checkpoint_asset(checkpoint_dir: epath.PathLike) -> int:
  checkpoint_dir = epath.Path(checkpoint_dir)
  if _is_tmp_checkpoint_asset(checkpoint_dir):
    return int(checkpoint_dir.suffix[len(CHECKPOINT_PREFIX):])
  return int(checkpoint_dir.stem[len(CHECKPOINT_PREFIX):])


def retrieve_checkpoint_type(
    maybe_use_persistence_checkpointing,
    task_p: base_task.BaseTask.HParams) -> CheckpointType:
  """Retrieves the CheckpointType given the input arguments."""
  using_pjit = task_p.model.mesh_shape is not None  # pytype: disable=attribute-error
  if using_pjit or py_utils.pmap_use_tensorstore():
    if using_pjit:
      assert py_utils.gda_or_jax_array(), 'pjit requires GDA or jax.Array'
    if maybe_use_persistence_checkpointing:
      return CheckpointType.CHECKPOINT_PERSISTENCE
    else:
      return CheckpointType.CHECKPOINT_GDA
  else:
    # pmap uses CHECKPOINT_FLAX, Persistence-based or not.
    return CheckpointType.CHECKPOINT_FLAX


def save_checkpoint(
    train_state: train_states.TrainState,
    checkpoint_dir: epath.PathLike,
    overwrite: bool = False,
    checkpoint_type: CheckpointType = CheckpointType.CHECKPOINT_FLAX,
    state_specs: Optional[train_states.TrainState] = None,
    async_checkpointer: Optional[AsyncCheckpointer] = None) -> None:
  """Saves a checkpoint into the provided base directory.

  This is typically called on a replicated TrainState instance.

  Args:
    train_state: The TrainState instance to save.
    checkpoint_dir: The base directory from where to retrieve checkpoints.
    overwrite: Whether to overwrite existing checkpoints files if a checkpoint
      at the current or a later step already exists.
    checkpoint_type: The checkpoint type (implementation) to save. Either
      `CHECKPOINT_FLAX`, `CHECKPOINT_GDA` or `CHECKPOINT_PERSISTENCE`.
    state_specs: Currently unused.
    async_checkpointer: When async checkpointing and Orbax are enabled, allows
      training to continue when checkpointing is going on as checkpointing
      happens in a different thread.

  Raises:
    ValueError: If the global step has an unexpected shape, if `state_specs`
    is not specified for persistence-based checkpointing or if
    `checkpoint_type` is invalid.
  """
  del state_specs

  checkpoint_dir = epath.Path(checkpoint_dir)
  step = int(py_utils.maybe_unreplicate_for_fully_replicated(train_state.step))

  if checkpoint_type == CheckpointType.CHECKPOINT_GDA:
    checkpoint_step_dir = _make_checkpoint_step_dir(checkpoint_dir, step)
    if async_checkpointer is not None:
      async_checkpointer.save(checkpoint_step_dir, train_state)
    else:
      checkpointer = orbax.checkpoint.Checkpointer(
          PaxCheckpointHandler(enable_aggregation=False))
      checkpointer.save(checkpoint_step_dir, train_state)
  elif checkpoint_type == CheckpointType.CHECKPOINT_FLAX:
    checkpointer = FlaxCheckpointer()
    checkpointer.save(checkpoint_dir, train_state, force=overwrite, step=step)
  else:
    raise ValueError(f'Unexpected checkpoint_type `{checkpoint_type}`.')


def latest_checkpoint(checkpoint_dir: epath.PathLike) -> Optional[epath.Path]:
  """Gets the path to the latest checkpoint.

  Args:
    checkpoint_dir: The base directory from where to retrieve checkpoints.

  Returns:
    Path to latest checkpoint or None if there is no checkpoint.
  """
  checkpoint_dir = epath.Path(checkpoint_dir)
  if not checkpoint_dir.exists():
    return None
  # Note: is_checkpoint_asset() already filters out flax temporary checkpoints
  # that would be ending with `tmp`.
  checkpoint_assets = [
      v for v in checkpoint_dir.iterdir() if is_checkpoint_asset(v)
  ]
  if not checkpoint_assets:
    return None
  checkpoint_assets = sorted(
      checkpoint_assets, key=get_step_from_checkpoint_asset)
  return checkpoint_dir / checkpoint_assets[-1]


def retrieve_latest_checkpoint_step(
    checkpoint_dir: epath.Path) -> Optional[int]:
  """Retrieves the latest checkpoint step if any.

  Note that this broadcasts the checkpoint step from host 0 to ensure that all
  processes get the exact same checkpoint step.

  Args:
    checkpoint_dir: The base directory from where to retrieve checkpoints.

  Returns:
    The latest checkpoint step as an integer or None if no checkpoint is found.
  """
  if not checkpoint_dir.exists():
    checkpoint_step = -1
  else:
    latest_checkpoint_path = latest_checkpoint(checkpoint_dir)
    if latest_checkpoint_path is None:
      checkpoint_step = -1
    else:
      checkpoint_step = get_step_from_checkpoint_asset(latest_checkpoint_path)
  np_checkpoint_step = multihost_utils.broadcast_one_to_all(
      np.array(checkpoint_step))
  multihost_utils.assert_equal(np_checkpoint_step,
                               "checkpoint_steps across hosts don't match.")
  step = int(np_checkpoint_step.item())
  if step == -1:
    return None
  return step


def restore_checkpoint(
    state_global_shapes: train_states.TrainState,
    checkpoint_dir: epath.PathLike,
    global_mesh: Optional[maps.Mesh] = None,
    checkpoint_type: CheckpointType = CheckpointType.CHECKPOINT_FLAX,
    state_specs: Optional[train_states.TrainState] = None,
    step: Optional[int] = None) -> Optional[train_states.TrainState]:
  """Restores a checkpoint from the provided base directory.

  This is typically called on an unreplicated TrainState instance.

  Args:
    state_global_shapes: The TrainState with variable names and corresponding
      ShapeDtypeStruct.
    checkpoint_dir: The base directory from where to retrieve checkpoints.
    global_mesh: The global mesh representing devices across multiple processes.
    checkpoint_type: The checkpoint type (implementation) to restore. Either
      `CHECKPOINT_FLAX`, `CHECKPOINT_GDA` or `CHECKPOINT_PERSISTENCE`.
    state_specs: If using a GDA-based checkpoint, the partition specs
      corresponding to this TrainState instance to restore.
    step: Step number to load a checkpoint from or None to load the latest.

  Returns:
    A restored `TrainState` instance. If no step specified and no checkpoint
    files present, return None.

  Raises:
    ValueError: When a mismatch between the current checkpoint structure and
    the saved checkpoint one is detected.
  """
  checkpoint_dir = epath.Path(checkpoint_dir)
  if checkpoint_type == CheckpointType.CHECKPOINT_GDA:
    if step is None:
      step = retrieve_latest_checkpoint_step(checkpoint_dir)
      if step is None:
        logging.info('No checkpoint found for restore in %s.', checkpoint_dir)
        return None
    checkpoint_step_dir = _make_checkpoint_step_dir(checkpoint_dir, step)
    checkpointer = orbax.checkpoint.Checkpointer(
        PaxCheckpointHandler(enable_aggregation=False))
    restored_train_state = checkpointer.restore(
        checkpoint_step_dir,
        item=state_global_shapes,
        specs=state_specs,
        mesh=global_mesh)
    return restored_train_state
  elif checkpoint_type == CheckpointType.CHECKPOINT_FLAX:
    checkpointer = FlaxCheckpointer()
    return checkpointer.restore(
        checkpoint_dir, item=state_global_shapes, step=step)
  else:
    raise ValueError(f'Unexpected checkpoint_type `{checkpoint_type}`.')


def _save_checkpoint_flax(train_state: train_states.TrainState,
                          checkpoint_dir: epath.Path, overwrite: bool,
                          step: int) -> None:
  """Saves a checkpoint using Flax serialization mechanism."""
  if not overwrite:
    previous_filename = latest_checkpoint(checkpoint_dir)
    if previous_filename:
      previous_step = int(str(previous_filename).rsplit('_', 1)[-1])
      if previous_step >= step:
        logging.warning(
            'A more recent checkpoint `%d` has already been saved compared '
            'to the current timestep `%d`. Skip saving a checkpoint.',
            previous_step, step)
        return

  # Extract/flatten data structure to store to disk. Flax requires a flattened
  # data structure to be passed to the checkpointer.
  flattened_state, pytree_state = jax.tree_util.tree_flatten(
      jax.device_get(train_state))
  checkpoint_target = {
      'flattened_state': flattened_state,
      # Saves a serialized version of the pytree structure to detect potential
      # mismatch caused by different versions of saver/restorer.
      'str_pytree_state': str(pytree_state),
  }

  flax_checkpoints.save_checkpoint(
      checkpoint_dir,
      checkpoint_target,
      step,
      prefix=CHECKPOINT_PREFIX,
      keep=_MAX_CHECKPOINT_FLAX,
      overwrite=overwrite)


def _restore_checkpoint_flax(
    state_global_shapes: train_states.TrainState,
    checkpoint_dir: epath.Path,
    step: Optional[int] = None) -> Optional[train_states.TrainState]:
  """Restores a checkpoint using Flax serialization mechanism."""
  # Input the same data structure as in save_checkpoint().
  flattened_state, pytree_state = jax.tree_util.tree_flatten(
      state_global_shapes)
  str_pytree_state = str(pytree_state)
  input_target = {
      'flattened_state': flattened_state,
      'str_pytree_state': str_pytree_state,
  }
  restored_target = flax_checkpoints.restore_checkpoint(
      os.fspath(checkpoint_dir), input_target, step=step)
  # Flax restore_checkpoint returned input_target unchanged if
  # no step specified and no checkpoint files present.
  if restored_target is input_target:
    return None
  restored_state = restored_target['flattened_state']
  restored_str_pytree_state = restored_target['str_pytree_state']
  if restored_str_pytree_state != str_pytree_state:
    # Could be spurious due to abbreviation of treedef printing added in
    # https://github.com/tensorflow/tensorflow/commit/aa21adc148c98c76f54ba5932ce34cf59da538c4
    logging.warning(
        'A possible mismatch (could be spurious) between the saved '
        'checkpoint structure and the current one has been detected '
        '(%s vs %s).', restored_str_pytree_state, str_pytree_state)
  return jax.tree_util.tree_unflatten(pytree_state, restored_state)


def _extract_nested_prefix_names(
    state: train_states.TrainState) -> train_states.TrainState:
  """Extracts prefix names from a TrainState data structure."""
  # CNS doesn't support square bracket in filenames.
  key_separator = '.'
  left_separator = '_'
  right_separator = ''
  return train_states.TrainState(
      step=py_utils.extract_prefixed_keys_from_nested_map(
          state.step,
          'step',
          key_separator=key_separator,
          left_separator=left_separator,
          right_separator=right_separator),
      mdl_vars=py_utils.extract_prefixed_keys_from_nested_map(
          state.mdl_vars,
          'mdl_vars',
          key_separator=key_separator,
          left_separator=left_separator,
          right_separator=right_separator),
      opt_states=py_utils.extract_prefixed_keys_from_nested_map(
          state.opt_states,
          'opt_states',
          key_separator=key_separator,
          left_separator=left_separator,
          right_separator=right_separator,
          is_leaf=py_utils.is_optax_masked_node))


def _masked_node_to_none(mask: Any, value: Any) -> Any:
  """Return value when `mask` is not a MaskedNode, or MaskedNode otherwise."""
  if py_utils.is_optax_masked_node(mask):
    return optax.MaskedNode()
  return value


def _tensorstore_prepare(
    train_state: train_states.TrainState,
    state_specs: Optional[train_states.TrainState] = None
) -> Tuple[Sequence[JTensorOrPartitionSpec], Sequence[str],
           Optional[Sequence[JTensorOrPartitionSpec]]]:
  """Prepares data prior to saving/restoring it from/to TensorStore.

  Args:
    train_state: A partitioned train_state that is a Pytree of
      GlobalDeviceArray.
    state_specs: [optional] The partition specs corresponding to this TrainState
      instance, when it is used for checkpoint restoring.

  Returns:
    A 3-tuple (flattened_traine_state, flattened_nested_names, out), where:
    - flattened_traine_state: A flattened version of the train state, where all
      MaskedNode instances have been filtered out.
    - flattened_nested_names: A flattened version of the nested names, where all
      entries corresponding to MaskedNode have been filtered out.
    - out: Either None when the input state_specs is None or the flattened
      version of the state_specs, where all entries corresponding to MaskedNode
      instances have been filtered out.
  """
  # This replaces MaskedNode instances by None values ...
  train_state_none = jax.tree_map(_masked_node_to_none, train_state,
                                  train_state)
  if state_specs is not None:
    state_specs_none = jax.tree_map(
        _masked_node_to_none,
        train_state,
        state_specs,
        is_leaf=py_utils.is_optax_masked_node)
  # ... that are filtered out when calling jax.tree_util.tree_flatten() here.
  flattened_train_state, _ = jax.tree_util.tree_flatten(train_state_none)
  if state_specs is not None:
    flattened_state_specs, _ = jax.tree_util.tree_flatten(state_specs_none)
  else:
    flattened_state_specs = None

  # _extract_nested_prefix_names() also replaces MaskedNode instances by None
  # values ...
  nested_names = _extract_nested_prefix_names(train_state)
  # ... that are filtered out when calling jax.tree_util.tree_flatten() here.
  flattened_nested_names, _ = jax.tree_util.tree_flatten(nested_names)
  return flattened_train_state, flattened_nested_names, flattened_state_specs


def _tensorstore_reconstruct(
    state_global_shapes: train_states.TrainState,
    restored_train_state: Sequence[JTensorOrPartitionSpec]
) -> train_states.TrainState:
  """Reconstructs a nested train state including MaskedNode.

  Args:
    state_global_shapes: The original nested train state with GDAs, which
      includes MaskedNode entries.
    restored_train_state: A flattened version of the restored train state, which
      does not include any MaskedNode entry.

  Returns:
    A nested version of `restored_train_state` after adding back the MaskedNode
    instances, based on the original structure of `state_global_shapes`.
  """
  c = 0
  restored_flattened_train_state = []
  flattened_state_global_shapes, treedef = jax.tree_util.tree_flatten(
      state_global_shapes)
  for l in flattened_state_global_shapes:
    if py_utils.is_optax_masked_node(l):
      restored_flattened_train_state.append(optax.MaskedNode())
    else:
      restored_flattened_train_state.append(restored_train_state[c])
      c += 1
  assert c == len(restored_train_state)
  return jax.tree_util.tree_unflatten(treedef, restored_flattened_train_state)


class PaxCheckpointHandler(orbax.checkpoint.PyTreeCheckpointHandler):
  """PaxCheckpointHandler override for Pax GDA checkpointing.

  Allows setting parameter names manually, which would normally be extracted
  from the train state itself. This is somewhat hacky, and we will aim to remove
  it eventually (see below).

  TODO(cpgaffney) Rework _extract_nested_prefix_names to allow extracting names
  from a state dict.
  """

  _param_names: PyTreeDef = None

  def _set_param_names(self, param_names: PyTreeDef):
    self._param_names = param_names

  def _get_param_names(self, item: PyTreeDef) -> PyTreeDef:
    return self._param_names

  async def async_save(self,
                       directory: epath.Path,
                       item: PyTreeDef,
                       save_args: Optional[PyTreeDef] = None) -> Any:
    """Filters optax.MaskedNode before calling superclass async_save."""
    flattened_train_state, flattened_nested_names, _ = _tensorstore_prepare(
        item)
    # At that point, the flattened entries do not contain any reference to
    # MaskedNode's.
    self._set_param_names(flattened_nested_names)
    return await super().async_save(
        directory, flattened_train_state, save_args=save_args)

  def restore(self,
              directory: epath.Path,
              item: Optional[PyTreeDef] = None,
              specs: Optional[PyTreeDef] = None,
              mesh: Optional[maps.Mesh] = None) -> PyTreeDef:
    """Restores by filtering optax.MaskedNode and adding it back after calling superclass restore."""
    flattened_train_state, flattened_nested_names, flattened_state_specs = (
        _tensorstore_prepare(item, specs))
    # At that point, the flattened entries do not contain any reference to
    # MaskedNode's.
    self._set_param_names(flattened_nested_names)

    def create_restore_args(pspec, shape_struct):
      return orbax.checkpoint.ArrayRestoreArgs(
          mesh=mesh, mesh_axes=pspec, global_shape=shape_struct.shape)

    restore_args = jax.tree_map(create_restore_args, flattened_state_specs,
                                flattened_train_state)

    # Consequently, we restore the checkpoint that does not contain any
    # reference to MaskedNode's.
    restored_train_state = super().restore(
        directory, item=flattened_train_state, restore_args=restore_args)

    # We add back the MaskedNode entries into the pytree.
    restored_train_state = _tensorstore_reconstruct(item, restored_train_state)

    return restored_train_state

  def structure(self, directory: epath.Path) -> PyTreeDef:
    return flax.serialization.to_state_dict(self._param_names)


class FlaxCheckpointer(orbax.checkpoint.AbstractCheckpointer):
  """Thin Orbax-compatible wrapper around flax_checkpoints.

  Since Flax-checkpoint support in Pax will eventually be deprecated, we do not
  plan to provide a more well-integrated interface.
  """

  def save(self,
           directory: epath.PathLike,
           item: Any,
           force: bool = False,
           step: Optional[int] = None):
    if not py_utils.pmap_use_tensorstore() and jax.process_index() != 0:
      return

    if step is None:
      raise ValueError('Required argument `step` for `FlaxCheckpointer.save`')
    _save_checkpoint_flax(item, epath.Path(directory), force, step)

  def restore(self,
              directory: epath.PathLike,
              *args: Any,
              item: Optional[Any] = None,
              step: Optional[int] = None) -> Any:
    if item is None:
      raise ValueError(
          'Required argument `item` for `FlaxCheckpointer.restore`')
    return _restore_checkpoint_flax(item, epath.Path(directory), step=step)

  def structure(self, directory: epath.PathLike) -> Optional[Any]:
    return NotImplementedError
