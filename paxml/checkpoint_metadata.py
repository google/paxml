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

"""Functions and classes for checkpoint metadata."""

import dataclasses
import functools
from typing import Any, Callable, Dict, Mapping, Optional

from absl import logging
from etils import epath
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from paxml import checkpoint_version
from paxml import train_states
from praxis import py_utils
from praxis import pytypes


PAX_METADATA_ITEM_NAME = 'pax_metadata'
METADATA_ITEM_NAME = ocp.checkpoint_manager.METADATA_ITEM_NAME

get_version_key = checkpoint_version.get_version_key
get_version = checkpoint_version.get_version
Checkpointer = ocp.Checkpointer

# string consts used in metadata
ARRAY_METADATA_TAG = '_array_metadata_tag'
IS_OPTAX_MASKED_NODE = 'is_optax_masked_node'
UNPADDED_SHAPE = 'unpadded_shape'
DTYPE = 'dtype'
TRAIN_STATE_METADATA = 'train_state_metadata'


def _get_shape_dtype_struct(nested: Any) -> Any:
  return jax.tree_util.tree_map(
      lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), nested
  )


def make_metadata(
    version: Optional[float] = None,
    train_state: Optional[train_states.TrainState] = None,
    train_state_unpadded_shape_dtype_struct: Optional[
        train_states.TrainState
    ] = None,
    tensorstore_use_ocdbt: Optional[bool] = None,
) -> Mapping[str, Any]:
  """Returns metadata dict."""
  if version is None:
    version = get_version(tensorstore_use_ocdbt=tensorstore_use_ocdbt)

  if version > 1:
    if train_state is None:
      raise ValueError(
          'train_state is required for version>1, to save the unpadded'
          ' shapes/dtypes/maskednodes in the checkpoint metadata.'
      )
    if train_state_unpadded_shape_dtype_struct is None:
      train_state_unpadded_shape_dtype_struct = _get_shape_dtype_struct(
          train_state
      )
      logging.warning(
          'train_state_unpadded_shape_dtype_struct is not provided. We assume'
          ' `train_state` is unpadded.'
      )
    metadata_dict = PaxMetadata.from_padded_and_unpadded(
        train_state, train_state_unpadded_shape_dtype_struct, version=version
    ).to_dict()
    return metadata_dict
  else:
    return {get_version_key(): version}


def metadata_exists(directory: epath.Path) -> bool:
  path = directory / METADATA_ITEM_NAME
  return path.is_dir() and path.exists()


def save_metadata(directory: epath.Path, metadata: Mapping[str, Any]):
  checkpointer = Checkpointer(ocp.JsonCheckpointHandler())
  path = directory / METADATA_ITEM_NAME
  checkpointer.save(path, metadata)


def restore_metadata(directory: epath.Path) -> Mapping[str, Any]:
  checkpointer = Checkpointer(ocp.JsonCheckpointHandler())
  path = directory / METADATA_ITEM_NAME
  return checkpointer.restore(path)


def _trees_are_equal(
    a_tree: pytypes.Nested[Any],
    b_tree: pytypes.Nested[Any],
    equal_fn: Optional[Callable[[Any, Any], bool]] = None,
    treedef: bool = False,
) -> bool:
  """Checks if the two trees are equal w.r.t. equal_fn."""
  a_flat, a_treedef = jax.tree_util.tree_flatten(a_tree)
  b_flat, b_treedef = jax.tree_util.tree_flatten(b_tree)

  if treedef and a_treedef != b_treedef:
    return False

  if equal_fn is None:
    equal_fn = lambda a, b: a == b

  for a_leaf, b_leaf in zip(a_flat, b_flat):
    if not equal_fn(a_leaf, b_leaf):
      return False
  return True


@dataclasses.dataclass
class ArrayMetadata:
  """Metadata of an array, to be saved in PaxMetadata."""

  unpadded_shape_dtype_struct: Optional[jax.ShapeDtypeStruct]
  is_optax_masked_node: bool

  def to_dict(self) -> Dict[str, Any]:
    """Returns a dict to be serialized."""
    d = {
        IS_OPTAX_MASKED_NODE: self.is_optax_masked_node,
        ARRAY_METADATA_TAG: True,  # tag needed to restore the ArrayMetadata
    }
    if self.unpadded_shape_dtype_struct is None:
      return d
    else:
      d.update({
          UNPADDED_SHAPE: self.unpadded_shape_dtype_struct.shape,
          DTYPE: str(self.unpadded_shape_dtype_struct.dtype),
      })
      return d

  @classmethod
  def from_dict(cls, d: Mapping[str, Any]) -> 'ArrayMetadata':
    """Constructs ArrayMetadata from a dict."""
    if UNPADDED_SHAPE in d and DTYPE in d:
      unpadded_shape_dtype_struct = jax.ShapeDtypeStruct(
          shape=tuple(d[UNPADDED_SHAPE]),
          dtype=jnp.dtype(d[DTYPE]),
      )
    else:
      unpadded_shape_dtype_struct = None

    return cls(
        unpadded_shape_dtype_struct=unpadded_shape_dtype_struct,
        is_optax_masked_node=d[IS_OPTAX_MASKED_NODE],
    )

  def equals(self, other: 'ArrayMetadata') -> bool:
    """Checks whether another ArrayMetadata is the same."""
    return (self.is_optax_masked_node == other.is_optax_masked_node) and (
        self.unpadded_shape_dtype_struct == other.unpadded_shape_dtype_struct
    )

  def is_compatible(self, other: 'ArrayMetadata') -> bool:
    """Checks whether another ArrayMetadata is compatible."""
    # if the node is masked, we do not check the shape/dtype
    if self.is_optax_masked_node and other.is_optax_masked_node:
      return True
    return self.equals(other)


@dataclasses.dataclass
class PaxMetadata:
  """Pax checkpoint metadata.

  This class is only to be used for version > 1.0.
  """

  version: float  # version of the checkpoint
  train_state_metadata: pytypes.Nested[ArrayMetadata]

  def to_dict(self) -> Dict[str, Any]:
    """Returns a dict to be serialized."""
    train_state_metadata = jax.tree_util.tree_map(
        lambda x: x.to_dict(),
        self.train_state_metadata,
    )
    if dataclasses.is_dataclass(train_state_metadata):
      train_state_metadata = dataclasses.asdict(train_state_metadata)

    # serialize to a nested dict so that it is json-serializable
    train_state_metadata = ocp.utils.serialize_tree(
        train_state_metadata,
        keep_empty_nodes=True,
    )

    return dict(
        version=self.version,
        train_state_metadata=train_state_metadata,
    )

  @classmethod
  def from_dict(cls, d: Mapping[str, Any]) -> 'PaxMetadata':
    """Constructs PaxMetadata from a dict."""

    # For version > 1.0, we require train_state_metadata.
    train_state_metadata = d[TRAIN_STATE_METADATA]
    if train_state_metadata is None:
      raise ValueError('train_state_metadata is required for version > 1.0.')

    def _is_array_metadata(d):
      return isinstance(d, dict) and d.get(ARRAY_METADATA_TAG)

    train_state_metadata = jax.tree_util.tree_map(
        ArrayMetadata.from_dict,
        train_state_metadata,
        is_leaf=_is_array_metadata,
    )
    return cls(
        version=d[get_version_key()],
        train_state_metadata=train_state_metadata,
    )

  @classmethod
  def from_padded_and_unpadded(
      cls,
      padded: train_states.TrainState,  # of ShapeDtypeStruct or jax.Array
      unpadded: Optional[train_states.TrainState],  # of ShapeDtypeStruct
      version: float,
      mdl_vars_only: bool = True,
  ) -> 'PaxMetadata':
    """Constructs PaxMetadata from padded and unpadded train state."""

    def _to_dict(train_state):
      return dict(
          step=train_state.step,
          mdl_vars=train_state.mdl_vars,
          opt_states=train_state.opt_states,
          extra_state=train_state.extra_state,
      )

    def _maybe_remove_keys(d):
      if mdl_vars_only:
        # Since we use json to serialize the metadata, we only save the mdl_vars
        # because many opt_states are not json-serializable.
        # Also some users have manipulated the train_state without manipulating
        # the metadata of the opt_states, leading to mismatch.
        if 'step' in d:
          del d['step']
        if 'opt_states' in d:
          del d['opt_states']
        if 'extra_state' in d:
          del d['extra_state']
      return d

    padded = _to_dict(padded)
    padded = _maybe_remove_keys(padded)
    if unpadded is not None:
      unpadded = _to_dict(unpadded)
      unpadded = _maybe_remove_keys(unpadded)

    def _get_array_metadata(padded_, unpadded_):
      padded_is_masked = py_utils.is_optax_masked_node(padded_)
      unpadded_is_masked = py_utils.is_optax_masked_node(unpadded_)
      if unpadded_is_masked:
        unpadded_shape_dtype_struct = None
      else:
        unpadded_shape_dtype_struct = unpadded_
      return ArrayMetadata(
          is_optax_masked_node=padded_is_masked,
          unpadded_shape_dtype_struct=unpadded_shape_dtype_struct,
      )

    if unpadded is None:
      train_state_metadata = jax.tree_util.tree_map(
          functools.partial(_get_array_metadata, unpadded_=None),
          padded,
          is_leaf=py_utils.is_optax_masked_node,
      )
    else:
      train_state_metadata = jax.tree_util.tree_map(
          _get_array_metadata,
          padded,
          unpadded,
          is_leaf=py_utils.is_optax_masked_node,
      )

    return cls(
        version=version,
        train_state_metadata=train_state_metadata,
    )

  def equals(self, other: 'PaxMetadata') -> bool:
    """Checks whether another PaxMetadata is the same."""
    return (self.version == other.version) and _trees_are_equal(
        self.train_state_metadata,
        other.train_state_metadata,
        equal_fn=lambda x, y: x.equals(y),
    )

  def is_compatible(self, other: 'PaxMetadata') -> bool:
    """Checks whether another PaxMetadata is compatible."""
    return _trees_are_equal(
        self.train_state_metadata,
        other.train_state_metadata,
        equal_fn=lambda x, y: x.is_compatible(y),
    )
