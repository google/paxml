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

"""Module for all partitioners and related utilities."""

import abc
import dataclasses
import functools
import json
from typing import Any, Callable, Dict, Optional, Protocol, Sequence, Tuple

from absl import logging
from clu import platform
from etils import epath
from flax.core import frozen_dict
import jax
from jax import core
from jax import numpy as jnp
from jax.experimental import pjit
from paxml import tasks_lib
from paxml import train_states
from paxml import trainer_lib
from praxis import asserts
from praxis import base_input
from praxis import base_layer
from praxis import py_utils
from praxis import pytypes

from paxml import checkpoints  # mapped to internal

PartitionSpec = jax.sharding.PartitionSpec

CheckpointType = checkpoints.CheckpointType
PRNGKey = pytypes.PRNGKey
NestedJTensor = pytypes.NestedJTensor
NestedPartitionSpec = pytypes.NestedPartitionSpec
NestedShapeDtypeLike = pytypes.NestedShapeDtypeLike
NestedWeightHParams = base_layer.NestedWeightHParams
TrainState = train_states.TrainState
TrainStateMetadata = trainer_lib.TrainStateMetadata
RunningMode = trainer_lib.RunningMode


def filter_nestedmap(full_specs, partial_specs):
  """Project full_specs into partial_specs."""
  if isinstance(full_specs, dict):
    result = type(full_specs)()
    for key in partial_specs.keys():  # pytype: disable=attribute-error  # jax-ndarray
      result[key] = filter_nestedmap(full_specs[key], partial_specs[key])
    return result
  elif isinstance(full_specs, list):
    # This handles the case where a list of children layers are added using
    # `BaseLayer.create_children()`.
    # Note we don't handle `tuple` since `PartitionSpec` is a subclass of it,
    # and we want to treat `ParttionSpec` as leaf.
    assert len(full_specs) == len(partial_specs), (
        f'Length mismatch. {len(full_specs)=} vs {len(partial_specs)=}. '
        f'Full content: {full_specs=} vs {partial_specs=}'
    )
    result = [
        filter_nestedmap(lhs, rhs)
        for lhs, rhs in zip(full_specs, partial_specs)
    ]
    return result
  else:
    return full_specs


def compile_for_auto_sharding(step_fn: Any,
                              train_state: TrainState,
                              step_key: PRNGKey,
                              inputs_shape_dtype: NestedShapeDtypeLike):
  """Compiles step_fn ahead of time to extract the shardings.

  The sharding is returned by the auto spmd partitioner and is attached on the
  compiled object.

  Args:
    step_fn: The step_fn function which will be compiled ahead of time.
    train_state: Train state which contains abstract values for ahead of time
      compilation.
    step_key: Prng key.
    inputs_shape_dtype: Inputs with shape/dtype attributes to be used for shape
      inference.

  Returns:
    * A compiled step_fn function
    * The input shardings returned by the auto spmd partitioner.
  """

  def _create_aval(x):
    # canonicalize_dtype is necessary to avoid errors like
    # data types are different when compiling and when being called.
    dtype = jax.dtypes.canonicalize_dtype(x.dtype)
    return core.ShapedArray(x.shape, dtype)

  inputs_shape_dtype = jax.tree_map(_create_aval, inputs_shape_dtype)
  compiled = step_fn.lower(train_state, step_key, inputs_shape_dtype).compile()
  return compiled, compiled.input_shardings[0]


def _remove_input_padding(
    inputs: NestedJTensor,
    unpadded_global_batch_size: int,
    input_partition_spec: NestedPartitionSpec,
    mesh_names: Optional[Sequence[str]] = None,
):
  """Removes input padding on the batch dimension."""
  padded_global_batch_size = jax.tree_util.tree_leaves(inputs)[0].shape[0]
  if padded_global_batch_size == unpadded_global_batch_size:
    return inputs

  def _remove_padding(x, pspec):
    x = x[:unpadded_global_batch_size]
    return base_layer.maybe_shard(x, pspec, mesh_names)

  return jax.tree_map(_remove_padding, inputs, input_partition_spec)


def _write_input_specs(
    input_specs: NestedShapeDtypeLike, job_log_dir: Optional[epath.Path]
) -> None:
  """Writes input specs as JSON to a file."""
  if job_log_dir is None or jax.process_index() != 0:
    return

  def _to_dict(array_like: Any) -> Dict[str, Any]:
    return {
        '_array': {
            'shape': list(array_like.shape),
            'dtype': str(array_like.dtype),
        }
    }

  input_specs_dict = frozen_dict.unfreeze(
      jax.tree_util.tree_map(_to_dict, input_specs)
  )
  fpath = job_log_dir / 'input_specs.json'
  with fpath.open('w') as f:
    json.dump(input_specs_dict, f, indent=2, sort_keys=True)

  work_unit = platform.work_unit()
  work_unit.create_artifact(
      platform.ArtifactType.FILE, str(fpath), 'Input specs'
  )


class Partitioner(metaclass=abc.ABCMeta):
  """Interface for partitioning computations.

  Example usage:

  ```
  # Create the partitioner.
  partitioner = create_partitioner()

  # Sets up the partitioner.
  train_input_pipeline = None

  # [Optional] Use the train input pipeline to get the shape/dtype information
  # for model.init. Needed only if train_inputs_shape_dtype is not available
  # (==None) when we call setup below.
  train_input_p = ...  # The config for training input pipeline.
  train_input_p = partitioner.preprocess_input_params(train_input_p)
  train_input_pipeline = instantiate(train_input_p)

  partitioner.setup(
      jax_task, init_key, train_inputs_shape_dtype, train_input_pipeline,
      job_log_dir)

  # Restore the train state.
  metadata = partitioner.get_train_state_metadata()
  train_state = restore(metadata, ...)

  # Create the PRNG key.
  root_prng_key = ...

  # Initialize the root prng key and train state. It'll initialize the train
  # state from scratch if no checkpoint is found (i.e. when train_state==None).
  root_prng_key, train_state = partitioner.initialize_prng_key_and_train_state(
      root_prng_key, train_state, checkpoint_type)

  # Partition the step function.
  partitioned_step_fn, input_pspec = partitioner.partition(
      step_fn, inputs_shape_dtype, is_eval)

  # Split and preprocess the prng key.
  prng_key, train_key = jax.random.split(root_prng_key)
  train_key = partitioner.preprocess_prng_key(train_key)

  # Get the inputs, preprocess and use it to run the partitioned function.
  inputs = train_input_pipeline.get_next_padded()
  inputs = partitioner.preprocess_inputs(
      train_input_pipeline, inputs, input_pspec)
  partitioned_step_fn(
      train_state, train_key, inputs, unpadded_global_batch_size)
  ```
  """

  def __init__(self, init_is_eval: bool):
    """Constructor.

    Args:
      init_is_eval: Whether it should set is_eval=True when running
        abstract_init_with_metadata.
    """
    # TODO(laigd): remove this option (it should always be False) once
    # always_use_train_for_model_init is enabled by default.
    self._init_is_eval = init_is_eval

    # States to set in .setup().
    self._jax_task = None
    self._job_log_dir = None
    self._init_key = None
    self._train_inputs_shape_dtype = None

    # The train state metadata, set in .get_train_state_metadata().
    self._train_state_metadata = None

  def setup(
      self,
      jax_task: tasks_lib.SingleTask,
      init_key: PRNGKey,
      train_inputs_shape_dtype: Optional[NestedShapeDtypeLike],
      # TODO(pax-dev): remove this arg and always use train_inputs_shape_dtype
      # once all experiments provide input specs.
      train_input_pipeline: Optional[base_input.BaseInput] = None,
      job_log_dir: Optional[epath.Path] = None,
  ) -> None:
    """Sets training shape/dtype using sample inputs from the input pipeline.

    Args:
      jax_task: The task which is an instance of tasks.SingleTask.
      init_key: PRNGKey for initializing the model variables.
      train_inputs_shape_dtype: Shape/dtype information of the training inputs
        to model.init, for use in getting params of model variables. If None,
        train_input_pipeline must be set.
      train_input_pipeline: The training input pipeline, used to get the
        shape/dtype information for model.init. If None,
        train_inputs_shape_dtype must be set.
      job_log_dir: Directory for the job logs.
    """
    self._jax_task = jax_task
    self._init_key = init_key
    self._job_log_dir = job_log_dir
    if train_inputs_shape_dtype is not None:
      assert train_input_pipeline is None
      self._train_inputs_shape_dtype = train_inputs_shape_dtype
    else:
      assert train_input_pipeline
      self._train_inputs_shape_dtype = self._get_train_inputs_shape_dtype(
          train_input_pipeline
      )

  @abc.abstractmethod
  def _get_train_inputs_shape_dtype(
      self, train_input_pipeline: base_input.BaseInput
  ) -> NestedShapeDtypeLike:
    """Get the shape/dtype information for model.init."""

  @property
  def train_inputs_shape_dtype(self) -> Optional[NestedShapeDtypeLike]:
    """Shape/dtype attributes of the training inputs to model.init."""
    assert self._train_inputs_shape_dtype
    return self._train_inputs_shape_dtype

  @property
  def global_mesh(self) -> Optional[jax.sharding.Mesh]:
    """The global mesh."""
    return None

  @abc.abstractmethod
  def preprocess_input_params(
      self, input_ps: base_input.BaseInput.HParams
  ) -> base_input.BaseInput.HParams:
    """Preprocess input hparam if necessary.

    This is necessary before using the hparam to create the input pipeline, so
    that the generated input batch can be used to run the partitioned function.

    Args:
      input_ps: The input hparam(s) to adjust.

    Returns:
      The adjusted input hparam(s), with the same nested structure.
    """

  @abc.abstractmethod
  def initialize_prng_key_and_train_state(
      self,
      root_prng_key: PRNGKey,
      train_state: Optional[TrainState],
      checkpoint_type: Optional[CheckpointType],
      discard_opt_states: Optional[bool] = False,
  ) -> Tuple[PRNGKey, TrainState]:
    """Initialize the root prng key and train state.

    Depending on the partitioner, this may involve actions like splitting the
    root_prng_key, replicating the train_state, etc.

    Args:
      proot_rng_key: The root prng key.
      train_state: The train state restored from checkpoint. If None, will
        initialize it from scratch.
      checkpoint_type: The checkpoint type.
      discard_opt_states: Whether to discard the part corresponding to the
        optimizer states or not, from train_state.

    Returns:
      The properly initialized root prng key and train state.
    """

  @abc.abstractmethod
  def preprocess_prng_key(self, prng_key: PRNGKey) -> PRNGKey:
    """Preprocess the key before using it to run the partitioned function.

    Args:
      prng_key: The prng key to preprocess.

    Returns:
      The preprocessed prng key that's ready to be used to run the partitioned
      function.
    """

  @abc.abstractmethod
  def preprocess_inputs(
      self,
      input_pipeline: base_input.BaseInput,
      padded_inputs: NestedJTensor,
      partition_specs: Optional[NestedPartitionSpec],
  ) -> NestedJTensor:
    """Preprocess the input batch before using it in the partitioned function.

    Args:
      input_pipeline: The input pipeline that generates `padded_inputs`.
      padded_inputs: The padded input batch used to run the partitioned
        function. Generated by `input_pipeline.get_next_padded()`.
      partition_specs: The partition spec of padded_inputs, returned by
        `self.partition()`.

    Returns:
      The preprocessed input batch that's ready to be used to run the
      partitioned function.
    """

  @abc.abstractmethod
  def get_train_state_metadata(
      self,
      discard_opt_states: bool = False,
  ) -> TrainStateMetadata:
    """Gets the TrainStateMetadata used for partitioning.

    Args:
      discard_opt_states: Whether to discard the part corresponding to the
        optimizer states or not.

    Returns:
      The TrainStateMetadata.
    """

  @property
  def _init_do_eval(self):
    """Whether to set do_eval=True when running abstract_init_with_metadata."""
    if self._jax_task.hparams.train.always_use_train_for_model_init:
      return False
    return self._init_is_eval

  def _get_train_state_metadata_default(self) -> TrainStateMetadata:
    """Helper method to get the TrainStateMetadata."""
    if not self._train_inputs_shape_dtype:
      raise ValueError('Train input spec is not set. It can be set in setup().')
    return trainer_lib.create_train_state_metadata(
        self._jax_task,
        self._train_inputs_shape_dtype,
        discard_opt_states=False,
        do_eval=self._init_do_eval,
    )

  def _maybe_discard_opt_states(
      self, metadata: TrainStateMetadata, discard_opt_states: bool
  ) -> TrainStateMetadata:
    if not discard_opt_states:
      # Make sure all the metadata has opt_states.
      for state in [
          metadata.padded_global_shapes,
          metadata.unpadded_global_shapes,
          metadata.partition_specs,
      ]:
        if state:
          assert state.opt_states
      return metadata

    # Discard the opt_states.
    to_eval_state = lambda state: state.to_eval_state() if state else None
    return TrainStateMetadata(
        input_shape_dtype=metadata.input_shape_dtype,
        var_weight_hparams=metadata.var_weight_hparams,
        padded_global_shapes=to_eval_state(metadata.padded_global_shapes),
        unpadded_global_shapes=to_eval_state(metadata.unpadded_global_shapes),
        partition_specs=to_eval_state(metadata.partition_specs),
    )

  # Signature of the function passed to `self.partition()`.
  #
  # Args:
  #   jax_task: A SingleTask instance.
  #   train_state: The current TrainState.
  #   prng_key: The PRNGKey.
  #   inputs: Inputs drawn from the input pipeline.
  #   fprop_dtype: Fprop datatype, can be either jnp.float32 or jnp.bfloat16.
  #   var_weight_hparams: A pytree of WeightHParams for the model variables.
  #
  # Returns:
  #   (new_train_state, ...) if is_eval==False, or just (...) otherwise, where
  #   new_train_state is the updated TrainState, and ... represents arbitrary
  #   number and structure of jax array.
  StepFn = Callable[
      [
          tasks_lib.SingleTask,
          TrainState,
          PRNGKey,
          NestedJTensor,
          jnp.dtype,
          NestedWeightHParams,
      ],
      Any,
  ]

  # Signature of the partitioned function returned by `self.partition()`.
  # typing.Protocol is used instead of Callable to better handle Optional args.
  #
  # Args:
  #   train_state: The current TrainState.
  #   prng_key: The PRNGKey.
  #   inputs: Inputs drawn from the input pipeline.
  #   unpadded_global_batch_size: (Optional) The unpadded size of global batch,
  #     and the padding is on the right side of each input.
  #
  # Returns:
  #   The same as StepFn.
  class PartitionedStepFn(Protocol):

    def __call__(
        self,
        train_state: TrainState,
        prng_key: PRNGKey,
        inputs: NestedJTensor,
        unpadded_global_batch_size: Optional[int] = None,
    ) -> Any:
      ...

  @abc.abstractmethod
  def partition(
      self,
      step_fn: StepFn,
      inputs_shape_dtype: NestedShapeDtypeLike,
      is_eval: bool,
  ) -> Tuple[PartitionedStepFn, Optional[NestedPartitionSpec]]:
    """Partitions the step function.

    Args:
      step_fn: Model step function to partition, whose signature is specified by
        StepFn.
      inputs_shape_dtype: Shape/dtype attributes of the inputs of step_fn.
      is_eval: A boolean indicating if it's a eval/decode task or not.

    Returns:
      (partitioned_step_fn, input_partition_spec):

      - partitioned_step_fn: The partitioned step function.
      - input_partition_spec: The partition spec for the inputs of the step
        function.
    """


class PmapPartitioner(Partitioner):

  def __init__(self, init_is_eval: bool):
    super().__init__(init_is_eval)
    logging.info('Using pmap for data parallelism.')

  def _get_train_inputs_shape_dtype(
      self, train_input_pipeline: base_input.BaseInput
  ) -> NestedShapeDtypeLike:
    sample_inputs = train_input_pipeline.peek_padded()
    # Reshard inputs and only keep the inputs corresponding to a given device.
    sample_inputs = self.preprocess_inputs(
        train_input_pipeline, sample_inputs, partition_specs=None
    )
    per_device_shape_dtype = jax.tree_map(
        lambda x: jax.ShapeDtypeStruct(shape=x.shape[1:], dtype=x.dtype),
        sample_inputs,
    )
    _write_input_specs(per_device_shape_dtype, self._job_log_dir)
    return per_device_shape_dtype

  def preprocess_input_params(
      self, input_ps: base_input.BaseInput.HParams
  ) -> base_input.BaseInput.HParams:
    """Preprocess input hparam if necessary."""
    return input_ps

  def initialize_prng_key_and_train_state(
      self,
      root_prng_key: PRNGKey,
      train_state: Optional[TrainState],
      checkpoint_type: Optional[CheckpointType],
      discard_opt_states: Optional[bool] = False,
  ) -> Tuple[PRNGKey, TrainState]:
    """Initialize the root prng key and train state."""
    root_prng_key, init_key = jax.random.split(root_prng_key)
    if train_state is None:
      # If no checkpoint was restored, initialize with random weights.
      metadata = self.get_train_state_metadata(discard_opt_states)
      train_state = trainer_lib.initialize_model_state(
          self._jax_task,
          init_key,
          metadata.input_shape_dtype,
          discard_opt_states=discard_opt_states,
          is_eval=self._init_do_eval,
          checkpoint_type=checkpoint_type,
      )

    logging.info(
        'train state shapes: %s', jax.tree_map(lambda x: x.shape, train_state)
    )
    replicated_train_state = trainer_lib.replicate_model_state(train_state)
    # Unreplicated model states are not needed anymore at that point.
    del train_state
    logging.info(
        'replicated train state shapes: %s',
        jax.tree_map(lambda x: x.shape, replicated_train_state),
    )

    # From now on, different replicas should use different random seeds.
    # Here, each process will have its unique prng key.
    # root_prng_key will be further split so that each core on a host will get
    # different key.
    root_prng_key = jax.random.fold_in(root_prng_key, jax.process_index())
    logging.info('root prng key: %s', root_prng_key)
    return root_prng_key, replicated_train_state

  def preprocess_prng_key(self, prng_key: PRNGKey) -> PRNGKey:
    """Preprocess the key before using it to run the partitioned function."""
    return jax.random.split(prng_key, num=jax.local_device_count())

  def preprocess_inputs(
      self,
      input_pipeline: base_input.BaseInput,
      padded_inputs: NestedJTensor,
      partition_specs: Optional[NestedPartitionSpec],
  ) -> NestedJTensor:
    """Preprocess the input batch before using it."""
    assert partition_specs is None
    return input_pipeline.reshard_for_pmap(padded_inputs)

  def get_train_state_metadata(
      self,
      discard_opt_states: bool = False,
  ) -> TrainStateMetadata:
    """Gets the TrainStateMetadata used for partitioning."""

    if not self._train_state_metadata:
      self._train_state_metadata = self._get_train_state_metadata_default()
    return self._maybe_discard_opt_states(
        self._train_state_metadata, discard_opt_states
    )

  def partition(
      self,
      step_fn: Partitioner.StepFn,
      inputs_shape_dtype: NestedShapeDtypeLike,
      is_eval: bool,
  ) -> Tuple[Partitioner.PartitionedStepFn, Optional[NestedPartitionSpec]]:
    """Partitions the step function."""
    del inputs_shape_dtype

    # Guard the case where get_train_state_metadata isn't called.
    train_state_metadata = self.get_train_state_metadata(is_eval)

    def _wrapped_step_fn(state, prng_key, inputs):
      return step_fn(
          self._jax_task,
          state,
          prng_key,
          inputs,
          self._jax_task.hparams.model.fprop_dtype,
          train_state_metadata.var_weight_hparams,
      )

    partitioned_step_fn = jax.pmap(
        _wrapped_step_fn,
        # For training, TrainState is the first argument and return value.
        # We setup donation/alias to minimize device memory usage.
        donate_argnums=() if is_eval else (0,),
        axis_name=base_layer.PMAP_PARALLEL_AXIS_NAME,
    )

    # unpadded_global_batch_size is not used for pmap'ed functions, so we
    # explicitly ignore it with a wrapper.
    def _wrapped_partitioned_step(
        state, prng_key, inputs, unpadded_global_batch_size=None
    ):
      del unpadded_global_batch_size
      return partitioned_step_fn(state, prng_key, inputs)

    return _wrapped_partitioned_step, None  # Input partition spec.


class PjitPartitioner(Partitioner):
  """Used for partitioning a step function of a SPMD model."""

  def __init__(
      self,
      init_is_eval: bool,
      reshard_inputs: bool,
      task_p: tasks_lib.SingleTask.HParams,
  ):
    """Constructor.

    Args:
      reshard_inputs: Whether to reshard model inputs before running the
        partitioned function. Only applicable for pjit.
      init_is_eval: Whether it should set is_eval=True when running
        abstract_init_with_metadata.
      task_p: The params for the task, needed to create global mesh.
    """
    super().__init__(init_is_eval)
    self._reshard_inputs = reshard_inputs
    logging.info('Using SPMD sharding for model parallelism.')

    # Creates global mesh.
    model_p = task_p.model
    self._mesh_names = model_p.mesh_axis_names
    device_mesh = py_utils.create_device_mesh(
        model_p.ici_mesh_shape,
        model_p.dcn_mesh_shape,
        contiguous_submeshes=model_p.contiguous_submeshes,
    )
    logging.info('device_mesh: %s', device_mesh)
    self._global_mesh = jax.sharding.Mesh(device_mesh, model_p.mesh_axis_names)

    # Pjit'ed function to preprocess the prng key.
    self._broadcast_key_fn = None

  def _get_train_inputs_shape_dtype(
      self, train_input_pipeline: base_input.BaseInput
  ) -> NestedShapeDtypeLike:
    sample_inputs = train_input_pipeline.peek_padded()
    global_shape_dtype = jax.tree_map(
        py_utils.get_global_input_shape_dtype, sample_inputs
    )
    perhost_inputs_shape_dtype = jax.tree_map(
        lambda x: jax.ShapeDtypeStruct(shape=x.shape, dtype=x.dtype),
        sample_inputs,
    )
    _write_input_specs(perhost_inputs_shape_dtype, self._job_log_dir)
    return global_shape_dtype

  @property
  def global_mesh(self) -> jax.sharding.Mesh:
    return self._global_mesh

  def preprocess_input_params(
      self, input_ps: base_input.BaseInput.HParams
  ) -> base_input.BaseInput.HParams:
    """Preprocess input hparam if necessary."""
    assert self.global_mesh
    return trainer_lib.adjust_input_params_for_small_batch(
        input_ps, self.global_mesh
    )

  def initialize_prng_key_and_train_state(
      self,
      root_prng_key: PRNGKey,
      train_state: Optional[TrainState],
      checkpoint_type: Optional[CheckpointType],
      discard_opt_states: Optional[bool] = False,
  ) -> Tuple[PRNGKey, TrainState]:
    """Initialize the root prng key and train state."""
    root_prng_key, init_key = jax.random.split(root_prng_key)
    # train_state should already be partitioned.
    partitioned_train_state = train_state
    if partitioned_train_state is None:
      # If no checkpoint was restored, initialize with random weights.
      metadata = self.get_train_state_metadata(discard_opt_states)
      # TODO(laigd): there is a potential bug here: when this is called in the
      # eval/decode pipeline, do_eval is not properly set (see the pmap
      # version). But since we're enabling always_use_train_for_model_init this
      # is probably fine.
      partitioned_train_state = trainer_lib.initialize_partitioned_model_states(
          self._jax_task,
          init_key,
          metadata.input_shape_dtype,
          metadata.partition_specs,
          global_mesh=self.global_mesh,
          # Note: We currently enforce that the checkpoint to reload via
          # init_checkpoint_rules are in the same format as the checkpoint
          # solution used by the experiment.
          checkpoint_type=checkpoint_type,
          discard_opt_states=discard_opt_states,
      )

    logging.info(
        'partitioned train state shapes (global shape): %s',
        jax.tree_map(lambda x: x.shape, partitioned_train_state),
    )

    # We do not fold in jax.process_index in contrast to the pmap version and
    # use a single global key instead to rely on pjit to split for different
    # replicas.
    logging.info('root prng key: %s', root_prng_key)
    return root_prng_key, partitioned_train_state

  def preprocess_prng_key(self, prng_key: PRNGKey) -> PRNGKey:
    """Preprocess the key before using it to run the partitioned function."""
    if not self._broadcast_key_fn:
      # The prng keys are already created on device with jax.random.split. We
      # broadcast it with an identity pjit function to avoid doing it in the
      # loop where a multi-slice program could be generated.
      def _broadcast_key(k):
        def _identity(x):
          return x

        with self._global_mesh:
          return pjit.pjit(_identity, in_shardings=None, out_shardings=None)(k)

      self._broadcast_key_fn = _broadcast_key

    return self._broadcast_key_fn(prng_key)

  def preprocess_inputs(
      self,
      input_pipeline: base_input.BaseInput,
      padded_inputs: NestedJTensor,
      partition_specs: Optional[NestedPartitionSpec],
  ) -> NestedJTensor:
    """Preprocess the input batch before using it."""
    if self._reshard_inputs or input_pipeline.hparams.experimental_remote_input:
      padded_inputs = input_pipeline.reshard_for_spmd(
          padded_inputs, self.global_mesh, partition_specs
      )
    return padded_inputs

  def get_train_state_metadata(
      self,
      discard_opt_states: bool = False,
  ) -> TrainStateMetadata:
    """Gets the TrainStateMetadata used for partitioning.

    Args:
      discard_opt_states: Whether to discard the part corresponding to the
        optimizer states or not.
    """
    if not self._train_state_metadata:
      self._train_state_metadata = self._get_train_state_metadata_default()
    return self._maybe_discard_opt_states(
        self._train_state_metadata, discard_opt_states
    )

  def partition(
      self,
      step_fn: Partitioner.StepFn,
      inputs_shape_dtype: NestedShapeDtypeLike,
      is_eval: bool,
  ) -> Tuple[Partitioner.PartitionedStepFn, Optional[NestedPartitionSpec]]:
    """Gets a sharded (pjit-ed) step function of the SPMD Model.

    Args:
      step_fn: Model step function to partition.
      inputs_shape_dtype: Shape/dtype attributes of the inputs of step_fn.
      is_eval: A boolean indicating if it's a eval/decode task or not.

    Returns:
      (partitioned_step_fn, input_partition_spec):

      - partitioned_step_fn: The partitioned step function.
      - input_partition_spec: The partition spec for the inputs of the step
        function.
    """
    metadata = self.get_train_state_metadata(discard_opt_states=is_eval)
    input_partition_spec = trainer_lib.get_input_partition_specs(
        self._mesh_names, inputs_shape_dtype
    )
    logging.info('step_fn inputs_partition_spec=%s', input_partition_spec)
    # Step function to be pjit-ed.
    wrapped_step_fn = self._get_step_fn(
        step_fn, is_eval, metadata, input_partition_spec
    )
    with self.global_mesh:
      return (
          self._partition_manual_shard(
              wrapped_step_fn,
              is_eval,
              inputs_shape_dtype,
              input_partition_spec,
              metadata,
          ),
          input_partition_spec,
      )

  def _get_step_fn(
      self,
      step_fn: Partitioner.StepFn,
      is_eval: bool,
      metadata: TrainStateMetadata,
      input_partition_spec: NestedPartitionSpec,
      is_auto_sharding: bool = False,
  ) -> Partitioner.PartitionedStepFn:
    """Returns a step function to apply the SPMD partition (pjit)."""
    task_p = self._jax_task.hparams
    model_p = task_p.model
    reshard_inputs_fn = functools.partial(
        trainer_lib.reshard_input_based_on_rank_fn,
        task_p.train.inputs_split_mapping,
        self._mesh_names,
    )

    def _wrapped_step_fn(
        state, prng_key, inputs, unpadded_global_batch_size=None
    ):
      # When auto-sharding is enabled, we can't pad the variables whose input
      # sharding may get changed by auto-sharding.
      # TODO(pax-dev): Add support for padding and unpadding inputs when auto
      # sharding is enabled.
      if not is_auto_sharding:
        # When there are input padding on multi-host, we use a different device
        # order in the program's input sharding. We now make sure they are
        # resharded back to the device order consistent with the global mesh.
        inputs = jax.tree_util.tree_map(
            lambda x, s: base_layer.maybe_shard(x, s, self._mesh_names), inputs,
            input_partition_spec)
        # Vars/inputs are padded at program entry/exit to avoid uneven sharding.
        # We slice the vars to remove padding before the step computation, and
        # pad them after the step computation to make user code independent of
        # paddings.
        # Internal uneven sharding in the step computation is supported by XLA.
        state = self._unpad_states(metadata, state)
        inputs = self._unpad_inputs(
            inputs, unpadded_global_batch_size, input_partition_spec
        )

      # Reshard inputs.
      # TODO(pax): when xla auto-sharding is enabled, it'll automatically figure
      # out the proper sharding of intermediate nodes, we can get rid of this
      # manual sharding now?
      inputs = jax.tree_map(reshard_inputs_fn, inputs)

      # When auto sharding is enabled, uneven sharding is not supported. This is
      # because the way padding is added is dependent on the provided train
      # state partition specs. This is not available during auto-sharding until
      # after the compilation is done.
      fn_out = step_fn(
          self._jax_task,
          state,
          prng_key,
          inputs,
          model_p.fprop_dtype,
          metadata.var_weight_hparams,
      )

      if is_eval:
        return fn_out
      assert len(fn_out) > 1

      # Pad the model states again for training step functions.
      if not is_auto_sharding:
        padded_states = self._pad_states(metadata, fn_out[0])
        fn_out = (padded_states,) + fn_out[1:]
      return fn_out

    return _wrapped_step_fn

  def _pjit(
      self,
      step_fn: Partitioner.PartitionedStepFn,
      is_eval: bool,
      fn_in_partition_specs: NestedPartitionSpec,
      fn_out_partition_specs: NestedPartitionSpec,
      use_pspec_on_array_inputs: bool = False,
  ):
    logging.info('step_fn fn_in_partition_specs=%s', fn_in_partition_specs)
    logging.info('step_fn fn_out_partition_specs=%s', fn_out_partition_specs)

    extra_kwargs = dict(in_axis_resources=fn_in_partition_specs)
    if not use_pspec_on_array_inputs:
      extra_kwargs = {}
    pjitted_fn = pjit.pjit(
        step_fn,
        out_axis_resources=fn_out_partition_specs,
        # For training, TrainState is the first argument and return value. We
        # setup donation/alias to minimize device memory usage.
        donate_argnums=() if is_eval else (0,),
        static_argnums=(3,),  # unpadded_global_batch_size is static.
        **extra_kwargs,
    )
    return trainer_lib.bind_mesh(pjitted_fn, self.global_mesh)

  def _get_state_unpadded_shapes(self, metadata: TrainStateMetadata):
    return jax.tree_map(lambda x: x.shape, metadata.unpadded_global_shapes)

  def _pad_states(
      self, metadata: TrainStateMetadata, unpadded_state: TrainState
  ):
    """Pad variables to avoid uneven sharding."""
    model_p = self._jax_task.hparams.model

    # Here the metadata is derived from input_spec which includes all possible
    # inputs. Thus metadata includes the full TrainState. The unpadded_state
    # here could be derived from a eval/decode dataset so it only includes a
    # subset of TrainState. Here we project the metadata according to the
    # actual state.
    partition_specs = metadata.partition_specs.replace(
        mdl_vars=filter_nestedmap(
            metadata.partition_specs.mdl_vars, unpadded_state.mdl_vars
        )
    )
    state_unpadded_shapes = self._get_state_unpadded_shapes(metadata)
    state_unpadded_shapes = state_unpadded_shapes.replace(
        mdl_vars=filter_nestedmap(
            state_unpadded_shapes.mdl_vars, unpadded_state.mdl_vars
        )
    )

    return py_utils.maybe_pad_uneven_sharding(  # pytype: disable=wrong-arg-types  # jax-ndarray
        unpadded_state,
        partition_specs,
        state_unpadded_shapes,
        model_p.mesh_shape,
        model_p.mesh_axis_names,
    )

  def _unpad_states(
      self, metadata: TrainStateMetadata, padded_state: TrainState
  ):
    """Remove paddings from variables."""
    # Similar to _pad_states above we need to project the metadata to match the
    # actual padded_state.
    partition_specs = metadata.partition_specs.replace(
        mdl_vars=filter_nestedmap(
            metadata.partition_specs.mdl_vars, padded_state.mdl_vars
        )
    )
    state_unpadded_shapes = self._get_state_unpadded_shapes(metadata)
    state_unpadded_shapes = state_unpadded_shapes.replace(
        mdl_vars=filter_nestedmap(
            state_unpadded_shapes.mdl_vars, padded_state.mdl_vars
        )
    )
    return py_utils.maybe_slice_uneven_sharding(  # pytype: disable=wrong-arg-types  # jax-ndarray
        padded_state,
        partition_specs,
        state_unpadded_shapes,
        is_leaf=py_utils.is_optax_masked_node,
    )

  def _unpad_inputs(
      self,
      padded_inputs: NestedJTensor,
      unpadded_global_batch_size: int,
      input_partition_spec: NestedPartitionSpec,
  ):
    """Remove paddings from inputs."""
    return _remove_input_padding(
        padded_inputs,
        unpadded_global_batch_size,
        input_partition_spec,
        self._mesh_names,
    )

  def _partition_manual_shard(
      self,
      step_fn: Partitioner.PartitionedStepFn,
      is_eval: bool,
      inputs_shape_dtype: NestedShapeDtypeLike,
      input_partition_spec: NestedPartitionSpec,
      metadata: TrainStateMetadata,
  ) -> Partitioner.PartitionedStepFn:
    prng_key_partition_spec = PartitionSpec(None)
    fn_in_partition_specs = (
        metadata.partition_specs,
        prng_key_partition_spec,
        input_partition_spec,
    )

    var_shapes = metadata.padded_global_shapes
    out_shapes = jax.eval_shape(
        step_fn, var_shapes, self._init_key, inputs_shape_dtype, None
    )
    # Currently, all the outputs are fully replicated.
    # TODO(yonghui): Somehow fetch the output sharding spec from _eval_step fn.
    fn_out_partition_specs = jax.tree_util.tree_map(
        lambda _: PartitionSpec(), out_shapes
    )
    if not is_eval:
      fn_out_partition_specs = tuple(
          [metadata.partition_specs] + list(fn_out_partition_specs[1:])
      )

    asserts.assert_same_structure(fn_out_partition_specs, out_shapes)

    # LINT.IfChange(UsePspecOnArrayInputs)
    use_pspec_on_array_inputs = (jax.process_count() == 1)
    # LINT.ThenChange(trainer_lib.py:PspecSharding)
    partitioned_step_fn = self._pjit(
        step_fn,
        is_eval,
        fn_in_partition_specs,
        fn_out_partition_specs,
        use_pspec_on_array_inputs=use_pspec_on_array_inputs)
    return partitioned_step_fn


class AutoShardingPjitPartitioner(PjitPartitioner):
  """PJIT partitioner that automatically select partition strategies."""

  @dataclasses.dataclass
  class AutoShardingInfo:
    """Info needed by auto-sharding to get train state partition specs."""

    # The step function to run auto-sharding on. This will be used to compute
    # the train state partition spec.
    step_fn: Partitioner.StepFn

    # Whether step_fn is used for evaluation.
    is_eval: bool

    # Whether to replicate the output when auto sharding is enabled.
    # TODO(pax-dev): support custom output partition spec.
    replicate_output: bool

  @dataclasses.dataclass
  class _AutoShardingResult:
    """Output of auto-sharding and related information."""

    # The partitioned step_fn generated by auto-sharding.
    partitioned_step_fn: Partitioner.PartitionedStepFn

    # Generated partition spec for the TrainState.
    train_state_partition_spec: TrainState

    # Generated partition spec for the data inputs of the step function.
    input_partition_spec: NestedPartitionSpec

    # Shape/dtype information for the inputs to partitioned_step_fn.
    inputs_shape_dtype: NestedShapeDtypeLike

  def __init__(
      self,
      init_is_eval: bool,
      reshard_inputs: bool,
      task_p: tasks_lib.SingleTask.HParams,
      auto_sharding_info: AutoShardingInfo,
  ):
    """Constructor.

    Args:
      init_is_eval: Whether it should set is_eval=True when running
        abstract_init_with_metadata.
      reshard_inputs: Whether to reshard model inputs before running the
        partitioned function. Only applicable for pjit.
      task_p: The params for the task, needed to create global mesh.
      auto_sharding_info: Information used for XLA auto-sharding. If None, it'll
        use the sharding information provided by the model config instead.
    """
    super().__init__(init_is_eval, reshard_inputs, task_p)
    self._auto_sharding_info = auto_sharding_info
    self._auto_sharding_result = None  # Used to cache auto-sharding results.

  def _get_train_inputs_shape_dtype(
      self, train_input_pipeline: base_input.BaseInput
  ) -> NestedShapeDtypeLike:
    global_shape_dtype = super()._get_train_inputs_shape_dtype(
        train_input_pipeline
    )
    # Extra checking in auto sharding case.
    train_input_p = train_input_pipeline.hparams
    if train_input_p.num_infeed_hosts < jax.process_count() or (
        train_input_p.cls.get_batch_size(train_input_p)
        < jax.local_device_count()
    ):
      raise NotImplementedError(
          'Per-device batch size < 1 not supported for auto sharding.'
      )
    logging.info('Auto sharding is enabled in PAX.')
    return global_shape_dtype

  def _partition_auto_shard(
      self,
      step_fn: Partitioner.PartitionedStepFn,
      is_eval: bool,
      inputs_shape_dtype: NestedShapeDtypeLike,
      input_partition_spec: NestedPartitionSpec,
      metadata: TrainStateMetadata,
  ) -> Tuple[Partitioner.PartitionedStepFn, NestedPartitionSpec, TrainState]:
    """Generates and returns the train state partition spec automatically."""
    # Workflow: create abstract train state and ahead of time compile the
    # `step_fn`. Then we can extract the input shardings returned by XLA's
    # auto spmd partitioner from the compiled object.

    # We provide input_partition_spec because Jax Array creation is specialized
    # to the input partition specs created here. If we use partition specs
    # returned by XLA, it errors out.
    prng_key_partition_spec = PartitionSpec(None)
    fn_in_partition_specs = (pjit.AUTO, prng_key_partition_spec,
                             input_partition_spec)
    fn_out_partition_specs = (
        PartitionSpec()
        if self._auto_sharding_info.replicate_output
        else pjit.AUTO
    )

    partitioned_step_fn = self._pjit(
        step_fn,
        is_eval,
        fn_in_partition_specs,
        fn_out_partition_specs,
        use_pspec_on_array_inputs=True)

    # NOTE(pax-dev): The following is currently incompatible with variable
    # uneven-sharding padding.
    (
        auto_sharded_step_fn,
        input_shardings,
    ) = compile_for_auto_sharding(
        partitioned_step_fn,
        metadata.unpadded_global_shapes,
        self._init_key,
        inputs_shape_dtype,
    )
    new_train_state_pspec = jax.tree_map(lambda x: x.spec, input_shardings[0])
    new_input_pspec = jax.tree_map(lambda x: x.spec, input_shardings[2])
    return auto_sharded_step_fn, new_input_pspec, new_train_state_pspec

  def get_train_state_metadata(
      self, discard_opt_states: bool = False
  ) -> TrainStateMetadata:
    if self._train_state_metadata:
      return self._maybe_discard_opt_states(
          self._train_state_metadata, discard_opt_states
      )
    train_state_metadata = self._get_train_state_metadata_default()
    assert self._train_inputs_shape_dtype
    # Currently we run auto-sharding only once to get the train state
    # partition spec, and reuse it for all subsequent get_train_state_metadata()
    # and partition() calls.
    # Since the structure of the partition spec needs to match the actual train
    # state passed to the auto-sharded step function, we need to discard the
    # opt_states if AutoShardingInfo.is_eval==True.
    train_state_metadata = self._maybe_discard_opt_states(
        train_state_metadata, self._auto_sharding_info.is_eval
    )
    if not self._auto_sharding_result:
      input_partition_spec = trainer_lib.get_input_partition_specs(
          self._mesh_names, self._train_inputs_shape_dtype
      )
      wrapped_step_fn = self._get_step_fn(
          self._auto_sharding_info.step_fn,
          self._auto_sharding_info.is_eval,
          train_state_metadata,
          input_partition_spec=input_partition_spec,
          is_auto_sharding=True,
      )

      with self.global_mesh:
        partitioned_step_fn, input_pspec, train_state_pspec = (
            self._partition_auto_shard(
                wrapped_step_fn,
                self._auto_sharding_info.is_eval,
                self._train_inputs_shape_dtype,
                input_partition_spec,
                train_state_metadata,
            )
        )
      # unpadded_global_batch_size is not used for AOT-compiled functions, so we
      # always pass None to avoid recompilation.
      def _wrapped_partitioned_step(
          state, prng_key, inputs, unpadded_global_batch_size=None
      ):
        del unpadded_global_batch_size
        # The unpadded_global_batch_size passed to partitioned_step_fn need to
        # be the same as the one used to run compile_for_auto_sharding.
        return partitioned_step_fn(state, prng_key, inputs)

      self._auto_sharding_result = (
          AutoShardingPjitPartitioner._AutoShardingResult(
              _wrapped_partitioned_step,
              train_state_pspec,
              input_pspec,
              jax.tree_util.tree_map(
                  lambda x: x, self._train_inputs_shape_dtype
              ),
          )
      )

    partition_specs = self._auto_sharding_result.train_state_partition_spec
    self._train_state_metadata = dataclasses.replace(
        train_state_metadata, partition_specs=partition_specs
    )
    return self._maybe_discard_opt_states(
        self._train_state_metadata, discard_opt_states
    )

  def partition(
      self,
      step_fn: Partitioner.StepFn,
      inputs_shape_dtype: NestedShapeDtypeLike,
      is_eval: bool,
  ) -> Tuple[Partitioner.PartitionedStepFn, Optional[NestedPartitionSpec]]:
    """Returns the auto-sharding partitioned step functions and input specs."""
    # Auto-sharding result is generated by self.get_train_state_metadata, so we
    # call it first if no result is cached.
    # The step function doesn't need opt_states when is_eval=True.
    if not self._auto_sharding_result:
      self.get_train_state_metadata(discard_opt_states=is_eval)
    if (
        step_fn is self._auto_sharding_info.step_fn
        and inputs_shape_dtype == self._auto_sharding_result.inputs_shape_dtype
    ):
      return (
          self._auto_sharding_result.partitioned_step_fn,
          self._auto_sharding_result.input_partition_spec,
      )
    # Only the first call can be used with pjit.Auto.
    # For any following partition calls, we fall back to manual sharding if
    # it is not already partitioned.
    return super().partition(step_fn, inputs_shape_dtype, is_eval)


def get_step_fn(mode: RunningMode) -> Tuple[Partitioner.StepFn, bool]:
  """Returns the step function to partition.

  Args:
    mode: One of TRAIN, EVAL, and DECODE, that determines the step function to
      use.

  Returns:
    (step_fn, is_eval), where step_fn is the step function to partition, and
    is_eval indicates whether step_fn is used for evaluation.
  """
  assert int(mode.has_train) + int(mode.has_eval) + int(mode.has_decode) == 1
  if mode.has_train:
    is_eval = False
    step_fn = trainer_lib.train_step_single_learner
  elif mode.has_eval:
    is_eval = True
    step_fn = trainer_lib.eval_step_single_learner
  else:
    is_eval = True
    step_fn = trainer_lib._decode_step_for_partitioner

  return step_fn, is_eval


def create_partitioner(
    jax_task: tasks_lib.SingleTask,
    init_is_eval: bool = False,
    reshard_inputs: bool = False,
    auto_sharding_mode: Optional[RunningMode] = None,
) -> Partitioner:
  """Return sharded train/eval/decode step function of the SPMD Model.

  Args:
    jax_task: The task which is an instance of tasks.SingleTask.
    init_is_eval: Whether it should set is_eval=True when running
      abstract_init_with_metadata.
    reshard_inputs: Whether to reshard model inputs before running the
      partitioned function. Only applicable for pjit.
    auto_sharding_mode: One of TRAIN, EVAL, and DECODE, that determines the step
      function to use for auto-sharding (when pjit is used). If None, it means
      to disable auto-sharding.

  Returns:
    A Partitioner instance.
  """
  if jax_task.hparams.model.ici_mesh_shape is None:
    partitioner = PmapPartitioner(init_is_eval)
  else:
    auto_sharding_info = None
    task_p = jax_task.hparams
    if auto_sharding_mode:
      step_fn, step_fn_is_eval = get_step_fn(auto_sharding_mode)
      replicate_output = auto_sharding_mode == RunningMode.DECODE
      auto_sharding_info = AutoShardingPjitPartitioner.AutoShardingInfo(
          step_fn, step_fn_is_eval, replicate_output
      )
      partitioner = AutoShardingPjitPartitioner(
          init_is_eval, reshard_inputs, task_p, auto_sharding_info
      )
    else:
      partitioner = PjitPartitioner(init_is_eval, reshard_inputs, task_p)
  return partitioner
