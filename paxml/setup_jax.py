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

r"""Utilities to set up JAX global configs."""

from typing import Optional

from absl import logging
import jax
from praxis import py_utils
import tensorflow.compat.v2 as tf


def setup_jax(globally_use_hardware_rng: bool,
              jax_backend_target: Optional[str], jax_xla_backend: Optional[str],
              jax_enable_checks: bool,
              jax_array: bool = False,
              jax_traceback_filtering_option: str = 'auto',
              should_initialize_jax_distributed: bool = False) -> None:
  """Setups JAX and logs information about this job."""
  # Hide any GPUs from TensorFlow. Otherwise TF might reserve memory and make
  # it unavailable to JAX.
  tf.config.experimental.set_visible_devices([], 'GPU')
  if globally_use_hardware_rng:
    py_utils.set_globally_use_rbg_prng_key()

  # We use xmap only with SPMD.
  jax.config.update('experimental_xmap_spmd_lowering', True)
  # Use the manual partitioning lowering of xmap to avoid vectorization.
  jax.config.update('experimental_xmap_spmd_lowering_manual', True)

  # Allow users to configure JAX traceback filtering.
  # https://github.com/google/jax/blob/main/jax/_src/config.py
  jax.config.update('jax_traceback_filtering', jax_traceback_filtering_option)

  if jax_array:
    # Always default to Array.
    jax.config.update('jax_array', True)
    jax.config.update('jax_parallel_functions_output_gda', False)
    logging.info('Using JAX Array for pjit, pmap, checkpointing and everywhere '
                 'else.')
  else:
    # Always default to GDA.
    jax.config.update('jax_parallel_functions_output_gda', True)
    jax.config.update('jax_array', False)
    logging.info('Using JAX GDA for pjit and checkpointing')

  if jax_enable_checks:
    jax.config.update('jax_enable_checks', True)
    logging.info('jax_enable_checks has been enabled.')

  if jax_backend_target:
    logging.info('Using JAX backend target %s', jax_backend_target)
    jax_xla_backend = 'None' if jax_xla_backend is None else jax_xla_backend
    logging.info('Using JAX XLA backend %s', jax_xla_backend)

  if should_initialize_jax_distributed:
    jax.distributed.initialize()

  logging.info('JAX process: %d / %d', jax.process_index(), jax.process_count())
  logging.info('JAX devices: %r', jax.devices())
  logging.info('jax.device_count(): %d', jax.device_count())
  logging.info('jax.local_device_count(): %d', jax.local_device_count())
  logging.info('jax.process_count(): %d', jax.process_count())
