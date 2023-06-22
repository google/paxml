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

r"""Utilities to set up JAX global configs."""

import dataclasses
from typing import Optional

from absl import logging
import jax
from praxis import py_utils
import tensorflow.compat.v2 as tf


@dataclasses.dataclass
class JaxDistributedOptions:
  coordinator_address: str
  num_processes: int
  process_id: int


@py_utils.benchmark('[PAX STATUS]: ')
def setup_jax(
    globally_use_hardware_rng: bool,
    jax_backend_target: Optional[str],
    jax_xla_backend: Optional[str],
    jax_enable_checks: bool,
    jax_traceback_filtering_option: str = 'auto',
    should_initialize_jax_distributed: bool = False,
    jax_distributed_options: Optional[JaxDistributedOptions] = None,
) -> None:
  """Setups JAX and logs information about this job."""

  # Hide any GPUs from TensorFlow. Otherwise TF might reserve memory and make
  # it unavailable to JAX.
  tf.config.experimental.set_visible_devices([], 'GPU')
  if globally_use_hardware_rng:
    py_utils.set_globally_use_rbg_prng_key()

  # Log tracing and compilation time.
  jax.config.update('jax_log_compiles', True)

  # Allow users to configure JAX traceback filtering.
  # https://github.com/google/jax/blob/main/jax/_src/config.py
  jax.config.update('jax_traceback_filtering', jax_traceback_filtering_option)

  if jax_enable_checks:
    jax.config.update('jax_enable_checks', True)
    logging.info('jax_enable_checks has been enabled.')

  if jax_backend_target:
    logging.info('Using JAX backend target %s', jax_backend_target)
    jax_xla_backend = 'None' if jax_xla_backend is None else jax_xla_backend
    logging.info('Using JAX XLA backend %s', jax_xla_backend)

  if should_initialize_jax_distributed:
    if jax_distributed_options:
      jax.distributed.initialize(
          jax_distributed_options.coordinator_address,
          jax_distributed_options.num_processes,
          jax_distributed_options.process_id,
      )
    else:
      jax.distributed.initialize()

  logging.info('JAX process: %d / %d', jax.process_index(), jax.process_count())
  logging.info('JAX devices: %r', jax.devices())
  logging.info('jax.device_count(): %d', jax.device_count())
  logging.info('jax.local_device_count(): %d', jax.local_device_count())
  logging.info('jax.process_count(): %d', jax.process_count())
