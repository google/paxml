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

"""Helpers for handling inputs unsupported by XLA."""

from absl import logging
import jax
import numpy as np


def split_out_xla_unsupported_batch(batch, partitioning_spec=None):
  """Splits out values not supported by XLA (such as strings) from the batch.

  This is used to pass the unsupported values through a different channel. See
  also `merge_back_xla_unsupported_batch`.

  Args:
    batch: The input (possibly nested) dictionary.
    partitioning_spec: The dictionary with partitioning information or None.

  Returns:
    A tuple of the following elements.
    batch: The original batch dictionary with unsupported elements removed.
    unsupported_batch: A dictionary with only the unsupported elements.
    partitioning_spec: The original partitioning_spec with unsupported elements
      removed.
  """
  unsupported_batch = {}
  new_partitioning_spec = {}

  for k, v in batch.items():
    if hasattr(v, 'items'):
      nested_batch, nested_unsupported_batch, nested_partitioning_spec = (
          split_out_xla_unsupported_batch(
              v,
              partitioning_spec=partitioning_spec.get(k)
              if partitioning_spec
              else None,
          )
      )
      if nested_unsupported_batch:
        batch[k] = nested_batch
        unsupported_batch[k] = nested_unsupported_batch
        if (
            partitioning_spec
            and k in partitioning_spec
            and nested_partitioning_spec
        ):
          new_partitioning_spec[k] = nested_partitioning_spec
      continue

    if not np.issubdtype(v.dtype, np.unicode_) and not np.issubdtype(
        v.dtype, np.object_
    ):
      continue

    unsupported_batch[k] = v

  # If no unsupported keys were detected, return out the original batch object
  # without modifying it.
  if not unsupported_batch:
    return batch, {}, partitioning_spec

  # Similarly for the multi-host case, which is not supported yet: return out
  # the original batch object without modifying it.
  if jax.process_count() > 1 and unsupported_batch:
    # TODO(b/279795947): Support xla passthrough for multihost eval.
    raise NotImplementedError(
        (
            'Unsupported inputs (with keys %s) were detected, but running with'
            ' more than one host. Forwarding these keys is currently not'
            ' supported (but may be supported in the future).'
        )
        % unsupported_batch.keys(),
    )

  batch = {k: v for k, v in batch.items() if k not in unsupported_batch}
  if partitioning_spec is not None:
    new_partitioning_spec.update(
        {
            k: v
            for k, v in partitioning_spec.items()
            if k not in unsupported_batch
        }
    )
  else:
    new_partitioning_spec = None
  return batch, unsupported_batch, new_partitioning_spec


def merge_back_xla_unsupported_batch(out, xla_unsupported_batch):
  """Adds back unsupported parts of the batch into out.

  This is done in case process_decode_out or other parts of the code want to
  make use of the unsupported parts.

  Args:
    out: The output dictionary without unsupported parts.
    xla_unsupported_batch: A dictionary with only the unsupported elements, if
      any.
  """
  if xla_unsupported_batch:
    out.update(xla_unsupported_batch)
