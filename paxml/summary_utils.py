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

"""Utils for TF Summaries."""

import collections.abc
import contextlib
import operator
import textwrap
import time
from typing import Any, Dict, Generator, List, Optional, Tuple

from absl import flags
from absl import logging
from clu import platform
import flax
import jax
from jax import numpy as jnp
import numpy as np
from praxis import base_layer
from praxis import py_utils
from praxis import pytypes
from praxis import train_states
import tensorflow.compat.v2 as tf
from tensorflow.compat.v2 import summary as tf_summary

flags.DEFINE_bool(
    'pax_only_aggregate_summaries', False,
    'If true, only output aggregate summary types (AGGREGATE_SCALAR, '
    'AGGREGATE_IMAGE).')
FLAGS = flags.FLAGS


JTensor = pytypes.JTensor
NestedJTensor = pytypes.NestedJTensor
TrainState = train_states.TrainState
SummaryType = base_layer.SummaryType
SummaryWriter = tf.summary.SummaryWriter
PMAP_PARALLEL_AXIS_NAME = base_layer.PMAP_PARALLEL_AXIS_NAME

# Maximum number of images written to a single summary entry.
MAX_IMAGES_PER_SUMMARY = 64
MAX_TEXTS_PER_SUMMARY = 64


# Copied from flax.core.FrozenDict and customized for lists.
def pretty_repr(values: NestedJTensor, num_spaces: int = 4) -> str:
  """Returns an indented representation of the nested dictionary."""

  def indent(txt: str) -> str:
    return textwrap.indent(txt, ' ' * num_spaces)

  if isinstance(values, dict):
    rep = []
    for key, val in values.items():
      rep.append(f'{key}: {pretty_repr(val)},\n')
    if rep:
      return '{\n' + indent(''.join(rep)) + '}'
    else:
      return '{}'
  elif isinstance(values, (list, tuple)):
    rep = []
    for v in values:
      rep.append(f'{pretty_repr(v)},\n')
    if rep:
      return '[\n' + indent(''.join(rep)) + ']'
    else:
      return '[]'
  else:
    return repr(values)


def pretty_repr_shapes(replicated_vars: NestedJTensor,
                       is_vars_replicated) -> str:
  """Returns a pretty representation of the variable shapes."""

  def pps(x: JTensor) -> str:
    """Remove leading dim from replicated model vars."""
    if is_vars_replicated:
      return 'x'.join(str(e) for e in x.shape[1:])
    else:
      # If var is not replicated, no need to remove the first dim.
      # Retrieves the global shape for GDA; otherwise host-local shape.
      x_sda = x
      return 'x'.join(str(e) for e in x_sda.shape)

  out = jax.tree_map(pps, replicated_vars)
  out = pretty_repr(out)
  for c in '{}(),[]':
    out = out.replace(c, '')
  return '\n'.join(l for l in out.splitlines() if (l and not l.isspace()))


def _yield_subtrees(
    root: NestedJTensor,
    max_level: int,
    level: int = 0,
    name: Tuple[str, ...] = (),
) -> Generator[Tuple[Tuple[str, ...], NestedJTensor], None, None]:
  """Yields subtrees up to max_level."""
  if level < max_level:
    if isinstance(root, dict):
      for key in root:
        for out in _yield_subtrees(root[key], max_level, level + 1,
                                   name + (key,)):
          yield out
    elif isinstance(root, (list, tuple)):
      list_len = len(root)
      for ii in range(list_len):
        for out in _yield_subtrees(root[ii], max_level, level + 1,
                                   name + (str(ii),)):
          yield out
    else:
      # TODO(yonghui): Support other common composite types.
      yield (name, root)
  else:
    if root is not None:
      yield (name, root)


def l2_mean(tree: NestedJTensor,
            prefix: str = '',
            max_level: int = 4,
            sep: str = '/') -> Dict[str, jnp.float32]:
  """L2 Norms over pytree."""

  def _sq(x):
    a = jnp.maximum(jnp.amax(jnp.abs(x)), 1.0)
    x = x / a
    return jnp.array([x.size, (a**2) * jnp.sum(x**2)])

  squares = jax.tree_map(_sq, tree)
  names, squares = zip(*_yield_subtrees(squares, max_level=max_level))
  names = [sep.join(name) for name in names]
  if prefix:
    names = [prefix + sep + n for n in names]

  def norm_fn(tree: NestedJTensor) -> jnp.float32:
    out = jax.tree_util.tree_reduce(operator.add, tree)
    # NOTE(yonghui): Here we normalize out[1] by out[0], instead of sqrt(out[1])
    # by out[0] so that l2_norm is more semantically meaningful: it means the
    # average scale of params. In addition, this normalization makes sure norm
    # is invariant to number of model replicas (in pmap training).
    # TODO(yonghui): In the future, compute mean and std instead.
    return jnp.sqrt(out[1] / out[0])

  norms = [norm_fn(tree) for tree in squares]
  return dict(zip(names, norms))


def flatten_flax_summaries(
    summary_tensors: NestedJTensor) -> Dict[str, JTensor]:
  """Flatten flax style summary pytree to a flat dict.

  summary_tensors is a nested dict, e.g.,
    summary_tensors = {'lm' : {'layer' : foo_array}}

  flatten_dict produces a flat dict
    summary_tensors = {('lm', 'layer') : foo_array}

  The returned summary dict is
    {'lm/layer' : foo_array}

  Args:
    summary_tensors: a nested summary dict.

  Returns:
    A flattened summary dict.
  """
  summary_tensors = flax.traverse_util.flatten_dict(summary_tensors)
  new_summary_tensors = {}
  for k, v in summary_tensors.items():
    assert isinstance(k, tuple)
    # Flax flatten_dict flattens nested dict to {(root, parent, self), value}.
    # The new key is 'root_parent_self'.
    k = '/'.join(k)
    # If summary is unpacked during repeat, we need to uniquify summary from
    # each layer.
    if isinstance(v, list):
      for i, u in enumerate(v):
        summary_type = base_layer.get_summary_type_from_key(k)
        root_name = base_layer.trim_summary_type_from_key(k)
        full_name = root_name + '_' + str(
            i) + base_layer.get_summary_type_suffix(summary_type)
        new_summary_tensors[full_name] = u
    else:
      new_summary_tensors[k] = v
  return new_summary_tensors


def aggregate_per_replica_summaries(summary_tensors: NestedJTensor):
  """Aggregates summaries from different replicas in pmap."""
  scalar_summaries = {}
  image_summaries = {}
  for k, v in summary_tensors.items():
    summary_type = base_layer.get_summary_type_from_key(k)
    if base_layer.get_summary_base_type(summary_type) == SummaryType.SCALAR:
      scalar_summaries[k] = v
    elif base_layer.get_summary_base_type(summary_type) == SummaryType.IMAGE:
      image_summaries[k] = v

  # Compute the mean of scalars.
  scalar_summaries = jax.lax.pmean(
      scalar_summaries, axis_name=PMAP_PARALLEL_AXIS_NAME)
  # Gather per-replica image results.
  image_summaries = jax.tree_map(
      lambda x: jax.lax.all_gather(x, axis_name=PMAP_PARALLEL_AXIS_NAME),
      image_summaries)
  max_entries = MAX_IMAGES_PER_SUMMARY
  image_summaries = jax.tree_map(
      lambda x: jnp.reshape(x, [-1] + list(x.shape)[-3:])[:max_entries],
      image_summaries)

  summary_tensors = summary_tensors.copy()
  for k, v in scalar_summaries.items():
    summary_tensors[k] = v
  for k, v in image_summaries.items():
    summary_tensors[k] = v
  return summary_tensors


@contextlib.contextmanager
def get_summary_writer(summary_dir: str) -> SummaryWriter:
  """Context manager around Tensorflow's SummaryWriter."""
  if jax.process_index() == 0:
    logging.info('Opening SummaryWriter `%s`...', summary_dir)
    summary_writer = tf_summary.create_file_writer(summary_dir)
  else:
    # We create a dummy tf.summary.SummaryWriter() on non-zero tasks. This will
    # return a mock object, which acts like a summary writer, but does nothing,
    # such as writing event to disk.
    logging.info('Opening a mock-like SummaryWriter.')
    summary_writer = tf_summary.create_noop_writer()
  try:
    yield summary_writer
  finally:
    summary_writer.close()
    if jax.process_index() == 0:
      logging.info('Closed SummaryWriter `%s`.', summary_dir)
    else:
      logging.info('Closed a mock-like SummaryWriter.')


def flatten_summary_dict(summary_dict: Dict[str, JTensor],
                         parent_key: Optional[str] = None) -> List[Any]:
  """Flattens a summary dictionary."""
  separator = '@'
  outputs = []
  for key, value in summary_dict.items():
    if parent_key is not None:
      key = f'{parent_key}{separator}{key}'
    if isinstance(value, collections.abc.MutableMapping):
      outputs.extend(flatten_summary_dict(value, key))
    else:
      outputs.append((key, value))
  return outputs


def write_summary_tensor(step_i: int, key: str, tensor: JTensor,
                         summary_type: SummaryType) -> bool:
  """Writes summary in relevant processes."""
  if FLAGS.pax_only_aggregate_summaries:
    if summary_type == SummaryType.SCALAR or summary_type == SummaryType.IMAGE:
      return
  if base_layer.get_summary_base_type(summary_type) == SummaryType.SCALAR:
    logging.info('summary tensor at step=%s %s %s', step_i, key, tensor)
    tensor = np.mean(tensor).item()
    tf_summary.scalar(key, tensor, step_i)
  elif base_layer.get_summary_base_type(summary_type) == SummaryType.IMAGE:
    # Some eval codepath adds a leading 'test split' dim.
    tensor = np.reshape(tensor, [-1] + list(tensor.shape)[-3:])
    # Create a separate key for each image to avoid RPC oversize issues.
    for i in range(min(tensor.shape[0], MAX_IMAGES_PER_SUMMARY)):
      tf_summary.image(f'{key}/{i}', tensor[i:i + 1], step_i)
  elif base_layer.get_summary_base_type(summary_type) == SummaryType.TEXT:
    for i in range(min(tensor.shape[0], MAX_TEXTS_PER_SUMMARY)):
      tf_summary.text(f'{key}/{i}', str(tensor[i:i + 1]), step_i)
  else:
    assert False, 'Unsupported summary type: ' + str(summary_type)


def write_summary_entry(summary_writer: SummaryWriter,
                        step_i: int,
                        loss: JTensor,
                        metrics: Dict[str, JTensor],
                        summary_tensors: NestedJTensor,
                        steps_per_sec: Optional[float] = None) -> None:
  """Writes a summary entry into the provided SummaryWriter."""
  work_unit = platform.work_unit()
  # Scalar values must be plain Python types rather than e.g. np.int / np.float.
  # SPMD training may produce GDA.
  loss = py_utils.maybe_unreplicate_for_fully_replicated(loss)
  metrics = py_utils.maybe_unreplicate_for_fully_replicated(metrics)
  summary_tensors = py_utils.maybe_unreplicate_for_fully_replicated(
      summary_tensors)

  mean_loss = np.mean(loss).item()
  with summary_writer.as_default():
    status_msg = f'step = {step_i}, loss= {mean_loss}'
    write_summary_tensor(step_i, 'loss', mean_loss, SummaryType.SCALAR)
    if steps_per_sec is not None:
      status_msg += f', steps/sec {steps_per_sec}'
      write_summary_tensor(step_i, 'Steps/sec', steps_per_sec,
                           SummaryType.SCALAR)
    logging.info('Metrics values at step %d:', step_i)
    logging.info('  loss=%f', mean_loss)
    for key, value in metrics.items():
      assert len(value) == 2, (
          'Metric value should be a pair of (value, weight).')
      metric_values = value[0]
      metric_weights = value[1]
      sum_metric_weights = np.sum(metric_weights)
      weighted_average = np.sum(
          metric_values * metric_weights) / sum_metric_weights
      logging.info('  %s=%f (weight=%f)', key, weighted_average.item(),
                   sum_metric_weights.item())
      status_msg += f', {key}={weighted_average.item()}'
      write_summary_tensor(step_i, f'Metrics/{key}', weighted_average.item(),
                           SummaryType.SCALAR)
      write_summary_tensor(step_i, f'Metrics/{key}-weight',
                           sum_metric_weights.item(), SummaryType.SCALAR)

    work_unit.set_task_status(status_msg)
    summaries = flatten_summary_dict(summary_tensors)
    for key, tensor in summaries:
      summary_type = base_layer.get_summary_type_from_key(key)
      write_summary_tensor(step_i, key, tensor, summary_type)

  # Lastly flush summaries.
  summary_writer.flush()
  logging.info('Wrote summary entry at step `%d` (loss=`%f`).', step_i,
               mean_loss)


def write_model_structure(train_summary_writer: SummaryWriter,
                          train_state: TrainState, is_vars_replicated):
  """Writes the Model Param structure to TB."""
  with train_summary_writer.as_default():
    out = pretty_repr_shapes(train_state.mdl_vars, is_vars_replicated)
    tf_summary.text('Model', out, step=0)
  train_summary_writer.flush()


def write_total_num_params(train_summary_writer: SummaryWriter,
                           total_num_params: int):
  """Writes the total number of parameters to TB."""
  with train_summary_writer.as_default():
    # Add whitespace every 3 digit for readability.
    num_params_str = '{:,}'.format(total_num_params).replace(',', ' ')
    tf_summary.text('Total Num Params', num_params_str, step=0)
  train_summary_writer.flush()


def write_global_batch_size(train_summary_writer: SummaryWriter,
                            global_batch_size: int):
  """Writes the global batch size to TB."""
  with train_summary_writer.as_default():
    batch_size_str = '{:,}'.format(global_batch_size).replace(',', ' ')
    tf_summary.text('Global batch size', batch_size_str, step=0)
  train_summary_writer.flush()


def write_summary_every_n_steps(train_summary_writer: SummaryWriter,
                                step_i: int, summary_every_n_steps: int,
                                loss: JTensor, metrics: NestedJTensor,
                                per_example_out: NestedJTensor,
                                summary_tensors: NestedJTensor,
                                summary_last_time: Optional[float],
                                summary_last_step: Optional[int]) -> bool:
  """Writes summaries at regular intervals."""
  if step_i % summary_every_n_steps != 0:
    return False
  loss = py_utils.maybe_unreplicate_for_fully_replicated(loss)
  metrics = py_utils.maybe_unreplicate_for_fully_replicated(metrics)
  per_example_out = py_utils.maybe_unreplicate_for_first_shard(per_example_out)
  summary_tensors = py_utils.maybe_unreplicate_for_fully_replicated(
      summary_tensors)
  logging.info('step_i: %d, training loss: %s', step_i, loss)
  logging.info('metrics: %s', metrics)
  logging.info('per_example_out: %s', per_example_out)
  logging.info('summary_tensors: %s', summary_tensors)

  duration_sec = time.time() - summary_last_time
  num_steps = step_i - summary_last_step
  steps_per_sec = num_steps / duration_sec
  logging.info('steps/sec: %f', steps_per_sec)

  write_summary_entry(train_summary_writer, step_i, loss, metrics,
                      summary_tensors, steps_per_sec)
  return True
