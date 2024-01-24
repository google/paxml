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

"""Utils for TF Summaries."""

import collections
import collections.abc
import concurrent
import contextlib
import operator
import textwrap
import typing
from typing import Any, Generator, Iterator, Mapping, Sequence

from absl import flags
from absl import logging
from clu import platform
import clu.values as clu_values
from etils import epath
import flax
import jax
from jax import numpy as jnp
import numpy as np
from paxml import metric_utils
from paxml import train_states
from praxis import base_layer
from praxis import py_utils
from praxis import pytypes
import seqio
import tensorflow.compat.v2 as tf
from tensorflow.compat.v2 import summary as tf_summary

flags.DEFINE_bool(
    'pax_only_aggregate_summaries', False,
    'If true, only output aggregate summary types (AGGREGATE_SCALAR, '
    'AGGREGATE_IMAGE).')
FLAGS = flags.FLAGS


JTensor = pytypes.JTensor
Nested = pytypes.Nested
NestedJTensor = pytypes.NestedJTensor
TrainState = train_states.TrainState
TensorProvenance = train_states.TensorProvenance
TrainStateProvenance = train_states.TrainStateProvenance
SummaryType = base_layer.SummaryType
SummaryWriter = tf.summary.SummaryWriter
WeightedScalars = pytypes.WeightedScalars
CluMetrics = pytypes.Metrics
WeightedScalarsList = pytypes.WeightedScalarsList
PMAP_PARALLEL_AXIS_NAME = base_layer.PMAP_PARALLEL_AXIS_NAME

# Maximum number of images written to a single summary entry.
MAX_IMAGES_PER_SUMMARY = 64
MAX_AUDIOS_PER_SUMMARY = 64
MAX_TEXTS_PER_SUMMARY = 64

# Used by tf_summary.audio.
AUDIO_SUMMARY_SAMPLE_RATE = 44_000

SummaryValueTypes = (
    clu_values.Scalar
    | clu_values.Image
    | clu_values.Text
    | clu_values.Summary
    | clu_values.Histogram
    | clu_values.Audio
)


_VALUES_TO_SUMMARY_TYPE = {
    clu_values.Scalar: SummaryType.SCALAR,
    clu_values.Text: SummaryType.TEXT,
    clu_values.Image: SummaryType.IMAGE,
    # Videos (GIFs) are written as clu.values.summary.
    clu_values.Summary: SummaryType.VIDEO,
    clu_values.Histogram: SummaryType.HISTOGRAM,
    clu_values.Audio: SummaryType.AUDIO,
}


def _get_summary_type(
    metric_value: SummaryValueTypes,
) -> SummaryType:
  """Infers metric summary type from the metric value type."""
  if type(metric_value) not in _VALUES_TO_SUMMARY_TYPE:
    raise ValueError(f'Unknown metric value type: {type(metric_value)}.')
  return _VALUES_TO_SUMMARY_TYPE[type(metric_value)]


def write_clu_metric_summaries(
    metric_values: dict[str, SummaryValueTypes], step_i: int
) -> None:
  """Given a dict of metric values, writes them out as summaries.

  This is expected to be called under a summary context.

  Args:
    metric_values: A dict[str, Any] objects with metric values. These values are
      one of the various clu_values.Value subtypes.
    step_i: An int representing the current step of decoding.
  """
  if not metric_values:
    return

  logging.info('Summarizing metrics.')
  for metric_name, metric_value in metric_values.items():
    logging.info('Summarizing metric %s', metric_name)
    summary_type = _get_summary_type(metric_value)
    # Pass both value and metadata to write_summary_tensor for video summary.
    if isinstance(metric_value, clu_values.Summary):
      write_summary_tensor(
          step_i,
          f'Metrics/{metric_name}',
          metric_value.value,
          summary_type,
          metric_value.metadata,
      )
    elif isinstance(metric_value, clu_values.Audio):
      write_summary_tensor(
          step_i,
          f'Metrics/{metric_name}',
          metric_value.value,
          summary_type,
          sample_rate=metric_value.sample_rate,
      )
    else:
      write_summary_tensor(
          step_i, f'Metrics/{metric_name}', metric_value.value, summary_type
      )


def compute_and_write_clu_metric_summaries(
    metrics: CluMetrics, step: int
) -> None:
  """Compute clu_metrics.Metric objects values and write them out as summaries.

  Args:
    metrics: A Dict[str, clu_metrics.Metric] objects with a compute_value()
      function implemented that returns either a clu_values.Value object, a
      Dict[str, clu_values.Value] objects, a Dict[str, List[clu_values.Value]]
      objects, or a List[clu_values.Value].
    step: An int representing the current step of decoding.
  """
  if metrics:
    # Convert metrics to Dict[str, clu_values.Value] for summary writing.
    clu_metric_values = metric_utils.compute_metric_values(metrics)
    write_clu_metric_summaries(clu_metric_values, step)


def write_seqio_metric_summaries(
    seqio_metrics: Sequence[Mapping[str, seqio.metrics.MetricValue | float]],
    metric_name_prefix: str,
    step: int,
) -> None:
  """Write seqio metric as summaries.

  Args:
    seqio_metrics: A sequence of dict of str to seqio metric value or float.
    metric_name_prefix: A prefix added to metric name.
    step: An int. representing the current step.
  """
  for m_dict in seqio_metrics:
    for k, v in m_dict.items():
      metric_name = f'{metric_name_prefix}/{k}'
      if isinstance(v, seqio.metrics.Text):
        metric_str = (
            v.textdata.decode() if isinstance(v.textdata, bytes) else v.textdata
        )
        logging.info(
            'Writing summary of %s with string value %s.',
            metric_name,
            metric_str,
        )
        tf_summary.text(metric_name, metric_str, step=step)
        continue

      if isinstance(v, seqio.metrics.Audio):
        logging.info('Writing summary of %s with audio.', metric_name)
        tf_summary.audio(
            metric_name,
            v.audiodata,
            v.sample_rate,
            step=step,
            max_outputs=v.max_outputs,
        )
        continue

      if isinstance(v, seqio.metrics.Image):
        logging.info('Writing summary of %s with image.', metric_name)
        tf_summary.image(
            metric_name, v.image, step=step, max_outputs=v.max_outputs
        )
        continue

      if isinstance(v, seqio.metrics.Histogram):
        tf_summary.histogram(metric_name, v.values, buckets=v.bins, step=step)
        continue

      if isinstance(v, seqio.metrics.Generic):
        tf_summary.write(metric_name, v.tensor, metadata=v.metadata, step=step)
        continue

      if isinstance(v, seqio.metrics.Scalar):
        v = float(v.value)
      else:
        v = float(v)
      logging.info('Writing summary of %s with value %.4f.', metric_name, v)
      write_summary_tensor(step, metric_name, v, SummaryType.AGGREGATE_SCALAR)


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


def pretty_format_iters(input_str: str) -> str:
  for c in '{}(),[]':
    input_str = input_str.replace(c, '')
  return '\n'.join(l for l in input_str.splitlines() if (l and not l.isspace()))


def pretty_repr_shapes(
    replicated_vars: NestedJTensor,
    is_vars_replicated: bool,
    with_dtype: bool = False,
) -> str:
  """Returns a pretty representation of the variable shapes and dtype."""

  def maybe_dtype_str(x: JTensor) -> str:
    if with_dtype:
      return f' {x.dtype}'
    else:
      return ''

  def pps(x: JTensor) -> str:
    """Remove leading dim from replicated model vars."""
    if is_vars_replicated:
      return 'x'.join(str(e) for e in x.shape[1:]) + maybe_dtype_str(x)
    else:
      # If var is not replicated, no need to remove the first dim.
      # Retrieves the global shape for Jax array; otherwise host-local shape.
      x_sda = x
      return 'x'.join(str(e) for e in x_sda.shape) + maybe_dtype_str(x)

  out = jax.tree_map(pps, replicated_vars)
  out = pretty_repr(out)
  return pretty_format_iters(out)


def pretty_repr_provenance(
    provenance: TensorProvenance | Nested[TensorProvenance],
) -> str:
  provenance_out = pretty_repr(provenance)
  return pretty_format_iters(provenance_out)


def _yield_subtrees(
    root: NestedJTensor,
    max_level: int,
    level: int = 0,
    name: tuple[str, ...] = (),
) -> Generator[tuple[tuple[str, ...], NestedJTensor], None, None]:
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


def l2_mean(
    tree: NestedJTensor, prefix: str = '', max_level: int = 4, sep: str = '/'
) -> dict[str, jnp.float32]:
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
    summary_tensors: NestedJTensor,
) -> dict[str, JTensor]:
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
  audio_summaries = {}
  video_summaries = {}
  for k, v in summary_tensors.items():  # pytype: disable=attribute-error  # jax-ndarray
    summary_type = base_layer.get_summary_type_from_key(k)
    if base_layer.get_summary_base_type(summary_type) == SummaryType.SCALAR:
      scalar_summaries[k] = v
    elif base_layer.get_summary_base_type(summary_type) == SummaryType.IMAGE:
      image_summaries[k] = v
    elif base_layer.get_summary_base_type(summary_type) == SummaryType.AUDIO:
      audio_summaries[k] = v
    elif base_layer.get_summary_base_type(summary_type) == SummaryType.VIDEO:
      video_summaries[k] = v

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
  audio_summaries = jax.tree_map(
      lambda x: jax.lax.all_gather(x, axis_name=PMAP_PARALLEL_AXIS_NAME),
      audio_summaries)
  max_entries = MAX_AUDIOS_PER_SUMMARY
  audio_summaries = jax.tree_map(
      lambda x: jnp.reshape(x, [-1] + list(x.shape[-2:]))[:max_entries],
      audio_summaries)
  video_summaries = jax.tree_map(
      lambda x: jax.lax.all_gather(x, axis_name=PMAP_PARALLEL_AXIS_NAME),
      video_summaries)

  summary_tensors = summary_tensors.copy()  # pytype: disable=attribute-error  # jax-ndarray
  for summary_dict in (
      scalar_summaries,
      image_summaries,
      audio_summaries,
      video_summaries,
  ):
    for k, v in summary_dict.items():
      summary_tensors[k] = v
  return summary_tensors


@contextlib.contextmanager
def get_summary_writer(summary_dir: epath.Path) -> Iterator[SummaryWriter]:
  """Context manager around Tensorflow's SummaryWriter."""
  if jax.process_index() == 0:
    logging.info('Opening SummaryWriter `%s`...', summary_dir)
    summary_writer = tf_summary.create_file_writer(str(summary_dir))
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


def flatten_summary_dict(
    summary_dict: dict[str, JTensor], parent_key: str | None = None
) -> list[Any]:
  """Flattens a summary dictionary."""
  outputs = []
  for key, value in summary_dict.items():
    if parent_key is not None:
      key = f'{parent_key}{key}'
    if isinstance(value, collections.abc.MutableMapping):
      outputs.extend(flatten_summary_dict(value, key))
    else:
      outputs.append((key, value))
  return outputs


def write_summary_tensor(
    step_i: int,
    key: str,
    tensor: float | JTensor | str | Sequence[JTensor],
    summary_type: SummaryType,
    metadata: Any | None = None,
    sample_rate: int = AUDIO_SUMMARY_SAMPLE_RATE,
) -> bool:
  """Writes summary in relevant processes."""
  if FLAGS.pax_only_aggregate_summaries:
    if summary_type not in {
        SummaryType.AGGREGATE_SCALAR, SummaryType.AGGREGATE_IMAGE
    }:
      return
  if isinstance(tensor, (list, tuple)):
    tensors = tensor
  else:
    tensors = [tensor]
  tensors = typing.cast(Sequence[JTensor], tensors)
  # Tensors are often pushed in step ascending order. Iterate over the most
  # recent ones. Only useful for non-aggregated summaries.
  tensors_it = reversed(tensors)
  base_summary_type = base_layer.get_summary_base_type(summary_type)
  if base_summary_type == SummaryType.SCALAR:
    # Force DeviceArray to NumPy array conversion before taking the mean.
    np_tensors = [np.array(t) for t in tensors_it]
    tensor = np.mean(np_tensors).item()
    logging.info('summary tensor at step=%s %s %s', step_i, key, tensor)
    tf_summary.scalar(key, tensor, step_i)
  elif base_summary_type == SummaryType.IMAGE:
    remaining_max_images = MAX_IMAGES_PER_SUMMARY
    for tensor in tensors_it:
      if remaining_max_images <= 0:
        break
      # Some eval codepath adds a leading 'test split' dim.
      tensor = np.reshape(tensor, [-1] + list(tensor.shape)[-3:])
      # Create a separate key for each image to avoid RPC oversize issues.
      for i in range(min(tensor.shape[0], remaining_max_images)):
        tf_summary.image(f'{key}/{i}', tensor[i:i + 1], step_i)
      remaining_max_images -= tensor.shape[0]
  elif base_summary_type == SummaryType.AUDIO:
    remaining_max_audios = MAX_AUDIOS_PER_SUMMARY
    for tensor in tensors_it:
      if remaining_max_audios <= 0:
        break
      tensor = np.reshape(tensor, [-1] + list(tensor.shape[-2:]))
      # TODO(nanxinchen): Make the sampling rate configurable
      for i in range(min(tensor.shape[0], remaining_max_audios)):
        tf_summary.audio(f'{key}/{i}', tensor[i : i + 1], sample_rate, step_i)
      remaining_max_audios -= tensor.shape[0]
  elif base_summary_type == SummaryType.TEXT:
    remaining_max_texts = MAX_TEXTS_PER_SUMMARY
    for tensor in tensors_it:
      if remaining_max_texts <= 0:
        break
      if isinstance(tensor, str):
        tf_summary.text(f'{key}', tensor, step_i)
      else:
        for i in range(min(tensor.shape[0], remaining_max_texts)):
          tf_summary.text(f'{key}/{i}', str(tensor[i:i + 1]), step_i)
        remaining_max_texts -= tensor.shape[0]
  elif base_layer.get_summary_base_type(summary_type) == SummaryType.VIDEO:
    # Metadata must exist for saving video summary.
    assert metadata is not None
    # Ensure that only one video summary is passed at a time.
    assert len(tensors) == 1
    tf_summary.write(tag=key, tensor=tensors[0], metadata=metadata, step=step_i)
  elif base_summary_type == SummaryType.HISTOGRAM:
    # Similar to the scalar case, we merge the histogram values. We expect the
    # same number of elements per tensor in `tensors`.
    tf_summary.histogram(key, np.concatenate(tensors), step_i)
  else:
    assert False, 'Unsupported summary type: ' + str(summary_type)


def get_summary_display_name_from_key(key: str) -> str:
  """Return a name for a summary using the base type.

  The internally tracked summary keys include type information about aggregate
  types, eg: AGGREGATE_SCALAR vs SCALAR, which is used for filtering. For
  externally facing summary names we omit the AGGREGATED_ and use only the base
  type suffix.
  """
  trimmed = base_layer.trim_summary_type_from_key(key)
  summary_type = base_layer.get_summary_type_from_key(key)
  base_summary_type = base_layer.get_summary_base_type(summary_type)
  return trimmed + base_layer.get_summary_type_suffix(base_summary_type)


def write_summary_entry(
    summary_writer: SummaryWriter,
    step_i: int,
    loss: JTensor,
    weighted_scalars_list: WeightedScalarsList,
    clu_metrics: CluMetrics,
    summary_tensors: NestedJTensor,
    steps_per_sec: float | None = None,
) -> None:
  """Writes a summary entry into the provided SummaryWriter."""
  work_unit = platform.work_unit()
  # Scalar values must be plain Python types rather than e.g. np.int / np.float.
  # SPMD training will produce a Jax Array.
  loss = py_utils.maybe_unreplicate_for_fully_replicated(loss)
  weighted_scalars_list = py_utils.maybe_unreplicate_for_fully_replicated(
      weighted_scalars_list)
  clu_metrics = py_utils.maybe_unreplicate_for_fully_replicated(clu_metrics)
  summary_tensors = py_utils.maybe_unreplicate_for_fully_replicated(
      summary_tensors)

  mean_loss = np.mean(loss).item()
  with summary_writer.as_default():
    status_msg = f'step = {step_i}, loss= {mean_loss}'
    write_summary_tensor(step_i, 'loss', mean_loss,
                         SummaryType.AGGREGATE_SCALAR)
    if steps_per_sec is not None:
      status_msg += f', steps/sec {steps_per_sec}'
      write_summary_tensor(step_i, 'Steps/sec', steps_per_sec,
                           SummaryType.AGGREGATE_SCALAR)
    logging.info('Metrics values at step %d:', step_i)
    logging.info('  loss=%f', mean_loss)
    for key, value_lst in weighted_scalars_list.items():
      sum_metric_weights = 0.
      weighted_sum_metric_values = 0.
      for value in value_lst:
        assert len(value) == 2, (
            'Metric value should be a pair of (value, weight).')
        metric_values = value[0]
        metric_weights = value[1]
        sum_metric_weights += np.sum(metric_weights)
        weighted_sum_metric_values += np.sum(metric_values * metric_weights)
      weighted_average = weighted_sum_metric_values / np.maximum(
          sum_metric_weights, 1e-6
      )
      logging.info('  %s=%f (weight=%f)', key, weighted_average,
                   sum_metric_weights)
      status_msg += f', {key}={weighted_average}'
      write_summary_tensor(step_i, f'Metrics/{key}', weighted_average,
                           SummaryType.AGGREGATE_SCALAR)
      write_summary_tensor(step_i, f'Metrics/{key}-weight', sum_metric_weights,
                           SummaryType.AGGREGATE_SCALAR)

    compute_and_write_clu_metric_summaries(clu_metrics, step_i)

    work_unit.set_task_status(status_msg)
    summaries = flatten_summary_dict(summary_tensors)
    for key, tensors in summaries:
      summary_type = base_layer.get_summary_type_from_key(key)
      key = get_summary_display_name_from_key(key)
      write_summary_tensor(step_i, key, tensors, summary_type)

  # Lastly flush summaries.
  summary_writer.flush()
  logging.info('Wrote summary entry at step `%d` (loss=`%f`).', step_i,
               mean_loss)


def write_model_structure(
    train_summary_writer: SummaryWriter,
    train_state: TrainState,
    is_vars_replicated,
):
  """Writes the Model Param structure to TB."""
  with train_summary_writer.as_default():
    out = pretty_repr_shapes(train_state.mdl_vars, is_vars_replicated)
    tf_summary.text(
        'Model', out, step=0
    )
  train_summary_writer.flush()


def write_model_provenance(
    train_summary_writer: SummaryWriter,
    train_state_provenance: TrainStateProvenance,
):
  """Writes the TrainStateProvenance to TB."""
  with train_summary_writer.as_default():
    mdl_vars_out = pretty_repr_provenance(train_state_provenance.mdl_vars)
    tf_summary.text(
        'Model vars provenance',
        mdl_vars_out,
        step=0,
    )
    opt_states_out = pretty_repr_provenance(train_state_provenance.opt_states)
    tf_summary.text(
        'Opt states provenance',
        opt_states_out,
        step=0,
    )
  train_summary_writer.flush()


def write_total_num_params(
    train_summary_writer: SummaryWriter, total_num_params: int
):
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


class SummaryHandler:
  """Handles summary writing to TensorBoard.

  This handler can be used to adjust the frequency of summary generation
  as well as enabling summary aggregation across several steps.
  """

  def __init__(
      self,
      summary_writer: SummaryWriter,
      write_interval_steps: int,
      accumulate_interval_steps: int | None = None,
      log_interval_steps: int | None = None,
      is_async: bool = False,
      name: str = '',
  ) -> None:
    """Constructor.

    Args:
      summary_writer: The SummaryWriter instance to use to write summaries to
        disk.
      write_interval_steps: The frequency at which to write summaries to disk.
      accumulate_interval_steps: The frequency at which to accumulate summaries
        across steps. If unset, do not accumulate and only use the values at a
        specific set.
      log_interval_steps: The interval to log step outputs. If not set, always
        log.
      is_async: Whether to process summaries asynchronously.
      name: Name of the handler.
    """
    self._summary_writer = summary_writer
    self._write_interval_steps = write_interval_steps
    self._accumulate_interval_steps = accumulate_interval_steps
    self._log_interval_steps = log_interval_steps
    self._name = name
    # When is_async is true, only update the following fields in the
    # SummaryHandler thread to make sure that they are thread safe.
    self._latest_step = -1
    self._losses = []
    self._weighted_scalars_list = collections.defaultdict(list)
    self._clu_metrics = {}
    self._summary_tensors = collections.defaultdict(list)
    self._steps_per_sec = []

    if is_async:
      self._summary_pool = concurrent.futures.ThreadPoolExecutor(
          max_workers=1, thread_name_prefix='SummaryHandler')
    else:
      self._summary_pool = None

  def __del__(self):
    # A blocking shutdown may prevent the main thread from exiting when an
    # exception happens.
    self.close(wait=False)

  def close(self, wait=True):
    """Shutdown the thread pool if processing summaries asynchronously."""
    if self._summary_pool:
      self._summary_pool.shutdown(wait=wait)

  @property
  def accumulate_over_steps(self) -> bool:
    """Indicates whether we should accumulate summaries over steps or not."""
    return self._accumulate_interval_steps is not None

  def should_accumulate(self, step: int) -> bool:
    """Indicates whether we should accumulate values for this step or not."""
    return (self.accumulate_over_steps and
            step % self._accumulate_interval_steps == 0)

  def should_write(self, step: int) -> bool:
    """Indicates whether we should write summaries to disk at this step."""
    return step % self._write_interval_steps == 0

  def should_log(self, step: int) -> bool:
    """Indicates whether to log the step outputs."""
    if self._log_interval_steps:
      return step % self._log_interval_steps == 0
    else:
      return True

  def process(
      self,
      step: int,
      loss: JTensor,
      weighted_scalars: WeightedScalars | None,
      summary_tensors: NestedJTensor,
      per_example_out: NestedJTensor | None = None,
      steps_per_sec: float | None = None,
      clu_metrics: CluMetrics | None = None,
  ) -> bool:
    """Adds summaries for a given step and indicates if summaries were written.

    Args:
      step: The step counter corresponding to these input values.
      loss: The loss tensor.
      weighted_scalars: The WeightedScalars instance, keyed by strings and
        valued by 2-tuples (value, weight).
      summary_tensors: The summary values keyed by summary names.
      per_example_out: A NestedMap of per example values.
      steps_per_sec: The estimate of steps per second to be added to the
        summary.
      clu_metrics: The CluMetrics instance, keyed by strings and valued by
        `Clu.Metrics` type.

    Returns:
      True if the summaries were written, False otherwise.
    """
    should_accumulate = self.should_accumulate(step)
    should_write_summary = self.should_write(step)
    should_log = self.should_log(step)
    if not (should_log or should_accumulate or should_write_summary):
      # Nothing to do, return immediately to avoid unnecessary work.
      return False
    loss = py_utils.maybe_unreplicate_for_fully_replicated(loss)
    weighted_scalars = py_utils.maybe_unreplicate_for_fully_replicated(
        weighted_scalars
    )
    if clu_metrics:
      clu_metrics = py_utils.maybe_unreplicate_for_fully_replicated(clu_metrics)
    summary_tensors = py_utils.maybe_unreplicate_for_fully_replicated(
        summary_tensors
    )
    if per_example_out and should_log:
      per_example_out = py_utils.maybe_unreplicate_for_first_shard(
          per_example_out)

    if self._summary_pool:

      def process_fn():
        # Copy the values from device to host first.
        (
            loss_copy,
            weighted_scalars_copy,
            clu_metrics_copy,
            summary_tensors_copy,
        ) = jax.device_get(
            (loss, weighted_scalars, clu_metrics, summary_tensors)
        )
        per_example_out_copy = None
        if per_example_out and should_log:
          per_example_out_copy = jax.device_get(per_example_out)

        # pytype: disable=wrong-arg-types
        self._process(
            step,
            loss_copy,
            weighted_scalars_copy,
            summary_tensors_copy,
            per_example_out_copy,
            steps_per_sec,
            should_log,
            clu_metrics_copy,
        )
        # pytype: enable=wrong-arg-types

      self._summary_pool.submit(process_fn)
    else:
      self._process(
          step,
          loss,
          weighted_scalars,
          summary_tensors,
          per_example_out,
          steps_per_sec,
          should_log,
          clu_metrics,
      )

    return self.should_write(step)

  def _process(
      self,
      step: int,
      loss: JTensor,
      weighted_scalars: WeightedScalars | None,
      summary_tensors: NestedJTensor,
      per_example_out: NestedJTensor | None = None,
      steps_per_sec: float | None = None,
      should_log: bool = False,
      clu_metrics: CluMetrics | None = None,
  ) -> bool:
    """Adds summaries for a given step."""

    if should_log:
      logging.info(
          '[PAX STATUS] step_i: %d, %s loss: %s', step, self._name, loss
      )
      if weighted_scalars:
        logging.info('weighted_scalars: %s', weighted_scalars)
      if clu_metrics:
        logging.info('clu_metrics: %s', clu_metrics)
      if per_example_out:
        logging.info('per_example_out: %s', per_example_out)
      logging.info('summary_tensors: %s', summary_tensors)

    if self.should_accumulate(step):
      self._add(
          step,
          loss,
          weighted_scalars,
          summary_tensors,
          steps_per_sec,
          clu_metrics,
      )

    if not self.should_write(step):
      return

    # No accumulation. Add at least the latest value.
    if not self.accumulate_over_steps:
      self._add(
          step,
          loss,
          weighted_scalars,
          summary_tensors,
          steps_per_sec,
          clu_metrics,
      )

    self._write()
    self._clear()

  def _add(
      self,
      step: int,
      loss: JTensor,
      weighted_scalars: WeightedScalars | None,
      summary_tensors: NestedJTensor,
      steps_per_sec: float | None = None,
      clu_metrics: CluMetrics | None = None,
  ) -> None:
    """Adds/accumulates the current summary values."""
    if self._latest_step >= 0 and step <= self._latest_step:
      logging.warning(
          'Step `%d` is smaller than the previously recorded one `%d`, while '
          'accumulating summaries.', step, self._latest_step)
    self._latest_step = step
    self._losses.append(loss)
    if weighted_scalars:
      for key, value in weighted_scalars.items():
        assert (
            len(value) == 2
        ), 'Metric value should be a pair of (value, weight).'
        self._weighted_scalars_list[key].append(value)
    if clu_metrics:
      self._clu_metrics = metric_utils.merge_clu_metrics(
          self._clu_metrics, clu_metrics
      )
    summaries = flatten_summary_dict(summary_tensors)
    for key, tensor in summaries:
      self._summary_tensors[key].append(tensor)
    if steps_per_sec:
      self._steps_per_sec.append(steps_per_sec)

  def _clear(self) -> None:
    """Clears the current summary values."""
    self._latest_step = -1
    self._losses = []
    self._weighted_scalars_list = collections.defaultdict(list)
    self._clu_metrics = {}
    self._summary_tensors = collections.defaultdict(list)
    self._steps_per_sec = []

  def _write(self) -> None:
    """Writes summaries, possibly accumulated across steps."""
    if self._latest_step == -1:
      raise ValueError('Cannot write an empty summary.')
    # Force DeviceArray to NumPy array conversion before taking the mean.
    losses = np.mean([np.array(l) for l in self._losses], axis=0)
    if self._steps_per_sec:
      steps_per_sec = np.mean(self._steps_per_sec)
    else:
      steps_per_sec = None

    write_summary_entry(
        self._summary_writer,
        self._latest_step,
        losses,
        self._weighted_scalars_list,
        self._clu_metrics,
        self._summary_tensors,
        steps_per_sec,
    )
