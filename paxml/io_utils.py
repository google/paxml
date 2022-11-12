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

"""Utility helpers for IO."""

import collections
import concurrent
import contextlib
import enum
import json
import os
import pickle
import re
import threading
from typing import Any, Iterable, Iterator, List, Optional, Sequence, Tuple

from absl import flags
from absl import logging
from etils import epath
import jax
import numpy as np
from praxis import py_utils
import tensorflow.compat.v2 as tf

from paxml import checkpoints  # mapped to internal

FLAGS = flags.FLAGS

NestedMap = py_utils.NestedMap

_PROGRESS_CKPT_STEP_KEY = 'restore_checkpoint_step'
_INTERNAL_ARTIFACTS_SUBDIR = '_internal_artifacts'


class EvaluationMode(str, enum.Enum):
  EVAL = 'eval'
  DECODE = 'decoder'

  @property
  def progress_filename(self) -> str:
    if 'xm_xid' in FLAGS and FLAGS.xm_xid > 0:
      return f'_{self.value}_{FLAGS.xm_xid}_progress.json'

    return f'_{self.value}_progress.json'


class OutputFormatType(enum.Enum):
  """Output's container format to write to.

  The name of the format will be also used as the written output filename's
  suffix.
  """
  TFRECORD = enum.auto()


class ShardedParallelWriter:
  """Round-robin sharded async example writer.

  There shouldn't be multiple instances of this writer for a given `fq_filename`
  as it does not support writing to the same shard from different processes.
  This means that when running in multi-controller (multi-process) jax, users
  should only have one instance of this writer writing to shards so
  instantiation, writing, and closing should be guarded by a
  `jax.process_index() == 0` check.

  Also, this writer is not re-usable. I.e. if an instance is created with the
  same `fq_filename` it'll replace existing records and after calling `close`
  one cannot "re-open".

  Note: When writing, a single chunk list gets written to a single shard, so for
  an evenly sharded output shards users should submit roughly evenly sized
  chunks consistently.
  """

  def __init__(self,
               fq_filename: epath.PathLike,
               num_shards: int,
               output_format: OutputFormatType = OutputFormatType.TFRECORD):
    """Constructor.

    Args:
      fq_filename: str, the fully-qualified filename prefix of the output to
        write to.
      num_shards: int, number of shards that we'll be writing the output to.
      output_format: an enum that defines the container format for serialized
        examples.
    """
    self._output_format = output_format
    self._num_shards = num_shards
    self._writer_fnames = [
        (f'{fq_filename}.{output_format.name.lower()}'
         f'-{i:05d}-of-{num_shards:05d}') for i in range(num_shards)]

    self._shard_idx = 0
    self._wpool = concurrent.futures.ThreadPoolExecutor(max_workers=num_shards)
    self._futures = []
    self._locks = [threading.Lock() for _ in range(num_shards)]

    if self._output_format == OutputFormatType.TFRECORD:
      self._writers = [
          tf.io.TFRecordWriter(fname) for fname in self._writer_fnames]
    else:
      raise NotImplementedError(f'{self._output_format} writer not implemented')

  def _maybe_reraise_errors(self) -> None:
    """Reraise exceptions since wpool doesn't raise until accessing result."""
    incomplete_futures = []
    for future in self._futures:
      if future.done():
        _ = future.result()  # raise any exceptions
      else:
        incomplete_futures.append(future)

    if len(incomplete_futures) >= self._num_shards:
      logging.warning('ShardedParallelWriter write speed is not able to keep '
                      'up with data produced. May eventually cause host OOM so '
                      'try increase the number of shards.')

    self._futures = incomplete_futures

  def _write_to_shard(self, shard_idx: int, chunk: Sequence[bytes]) -> None:
    with self._locks[shard_idx]:
      writer = self._writers[shard_idx]
      for ser_bytes in chunk:
        writer.write(ser_bytes)

  def write(self, chunk: Sequence[bytes]) -> None:
    """Async write: submits a write job to worker pool.

    Args:
      chunk: a list of bytes to be written to a single shard.
    """
    self._maybe_reraise_errors()
    future = self._wpool.submit(self._write_to_shard, self._shard_idx, chunk)
    self._futures.append(future)
    self._shard_idx = (self._shard_idx + 1) % self._num_shards

  def close(self) -> None:
    # shutdown worker pool after waiting for all futures to complete
    logging.info('waiting for all workers to join')
    self._wpool.shutdown(wait=True)

    logging.info('closing all writers')
    for writer in self._writers:
      writer.close()


def _to_ndarray(x: Any) -> Any:
  # Values in key_value_pairs may contain numpy ndarrays, but may also
  # contain DeviceArray types -- which are equivalent to ndarray in all
  # respects, except one: they require additional dependencies in code
  # that attempts to load the .pickle file.
  # For such types the _value property returns a proper np.ndarray.
  if hasattr(x, '_value') and isinstance(x._value, np.ndarray):  # pylint: disable=protected-access
    x = x._value  # pylint: disable=protected-access

  return x


class JnpEncoder(json.JSONEncoder):
  """jax.numpy compatible encoder: https://github.com/mpld3/mpld3/issues/434."""

  def default(self, o: Any) -> Any:
    if isinstance(o, jax.numpy.DeviceArray):
      return _to_ndarray(o)
    elif isinstance(o, np.integer):
      return int(o)
    elif isinstance(o, np.floating):
      return float(o)
    elif isinstance(o, np.ndarray):
      return o.tolist()
    elif isinstance(o, bytes):
      return o.decode('utf-8')

    return super().default(o)


def write_key_value_pairs(filename: epath.PathLike,
                          key_value_pairs: Sequence[Tuple[str, Any]],
                          cast_to_ndarray: bool = True,
                          write_pickle: bool = True) -> None:
  """Writes `key_value_pairs` to pkl and jsonl files."""
  filename = epath.Path(filename)

  if cast_to_ndarray:
    key_value_pairs = jax.tree_map(_to_ndarray, key_value_pairs)

  if write_pickle:
    with filename.with_suffix('.pickle').open('wb') as pkl_f:
      pickle.dump(key_value_pairs, pkl_f, protocol=pickle.HIGHEST_PROTOCOL)

  with filename.with_suffix('.jsonl').open('w') as jsonl_f:
    for _, v in key_value_pairs:
      jsonl_f.write(json.dumps(v, cls=JnpEncoder) + '\n')


def _validate_filenames(filenames: Iterable[epath.PathLike],
                        step: Optional[int] = None) -> Tuple[int, int]:
  """Validates the list of file names."""
  if not filenames:
    raise ValueError('Expecting at least one file. Found none.')
  filenames = [epath.Path(p) for p in filenames]
  # filename format: {eval,decoder}_out_400000_shard_0.pickle
  shard_fname_regex = re.compile(
      '(eval|decoder)_out_([0-9]*)_shard_([0-9]*).pickle')
  steps_to_shards = collections.defaultdict(set)
  for fname in filenames:
    filename = os.path.basename(fname)
    if (m := shard_fname_regex.fullmatch(filename)) is None:
      raise ValueError(f'filename {filename} is not recognized as valid')
    steps_to_shards[int(m.group(2))].add(int(m.group(3)))
  available_steps = ', '.join(str(x) for x in steps_to_shards)

  if step is None:
    if len(steps_to_shards) > 1:
      raise ValueError(
          'Requiring explicit step= argument, as eval/decode outputs from'
          f' multiple steps are found: {available_steps}')
    step = list(steps_to_shards)[0]

  if step not in steps_to_shards:
    raise ValueError(
        f'step={step} not found in eval/decode outputs. Available steps: '
        f'{available_steps}')

  full_shards = set()
  num_shards = len(steps_to_shards[step])
  for i in range(num_shards):
    full_shards.add(i)
  if full_shards - steps_to_shards[step]:
    raise ValueError(f'step={step} missing the following shards: '
                     f'{full_shards - steps_to_shards[step]}')
  return step, num_shards


def load_outputs(basedir: epath.Path,
                 pname: str,
                 fname_prefix: str,
                 step: Optional[int] = None) -> List[Any]:
  """Loads and returns the eval/decode outputs.

  Args:
    basedir: The job log dir used when running the eval/decoder.
    pname: name of the dataset, usually `p.name` of the input param.
    fname_prefix: prefix of the filename ('eval' or 'decoder').
    step: the model step at which the model ran. If there are results only for
      one step present, this argument can be omitted.

  Returns:
    A list of the decoder outputs.
  """
  if basedir.parts[-1] == '1':
    dirname = basedir / f'{fname_prefix}_out/{pname}/'
  else:
    dirname = basedir / f'1/{fname_prefix}_out/{pname}/'
  filenames = dirname.glob(f'{fname_prefix}_out_*.pickle')

  try:
    step, num_shards = _validate_filenames(filenames, step)
  except ValueError as e:
    raise ValueError(f'Failed to read outputs under "{basedir}"') from e

  # load the data
  ret = list()
  for shard_idx in range(num_shards):
    fname = dirname / f'{fname_prefix}_out_{step}_shard_{shard_idx}.pickle'
    with fname.open('rb') as f:
      ret.extend(pickle.load(f))
  logging.info('Loaded %s outputs from "%s", from %d shards, step=%d',
               fname_prefix, dirname, num_shards, step)
  return ret


@contextlib.contextmanager
def checkpoint_progress(job_log_dir: epath.Path, checkpoint_step: Optional[int],
                        mode: EvaluationMode) -> Iterator[None]:
  """Checkpoints eval/decode status by writing current step to job_log_dir.

  If checkpoint_step is None, that means we haven't restored from a checkpoint
  and using random weights. If using random weights, we skip checkpointing
  eval progress since it's not based on a written checkpoint.

  Args:
    job_log_dir: the jobs log directory to write eval progress under.
    checkpoint_step: current checkpoint step we're restoring weights from. If
      None, it means we haven't restored from a checkpoint and using random
      weights instead. If using random weights, we skip checkpointing eval
      progress since it's not based on a written checkpoint.
    mode: a EvaluationMode enum type indicating the mode in which the model is
      being evaluated

  Yields:
    None, purely for context manager api purposes.
  """
  if checkpoint_step is None or jax.process_index() != 0:
    # Do nothing for random weights or non-leader processes.
    yield
  else:
    # If we're resuming after being preempted mid evaluation, don't overwrite.
    fname = job_log_dir / _INTERNAL_ARTIFACTS_SUBDIR / mode.progress_filename
    fname.parent.mkdir(parents=True, exist_ok=True)
    if not fname.exists():
      logging.info('Writing %s progress to %s for step %d.',
                   mode.value, fname, checkpoint_step)
      with fname.open('w') as f:
        json.dump({_PROGRESS_CKPT_STEP_KEY: checkpoint_step}, f)
    else:
      logging.info('Not writing %s progress to %s since resuming.',
                   mode.value, fname)

    try:
      yield
    finally:
      logging.info('Completed %s for step %d.', mode.value, checkpoint_step)
      fname.unlink()


def get_checkpoint_step(job_log_dir: epath.Path,
                        restore_checkpoint_dir: epath.Path,
                        mode: EvaluationMode) -> Optional[int]:
  """Gets the latest checkpoint step to eval/decode on.

  Args:
    job_log_dir: the jobs log directory to search for checkpoints and eval
      progress under.
    restore_checkpoint_dir: the directory from which we're reading checkpoints.
      Note that this may not necessarily be the same as job_log_dir.
    mode: a EvaluationMode enum type indicating the mode in which the model is
      being evaluated

  Returns:
    Returns the step with partially completed eval/decode if a job was preempted
    mid-way through a eval/decode on a checkpoint that is no longer the newest
    written to the `job_log_dir/checkpoints`. Otherwise, returns the latest
    checkpoint step written.
  """
  progress_fname = (
      job_log_dir / _INTERNAL_ARTIFACTS_SUBDIR / mode.progress_filename)
  if progress_fname.exists():
    with progress_fname.open() as f:
      progress_json = json.load(f)

    step = progress_json[_PROGRESS_CKPT_STEP_KEY]
    logging.info('Resuming %s from step %d.', mode.value, step)
    return step

  return checkpoints.retrieve_latest_checkpoint_step(
      restore_checkpoint_dir)
