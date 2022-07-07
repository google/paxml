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
import enum
import os
import pickle
import re
import threading
from typing import Any, List, Optional, Sequence, Tuple

from absl import logging
from praxis import py_utils
import tensorflow.compat.v2 as tf

NestedMap = py_utils.NestedMap


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

  def __init__(self, fq_filename: str, num_shards: int,
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

  def _write_to_shard(self, shard_idx: int, chunk: List[bytes]) -> None:
    with self._locks[shard_idx]:
      writer = self._writers[shard_idx]
      for ser_bytes in chunk:
        writer.write(ser_bytes)

  def write(self, chunk: List[bytes]) -> None:
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


def write_key_value_pairs(
    filename: str, key_value_pairs: Sequence[Tuple[str, Any]]) -> None:
  """Writes `key_value_pairs` to file."""
  if not filename.endswith('.pickle'):
    filename += '.pickle'
  with tf.io.gfile.GFile(filename, 'wb') as f:
    pickle.dump(key_value_pairs, f, protocol=pickle.HIGHEST_PROTOCOL)


def _validate_filenames(filenames: Sequence[str],
                        step: Optional[int] = None) -> Tuple[int, int]:
  """Validates the list of file names."""
  if not filenames:
    raise ValueError('Expecting at least one file. Found none.')
  # filename format: decoder_out_400000_shard_0.pickle
  shard_fname_regex = 'decoder_out_([0-9]*)_shard_([0-9]*).pickle'
  steps_to_shards = collections.defaultdict(set)
  for fname in filenames:
    filename = os.path.split(fname)[1]
    m = re.fullmatch(shard_fname_regex, filename)
    if m is None:
      raise ValueError(f'filename {filename} is not recognized as valid')
    steps_to_shards[int(m.group(1))].add(int(m.group(2)))
  available_steps = ', '.join([str(x) for x in steps_to_shards.keys()])
  if step is None and len(steps_to_shards) > 1:
    raise ValueError('Requiring explicit step= argument, as decode outputs from'
                     f' multiple steps are found: {available_steps}')
  if step is None:
    step = list(steps_to_shards.keys())[0]
  if step not in steps_to_shards:
    raise ValueError(
        f'step={step} not found in decode outputs. Available steps: '
        f'{available_steps}')
  full_shards = set()
  num_shards = len(steps_to_shards[step])
  for i in range(num_shards):
    full_shards.add(i)
  if full_shards - steps_to_shards[step]:
    raise ValueError(f'step={step} missing the following shards: '
                     f'{full_shards - steps_to_shards[step]}')
  return step, num_shards


def load_outputs(basedir: str,
                 pname: str,
                 step: Optional[int] = None) -> List[Any]:
  """Loads and returns the decode outputs.

  Args:
    basedir: The job log dir used when running the decoder.
    pname: name of the decoding dataset, usually `p.name` of the input param.
    step: the model step at which the decoder ran. If there are results only for
      one step present, this argument can be omitted.

  Returns:
    A list of the decoder outputs.
  """
  if basedir.endswith('/1') or basedir.endswith('/1/'):
    dirname = os.path.join(basedir, f'decoder_out/{pname}/')
  else:
    dirname = os.path.join(basedir, f'1/decoder_out/{pname}/')
  filenames = tf.io.gfile.glob(os.path.join(dirname, 'decoder_out_*.pickle'))
  try:
    step, num_shards = _validate_filenames(filenames, step)
  except ValueError as e:
    raise ValueError(f'Failed to read decode outputs under "{basedir}"') from e
  # load the data
  ret = list()
  for shard_idx in range(num_shards):
    fname = os.path.join(dirname,
                         f'decoder_out_{step}_shard_{shard_idx}.pickle')
    with tf.io.gfile.GFile(fname, 'rb') as f:
      ret.extend(pickle.load(f))
  logging.info('Loaded decoded outputs from "%s", from %d shards, step=%d',
               dirname, num_shards, step)
  return ret
