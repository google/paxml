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

"""MetricTracker class and associated utilities."""

import os
import re
import sys

from absl import logging

import tensorflow.compat.v2 as tf


class MetricTracker:
  """Stateful tracker for metrics , e.g. WER.

  Value is stored on disk such that random re-starts do not lose the
  tracked value.

  Attributes:
    metric_name: str storing the name of the tracked metric, e.g. 'wer'.
    metric_partition: str name of the data partition on which we track the
      metric, e.g. 'dev-clean', 'test-other'.
    metric_value: float storing the current best value for the tracked metric;
      logic on whether this is a min or a max tracker is left to the user of the
      class.
    dir_name: str storing the full path to the directory where the tracker state
      is stored in a file. If model checkpoints achieving the best value are
      deemed useful, they should be stored by the user of the tracker in the
      same directory; the tracker manages garbage collection (deletion of
      previous best checkpoint when the new best metric value is updated).
    global_step: int specifying the training step at which the current best
      value was recorded.
  """

  def __init__(self,
               dir_name: str,
               metric_name: str,
               metric_partition: str,
               initial_metric_value: float):
    self._dir_name: str = dir_name
    self._global_step: int = -1
    self._metric_name: str = metric_name
    self._metric_partition: str = metric_partition
    self._metric_value: float = initial_metric_value
    self._init()

  def _init(self, default_value: float = sys.float_info.max) -> None:
    """Read metric value from file on disk.

    Args:
      default_value: if file does not exist: first time use; set value to
        default_value and write file.
    """
    if tf.io.gfile.exists(self.metric_filename):
      self._restore_from_file()
    else:
      self._set(default_value, -1)

  def _restore_from_file(self) -> None:
    """Restore metric value from file."""
    assert tf.io.gfile.exists(self.metric_filename)
    with tf.io.gfile.GFile(self.metric_filename, mode='r') as summary_file:
      for line in summary_file:
        m = re.fullmatch(r'Checkpoint at step=(\S+)\s+(\S+)\s+(\S+)\: (\S+);',
                         line.strip())
        if m:
          assert len(
              m.groups()) == 4, f'm.groups()={m.groups()}; line={line.strip()}'
          assert self._metric_name == str(m.group(3))
          assert self._metric_partition == str(m.group(2))
          self._metric_value = float(m.group(4))
          self._global_step = int(m.group(1))
          return
    # Should never get here.
    raise ValueError(
        f'Unable to restore metric value from file {self.metric_filename}')

  def _set(self, value: float, global_step: int) -> None:
    self._metric_value = value
    self._global_step = global_step
    # Also update file.
    with tf.io.gfile.GFile(self.metric_filename, mode='w') as summary_file:
      summary_file.write(
          ''.join((f'Checkpoint at step={self._global_step}',
                   f'\t{self._metric_partition} ',
                   f'{self._metric_name}: ',
                   f'{self._metric_value};\n')))

  @property
  def global_step(self) -> int:
    return self._global_step

  @property
  def metric_filename(self) -> str:
    return os.path.join(self._dir_name,
                        self._metric_name + '-tracker')

  @property
  def metric_value(self) -> float:
    return self._metric_value

  def update(self, value: float, global_step: int) -> None:
    """Update metric value."""
    if self._metric_value != value:
      # Remove existing checkpoint, if there.
      # TODO(ciprianchelba): Make this checkpoint removal process more
      # efficient, since this has been an issue with TensorStore in the
      # past for checkpoints with many small files.
      # TODO(ciprianchelba): Update the path to include global-step with
      # leading zeros, once TensorStore-base checkpoint will be used.
      checkpoint_assets = os.path.join(self._dir_name,
                                       f'checkpoint_{self._global_step}')
      if tf.io.gfile.exists(checkpoint_assets):
        logging.info('Removing existing checkpoint %s.', checkpoint_assets)
        tf.io.gfile.rmtree(checkpoint_assets)
      # Then set the new best values.
      self._set(value, global_step)
