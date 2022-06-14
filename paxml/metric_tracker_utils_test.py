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

"""Tests for metric_tracker_utils."""

import pathlib
import sys

from absl import flags
from absl.testing import absltest

from paxml import metric_tracker_utils as tk_utils

FLAGS = flags.FLAGS


class MetricTrackerTest(absltest.TestCase):

  def test_tracker(self):
    tracker = tk_utils.MetricTracker(
        dir_name=FLAGS.test_tmpdir,
        metric_name='wer',
        metric_partition='test',
        initial_metric_value=sys.float_info.max)
    self.assertTrue(pathlib.Path(tracker.metric_filename).exists())
    self.assertEqual(sys.float_info.max, tracker.metric_value)

    tracker.update(value=0.13, global_step=10)
    self.assertEqual(0.13, tracker.metric_value)

    # Another tracker instance on the same metric, data partition,
    # and tracking directory will have the correct metric value as
    # stored with the state on disk.
    new_tracker = tk_utils.MetricTracker(
        dir_name=FLAGS.test_tmpdir,
        metric_name='wer',
        metric_partition='test',
        initial_metric_value=sys.float_info.max)
    self.assertEqual(0.13, new_tracker.metric_value)
    new_tracker.update(value=0.07, global_step=20)
    self.assertEqual(0.07, new_tracker.metric_value)


if __name__ == '__main__':
  absltest.main()
