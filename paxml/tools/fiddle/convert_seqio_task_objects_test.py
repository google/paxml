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

"""Tests for convert_seqio_task_objects."""

from absl.testing import absltest
import fiddle as fdl
from paxml.tools.fiddle import convert_seqio_task_objects
from praxis import pax_fiddle
import seqio
from seqio import test_utils


class ConvertSeqioTaskObjectsTest(test_utils.FakeTaskTest):

  def test_convert_seqio_tasks(self):
    config = {"task": seqio.get_mixture_or_task("tfds_task")}
    transformed = convert_seqio_task_objects.convert_seqio_tasks(config=config)
    self.assertIsInstance(transformed["task"], pax_fiddle.Config)
    self.assertEqual(
        fdl.ordered_arguments(transformed["task"]),  # pytype: disable=wrong-arg-types
        {"task_or_mixture_name": "tfds_task"},
    )


if __name__ == "__main__":
  absltest.main()
