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

"""Tests for graphviz."""

from absl.testing import absltest
from paxml.tools.fiddle import graphviz_utils
from paxml.tools.fiddle import test_fixtures


class GraphvizTest(absltest.TestCase):

  def test_smoke_render(self):
    config = test_fixtures.SampleExperimentNewBaseline().experiment_fixture()
    graphviz_utils.render(config=config)


if __name__ == "__main__":
  absltest.main()
