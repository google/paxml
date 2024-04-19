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

"""Unit tests for creating and parsing experiment class variables summaries."""

import unittest

from paxml import base_experiment
from paxml import experiment_utils
from paxml import experiment_vars_summary_parser


class TestExperimentA(base_experiment.BaseExperiment):
  INT_VAR = 0
  STR_VAR = 'A'
  TUPLE_VAR = (0, 'A')


class TestExperimentB(TestExperimentA):
  STR_VAR = 'B'
  BOOL_VAR = True


class ExperimentVarsSummaryTest(unittest.TestCase):

  def test_create_and_parse_cls_vars_summary(self):
    summary = experiment_utils.get_cls_vars_summary(TestExperimentB)

    summary_lines = summary.splitlines()
    self.assertEqual(summary_lines[0], 'paxml.base_experiment.BaseExperiment:')
    self.assertRegex(
        summary_lines[1], '   _abc_impl: <_abc._abc_data object at .*>'
    )
    self.assertEqual(summary_lines[2], '')
    self.assertEqual(summary_lines[3], '__main__.TestExperimentA:')
    self.assertEqual(summary_lines[4], '    INT_VAR: 0')
    self.assertEqual(summary_lines[5], "    TUPLE_VAR: (0, 'A')")
    self.assertEqual(summary_lines[6], '')
    self.assertEqual(summary_lines[7], '__main__.TestExperimentB:')
    self.assertEqual(summary_lines[8], '    STR_VAR: B')
    self.assertEqual(summary_lines[9], '    BOOL_VAR: True')

    cls_vars = experiment_vars_summary_parser.parse(summary)
    self.assertCountEqual(
        cls_vars.keys(),
        ['_abc_impl', 'INT_VAR', 'STR_VAR', 'TUPLE_VAR', 'BOOL_VAR'],
    )
    self.assertEqual(cls_vars['INT_VAR'], 0)
    self.assertEqual(cls_vars['TUPLE_VAR'], (0, 'A'))
    self.assertEqual(cls_vars['STR_VAR'], 'B')
    self.assertEqual(cls_vars['BOOL_VAR'], True)


if __name__ == '__main__':
  unittest.main()
