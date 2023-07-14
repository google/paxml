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

"""Tests for trainer_lib."""

from absl.testing import absltest
from absl.testing import parameterized
from paxml import trainer_lib


class RunningModeTest(parameterized.TestCase):

  def test_unknown_mode(self):
    self.assertEqual(
        trainer_lib.RunningMode.detect(False, False, False),
        trainer_lib.RunningMode.UNKNOWN,
    )

  @parameterized.parameters(
      ('has_train', True, False, False),
      ('has_train', True, True, False),
      ('has_train', True, False, True),
      ('has_train', True, True, True),
      ('has_eval', False, True, False),
      ('has_eval', True, True, False),
      ('has_eval', False, True, True),
      ('has_eval', True, True, True),
      ('has_decode', False, False, True),
      ('has_decode', False, True, True),
      ('has_decode', True, False, True),
      ('has_decode', True, True, True),
  )
  def test_valid_modes(
      self, running_mode, has_train_metrics, has_eval_metrics, has_test_metrics
  ):
    self.assertTrue(
        getattr(
            trainer_lib.RunningMode.detect(
                has_train_metrics, has_eval_metrics, has_test_metrics
            ),
            running_mode,
        )
    )


if __name__ == '__main__':
  absltest.main()
