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

"""Tests for input_generator."""

from absl.testing import absltest
from paxml.tasks.vision import input_generator


class InputGeneratorTest(absltest.TestCase):

  def _check_data_shape(self,
                        p,
                        is_multlabels=False,
                        batch_size=8,
                        image_size=33):
    p.data_shape = (image_size, image_size, 3)
    p.batch_size = batch_size
    p.num_batcher_threads = 1
    p.file_parallelism = 1
    p.file_buffer_size = 1
    inp = p.Instantiate()
    batch = inp.GetPreprocessedInputBatch()
    b, (h, w, d) = p.batch_size, p.data_shape
    self.assertEqual(batch.image.shape, (b, h, w, d))
    self.assertEqual(batch.label_probs.shape, (b, p.num_classes))

  def testImageNetValidation(self):
    p = input_generator.ImageNetValidation.Params()
    self._check_data_shape(p)

  def testImageNetTrain(self):
    p = input_generator.ImageNetTrain.Params()
    self._check_data_shape(p)


if __name__ == '__main__':
  absltest.main()
