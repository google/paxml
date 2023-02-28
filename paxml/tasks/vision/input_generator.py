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

"""Input generator for image data."""

from typing import Any, Dict

from absl import logging
from lingvo.core import base_input_generator
from lingvo.core import generic_input
from lingvo.core import py_utils
from paxml.tasks.vision import resnet_preprocessing
import tensorflow.compat.v2 as tf

TRAIN_EXAMPLES = 1281167  # The size of ImageNet training dataset.
EVAL_EXAMPLES = 50000  # The size of ImageNet validation dataset.
IMAGENET_DATA_DIR = '/PATH/TO/IMAGENET_DATA/'


class BaseImageNetInput(base_input_generator.BaseInputGeneratorFromFiles):
  """Base imagenet input generator.

  Input batch (b: batch size, h: height, w: width, d: depth):
    image: Preprocessed images. [b, h, w, d].
    label_probs: Labels. [b, num_classes].
    weight: [b]. weight[i] is 1.0 if i-th sample is considered to
      be a real example. Otherwise, weight[i] is 0.0.
  """

  @classmethod
  def Params(cls):
    """Defaults params."""
    p = super().Params()
    p.Define('num_classes', 1000, 'An integer as number of classes.')
    p.Define('data_shape', (224, 224, 3),
             'A tuple of 3 ints (height, weight, depth).')
    p.Define('label_id_offset', -1,
             'Shifts the label ids in the raw data by this amount.')
    p.batch_size = 128
    p.num_batcher_threads = 64
    p.file_parallelism = 64
    p.use_per_host_infeed = True
    p.file_random_seed = 0
    return p

  def _DataSourceFromFilePattern(self,
                                 file_pattern: Any,
                                 input_source_weights: Any = None,
                                 input_source_id_offset: Any = None,
                                 **extra_input_kwargs: Dict[str, Dict[Any,
                                                                      Any]]):

    def proc(record):
      """Process a tf record into tensors."""
      keys_to_features = {
          'image/encoded':
              tf.io.FixedLenFeature((), tf.string, default_value=''),
          'image/class/label':
              tf.io.FixedLenFeature([], dtype=tf.int64, default_value=-1),
      }

      features = tf.io.parse_single_example(record, keys_to_features)

      image = features['image/encoded']

      # Preprocess the image to the train/eval.
      image = self._preprocess(image)
      label = tf.cast(
          tf.reshape(features['image/class/label'], shape=[]), dtype=tf.int32)
      return [image, label], 1  # batch key is always 1.

    (images, labels), bucket_keys = generic_input.GenericInput(
        processor=proc, file_pattern=file_pattern, **self.CommonInputOpArgs())

    p = self.params

    logging.info('input image shape: %s', p.data_shape)
    if p.fprop_dtype and p.fprop_dtype != p.dtype:
      images = tf.cast(images, p.fprop_dtype)

    n = tf.shape(labels)[0]
    b, (h, w, d) = self.InfeedBatchSize(), p.data_shape

    labels += p.label_id_offset
    label_probs = tf.one_hot(labels, p.num_classes)
    if self.do_eval:
      return py_utils.NestedMap(
          bucket_keys=py_utils.PadOrTrimTo(bucket_keys, [b]),
          image=py_utils.PadOrTrimTo(images, [b, h, w, d]),
          label_probs=py_utils.PadOrTrimTo(label_probs, [b, p.num_classes]),
          weight=py_utils.PadOrTrimTo(tf.ones([n]), [b]))
    else:
      return py_utils.NestedMap(
          bucket_keys=py_utils.PadOrTrimTo(bucket_keys, [b]),
          image=py_utils.PadOrTrimTo(images, [b, h, w, d]),
          label_probs=py_utils.PadOrTrimTo(label_probs, [b, p.num_classes]),
          weight=py_utils.PadOrTrimTo(tf.ones([n]), [b]))

  def _preprocess(self, image):
    """Process image and bounding box to return an image tensor."""
    p = self.params
    return resnet_preprocessing.preprocess_image(
        image, self._is_training(), image_size=p.data_shape[0])

  def _is_training(self):
    raise NotImplementedError('Abstract method of %s' % self)

  def ImageBytesToBatch(self, image_bytes):
    """Returns an input batch containing the image_bytes."""
    image = self._preprocess(image_bytes)
    # Adds dim 0
    image = image[tf.newaxis, ...]
    return py_utils.NestedMap(image=image)


class ImageNetTrain(BaseImageNetInput):
  """Imagenet train set."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.file_pattern = (f'tfrecord:{IMAGENET_DATA_DIR}train-*-of-*')
    p.num_samples = TRAIN_EXAMPLES  # The size of ImageNet training dataset.
    return p

  def _is_training(self):
    return True


class ImageNetValidation(BaseImageNetInput):
  """Imagenet validation set."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.file_pattern = (f'tfrecord:{IMAGENET_DATA_DIR}validation-*-of-*')
    p.num_samples = EVAL_EXAMPLES  # The size of ImageNet validation dataset.
    return p

  def _is_training(self):
    return False
