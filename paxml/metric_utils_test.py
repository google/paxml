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

"""Tests for metric_utils."""
import os

from typing import Any, Dict, List, Tuple
# Internal unittest mock import

from absl.testing import absltest

import clu.metrics as clu_metrics
import clu.values as clu_values
import flax
import numpy as np
import numpy.testing as npt
from paxml import metric_utils
from paxml import summary_utils
# Internal platform import
import seqio

# internal runtime import


@flax.struct.dataclass
class MockMetric(clu_metrics.Metric):
  """A mock metric where all the data is empty."""

  @classmethod
  def from_model_output(cls) -> clu_metrics.Metric:
    return MockMetric()

  def merge(self, other: clu_metrics.Metric) -> clu_metrics.Metric:
    return MockMetric()

  def compute(self) -> None:
    return None

  def compute_value(self) -> clu_values.Value:
    raise NotImplementedError('Other mock metrics should define this.')


def _mock_clu_image(batch_size=None) -> clu_values.Image:
  base_shape = [12, 12, 3]
  if batch_size:
    base_shape = [batch_size] + base_shape
  return clu_values.Image(np.ones(base_shape))


def _mock_seqio_image(batch_size=None) -> seqio.metrics.Image:
  base_shape = [12, 12, 3]
  if batch_size:
    base_shape = [batch_size] + base_shape
  return seqio.metrics.Image(np.ones(base_shape))


def _mock_clu_audio(batch_size=None) -> clu_values.Audio:
  base_shape = [12, 1]  # [time, channel]
  if batch_size:
    base_shape = [batch_size] + base_shape
  return clu_values.Audio(np.ones(base_shape), sample_rate=44_100)


def _mock_seqio_audio(batch_size=None) -> seqio.metrics.Audio:
  base_shape = [12, 1]  # [time, channel]
  if batch_size:
    base_shape = [batch_size] + base_shape
  return seqio.metrics.Audio(np.ones(base_shape), sample_rate=44_100)


def _mock_clu_video(batch_size=None) -> clu_values.Summary:
  class Metadata(object):
    message: str = 'dummy metadata'

  base_shape = [5, 12, 12, 3]
  video = np.array([
      str(base_shape[1]),
      str(base_shape[2]),
      np.ones(base_shape).astype(np.uint8).tobytes()])
  video_summary = [clu_values.Summary(value=video, metadata=Metadata())]
  if batch_size is not None:
    return video_summary * batch_size
  else:
    return video_summary


class MetricUtilsTest(absltest.TestCase):

  def _test_dir(self):
    return os.path.join(absltest.get_default_test_tmpdir(), 'summary')

  def test_scalar_compute_metric_values(self):
    @flax.struct.dataclass
    class ScalarMetric(MockMetric):

      def compute_value(self) -> clu_values.Scalar:
        return clu_values.Scalar(5)

    metrics = {'test': ScalarMetric()}
    test_dir = self._test_dir()

    metric_values = metric_utils.compute_metric_values(metrics)
    self.assertIn('test', metric_values)
    self.assertEqual(metric_values['test'].value, 5)

    with summary_utils.get_summary_writer(test_dir):
      metric_utils.write_clu_metric_summaries(metric_values, step_i=0)

  def test_list_compute_metric_values(self):
    @flax.struct.dataclass
    class ScalarListMetric(MockMetric):

      def compute_value(self) -> List[Any]:
        return [
            clu_values.Scalar(5),
            clu_values.Text('hi'),
            _mock_clu_image(batch_size=None),
            _mock_clu_image(batch_size=2),
            _mock_clu_video(batch_size=None),
            _mock_clu_video(batch_size=2),
            _mock_clu_audio(batch_size=None),
            _mock_clu_audio(batch_size=2),
        ]
    metrics = {'test': ScalarListMetric()}

    metric_values = metric_utils.compute_metric_values(metrics)
    self.assertEqual(metric_values['test/test_0'].value, 5)
    self.assertEqual(metric_values['test/test_1'].value, 'hi')
    self.assertEqual(metric_values['test/test_2'].value.shape, (12, 12, 3))
    self.assertEqual(metric_values['test/test_3'].value.shape, (2, 12, 12, 3))
    self.assertLen(metric_values['test/test_4'], 1)
    self.assertEqual(metric_values['test/test_4'][0].value.shape, (3,))
    self.assertLen(metric_values['test/test_5'], 2)
    self.assertEqual(metric_values['test/test_6'].value.shape, (12, 1))
    self.assertEqual(metric_values['test/test_6'].sample_rate, 44_100)
    self.assertEqual(metric_values['test/test_7'].value.shape, (2, 12, 1))
    self.assertEqual(metric_values['test/test_7'].sample_rate, 44_100)

  def test_tuple_compute_metric_values(self):
    @flax.struct.dataclass
    class ScalarTupleMetric(MockMetric):

      def compute_value(self) -> Tuple[Any]:
        return (clu_values.Scalar(5), clu_values.Text('hi'))
    metrics = {'test': ScalarTupleMetric()}

    metric_values = metric_utils.compute_metric_values(metrics)
    self.assertEqual(metric_values['test/test_0'].value, 5)
    self.assertEqual(metric_values['test/test_1'].value, 'hi')

  def test_scalar_dict_comptue_metric_values(self):
    @flax.struct.dataclass
    class ScalarDictMetric(MockMetric):

      def compute_value(self) -> Dict[str, Any]:
        return {
            'scalar_0': clu_values.Scalar(1),
            'scalar_1': clu_values.Scalar(2),
            'text_0': clu_values.Text('test3'),
            'image_0': _mock_clu_image(None),
            'image_1': _mock_clu_image(5),
            'video_0': _mock_clu_video(None),
            'video_1': _mock_clu_video(2),
            'audio_0': _mock_clu_audio(None),
            'audio_1': _mock_clu_audio(2),
        }

    metrics = {'test': ScalarDictMetric()}

    metric_values = metric_utils.compute_metric_values(metrics)
    self.assertEqual(metric_values['test/scalar_0'].value, 1)
    self.assertEqual(metric_values['test/scalar_1'].value, 2)
    self.assertEqual(metric_values['test/text_0'].value, 'test3')
    self.assertEqual(metric_values['test/image_0'].value.shape, (12, 12, 3))
    self.assertEqual(metric_values['test/image_1'].value.shape, (5, 12, 12, 3))
    self.assertEqual(metric_values['test/video_0_0'].value.shape, (3,))
    self.assertEqual(metric_values['test/video_1_0'].value.shape, (3,))
    self.assertEqual(metric_values['test/video_1_1'].value.shape, (3,))
    self.assertEqual(metric_values['test/audio_0'].value.shape, (12, 1))
    self.assertEqual(metric_values['test/audio_0'].sample_rate, 44_100)
    self.assertEqual(metric_values['test/audio_1'].value.shape, (2, 12, 1))
    self.assertEqual(metric_values['test/audio_1'].sample_rate, 44_100)

  def test_mixed_dict_compute_metric_values(self):
    @flax.struct.dataclass
    class MixedDictMetric(MockMetric):

      def compute_value(self) -> Dict[str, Any]:
        return {
            'scalar_0': clu_values.Scalar(1),
            'list_0': [clu_values.Scalar(1), clu_values.Scalar(2)],
            'list_1': [clu_values.Scalar(1), clu_values.Text('test')],
            'tuple_0': (_mock_clu_image(None), _mock_clu_image(5)),
            'image_0': _mock_clu_image(5),
            'histogram_0': clu_values.Histogram(
                np.array([1, 2, 3]), num_buckets=2
            ),
        }

    metrics = {'test': MixedDictMetric()}
    metric_values = metric_utils.compute_metric_values(metrics)
    self.assertEqual(metric_values['test/scalar_0'].value, 1)
    # First list is two scalars.
    self.assertEqual(metric_values['test/list_0_0'].value, 1)
    self.assertEqual(metric_values['test/list_0_1'].value, 2)
    # Secont list is a scalar and text.
    self.assertEqual(metric_values['test/list_1_0'].value, 1)
    self.assertEqual(metric_values['test/list_1_1'].value, 'test')
    # Third list is two images, one unbatched, and one batched.
    self.assertEqual(metric_values['test/tuple_0_0'].value.shape, (12, 12, 3))
    self.assertEqual(
        metric_values['test/tuple_0_1'].value.shape, (5, 12, 12, 3))
    # We have an image.
    self.assertEqual(metric_values['test/image_0'].value.shape, (5, 12, 12, 3))
    # And a histogram.
    npt.assert_array_equal(metric_values['test/histogram_0'].value, [1, 2, 3])

  def test_write_clu_metric_summaries(self):
    @flax.struct.dataclass
    class MixedDictMetric(MockMetric):

      def compute_value(self) -> Dict[str, Any]:
        return {
            'scalar_0': clu_values.Scalar(1),
            'list_0': [clu_values.Scalar(1), clu_values.Scalar(2)],
            'list_1': [clu_values.Scalar(1), clu_values.Text('test')],
            'image_tuple_0': (_mock_clu_image(None), _mock_clu_image(5)),
            'image_0': _mock_clu_image(5),
            'video_0': _mock_clu_video(None),
            'video_1': _mock_clu_video(2),
            'audio_0': _mock_clu_audio(None),
            'audio_1': _mock_clu_audio(2),
            'histogram_0': clu_values.Histogram(
                np.array([1, 2, 3]), num_buckets=2
            ),
        }

    metrics = {'test': MixedDictMetric()}
    metric_values = metric_utils.compute_metric_values(metrics)
    test_dir = self._test_dir()
    with summary_utils.get_summary_writer(test_dir):
      metric_utils.write_clu_metric_summaries(metric_values, step_i=0)

  def test_write_seqio_metric_summaries(self):
    @flax.struct.dataclass
    class MixedDictMetric(MockMetric):

      def compute_value(self) -> Dict[str, Any]:
        return {
            'scalar_0': seqio.metrics.Scalar(1),
            'list_0': [seqio.metrics.Scalar(1), seqio.metrics.Scalar(2)],
            'list_1': [seqio.metrics.Scalar(1), seqio.metrics.Text('test')],
            'image_tuple_0': (_mock_seqio_image(None), _mock_seqio_image(5)),
            'image_0': _mock_seqio_image(5),
            'audio_0': _mock_seqio_audio(None),
            'audio_1': _mock_seqio_audio(2),
            'histogram_0': seqio.metrics.Histogram(
                np.array([1, 2, 3]), bins=2
            ),
        }

    metrics = {'test': MixedDictMetric()}
    metric_values = metric_utils.compute_metric_values(metrics)
    test_dir = self._test_dir()
    with summary_utils.get_summary_writer(test_dir):
      metric_utils.write_seqio_metric_summaries(
          [metric_values], step=0, metric_name_prefix='test_prefix')

  def is_float_convertible(self):
    self.assertTrue(metric_utils.is_float_convertible(0.1))
    self.assertTrue(metric_utils.is_float_convertible(np.float32(0.1)))
    self.assertTrue(metric_utils.is_float_convertible(clu_values.Scalar(0.1)))
    self.assertTrue(
        metric_utils.is_float_convertible(seqio.metrics.Scalar(0.1)))
    self.assertTrue(
        metric_utils.is_float_convertible((np.array([1.0]), np.array([0.1]))))
    self.assertTrue(
        metric_utils.is_float_convertible([(np.array([1.0]), np.array([0.1]))]))
    self.assertFalse(metric_utils.is_float_convertible('abc'))
    self.assertFalse(
        metric_utils.is_float_convertible(seqio.metrics.Text('abc')))
    self.assertFalse(metric_utils.is_float_convertible(clu_values.Text('abc')))
    self.assertFalse(
        metric_utils.is_float_convertible(
            clu_values.Histogram(np.array([1, 2, 3]))
        )
    )

  def test_as_float(self):
    self.assertEqual(metric_utils.as_float(0.2), 0.2)
    self.assertEqual(metric_utils.as_float(np.float32(1.0)), 1.0)
    self.assertEqual(metric_utils.as_float(clu_values.Scalar(0.2)), 0.2)
    self.assertEqual(
        metric_utils.as_float(clu_values.Scalar(np.float32(1.0))), 1.0)

    self.assertEqual(
        metric_utils.as_float((np.array([1.0]), np.array([0.1]))), 1.0)
    self.assertEqual(
        metric_utils.as_float([(np.array([1.0]), np.array([0.1])),
                               (np.array([3.0]), np.array([0.1]))]), 2.0)
    self.assertEqual(metric_utils.as_float((0.2, 1.0)), 0.2)

  def test_as_float_dict(self):
    self.assertEqual(metric_utils.as_float_dict({'x': 0.2}), {'x': 0.2})
    self.assertEqual(
        metric_utils.as_float_dict({'x': np.float32(1.0)}), {'x': 1.0})
    self.assertEqual(
        metric_utils.as_float_dict({
            'x': clu_values.Scalar(np.float32(1.0)),
            'y': 0.3,
            'z': clu_values.Text('abc')
        }), {
            'x': 1.0,
            'y': 0.3
        })

  def test_merge_float_dict(self):
    m1 = {'a': 1, 'b': 2}
    self.assertEqual(
        metric_utils.update_float_dict(m1, {'a': 2}), {
            'a': 2,
            'b': 2
        })
    self.assertEqual(
        metric_utils.update_float_dict(m1, {'a': 2}, prefix='x'), {
            'a': 2,
            'b': 2,
            'x/a': 2
        })

    # Internal scalar types test


if __name__ == '__main__':
  absltest.main()
