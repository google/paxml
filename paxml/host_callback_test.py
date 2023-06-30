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

"""Unit tests for callback."""

from absl.testing import absltest
from paxml import host_callback


class RepositoryTest(absltest.TestCase):

  def test_namespace(self):
    regex_id = host_callback.repository('namespace_1').add('regex')
    self.assertEqual(
        host_callback.repository('namespace_1').get(regex_id), 'regex'
    )

  def test_namespace_same_object(self):
    repository1 = host_callback.repository('namespace_1')
    repository2 = host_callback.repository('namespace_1')
    self.assertIs(repository1, repository2)

  def test_namespace_different_object(self):
    repository1 = host_callback.repository('namespace_2')
    repository2 = host_callback.repository('namespace_3')
    self.assertIsNot(repository1, repository2)

  def test_add_and_pop(self):
    repository = host_callback.Repository()
    self.assertEqual(repository.size, 0)
    regex_id = repository.add('regex')
    self.assertEqual(repository.size, 1)
    self.assertTrue(repository.pop(regex_id))
    self.assertEqual(repository.size, 0)

  def test_pop_unknown_id(self):
    repository = host_callback.Repository()
    self.assertEqual(repository.size, 0)
    self.assertFalse(repository.pop(0))
    self.assertEqual(repository.size, 0)

  def test_add_and_get(self):
    repository = host_callback.Repository()
    regex_id = repository.add('regex')
    self.assertEqual(repository.get(regex_id), 'regex')

  def test_get_unknown_id(self):
    repository = host_callback.Repository()
    with self.assertRaises(KeyError):
      repository.get(0)

  def test_eviction(self):
    repository = host_callback.Repository(max_size=1)
    self.assertEqual(repository.size, 0)
    regex_ids = []
    for i in range(4):
      regex_ids.append(repository.add(f'regex_{i}'))
    self.assertEqual(repository.size, 1)

    # The last regex still remains.
    self.assertEqual(repository.get(regex_ids[-1]), 'regex_3')

    # The others were evicted.
    for regex_id in regex_ids[:-1]:
      with self.assertRaises(KeyError):
        repository.get(regex_id)


if __name__ == '__main__':
  absltest.main()
