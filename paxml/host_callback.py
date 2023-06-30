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

"""Utilities for host callbacks."""

import collections
import threading


class Repository:
  """Thread-safe container of ID-keyed strings to pass strings to host_callback.

  The intended usage is as follows:
    1. Set up a `Repository` that is accessible from both pre-processing
       and the model.  For convenience, the function `repository(namespace)`
       in this module returns a per-namespace singleton `Repository`.
       Other approaches include a module-level `Repository` variable or a
       `Repository` injected via hparams.
    2. In pre-processing, use `repository(namespace).add()` to add strings
       and pass the resulting string IDs to the model.
    3. In the model/accelerator, use `repository(namespace).get()` to fetch
       strings by ID.
    4. Set the `device_index` argument in host_callback.call match the device
       that runs pre-processing.
    5. In post-processing, use `repository(namespace).pop()` to remove strings
       by ID.  There is also a last-resort eviction policy, see `MAX_SIZE`.

  To avoid OOM when the caller does not promptly pop() the strings they add(),
  there is a limit on size.  If this grows beyond that limit, then strings are
  evicted in least-recently-added order.

  TODO(terrykoo): Define string ID using fingerprints to allow caching and
  reuse. If we do this, however, we will need to add refcounts so pop() only
  removes an ID when all usages of it have subsided.
  """

  # Maximum number of strings held in the repository of each namespace.
  MAX_SIZE = 10000

  def __init__(self, max_size: int = MAX_SIZE):
    """Creates an empty repository.

    If you use a non-singleton `Repository`, the generated string IDs might not
    be sufficiently unique.

    Args:
      max_size: Maximum number of strings to hold.
    """
    self._max_size = max_size
    self._lock = threading.Lock()
    self._string_by_id = dict()
    self._next_id_to_assign = 0
    self._next_id_to_evict = 0

  def add(self, value: str) -> int:
    """Adds new string to the mapping and returns its global ID.

    If necessary, also evicts old regexes to keep this under the maximum size.

    Args:
      value: String to add.

    Returns:
      ID of the string.  IDs are unique per LM server provided the caller uses
      the singleton.
    """
    with self._lock:
      string_id = self._next_id_to_assign
      self._next_id_to_assign += 1
      self._string_by_id[string_id] = value

      while len(self._string_by_id) > self._max_size:
        self._string_by_id.pop(self._next_id_to_evict)
        self._next_id_to_evict += 1

      return string_id

  def pop(self, value_id: int) -> bool:
    """Attempts to remove the `string_id`.

    The regex might not be removed if the `string_id` is unknown.

    Args:
      value_id: ID of the string to remove, as returned by add().

    Returns:
      True if the string was removed.
    """
    with self._lock:
      return self._string_by_id.pop(value_id, None) is not None

  def get(self, value_id: int) -> str:
    """Returns the string mapped to `string_id`.

    Args:
      value_id: ID of the string to fetch, as returned by add().

    Returns:
      String associated with the `value_id`.

    Raises:
      KeyError: If the `value_id` is not mapped.
    """
    with self._lock:
      return self._string_by_id[value_id]

  @property
  def size(self) -> int:
    """Returns the number of strings in this."""
    with self._lock:
      return len(self._string_by_id)


# This is defined and instantiated after the class, because (unlike languages
# like C++, Java, or TypeScript) Python classes don't exist until after their
# definition ends.
_global_lock = threading.Lock()
_global_repository_by_namespace = collections.defaultdict(Repository)


def repository(namespace: str) -> Repository:
  with _global_lock:
    return _global_repository_by_namespace[namespace]
