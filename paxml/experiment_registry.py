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

"""A registry of experiment configurations."""
import collections
import functools
import importlib
import traceback
from typing import Dict, List, Mapping, Optional

from absl import logging

from paxml import base_experiment

BaseExperimentT = base_experiment.BaseExperimentT


def _being_reloaded() -> bool:
  """Returns whether we are being called from importlib.reload."""
  for s in traceback.extract_stack():
    if s.name == 'reload' and s.filename == importlib.__file__:
      # A conservative guess.
      return True
  return False


class _ExperimentRegistryHelper:
  """Helper class encapsulating a global registry keyed by experiment name."""

  # Global variable for the experiment configuration registry
  # A mapping from a canonical key to the BaseExperimentT class.
  _registry = {}
  _registry_tags: Dict[str, List[str]] = {}
  # A mapping from a secondary key to all matching canonical keys.
  _secondary_keys = collections.defaultdict(list)

  # If to allow exp param override. This is convenient for colab debugging
  # where reloading a exp params module would otherwise trigger duplicated
  # exp registration error.
  # e.g.
  # from paxml import experiment_registry
  # experiment_registry._ExperimentRegistryHelper._allow_overwrite = True
  _allow_overwrite = False

  @classmethod
  def custom_secondary_keys(cls, canonical_key) -> List[str]:
    """Returns the list of custom secondary keys."""
    ret = []
    parts = canonical_key.split('.')
    try:
      idx = parts.index('params')
      if idx > 0:
        del parts[idx]
        new_key = '.'.join(parts[idx - 1:])
        ret.append(new_key)
    except ValueError:
      pass
    return ret

  @classmethod
  def register(cls,
               experiment_class: Optional[BaseExperimentT] = None,
               *,
               tags: Optional[List[str]] = None,
               allow_overwrite=False):
    """Registers an experiment configuration class into the global registry.

    If allow_overwrite is True, repeated registering of an existing class
    refreshes its definition in the global registry. Otherwise re-registering
    raises an error.

    Usage example:
      @experiment_registry.register
      class MyExperiment(base_experiment.BaseExperiment):
        pass

      # Allows re-registering.
      @experiment_registry.register(allow_overwrite=True)
      class MyExperiment(base_experiment.BaseExperiment):
        pass

    Args:
      experiment_class: a BaseExperimentT class.
      tags: String tags, which can be used to mark experiments as important or
        unit-testable. Currently we do not prescribe any semantics.
      allow_overwrite: bool, whether re-register an existing class is allowed.
        It's always set to True when reloading modules.

    Returns:
      experiment_class itself, so this can be used as a class decorator, or
      a decorator.

    Raises:
      ValueError: if allow_overwrite is False and the same class (same module
      and class name) has already been registered.
    """
    if experiment_class is None:
      # decorated with @register(...)
      return functools.partial(
          cls.register, allow_overwrite=allow_overwrite, tags=tags)
    if _being_reloaded():
      # Allow overwrite when we're reloading modules. This often happens when
      # developing in notebooks.
      allow_overwrite = True

    # canonical key is the full path.
    canonical_key = (
        experiment_class.__module__ + '.' + experiment_class.__name__)
    preexisting = canonical_key in cls._registry
    if preexisting and not (cls._allow_overwrite or allow_overwrite):
      raise ValueError(f'Experiment already registered: {canonical_key}')
    cls._registry[canonical_key] = experiment_class
    cls._registry_tags[canonical_key] = list(tags or [])
    logging.info('Registered experiment `%s`%s', canonical_key,
                 ' (overwritten)' if preexisting else '')
    if preexisting:
      # No need to update secondary keys.
      return experiment_class
    # Add secondary keys, which can be any partial paths.
    secondary_keys = set()
    parts = canonical_key.split('.')
    for i in range(len(parts)):
      # Any partial path is a valid secondary key.
      secondary_keys.add('.'.join(parts[i:]))

    for k in cls.custom_secondary_keys(canonical_key):
      secondary_keys.add(k)
    for k in secondary_keys:
      cls._secondary_keys[k].append(canonical_key)
    return experiment_class

  @classmethod
  def get(cls, key: str) -> Optional[BaseExperimentT]:
    """Retrieves an experiment from the global registry from the input key.

    Args:
      key: string, a (secondary) key for the experiment. Any partial path works
        as long as it uniquely identifies the experiment.

    Returns:
      None if no match is found, or the unique BaseExperimentT that
      matches the provided key.

    Raises:
      If the provided key does not uniquely identifies an experiment.
    """
    canonical_keys = cls._secondary_keys.get(key)
    if not canonical_keys:
      return None
    if len(canonical_keys) > 1:
      matches = ', '.join(canonical_keys)
      raise ValueError(f'key={key} does not uniquely identify an experiment. '
                       f'possible matches: {matches}')
    return cls._registry.get(canonical_keys[0])

  @classmethod
  def get_registry_tags(cls, key: str) -> List[str]:
    return cls._registry_tags.get(key, [])

  @classmethod
  def get_all(cls) -> Mapping[str, BaseExperimentT]:
    """Retrieves all the experiment configurations from the global registry.

    Returns:
      A dict of {experiment canonical key, experiment class}.
    """
    return cls._registry


register = _ExperimentRegistryHelper.register

# Unless you are writing a binary target like main.py that is part of
# the BUILD macro pax_targets(), you should not be calling get().
# Explicitly import the module that defines the experiment config like a regular
# Python class instead.
get = _ExperimentRegistryHelper.get

get_all = _ExperimentRegistryHelper.get_all
get_registry_tags = _ExperimentRegistryHelper.get_registry_tags
