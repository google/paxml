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

"""High-level API for doing common normalization to Pax configs."""

import dataclasses
from typing import TypeVar

import fiddle as fdl
from fiddle.experimental import auto_config
from fiddle.experimental import dataclasses as fdl_dataclasses
from fiddle.experimental import visualize
from paxml import base_task
from paxml import parameterized_experiment
from paxml.tools.fiddle import convert_seqio_task_objects as convert_seqio_task_objects_lib
from paxml.tools.fiddle import remove_sharding
from paxml.tools.fiddle import unshare_sharding
from paxml.tools.fiddle import wrap_nested_maps as wrap_nested_maps_lib
from praxis import pax_fiddle


@dataclasses.dataclass(frozen=True)
class ComponentNormalizer:
  """Mid-level API for normalizing tasks, datasets, or other configs."""

  # (Defaults for the below are in ConfigNormalizer.)
  remove_defaults: bool
  convert_dataclasses: bool
  wrap_nested_maps: bool

  def __call__(self, config: pax_fiddle.Config[base_task.BaseTask]):
    if self.remove_defaults:
      config = visualize.with_defaults_trimmed(
          config, remove_deep_defaults=True
      )
    if self.convert_dataclasses:
      config = fdl_dataclasses.convert_dataclasses_to_configs(config)
    if self.wrap_nested_maps:
      config = wrap_nested_maps_lib.wrap_nested_maps(config)
    return config


@dataclasses.dataclass(frozen=True)
class TaskNormalizer(ComponentNormalizer):
  """Mid-level API for normalizing tasks."""

  # (Defaults for the below are in ConfigNormalizer.)
  remove_sharding_annotations: bool
  unshare_sharding_config: bool

  def __call__(self, task_config: pax_fiddle.Config[base_task.BaseTask]):
    task_config = super().__call__(task_config)
    if self.remove_sharding_annotations:
      task_config = remove_sharding.remove_sharding(task_config)
    elif self.unshare_sharding_config:
      task_config = unshare_sharding.unshare_sharding(task_config)
    return task_config


_T = TypeVar("_T")


@dataclasses.dataclass(frozen=True)
class DatasetNormalizer(ComponentNormalizer):
  """Mid-level API for normalizing dataset configs."""

  # (Defaults for the below are in ConfigNormalizer.)
  convert_seqio_task_objects: bool

  def __call__(self, dataset_config: _T) -> _T:
    """Normalizes a dataset or list of datasets."""
    dataset_config = super().__call__(dataset_config)
    if self.convert_seqio_task_objects:
      dataset_config = convert_seqio_task_objects_lib.convert_seqio_tasks(
          dataset_config
      )
    return dataset_config


@dataclasses.dataclass(frozen=True)
class ExperimentNormalizer:
  """Mid-level API for normalizing experiments."""

  task_normalizer: TaskNormalizer
  dataset_normalizer: DatasetNormalizer
  remove_eval_datasets: bool
  remove_decoder_datasets: bool

  def __call__(
      self,
      experiment_config: pax_fiddle.Config[
          parameterized_experiment.ParameterizedExperiment
      ],
  ) -> pax_fiddle.Config[parameterized_experiment.ParameterizedExperiment]:
    kwargs = fdl.ordered_arguments(experiment_config)
    new_kwargs = {}
    if "task" in kwargs:
      new_kwargs["task"] = self.task_normalizer(experiment_config.task)
    if "training_dataset" in kwargs:
      new_kwargs["training_dataset"] = self.dataset_normalizer(
          experiment_config.training_dataset
      )
    # Note: The below just skips the normalizer for `remove_eval_datasets`, we
    # have to actually remove it below.
    if "eval_datasets" in kwargs and not self.remove_eval_datasets:
      new_kwargs["eval_datasets"] = self.dataset_normalizer(
          experiment_config.eval_datasets
      )
    if "decoder_datasets" in kwargs and not self.remove_decoder_datasets:
      new_kwargs["decoder_datasets"] = self.dataset_normalizer(
          experiment_config.decoder_datasets
      )

    result = fdl.copy_with(experiment_config, **new_kwargs)
    if "eval_datasets" in kwargs and self.remove_eval_datasets:
      del result.eval_datasets
    if "decoder_datasets" in kwargs and self.remove_decoder_datasets:
      del result.decoder_datasets
    return result


@dataclasses.dataclass(frozen=True)
class ConfigNormalizer:
  """Mid-level API for normalizing configs.

  For most users this will be too flexible; please see normalize(). Please see
  normalize() for attribute documentation.

  Attributes:
    remove_defaults: Whether to remove default values. Often with Pax configs,
      dataclass field defaulting magic means that you get large, expanded
      templates that may actually be unused or equal to their default values.
    convert_dataclasses: Whether to convert dataclass instances to configs. This
      will only be applied if the dataclasses do not have __post_init__
      functions, as __post_init__ can obscure the initial call values.
    wrap_nested_maps: Whether to wrap NestedMap instances in a helper function,
      to remove these custom objects from the config.
    remove_sharding_annotations: Whether to remove sharding annotations.
    unshare_sharding_config: If remove_sharding_annotations=False, whether to
      unshare values in sharding configuration. Fiddle generally retains
      information when mutables like lists or sub-config objects are shared, but
      for sharding annotations this shouldn't matter; only the values matter.
      Generated code is generally prettier when you unshare sharding configs.
      However, if you later write a test asserting equality with the original
      config, please make sure to run unshare_sharding.unshare_sharding() on the
      original config.
    convert_seqio_task_objects: Replaces SeqIO Task objects with a config object
      that will load them by name. Normalized configurations should not contain
      custom objects, only Fiddle buildables and primitives. However, if you
      have a better way of configuring SeqIO objects than using the global
      registry, please use that.
    remove_eval_datasets: Whether to remove/clear eval_datasets, even if they
      exist.
    remove_decoder_datasets: Whether to remove/clear decoder_datasets, even if
      they exist.
  """

  remove_defaults: bool = True
  convert_dataclasses: bool = True
  wrap_nested_maps: bool = True
  remove_sharding_annotations: bool = False
  unshare_sharding_config: bool = True
  convert_seqio_task_objects: bool = True
  remove_eval_datasets: bool = False
  remove_decoder_datasets: bool = False

  @auto_config.auto_config
  def experiment_normalizer(self):
    return ExperimentNormalizer(
        task_normalizer=self.task_normalizer(),
        dataset_normalizer=self.dataset_normalizer(),
        remove_eval_datasets=self.remove_eval_datasets,
        remove_decoder_datasets=self.remove_decoder_datasets,
    )

  @auto_config.auto_config
  def task_normalizer(self):
    return TaskNormalizer(
        remove_defaults=self.remove_defaults,
        convert_dataclasses=self.convert_dataclasses,
        wrap_nested_maps=self.wrap_nested_maps,
        remove_sharding_annotations=self.remove_sharding_annotations,
        unshare_sharding_config=self.unshare_sharding_config,
    )

  @auto_config.auto_config
  def dataset_normalizer(self):
    return DatasetNormalizer(
        remove_defaults=self.remove_defaults,
        convert_dataclasses=self.convert_dataclasses,
        wrap_nested_maps=self.wrap_nested_maps,
        convert_seqio_task_objects=self.convert_seqio_task_objects,
    )

  def __call__(
      self,
      config: pax_fiddle.Config[
          parameterized_experiment.ParameterizedExperiment
      ],
  ) -> pax_fiddle.Config[parameterized_experiment.ParameterizedExperiment]:
    return self.experiment_normalizer()(config)

  def lowlevel_config(self):
    return self.experiment_normalizer.as_buildable()


def noop_normalizer() -> ConfigNormalizer:
  return ConfigNormalizer(
      remove_sharding_annotations=False,
      unshare_sharding_config=False,
      remove_defaults=False,
      wrap_nested_maps=False,
      convert_seqio_task_objects=False,
  )


def default_normalizer() -> ConfigNormalizer:
  return ConfigNormalizer()


def aggressive_normalizer() -> ConfigNormalizer:
  return ConfigNormalizer(
      remove_sharding_annotations=True,
      remove_defaults=True,
      wrap_nested_maps=True,
      convert_seqio_task_objects=True,
  )
