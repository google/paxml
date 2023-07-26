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

"""New API for baseline experiments.

The new API attempts to be more Fiddle-native, and uses some tricks to allow
overrides of both high-level and low-level settings, while remaining in Fiddle.
This is a reasonably nice formulation that allows us to have high-level settings
as shared/derived/computed values, but not need tags. We simply "lower" the
fixtures at the correct time.

Please see `baseline_experiment_test.py` for some examples.
"""

import abc
import copy
import dataclasses
import inspect
import typing
from typing import Type

import fiddle as fdl
from paxml import parameterized_experiment
from praxis import pax_fiddle


# Note: It seems that we have to choose in the base class whether to make all
# subclasses frozen or not. While we mostly mutate values when we have Fiddle
# configs of these baselines, it shouldn't be a huge issue to directly handle
# instances, so we've chosen to allow mutation.
@dataclasses.dataclass
class BaselineExperiment(metaclass=abc.ABCMeta):
  """New baseline experiment.

  Subclasses should (a) remember to use the dataclass decorator (b) provide the
  experiment_fixture() method (c) use dataclass fields for high-level settings.
  """

  @abc.abstractmethod
  def experiment_fixture(
      self,
  ) -> pax_fiddle.Config[parameterized_experiment.ParameterizedExperiment]:
    """Returns configuration for the experiment."""

  @classmethod
  def make_experiment(
      cls, **highlevel_settings
  ) -> parameterized_experiment.ParameterizedExperiment:
    """Returns a ParameterizedExperiment.

    This method is necessary for building a Fiddle config in
    `highlevel_config()` whose attributes are the dataclass fields, but has the
    desired return type. It is generally not useful to call this method by
    itself.

    Args:
      **highlevel_settings: Settings which will be used to construct an instance
        of this class.

    Returns:
      A ParameterizedExperiment.
    """
    return pax_fiddle.build(cls(**highlevel_settings).experiment_fixture())

  @classmethod
  def highlevel_config(cls, **highlevel_settings):
    """Returns the high-level configuration.

    Args:
      **highlevel_settings: Settings which will be used when constructing an
        instance of this class.
    """
    return pax_fiddle.Config(cls.make_experiment, **highlevel_settings)

  @classmethod
  def lowlevel_config(cls, **highlevel_settings):
    """Returns the low-level configuration.

    For the CLI API, one can set

    --fdl_config=path.to.MyBaseline.lowlevel_config(highlevel_setting=1234)

    and then proceed to set low-level options, like

    --fdl_set=task.train.learner.optimizer.lr_schedule.decay_end=10000

    Args:
      **highlevel_settings: Settings which will be used when constructing an
        instance of this class.
    """
    return lower(cls.highlevel_config(**highlevel_settings))


def lower(
    config: pax_fiddle.Config[parameterized_experiment.ParameterizedExperiment],
) -> pax_fiddle.Config[parameterized_experiment.ParameterizedExperiment]:
  """Converts high-level config to low-level config.

  We recommend aliasing this method in baseline configuration files, so that
  it can be conveniently referenced by the Fiddle CLI API (instead of having to
  use a fully-qualified name).

  Args:
    config: High-level configuration, which normally has attributes as fields of
      a given BaselineExperiment subclass.

  Returns:
    Low-level configuration, which normally has `task`, `training_dataset`,
    `eval_datasets`, `decoder_datasets`, and `input_specs_provider` attributes.
  """
  make_experiment = fdl.get_callable(config)
  if not inspect.ismethod(make_experiment):
    raise ValueError(
        "Expected config of BaselineExperiment.make_experiment() classmethod,"
        f" got {config}"
    )
  exp_cls = typing.cast(Type[BaselineExperiment], make_experiment.__self__)
  if not isinstance(exp_cls, type) or not issubclass(
      exp_cls, BaselineExperiment
  ):
    raise ValueError(
        "Expected config of BaselineExperiment.make_experiment() classmethod,"
        f" got {config}"
    )
  config = copy.deepcopy(config)
  pax_fiddle.update_callable(config, exp_cls)
  config = typing.cast(pax_fiddle.Config[BaselineExperiment], config)
  return pax_fiddle.build(config).experiment_fixture()
