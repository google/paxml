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

"""Provides a utility function for rendering configs with normalization."""

from typing import Optional

from fiddle import graphviz as fiddle_graphviz
import graphviz
from paxml.tools.fiddle import config_normalization


def render(
    config,
    *,
    max_depth: Optional[int] = 4,
    max_str_length: Optional[int] = 100,
    remove_defaults: bool = True,
    convert_dataclasses: bool = True,
    remove_sharding_annotations: bool = False,
    unshare_sharding_config: bool = True,
) -> graphviz.Graph:
  """Renders a config with normalization.

  Args:
    config: The config to render.
    max_depth: The maximum depth of the rendered graph.
    max_str_length: The maximum length of the rendered strings.
    remove_defaults: Whether to remove default values. Often with Pax configs,
      dataclass field defaulting magic means that you get large, expanded
      templates that may actually be unused or equal to their default values.
    convert_dataclasses: Whether to convert dataclass instances to configs. This
      will only be applied if the dataclasses do not have __post_init__
      functions, as __post_init__ can obscure the initial call values.
    remove_sharding_annotations: Whether to remove sharding annotations.
    unshare_sharding_config: If remove_sharding_annotations=False, whether to
      unshare values in sharding configuration. If
      remove_sharding_annotations=True, this should be False.

  Returns:
    A rendered graph.
  """
  normalizer = config_normalization.ConfigNormalizer(
      remove_defaults=remove_defaults,
      convert_dataclasses=convert_dataclasses,
      remove_sharding_annotations=remove_sharding_annotations,
      unshare_sharding_config=unshare_sharding_config,
  )
  config = normalizer(config)
  return fiddle_graphviz.render(
      config=config, max_depth=max_depth, max_str_length=max_str_length
  )
