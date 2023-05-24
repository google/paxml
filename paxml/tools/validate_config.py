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

r"""A simple utility to validate an experiment config.

The binary target `:validate_config` is defined by `pax_targets()` in the `BUILD`
file.

Example commandline:
bazel run //PATH/TO/PAX/TARGETS:validate_config -- \
    --exp=lm1b.Lm1bTransformerL32H8kSPMD8x8Repeat \
    --completeness=light
"""

from paxml.tools import validate_config_lib


if __name__ == '__main__':
  validate_config_lib.main()

