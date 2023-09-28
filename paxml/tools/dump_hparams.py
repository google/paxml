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

r"""A simple utility to dump experiment hparams to stdout or a txt file.

The binary target `:dump_hparams` is defined by `pax_targets()` in the `BUILD`
file.

Example commandline:
python paxml/tools/dump_hparams.py \
    --exp=tasks.lm.params.lm_cloud.LmCloudTransformerAdamTest \
    --params_ofile=/tmp/bert.txt

Alternatively, omitting --params_ofile just prints to stdout, which might be
useful in shell pipelines.

You may also additionally specify --post_init_params_ofile, to write a second
file consisting of the post-init model params (this takes longer to generate):
  --post_init_params_ofile=/tmp/lm_post.txt
"""

from paxml.tools import dump_hparams_lib


if __name__ == '__main__':
  dump_hparams_lib.main()
