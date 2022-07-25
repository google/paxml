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

"""Dumps the input specs of an experiment config."""

from collections.abc import Sequence

from absl import app
from absl import flags
from paxml import experiment_registry
from paxml.tools import dump_input_specs_lib
import tensorflow.compat.v2 as tf

_EXP = flags.DEFINE_string('exp', None, 'A registered experiment name.')
_OUTPUT_FILENAME = flags.DEFINE_string(
    'output_filename', None, 'Output filename for dumping the input_specs.')

FLAGS = flags.FLAGS


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  experiment_config = experiment_registry.get(_EXP.value)()

  specs = dump_input_specs_lib.extract_input_specs(experiment_config)
  out_str = dump_input_specs_lib.specs_to_string(FLAGS.exp, specs)
  with tf.io.gfile.GFile(_OUTPUT_FILENAME.value, 'w') as fout:
    fout.write(out_str)
  print(out_str)


if __name__ == '__main__':
  flags.mark_flags_as_required(['exp', 'output_filename'])
  app.run(main)
