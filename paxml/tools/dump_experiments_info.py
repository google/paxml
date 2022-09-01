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

"""A simple utility to print info about all experiments."""

import inspect
import json
from absl import app
from absl import flags
from paxml import experiment_registry
import tensorflow.compat.v2 as tf

_OUTPUT_FILE = flags.DEFINE_string(
    "output_file",
    None,
    "Output filename (usually should end with .json)",
    required=True)


def get_experiment_info(experiment_cls):
  module = inspect.getmodule(experiment_cls).__name__
  name = experiment_cls.__qualname__
  return {"qualified_name": f"{module}.{name}"}


def main(unused_argv) -> None:
  all_classes = set(experiment_registry.get_all().values())
  infos = list(map(get_experiment_info, all_classes))

  with tf.io.gfile.GFile(_OUTPUT_FILE.value, "w") as f:
    json.dump(infos, f, indent=2)


if __name__ == "__main__":
  app.run(main)
