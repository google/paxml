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

"""TrainState class for encapsulating model weights and optimizer states.

TODO(b/259501483): This is is currently aliasing Praxis train_states.py's module
symbol(s), until train_states.py gets fully migrated into Paxml.
"""

from praxis import train_states

TrainState = train_states.TrainState
