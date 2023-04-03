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

#!/bin/bash

set -x

CONTAINER=${1}
echo $CONTAINER
docker pull $CONTAINER

DATASET_PATH=${2} 
VOCAB_PATH=${3}

## !! Uncomment this to add a custom path to workspace dir !!##
## By default `<CURRENT_WORKING_DIRECTORY>/workspace` is selected
# WORKSPACE_PATH=<ADD CUSTOM PATH TO `workspace` dir>

docker run -ti --runtime=nvidia --net=host --ipc=host -v ${DATASET_PATH}:/pax/paxml/datasets -v ${WORKSPACE_PATH:-${PWD}/workspace}:/pax/paxml/workspace -v ${VOCAB_PATH}:/pax/paxml/vocab --privileged $CONTAINER /bin/bash

set +x
