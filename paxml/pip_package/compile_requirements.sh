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

#!/bin/bash
# This script generates new requirements.txt for Praxis and Paxml
# It pulls the nightly build docker image, and re-compile requirements.in

set -e -x

export TMP_FOLDER="$HOME/tmp/requirements"

[ -f $TMP_FOLDER ] && rm -rf $TMP_FOLDER
mkdir -p $TMP_FOLDER
cp ../../paxml/pip_package/requirements.in $TMP_FOLDER/paxml-requirements.in
cp ../../praxis/pip_package/requirements.in $TMP_FOLDER/praxis-requirements.in
cp ./compile_requirements_helper.sh $TMP_FOLDER/
sed -i 's/praxis/#praxis/' $TMP_FOLDER/paxml-requirements.in

docker pull gcr.io/pax-on-cloud-project/paxml_nightly_3.8:latest
docker run --rm -a stdin -a stdout -a stderr -v $TMP_FOLDER:/tmp/requirements \
  --name container1 gcr.io/pax-on-cloud-project/paxml_nightly_3.8:latest \
  bash /tmp/requirements/compile_requirements_helper.sh

cp $TMP_FOLDER/paxml-requirements.txt ../../paxml/pip_package/requirements.txt
cp $TMP_FOLDER/praxis-requirements.txt ../../praxis/pip_package/requirements.txt

rm -rf $TMP_FOLDER
