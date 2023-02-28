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

function collect_wheels() {
  release_version=$1
  wheel_version="${release_version}"
  wheel_folder=$2
  if [ "${release_version}" != "nightly" ]; then
    wheel_version=$( echo "${release_version}" | grep -oP '\d+.\d+(.\d+)?' )
  fi

  mkdir /tmp/staging-wheels
  pushd /tmp/staging-wheels
  cp $wheel_folder/*.whl .
  rename -v "s/^paxml-(.*?)-py3/paxml-${wheel_version}+$(date -u +%Y%m%d)-py3/" *.whl
  rename -v "s/^praxis-(.*?)-py3/praxis-${wheel_version}+$(date -u +%Y%m%d)-py3/" *.whl
  popd
  mv /tmp/staging-wheels/* .
  mv $wheel_folder/*.txt .
}
