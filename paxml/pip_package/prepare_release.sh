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

# This script prepare a new release by:
# 1) update version number in setup.py and cloudbuild-release.yaml
# 2) add a new section in RELEASE.md with version and corresponding commit

set -e -x

function print_help_and_exit {
 echo "Usage: prepare_release.sh -v <paxml_version> -x <praxis_version> -d <build_date:YYYYMMDD> "
 echo "exp: bash prepare_release.sh -v 0.2.0 -x 0.2.0 -d 20221114"
 exit 0
}

while getopts "hv:d:x:" opt; do
  case $opt in
    v)
      PAXML_VERSION=${OPTARG}
      ;;
    x)
      PRAXIS_VERSION=${OPTARG}
      ;;
    d)
      BUILD_DATE=${OPTARG}
      ;;
    *)
      print_help_and_exit
      ;;
  esac
done

RELEASE_NOTE="../RELEASE.md"
RELEASE_NOTE_NEW="release_new.md"

if [[ -z "$BUILD_DATE" ]]; then
  echo "Build date is required!"
  exit 1
fi

if [[ -z "$PAXML_VERSION" ]]; then
  echo "paxml version is required!"
  exit 1
fi

echo "Build date: "$BUILD_DATE
echo "PAXML version: "$PAXML_VERSION

if [[ ! -z "$PRAXIS_VERSION" ]]; then
    sed -i "s/_PRAXIS_VERSION: '[0-9.]*'/_PRAXIS_VERSION: '$PRAXIS_VERSION'/" cloudbuild-release.yaml
fi

sed -i "s/version='[0-9.]*'/version='$PAXML_VERSION'/" setup.py
sed -i "s/_RELEASE_VERSION: '[0-9.]*'/_RELEASE_VERSION: '$PAXML_VERSION'/" cloudbuild-release.yaml
gsutil cp gs://pax-on-cloud-tpu-project/wheels/"$BUILD_DATE"/paxml_commit.txt ./
gsutil cp gs://pax-on-cloud-tpu-project/wheels/"$BUILD_DATE"/praxis_commit.txt ./
PAXML_COMMIT=$(<paxml_commit.txt)
PRAXIS_COMMIT=$(<praxis_commit.txt)
rm paxml_commit.txt
rm praxis_commit.txt
echo "PAXML_COMMIT: " $PAXML_COMMIT
[ -e $RELEASE_NOTE_NEW ] && rm $RELEASE_NOTE_NEW
echo "# Version: $PAXML_VERSION" >> $RELEASE_NOTE_NEW
echo "## Major Features and Improvements" >> $RELEASE_NOTE_NEW
echo "## Breaking changes" >> $RELEASE_NOTE_NEW
echo "## Deprecations" >> $RELEASE_NOTE_NEW
echo "## Note" >> $RELEASE_NOTE_NEW
echo "*   Version: $PAXML_VERSION" >> $RELEASE_NOTE_NEW
echo "*   Build Date: $BUILD_DATE" >> $RELEASE_NOTE_NEW
echo "*   Paxml commit: $PAXML_COMMIT" >> $RELEASE_NOTE_NEW
echo "*   Praxis version: $PRAXIS_VERSION" >> $RELEASE_NOTE_NEW
echo "*   Praxis commit: $PRAXIS_COMMIT" >> $RELEASE_NOTE_NEW
RELEASE_NOTE_TMP="RELEASE.tmp.md"
cat $RELEASE_NOTE_NEW $RELEASE_NOTE >> $RELEASE_NOTE_TMP
rm $RELEASE_NOTE_NEW
rm $RELEASE_NOTE
mv $RELEASE_NOTE_TMP $RELEASE_NOTE
