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

#! /bin/bash
# Assumes you are using a SLURM cluster. Edit flags under --multiprocess_gpu below to suit your setup

TFDS_DATA_DIR=$1
VOCAB_PATH=$2
LOG_DIR=${3:-"test_logdir"}

export VOCAB_PATH=$VOCAB_PATH

mkdir -p ${LOG_DIR}
python3 /pax/paxml/paxml/main.py \
    --job_log_dir=${LOG_DIR} \
    --exp=paxml.contrib.gpu.scripts_gpu.configs.Pile126M \
    --tfds_data_dir=$TFDS_DATA_DIR \
    --multiprocess_gpu \
    --server_addr=${SLURM_LAUNCH_NODE_IPADDR}:12345 \
    --num_hosts=$SLURM_NTASKS \
    --host_idx=$SLURM_PROCID \
    --alsologtostderr

