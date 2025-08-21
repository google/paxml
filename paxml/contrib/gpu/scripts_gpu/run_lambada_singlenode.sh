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
set -u
set -o pipefail

TFDS_DATA_DIR=$1
VOCAB_PATH=$2
PREC=${3:-"bfloat16"}        # Precision (float32, bfloat16)
NUM_GPUS=${4:-8}      # Number of GPUs (1, 2, 4, 8)
PERCORE_BATCH_SIZE=${5:-4}
### path to pretrained log_dir
LOG_DIR=$6

export VOCAB_PATH=$VOCAB_PATH
BASE_XLA_FLAGS=${BASE_XLA_FLAGS:-"--xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_enable_triton_gemm=false
                       --xla_gpu_enable_highest_priority_async_stream=true
                       --xla_gpu_all_reduce_combine_threshold_bytes=51200
                       --xla_gpu_enable_command_buffer=''"}
export XLA_FLAGS="$BASE_XLA_FLAGS ${XLA_FLAGS:-}"


mkdir -p ${LOG_DIR}
python3 -u -m paxml.main \
    --job_log_dir=${LOG_DIR} \
    --fdl_config=paxml.contrib.gpu.scripts_gpu.configs.Lambada126M \
    --fdl.FPROP_DTYPE=\"${PREC}\" \
    --fdl.ICI_MESH_SHAPE="[${NUM_GPUS}, 1, 1]" \
    --fdl.DCN_MESH_SHAPE="[1,1,1]" \
    --fdl.PERCORE_BATCH_SIZE=$PERCORE_BATCH_SIZE \
    --tfds_data_dir=$TFDS_DATA_DIR \
    --mode='eval' \
    --alsologtostderr \
    2>&1 | tee ${LOG_DIR}/lambada_output.log

EXP_STATUS=$?

if [ $EXP_STATUS != 0 ]; then
  echo "Run failed"
else
  echo "Run succeeded!"
fi

echo Output written to ${LOG_DIR}/lambada_output.log
