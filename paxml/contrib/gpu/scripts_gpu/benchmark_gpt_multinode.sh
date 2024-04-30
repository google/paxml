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
set -u

CONFIG=${1:-"Synthetic126M"}
PREC=${2:-"bfloat16"}        # Precision (float32, bfloat16)
NUM_GPUS=${3:-8}      # Number of GPUs (1, 2, 4, 8)
PERCORE_BATCH_SIZE=${4:-4}
LOG_DIR=${5:-"test_logdir"}

### vocab path is unused. Just set to avoid assertion error
export VOCAB_PATH=gs://t5-data/vocabs/cc_all.32000.100extra/sentencepiece.model

export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.85}
BASE_XLA_FLAGS=${BASE_XLA_FLAGS:-"--xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_enable_triton_gemm=false
                       --xla_gpu_enable_async_all_gather=true
                       --xla_gpu_enable_async_reduce_scatter=true  --xla_gpu_enable_highest_priority_async_stream=true
                       --xla_gpu_enable_triton_softmax_fusion=false  --xla_gpu_all_reduce_combine_threshold_bytes=51200
                       --xla_gpu_graph_level=0 --xla_gpu_enable_async_all_reduce=true"}
export XLA_FLAGS="$BASE_XLA_FLAGS ${XLA_FLAGS:-}"


mkdir -p $LOG_DIR
python3 -u -m paxml.main \
    --job_log_dir=$LOG_DIR \
    --fdl_config=paxml.contrib.gpu.scripts_gpu.configs.${CONFIG} \
    --fdl.FPROP_DTYPE=\"${PREC}\" \
    --fdl.PERCORE_BATCH_SIZE=$PERCORE_BATCH_SIZE \
    --multiprocess_gpu \
    --server_addr=${SLURM_LAUNCH_NODE_IPADDR}:12345 \
    --num_hosts=$SLURM_NTASKS \
    --host_idx=$SLURM_PROCID \
    --enable_checkpoint_saving=False \
    --fdl.MAX_STEPS=100 \
    --alsologtostderr

