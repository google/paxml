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
set -eou pipefail

VOCAB_PATH=${VOCAB_PATH:-}
TFDS_DATA_DIR=${TFDS_DATA_DIR:-}
EVAL_ONLY=${EVAL_ONLY:-0}

if [[ -z "$VOCAB_PATH" || -z "$TFDS_DATA_DIR" ]]; then
    echo "Need to set both VOCAB_PATH and TFDS_DATA_DIR for the script."
    exit 1
fi

LOG_DIR=${LOG_DIR:-/tmp/llama}
mkdir -p $LOG_DIR/logs

USE_MULTIPROCESS=${USE_MULTIPROCESS:-"0"}
MULTIPROCESS_FLAGS=""
if [[ "$USE_MULTIPROCESS" = "1" ]]; then
    MULTIPROCESS_FLAGS="--multiprocess_gpu"
fi

USE_LORA=${USE_LORA:-0}
LORA_FLAGS=""
HYPERPARAM_FLAGS=""
if [[ "$USE_LORA" = "1" ]]; then
    LORA_RANK=${LORA_RANK:-32}
    LORA_FLAGS+="--fdl.USE_LORA=True \
        --fdl.LORA_RANK=${LORA_RANK:-32} \
        --fdl.LORA_TARGET_LAYERS=\"${LORA_TARGET_LAYERS:-all}\""

    ## single node
    HYPERPARAM_FLAGS="--fdl.LEARNING_RATE=${LEARNING_RATE:-1e-4} \
        --fdl.LR_SCHEDULE=\"${LR_SCHEDULE:-constant}\" \
        --fdl.MAX_STEPS=${MAX_STEPS:-600} \
        --fdl.PERCORE_BATCH_SIZE=${PERCORE_BATCH_SIZE:-2} \
        --fdl.MAX_SEQ_LEN=${MAX_SEQ_LEN:-4096} \
        --fdl.ICI_MESH_SHAPE=${ICI_MESH_SHAPE:-[8,1,1]} \
        --fdl.DCN_MESH_SHAPE=${DCN_MESH_SHAPE:-[1,1,1]} "
else
    ## full SFT requires 2 nodes
    HYPERPARAM_FLAGS="--fdl.LEARNING_RATE=${LEARNING_RATE:-1e-6} \
        --fdl.LR_SCHEDULE=\"${LR_SCHEDULE:-linear_rampup_cosine_decay}\" \
        --fdl.MAX_STEPS=${MAX_STEPS:-1000} \
        --fdl.LR_COS_WARMUP=${LR_COS_WARMUP:-500} \
        --fdl.PERCORE_BATCH_SIZE=${PERCORE_BATCH_SIZE:-2} \
        --fdl.MAX_SEQ_LEN=${MAX_SEQ_LEN:-4096} \
        --fdl.ICI_MESH_SHAPE=${ICI_MESH_SHAPE:-[1,8,1]} \
        --fdl.DCN_MESH_SHAPE=${DCN_MESH_SHAPE:-[1,2,1]} "
fi

export VOCAB_PATH=$VOCAB_PATH

export NCCL_IB_SL=0
export NCCL_PROTO=LL128

export NVTE_FWD_LAYERNORM_SM_MARGIN=4
export NVTE_BWD_LAYERNORM_SM_MARGIN=4

## NOTE: only ENABLE_TE=0 is supported by default. Please see
## https://github.com/NVIDIA/JAX-Toolbox/tree/main/rosetta/rosetta/projects/pax
## for information about how to enable transformer engine (TE) in your runs
export NVTE_FUSED_ATTN=${NVTE_FUSED_ATTN:-0}
export ENABLE_TE=${ENABLE_TE:-0}

BASE_XLA_FLAGS=${BASE_XLA_FLAGS:-"\
	--xla_gpu_enable_latency_hiding_scheduler=true \
	--xla_gpu_enable_triton_gemm=false \
	--xla_gpu_enable_highest_priority_async_stream=true \
	--xla_gpu_all_reduce_combine_threshold_bytes=51200 \
	--xla_gpu_graph_level=0"}
export XLA_FLAGS="$BASE_XLA_FLAGS ${XLA_FLAGS:-}"

ENABLE_FP8=0
JOB_DIR=$LOG_DIR/llama-${CONFIG:-LLaMA7B}-mbs-${PERCORE_BATCH_SIZE:-2}-te-${ENABLE_TE}-lora-${USE_LORA}
LOG_PATH=$LOG_DIR/logs/llama-${CONFIG:-LLaMA7B}-mbs-${PERCORE_BATCH_SIZE:-2}-te-${ENABLE_TE}-lora-${USE_LORA}.txt

if [[ -d "$JOB_DIR/checkpoints" ]]; then
    echo "WARNING: \"checkpoints\" directory already exists in $JOB_DIR. Training will resume from this directory."
fi


CHECKPOINT_TO_RESTORE=${CHECKPOINT_RESTORE_PATH:-}
if [[ ! -z $CHECKPOINT_TO_RESTORE ]]; then
    CHECKPOINT_TO_RESTORE+="/checkpoints"
fi

# Start training
if [[ "$EVAL_ONLY" = "0" ]]; then
  mkdir -p $JOB_DIR
  ENABLE_TE=$ENABLE_TE ENABLE_FP8=$ENABLE_FP8 NVTE_FUSED_ATTN=${NVTE_FUSED_ATTN:-1} XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.85} python3 -u -m paxml.main \
      --job_log_dir=$JOB_DIR \
      --tfds_data_dir=$TFDS_DATA_DIR \
      --fdl_config=paxml.contrib.gpu.scripts_gpu.configs.${CONFIG:-LLaMA7B} \
      --fdl.CHECKPOINT_RESTORE_PATH="\"$CHECKPOINT_TO_RESTORE\"" \
      --fdl.USE_REPEATED_LAYER=${USE_REPEATED_LAYER:-False} \
      $LORA_FLAGS \
      $HYPERPARAM_FLAGS \
      $MULTIPROCESS_FLAGS \
      --alsologtostderr 2>&1 | tee $LOG_PATH
fi

EVAL_CHECKPOINT=$JOB_DIR
if [[  "$EVAL_ONLY" = "1" ]]; then
    EVAL_CHECKPOINT=$CHECKPOINT_RESTORE_PATH
fi

# Start eval
ENABLE_TE=$ENABLE_TE ENABLE_FP8=$ENABLE_FP8 NVTE_FUSED_ATTN=${NVTE_FUSED_ATTN:-1} XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.85} python3 -u -m paxml.main \
    --job_log_dir=$EVAL_CHECKPOINT \
    --tfds_data_dir=$TFDS_DATA_DIR \
    --fdl_config=paxml.contrib.gpu.scripts_gpu.configs.${CONFIG:-LLaMA7B} \
    --fdl.USE_REPEATED_LAYER=${USE_REPEATED_LAYER:-False} \
    $LORA_FLAGS \
    $HYPERPARAM_FLAGS \
    $MULTIPROCESS_FLAGS \
    --mode="eval" \
    --alsologtostderr 2>&1 | tee -a $LOG_PATH
