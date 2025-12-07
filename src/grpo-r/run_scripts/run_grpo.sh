#!/bin/bash

# ==================== Configuration Section ====================
WORK_DIR="JarvisArt/src/grpo-r" # Project root directory
SAVE_CKPT_PATH="JarvisArt/checkpoints/jarvisart_rl" # Checkpoint save root path
SCRIPT_PATH="JarvisArt/src/grpo-r/run_scripts/training_args.yaml"
EXP_NAME="EXP1"  # Experiment name
NPROC_PER_NODE=8
# ================================================================

# Create output directory
OUTPUT_DIR="${SAVE_CKPT_PATH}/${EXP_NAME}"
mkdir -p "$OUTPUT_DIR"

# Build training arguments from YAML file
build_train_args() {
    local args=""
    while IFS=": " read -r key value; do
        [[ -z "$key" || "$key" =~ ^[[:space:]]*# ]] && continue
        value=$(echo "$value" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' -e 's/^"//' -e 's/"$//')
        args="$args --$key $value"
    done < "${SCRIPT_PATH}"
    echo "$args"
}

TRAIN_ARGS=$(build_train_args)
LOG_DIR="${WORK_DIR}/runs/${EXP_NAME}"
mkdir -p "$LOG_DIR"

export PYTHONPATH="${WORK_DIR}/src/open-r1-multimodal/src:$PYTHONPATH"

cd "${WORK_DIR}/src/open-r1-multimodal"

echo "🚀 Starting training: ${EXP_NAME} with ${NPROC_PER_NODE} GPUs"

torchrun --nproc_per_node=${NPROC_PER_NODE} \
    src/open_r1/grpo_lr.py \
    --output_dir ${OUTPUT_DIR} \
    --run_name ${EXP_NAME} \
    --dataset-name this_is_not_used \
    --deepspeed ${WORK_DIR}/src/open-r1-multimodal/configs/zero2.json \
    $TRAIN_ARGS 2>&1 | tee "${LOG_DIR}/training.log"

echo "✅ Training finished."
