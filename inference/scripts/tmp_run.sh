#!/bin/bash
MODEL_PATH=${1:-"/mnt/model/DeepSeek-R1-Distill-Qwen-1.5B"}
TASKS=${2:-"arc_challenge"}
BATCH_SIZE=${3:-"16"}
SHOTS=${4:-"0"}
QUANT_MODE=${5:-"pseudo"}
EVAL_LIB=${6:-"lm-eval"}

OUTPUT_NAME=DeepSeek-R1-Distill
OUTPUT_DIR="output/$OUTPUT_NAME"

mkdir -p $OUTPUT_DIR

python -m inference_test_v2 \
  --model_path $MODEL_PATH \
  --task $TASKS \
  --dtype fp16 \
  --batch_size $BATCH_SIZE \
  --quantize \
  --quant-mode $QUANT_MODE \
  --eval_lib $EVAL_LIB \
  --nvfp \
| tee $OUTPUT_DIR/${OUTPUT_NAME}_${TASKS}_${SHOTS}shots_${QUANT_MODE}_$(date +%m%d%H%M).log 2>&1