#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash inference/scripts/run_lmeval_acc.sh [MODEL_NAME] [TASKS] [MODE] [BATCH_SIZE] [DTYPE]
#
# Example:
#   bash inference/scripts/run_lmeval_acc.sh llama-2-7b-hf arc_challenge,boolq emulation 16 fp16

MODEL_NAME="${1:-llama-2-7b-hf}"
TASKS="${2:-arc_challenge}"
MODE="${3:-emulation}"     # real | pseudo | emulation
BATCH_SIZE="${4:-16}"
DTYPE="${5:-fp16}"

MODEL_ROOT="${MODEL_ROOT:-/mnt/model}"
OUTPUT_DIR="${OUTPUT_DIR:-./output/output_$(date +%m%d)}"
mkdir -p "${OUTPUT_DIR}"

case "${MODE}" in
  real|pseudo|emulation) ;;
  *)
    echo "[Error] MODE must be one of: real, pseudo, emulation"
    exit 1
    ;;
esac

TASK_TAG="${TASKS//,/+}"
LOG_FILE="${OUTPUT_DIR}/${MODEL_NAME}_${MODE}_${TASK_TAG}_bch${BATCH_SIZE}_$(date +%m%d%H%M).log"

CMD=(
  python -m inference.inference_test
  --model_path "${MODEL_ROOT}/${MODEL_NAME}"
  --tasks "${TASKS}"
  --batch_size "${BATCH_SIZE}"
  --quantize
  --quant-mode "${MODE}"
  --nvfp
  --dtype "${DTYPE}"
)

# Emulation mode uses Triton acceleration by default.
if [[ "${MODE}" == "emulation" ]]; then
  CMD+=(--use-triton-emu)
fi

echo "[Run] Model=${MODEL_NAME} Tasks=${TASKS} Mode=${MODE} Batch=${BATCH_SIZE} DType=${DTYPE}"
echo "[Run] Log=${LOG_FILE}"
"${CMD[@]}" 2>&1 | tee "${LOG_FILE}"
