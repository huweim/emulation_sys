#!/usr/bin/env bash
set -euo pipefail

source /home/wmhu/miniconda3/etc/profile.d/conda.sh
conda activate llm_inference

REPO_ROOT="/home/wmhu/emulation_workspace/numerical_emulation_system"
cd "${REPO_ROOT}"

DEVICE_LABEL="${1:-auto}"
ITERS="${2:-100}"
WARMUP="${3:-20}"
RUN_REAL="${4:-auto}"   # auto | on | off

M="${M:-512}"
N="${N:-2048}"
K_VALUES="${K_VALUES:-512,1024,2048,4096,8192}"
M_CHUNK_SIZE="${M_CHUNK_SIZE:-512}"
OUTPUT_DIR="${OUTPUT_DIR:-output}"

mkdir -p "${OUTPUT_DIR}"

case "${RUN_REAL}" in
  auto|on|off) ;;
  *)
    echo "[Error] RUN_REAL must be one of: auto, on, off"
    exit 1
    ;;
esac

if [[ "${DEVICE_LABEL}" == "auto" ]]; then
  DEVICE_LABEL="$(python - <<'PY'
import torch
name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
san = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name).strip("_")
print(san or "gpu")
PY
)"
fi

LOG_FILE="${OUTPUT_DIR}/${DEVICE_LABEL}_gemm_profile_table3_$(date +%m%d%H%M).log"

CMD=(
  python inference/quant/nvfp_kernel/gemm_profile.py
  --m "${M}"
  --n "${N}"
  --k-values "${K_VALUES}"
  --iters "${ITERS}"
  --warmup "${WARMUP}"
  --run-real "${RUN_REAL}"
  --device-label "${DEVICE_LABEL}"
  --m-chunk-size "${M_CHUNK_SIZE}"
  --output-dir "${OUTPUT_DIR}"
)

{
  echo "[Run] DeviceLabel=${DEVICE_LABEL} Iters=${ITERS} Warmup=${WARMUP} RunReal=${RUN_REAL}"
  echo "[Run] M=${M} N=${N} K_VALUES=${K_VALUES} M_CHUNK_SIZE=${M_CHUNK_SIZE}"
  echo "[Run] Log=${LOG_FILE}"
  "${CMD[@]}"
} 2>&1 | tee "${LOG_FILE}"
