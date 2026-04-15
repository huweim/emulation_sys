#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash inference/scripts/run_gemm_profile.sh [DEVICE_LABEL] [ITERS] [WARMUP] [RUN_REAL]
#
# Examples:
#   bash inference/scripts/run_gemm_profile.sh RTX5090 1000 100 auto
#   bash inference/scripts/run_gemm_profile.sh H100 1000 100 off

DEVICE_LABEL="${1:-auto}"
ITERS="${2:-1000}"
WARMUP="${3:-100}"
RUN_REAL="${4:-auto}"   # auto | on | off

OUTPUT_DIR="${OUTPUT_DIR:-./output/output_$(date +%m%d)}"
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

LOG_FILE="${OUTPUT_DIR}/${DEVICE_LABEL}_gemm_profile_$(date +%m%d%H%M).log"

CMD=(
  python inference/quant/nvfp_kernel/gemm_profile.py
  --device-label "${DEVICE_LABEL}"
  --iters "${ITERS}"
  --warmup "${WARMUP}"
  --run-real "${RUN_REAL}"
  --output-dir "${OUTPUT_DIR}"
  --m-chunk-size 512
)

{
  echo "[Run] DeviceLabel=${DEVICE_LABEL} Iters=${ITERS} Warmup=${WARMUP} RunReal=${RUN_REAL}"
  echo "[Run] Log=${LOG_FILE}"
  "${CMD[@]}"
} 2>&1 | tee "${LOG_FILE}"
