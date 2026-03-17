OUTPUT_DIR="./output/output_$(date +%m%d)"
mkdir -p "$OUTPUT_DIR"
MODEL_NAME="llama-2-7b-hf"

LOG_FILE="$OUTPUT_DIR/${MODEL_NAME}_emulation_$(date +%m%d%H%M).log"
python -m inference.inference_test \
  --model_path /mnt/model/llama-2-7b-hf \
  --task arc_challenge,boolq \
  --batch_size 16 \
  --quantize \
  --quant-mode emulation \
  --nvfp \
  --dtype fp16 \
  --limit 300 \
  | tee "$LOG_FILE"

LOG_FILE="$OUTPUT_DIR/${MODEL_NAME}_real_$(date +%m%d%H%M).log"
python -m inference.inference_test \
  --model_path /mnt/model/llama-2-7b-hf \
  --task arc_challenge,boolq \
  --batch_size 16 \
  --quantize \
  --quant-mode real \
  --nvfp \
  --dtype fp16 \
  --limit 300 \
  | tee "$LOG_FILE"

LOG_FILE="$OUTPUT_DIR/${MODEL_NAME}_pseudo_$(date +%m%d%H%M).log"
python -m inference.inference_test \
  --model_path /mnt/model/llama-2-7b-hf \
  --task arc_challenge,boolq \
  --batch_size 16 \
  --quantize \
  --quant-mode pseudo \
  --nvfp \
  --dtype fp16 \
  --limit 300 \
  | tee "$LOG_FILE"