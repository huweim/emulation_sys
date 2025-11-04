OUTPUT_DIR="./output/output_$(date +%m%d)"
mkdir -p "$OUTPUT_DIR"
MODEL_NAME="llama-2-7b-hf"
LOG_FILE="$OUTPUT_DIR/${MODEL_NAME}_$(date +%m%d%H%M).log"

python -m inference.inference_test_v2 \
  --model_path /mnt/model/$MODEL_NAME \
  --task wikitext \
  --batch_size 32 \
  2>&1 | tee "$LOG_FILE"

python -m inference.inference_test_v2 \
  --model_path /mnt/model/$MODEL_NAME \
  --task wikitext \
  --batch_size 32 \
  --quantize \
  2>&1 | tee "$LOG_FILE"

python -m inference.inference_test_v2 \
  --model_path /mnt/model/$MODEL_NAME \
  --use_prompt \
  --batch_size 32 \
  --quantize \
  2>&1 | tee "$LOG_FILE"
