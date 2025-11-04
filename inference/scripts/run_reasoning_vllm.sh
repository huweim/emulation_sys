
OUTPUT_DIR="./output/output_$(date +%m%d)"
mkdir -p "$OUTPUT_DIR"
MODEL_NAME="DeepSeek-R1-Distill-Qwen-1.5B"
LOG_FILE="$OUTPUT_DIR/${MODEL_NAME}_$(date +%m%d%H%M).log"

# config from eval/lighteval/model_configs/vllm_model_config.yaml
python -m inference.inference_test_v2 \
  --model_path /mnt/model/$MODEL_NAME \
  --task AIME-90 \
  --batch_size 4 \
  --quantize \
  --quant-mode pseudo \
  --eval_lib lighteval \
  --nvfp \
  --backend vllm \
  2>&1 | tee "$LOG_FILE"