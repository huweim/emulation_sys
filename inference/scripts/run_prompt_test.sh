# inference_test - 简单生成测试
python -m inference.inference_test \
  --model_path /mnt/model/llama-2-7b-hf \
  --quantize --quant-mode emulation --nvfp \
  --max_new_tokens 1024

python -m inference.inference_test \
  --model_path /mnt/model/llama-2-7b-hf \
  --quantize --quant-mode real --nvfp \
  --max_new_tokens 1024