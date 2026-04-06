# 无量化
python -m inference.inference_prompt \
    --model_path /mnt/model/llama-2-7b-hf \
    --prompt "Your prompt here"

# Pseudo 量化
python -m inference.inference_prompt \
    --model_path /mnt/model/llama-2-7b-hf \
    --prompt "Your prompt here" \
    --quant-mode pseudo

# Real 量化 (需要 RTX 5090)
python -m inference.inference_prompt \
    --model_path /mnt/model/llama-2-7b-hf \
    --prompt "Your prompt here" \
    --quant-mode real

# Emulation 模式 (A100/H100 可用)
python -m inference.inference_prompt \
    --model_path /mnt/model/llama-2-7b-hf \
    --prompt "Your prompt here" \
    --quant-mode emulation

python -m inference.inference_prompt \
  --model_path /mnt/model/llama-2-7b-hf \
  --prompt "Your prompt here" \
  --quant-mode emulation \
  --use-triton-emu \
  --max_new_tokens 5