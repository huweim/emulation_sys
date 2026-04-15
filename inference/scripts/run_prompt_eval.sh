python -m inference.inference_eval_prompt \
  --model_path /mnt/model/Qwen3-8B \
  --dataset-name wikitext \
  --dataset-config wikitext-2-raw-v1 \
  --dataset-split test \
  --dataset-index 0 \
  --min-prompt-tokens 100 \
  --max-prompt-tokens 200 \
  --max-new-tokens 64 \
  --temperature 0 \
  --quant-mode pseudo \
  --save-json ./output/token_eval_qwen3_pseudo.json

python -m inference.inference_eval_prompt \
  --model_path /mnt/model/Qwen3-8B \
  --dataset-name wikitext \
  --dataset-config wikitext-2-raw-v1 \
  --dataset-split test \
  --dataset-index 0 \
  --min-prompt-tokens 100 \
  --max-prompt-tokens 200 \
  --max-new-tokens 64 \
  --temperature 0 \
  --quant-mode real \
  --save-json ./output/token_eval_qwen3_real.json

python -m inference.inference_eval_prompt \
  --model_path /mnt/model/Qwen3-8B \
  --dataset-name wikitext \
  --dataset-config wikitext-2-raw-v1 \
  --dataset-split test \
  --dataset-index 0 \
  --min-prompt-tokens 100 \
  --max-prompt-tokens 200 \
  --max-new-tokens 64 \
  --temperature 0 \
  --quant-mode emulation \
  --use-triton-emu \
  --save-json ./output/token_eval_qwen3_emu.json

python -m inference.inference_eval_prompt \
  --model_path /mnt/model/Meta-Llama-3-8B \
  --prompt "your fixed prompt here" \
  --max-new-tokens 64 \
  --temperature 0 \
  --quant-mode real \
  --fp64-override

python -m inference.inference_eval_prompt \
  --model_path /mnt/model/Qwen3-8B \
  --dataset-name wikitext \
  --dataset-config wikitext-2-raw-v1 \
  --dataset-split test \
  --dataset-index 0 \
  --min-prompt-tokens 100 \
  --max-prompt-tokens 200 \
  --max-new-tokens 64 \
  --temperature 0 \
  --quant-mode real \
  --use-triton-emu \
  --fp64-override \
  --save-json ./output/token_eval_qwen3_emu_fp64.json