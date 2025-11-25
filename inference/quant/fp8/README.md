In real mode, x quant block is [1, 128], w quant block is [128, 128]
```shell
pip install -e .
# --- Run 1: Pseudo Quant ---
# This simulates FP8 quantization but performs GEMM in float32.
# PPL on wikitext: 5.484800338745117
python -m inference.inference_test_v2 \
  --model_path /mnt/model/llama-2-7b-hf \
  --task wikitext \
  --dtype fp16 \
  --batch_size 32 \
  --quantize \
  --quant-mode pseudo \
  --fp8

# --- Run 2: Real Quant ---
# This dequant fp8 tp bf16 and gemm(fp8 gemm is todo)
#PPL on wikitext: 5.488485336303711
python -m inference.inference_test_v2 \
  --model_path /mnt/model/llama-2-7b-hf \
  --task wikitext \
  --dtype fp16 \
  --batch_size 32 \
  --quantize \
  --quant-mode real \
  --fp8
```