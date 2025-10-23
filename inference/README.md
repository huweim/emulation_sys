

## Installation

```shell
conda create -n llm_inference python=3.12 -y
conda activate llm_inference

pip install vllm==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu128

pip install -e ./inference

git clone --recursive https://github.com/huweim/emulation_sys
# install nvfp kernel
git submodule update --init --recursive

pip install lighteval==0.9.0

```

## Run

### Difference between vllm and hf backend

Run a determinism test to compare the output consistency of the vllm and hf backends over many identical runs.

```shell
python inference_test.py --backend vllm --num-runs 1000 --model_path /mnt/model/llama-2-7b-hf  | tee vllm_1000_runs.log
python inference_test.py --backend hf --num-runs 1000 --model_path /mnt/model/llama-2-7b-hf  | tee hf_1000_runs.log

python inference_test.py --backend vllm --num-runs 10 --model_path /mnt/model/llama-2-7b-hf  | tee vllm_10_runs.log
python inference_test.py --backend hf --num-runs 10 --model_path /mnt/model/llama-2-7b-hf  | tee hf_10_runs.log
```

### Quantization Fidelity (Pseudo vs. Real)

Run a baseline PPL (Perplexity) evaluation on the wikitext task using the `lm-eval`.

```shell
MODEL_PATH=/mnt/model/llama-2-7b-hf

# Run the wikitext with lm-eval
python -m inference.inference_test_v2 \
  --model_path /mnt/model/llama-2-7b-hf \
  --task wikitext \
  --dtype fp16 \
  --batch_size 32
```

Run the NVFP quantization fidelity test. This compares the PPL score of a pseudo (simulated) quantized model against a real (kernel-level) quantized model to measure the numerical gap.

```shell
# --- Run 1: Pseudo Quant ---
# This simulates NVFP4 quantization but performs GEMM in high-precision (BF16/FP16).
python -m inference.inference_test_v2 \
  --model_path /mnt/model/llama-2-7b-hf \
  --task wikitext \
  --dtype fp16 \
  --batch_size 32 \
  --quantize \
  --quant-mode pseudo \
  --nvfp

# --- Run 2: Real Quant ---
# This uses the real low-precision NVFP4 kernels.
python -m inference.inference_test_v2 \
  --model_path /mnt/model/llama-2-7b-hf \
  --task wikitext \
  --dtype fp16 \
  --batch_size 32 \
  --quantize \
  --quant-mode real \
  --nvfp
```

## Roadmap and TODO

+ [ ] **Full real Quantization Support**: Expand beyond NVFP to include other real quantization kernels (e.g., INT8/INT4 GEMM) and measure their fidelity gap.
+ [ ] **Emulation** of real quant through C/CUDA/Python (`emulation mode`)
+ [ ] **Advanced Difference Metrics**: Add tooling to the `--use_prompt` mode to automatically compute and report:
  + Token-by-token difference (first point of divergence).
  + Logit/softmax distribution diff at the point of divergence.
  + Numerical diff of evaluation tasks (ppl, accuracy)
+ [ ] **Full vLLM Backend Integration**: Resolve dependency conflicts (e.g., vllm==0.7.0 vs lighteval==0.9.0) to enable high-speed baseline evaluation for reasoning tasks.