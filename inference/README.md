
```shell
conda create -n llm_inference python=3.12 -y
conda activate llm_inference

pip install torch==2.7.0 --extra-index-url https://download.pytorch.org/whl/cu128
pip install vllm==0.9.2
pip install -e .
```

## Run

### Difference between vllm and hf backend

```shell
python inference_test.py --backend vllm --num-runs 1000 --model_path /mnt/model/llama-2-7b-hf  | tee vllm_1000_runs.log
python inference_test.py --backend hf --num-runs 1000 --model_path /mnt/model/llama-2-7b-hf  | tee hf_1000_runs.log

python inference_test.py --backend vllm --num-runs 10 --model_path /mnt/model/llama-2-7b-hf  | tee vllm_10_runs.log
python inference_test.py --backend hf --num-runs 10 --model_path /mnt/model/llama-2-7b-hf  | tee hf_10_runs.log
```

### Diff between pseudo and real quant

```shell
python -m inference.inference_test_v2 \
  --model_path /mnt/model/llama-2-7b-hf \
  --task wikitext \
  --dtype fp16 \
  --batch_size 32
```

TODO: 
+ [ ] real quant implementation
+ [ ] metrics difference evaluation
  + token diff of generation
  + numerical diff of evaluation tasks (ppl, accuracy)
+ [ ] emulation of real quant through C/CUDA/Python