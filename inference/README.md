

python inference_test.py --backend vllm --num-runs 1000 --model_path /mnt/model/llama-2-7b-hf  | tee vllm_1000_runs.log
python inference_test.py --backend hf --num-runs 1000 --model_path /mnt/model/llama-2-7b-hf  | tee hf_1000_runs.log

python inference_test.py --backend vllm --num-runs 10 --model_path /mnt/model/llama-2-7b-hf  | tee vllm_10_runs.log
python inference_test.py --backend hf --num-runs 10 --model_path /mnt/model/llama-2-7b-hf  | tee hf_10_runs.log


```shell
conda create -n llm_inference python=3.12 -y
conda activate llm_inference

pip install torch==2.7.0 --extra-index-url https://download.pytorch.org/whl/cu128
pip install .
```