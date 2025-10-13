python -m inference.inference_test_v2 \
  --model_path /mnt/model/llama-2-7b-hf \
  --task wikitext \
  --dtype fp16 \
  --batch_size 32

python -m inference.inference_test_v2 \
  --model_path /mnt/model/llama-2-7b-hf \
  --task wikitext \
  --dtype fp16 \
  --batch_size 32 \
  --quantize \

python -m inference.inference_test_v2 \
  --model_path /mnt/model/llama-2-7b-hf \
  --use_prompt \
  --dtype fp16 \
  --batch_size 32 \
  --quantize 
