import os
import json
import random
from tqdm import tqdm

import torch
import transformers
from vllm import LLM
from vllm.engine.arg_utils  import PoolerConfig

from lighteval.models.model_input  import GenerationParameters
from lighteval_custom.models.vllm.vllm_model  import VLLMModelConfig
from lighteval_custom.main_vllm  import vllm

# ========== 固定参数（原 argparse 的默认值） ==========
DEBUG                 = False
OVERWRITE             = False
LOAD_RESPONSES_JSON   = None          # 不从外部文件加载 response

DTYPE                 = 'bfloat16'    # 若模型路径含 "gptqmodel" 会被强制改为 float16
MAX_SAMPLES           = None          # 不限制样本数（debug 时会强制改 2）
TEMPERATURE           = 0.6
TOP_P                 = 0.95
SEED                  = 42
MAX_NEW_TOKENS        = 32768
MAX_MODEL_LENGTH      = 32768

# ========== 根据上面参数派生出的变量 ==========

TENSOR_PARALLEL_SIZE  = torch.cuda.device_count() 



# ========== main ==========
def main(model_path='/localssd/models/DeepSeek-R1-Distill-Qwen-1.5B', task="AIME-90"):

    # gptqmodel 强制 float16
    if "gptqmodel" in model_path:
        global DTYPE
        DTYPE = "float16"
        
    random.seed(SEED) 
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"]  = "spawn"

    effective_max_new_tokens = MAX_NEW_TOKENS
    effective_max_samples   = MAX_SAMPLES

    generation_parameters = GenerationParameters(
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=30 if "QwQ" in model_path else None,
        max_new_tokens=effective_max_new_tokens,
        seed=SEED,
    )

    model_config = VLLMModelConfig(
        pretrained=model_path,
        dtype=DTYPE,
        max_model_length=MAX_MODEL_LENGTH,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        gpu_memory_utilization=0.9,
        enforce_eager=True,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        generation_parameters=generation_parameters,
        init_model=(LOAD_RESPONSES_JSON is None),
    )

    # mapping from dataset string to lighteval task spec
    # The task string format is: "suite|task_name|num_fewshot|version"
    task_map = {
        "AIME-2024":      ("custom|aime24|0|0",       "lighteval_custom.tasks.reasoning"), 
        "AIME-2025":      ("custom|aime25|0|0",       "lighteval_custom.tasks.reasoning"), 
        "AIME-90":        ("custom|aime90|0|0",       "lighteval_custom.tasks.reasoning"), 
        "MATH-500":       ("custom|math_500|0|0",     "lighteval_custom.tasks.reasoning"), 
        "NuminaMath-1.5": ("custom|numina_math|0|0",  "lighteval_custom.tasks.reasoning"), 
        "GSM8K":          ("custom|gsm8k|0|0",        "lighteval_custom.tasks.reasoning"), 
        "GPQA-Diamond":   ("custom|gpqa:diamond|0|0", "lighteval_custom.tasks.reasoning"), 
        "LiveCodeBench":  ("custom|lcb:codegeneration|0|0",
                           "lighteval_custom.tasks.livecodebench"), 
    }

    tasks_str, custom_tasks_path = task_map[task]
    vllm(
        model_config=model_config,
        use_chat_template=True,
        max_samples=effective_max_samples,
        load_responses_from_json_file=LOAD_RESPONSES_JSON,
        tasks=tasks_str,
        custom_tasks=custom_tasks_path,
    )

if __name__ == "__main__":
    main()