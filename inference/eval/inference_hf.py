import os
import json
import random
from tqdm import tqdm

import torch
import transformers

# --- Imports (Modified) ---
# 我们不再需要 TransformersModelConfig, 因为我们将构建一个字符串
from lighteval.models.model_input import GenerationParameters # (保留)
# 我们假设您的 custom_accelerate 仍然是您要调用的目标
from lighteval_custom.main_accelerate import accelerate

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
BATCH_SIZE            = 1
OUTPUT_DIR            = "./lighteval_results" # (Added) accelerate 函数需要这个

# ========== main ==========
def main(model_path='/localssd/models/DeepSeek-R1-Distill-Qwen-1.5B', task="AIME-90"):

    # gptqmodel 强制 float16
    if "gptqmodel" in model_path:
        global DTYPE
        DTYPE = "float16"
        
    random.seed(SEED) 
    
    effective_max_new_tokens = MAX_NEW_TOKENS
    effective_max_samples   = MAX_SAMPLES

    # --- 1. (Modified) 将所有参数构建为字典 ---
    # `accelerate` 函数期望一个 `model_args` *字符串*，
    # 格式为 "key1=val1,key2=val2,..."
    # 我们首先把所有模型和生成参数收集到一个字典中。

    # (a) 模型参数
    model_params = {
        "model_name": model_path,  # <--- THIS IS THE FIX
        "dtype": DTYPE,
        "max_model_length": MAX_MODEL_LENGTH,
        "batch_size": BATCH_SIZE,
        "device_map": "auto",
        "use_chat_template": True, 
    }

    # (b) 生成参数
    generation_params = {
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": 30 if "QwQ" in model_path else None, # None 值将在下一步被过滤掉
        "max_new_tokens": effective_max_new_tokens,
        "seed": SEED,
    }

    # (c) 合并并格式化为字符串
    all_params = {**model_params, **generation_params}
    
    model_args_list = []
    for key, value in all_params.items():
        if value is not None: # 过滤掉 None 值的键
            model_args_list.append(f"{key}={value}")
    
    # 这就是 accelerate 函数期望的最终 `model_args` 字符串
    model_args_str = ",".join(model_args_list)
    
    print(f"[Info] Calling accelerate with model_args: {model_args_str}")

    # --- 2. (Unchanged) 任务映射 ---
    # (这部分完全不变，因为它只与您的 custom task loader 相关)
    task_map = {
        "AIME-2024":      ("custom|aime24|0|0",       "lighteval_custom.tasks.reasoning"), 
        "AIME-2025":      ("custom|aime25|0|0",       "lighteval_custom.tasks.reasoning"), 
        "AIME-90":        ("custom|aime90|0|0",       "lighteval_custom.tasks.reasoning"), 
        "MATH-500":       ("custom|math_500|0|0",     "lighteval_custom.tasks.reasoning"), 
        "Numina-Math-1.5": ("custom|numina_math|0|0",  "lighteval_custom.tasks.reasoning"), 
        "GSM8K":          ("custom|gsm8k|0|0",        "lighteval_custom.tasks.reasoning"), 
        "GPQA-Diamond":   ("custom|gpqa:diamond|0|0", "lighteval_custom.tasks.reasoning"), 
        "LiveCodeBench":  ("custom|lcb:codegeneration|0|0",
                           "lighteval_custom.tasks.livecodebench"), 
    }
    tasks_str, custom_tasks_path = task_map[task]

    # --- 3. (Modified) 调用 accelerate 主函数 ---
    # 现在我们传递的是 `model_args_str`，并匹配官方签名
    accelerate(
        model_args=model_args_str,
        tasks=tasks_str,
        custom_tasks=custom_tasks_path,
        max_samples=effective_max_samples,
        # 匹配官方参数名: load_responses_from_json_file -> load_responses_from_details_date_id
        load_responses_from_details_date_id=LOAD_RESPONSES_JSON, 
        output_dir=OUTPUT_DIR,
        # 您还可以从上面的 "固定参数" 区传入更多参数
        # ...
    )

if __name__ == "__main__":
    main()