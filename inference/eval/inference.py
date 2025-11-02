import os

from lighteval_custom import patch
from lighteval_custom.main_vllm  import vllm

MODELS_ARGS = [
    {
        "model_name": "examples/model_configs/vllm_model_config.yaml",
        "results_file": "tests/reference_scores/SmolLM2-1.7B-Instruct-results-vllm.json",
    }
]

# MAX_SAMPLES = 5
MAX_SAMPLES = None

def main(model_path_vllm='/localssd/models/DeepSeek-R1-Distill-Qwen-1.5B', task="AIME-90"
):

    os.environ["VLLM_WORKER_MULTIPROC_METHOD"]  = "spawn"

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

    effective_max_samples = MAX_SAMPLES

    tasks_str, custom_tasks_path = task_map[task]

    model_name = '/home/wmhu/emulation_workspace/numerical_emulation_system/inference/eval/lighteval_custom/model_configs/vllm_model_config.yaml'

    vllm(
        # model_config=model_config,
        model_args=model_name,
        use_chat_template=True,
        max_samples=effective_max_samples,
        # load_responses_from_json_file=LOAD_RESPONSES_JSON,
        tasks=tasks_str,
        custom_tasks=custom_tasks_path,
        model_path_config=model_path_vllm,
    )

if __name__ == "__main__":
    main()