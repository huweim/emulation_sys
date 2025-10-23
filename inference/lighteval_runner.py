import torch
from transformers import AutoModelForCausalLM

from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.transformers.transformers_model import TransformersModel, TransformersModelConfig
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters

# Task map moved to the module level as a constant
TASK_MAP = {
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

def run_lighteval_evaluation(
    model: torch.nn.Module, 
    model_path: str, 
    task_name: str, 
    limit: int | None,
    num_fewshot: int,
    batch_size: int
):
    """
    Runs evaluation using lighteval on an in-memory model.

    Args:
    - model: The loaded (and possibly quantized) torch.nn.Module
    - model_path: Original model path (used for loading config/tokenizer by lighteval)
    - task_name: The task to run (e.g., "AIME-90")
    - limit: Max samples to evaluate (None for all)
    - num_fewshot: Number of few-shot examples
    - batch_size: Evaluation batch size
    """

    if task_name not in TASK_MAP:
        raise ValueError(f"Task '{task_name}' not in TASK_MAP. Available: {list(TASK_MAP.keys())}")

    base_task_str, custom_tasks_path = TASK_MAP[task_name]

    # --- Dynamically build the task string with the correct few-shot number ---
    try:
        parts = base_task_str.split('|')
        parts[2] = str(num_fewshot) # Replace the hardcoded '0'
        tasks_str = "|".join(parts)
    except IndexError:
        print(f"Warning: Could not set fewshot for task string '{base_task_str}'. Using default.")
        tasks_str = base_task_str
    
    print(f"[Lighteval Runner] Using task string: {tasks_str}")

    evaluation_tracker = EvaluationTracker(output_dir="./results")
    pipeline_params = PipelineParameters(
        launcher_type=ParallelismManager.NONE,
        max_samples=limit if limit is not None else None, # Use the passed limit
        custom_tasks_directory=custom_tasks_path,
    )

    # We assume `model` is always passed in; no need for `if model is None:`
    
    config = TransformersModelConfig(model_name=model_path, batch_size=batch_size)
    # Use the in-memory, potentially-quantized `model` object
    lighteval_model = TransformersModel.from_model(model, config)

    pipeline = Pipeline(
        model=lighteval_model,
        pipeline_parameters=pipeline_params,
        evaluation_tracker=evaluation_tracker,
        tasks=tasks_str,
    )

    print(f"[Lighteval Runner] Starting evaluation for task: {task_name}...")
    results = pipeline.evaluate()
    pipeline.show_results()
    results = pipeline.get_results()
    print(f"[Lighteval Runner] Evaluation finished.")
    return results
