import torch
import argparse
import time
from collections import defaultdict
import textwrap
from tqdm import tqdm

import torch
import torch.nn as nn

from transformers import AutoConfig

from lighteval_custom import patch
from .utils.vllm_utils import prepare_vllm_temp_model

from .quant.pre_quant import replace_quant_linear 

from .eval.lighteval_runner_hf import run_lighteval_evaluation
from .eval.lmeval_runner import run_lm_eval
from inference.eval.prompt_runner import run_determinism_test
from .eval.inference import main as lighteval_main

from .models.qwen_vllm.qwen_mxfp import Qwen2ForCausalLM_nvfp

from vllm import ModelRegistry
ModelRegistry.register_model("Qwen2ForCausalLM_nvfp", Qwen2ForCausalLM_nvfp)


def load_hf_model(model_path: str, dtype: torch.dtype): # Added dtype param
    from transformers import AutoTokenizer, AutoModelForCausalLM

    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        raise RuntimeError("This script requires a CUDA-enabled GPU.")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype, # Use the passed dtype
        device_map="auto"
    )
    model.to(device)
    model.eval()
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Advanced LLM Inference Determinism Test Script")
    parser.add_argument("--model_path", type=str, default="/mnt/model/llama-2-7b-hf")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max tokens for prompt generation")
    parser.add_argument("--seed", type=int, default=42)

    # --- Eval flags ---
    parser.add_argument("--tasks", type=str, default=None, help='Task name(s) for evaluation (e.g., "AIME-90" or "arc_easy,hellaswag")')
    parser.add_argument("--eval_lib", type=str, default="lm-eval", choices=["lm-eval", "lighteval"],
                        help="Evaluation library to use. 'lighteval' handles custom reasoning tasks.")
    parser.add_argument("--backend", type=str, default="hf", choices=["hf", "vllm"], help="Specify the conceptual backend (hf required for quantization tests)")
    parser.add_argument("--batch_size", type=int, default=1, help="Evaluation batch size (default: 1)") # Changed default to 1
    parser.add_argument("--num_fewshot", type=int, default=0, help="Number of few-shot examples")
    parser.add_argument("--limit", type=int, default=0, help="Limit eval samples (0 for no limit)") # Changed default

    # --- Prompting flags ---
    parser.add_argument("--use_prompt", action="store_true", help="Generate based on single prompt (if no tasks specified, default False)")
    parser.add_argument("--num-runs", type=int, default=1, help="Number of inference runs for determinism test")
    parser.add_argument("--prompt", type=str, default=(
        "You are a supply chain analyst for a company that manufactures high-end electric bicycles. "
        "A critical shipment of lithium-ion battery packs, originating from a supplier in South Korea, "
        "is delayed by 3 weeks due to unforeseen port congestion in Singapore. The bicycles are "
        "assembled in Germany and are scheduled for a major product launch in France in 6 weeks. "
        "The battery packs are the only missing component.\n\n"
        "Please outline a detailed plan of action. Structure your response with clear headings "
        "for each of the following sections:\n"
        "1. Immediate Information Verification: What are the first steps to confirm the delay and the new timeline?\n"
        "2. Alternative Logistics Analysis: What alternative shipping options should be explored (e.g., air freight vs. rerouting sea freight), and what are their cost-benefit trade-offs?\n"
        "3. Production Schedule Adjustment: How should the assembly line schedule in Germany be adapted to minimize downtime?\n"
        "4. Stakeholder Communication Plan: What is the communication strategy for internal teams (Marketing, Sales, Leadership) and potentially external partners?"
    ))
    # === quantization flags ===
    parser.add_argument("--quantize", action="store_true", help="wrap Linear with fake-quant QuantLinear")
    parser.add_argument("--quant-mode", type=str, default="pseudo", choices=["pseudo", "real", "emulation"],
                        help="Use 'pseudo' (fake-quant + F.linear), 'real' (int GEMM API), or 'emulation' (our simulator)")
    parser.add_argument("--w-bit", type=int, default=4, help="weight bitwidth, e.g., 4 for W4")
    parser.add_argument("--a-bit", type=int, default=4, help="activation bitwidth, e.g., 4 for A4")
    parser.add_argument("--q-group-size", type=int, default=16, help="group size along in_features for weight per-group quant")
    parser.add_argument("--zero-point", action="store_true", help="use asymmetric (with zero-point) quant; default symmetric")
    parser.add_argument("--nvfp", action="store_true", help="use nvfp quant")

    args = parser.parse_args()

    # --- 1) Load model and tokenizer ---
    model, tokenizer = None, None
    load_model_needed = (args.backend == 'hf') or (args.eval_lib == 'lm-eval') or args.use_prompt

    if load_model_needed:
        print(f"[Main] Loading HF model into memory for {args.backend} backend...")
        model_dtype = torch.bfloat16 if args.dtype == 'bf16' else (torch.float16 if args.dtype == 'fp16' else torch.float32)
        model, tokenizer = load_hf_model(args.model_path, model_dtype)
    else:
        print("[Main] Model loading deferred to vLLM backend.")

    # model_path_vllm is the real path to lighteval_main
    model_path_vllm = args.model_path 
    temp_vllm_dir = None

    # --- 2) (Optional) Apply Quantization ---
    # The `model` object is modified in-place
    if args.quantize:
        if args.backend == "vllm" and args.tasks and args.eval_lib == "lighteval":
            if args.tasks in ["AIME-2025", "AIME-90","MATH-500", "GSM8K", "GPQA-Diamond", "LiveCodeBench"]:
                config = AutoConfig.from_pretrained(args.model_path)
                name = config.architectures[0]
                if not name.endswith("_nvfp"):
                    name += "_nvfp"
                
                if name != "Qwen2ForCausalLM_nvfp":
                    raise Exception(f"{name} is not supported yet")
                
                vllm_model_type_override = name

                # The value of `quant_method` must be the same as `NVFPQuantConfig @register_quantization_config`
                quant_params = {
                    "quant_method": "nvfp_quant", # TODO: Support other quant method [FP8, INT8, INT4]
                    "w_bit": args.w_bit,
                    "a_bit": args.a_bit,
                    "group_size": args.q_group_size,
                    "quant_mode": args.quant_mode,
                    "use_zero_point": args.zero_point,
                }
                temp_vllm_dir = prepare_vllm_temp_model(args.model_path, vllm_model_type_override, quant_params)
                model_path_vllm = temp_vllm_dir # Update the path used by vLLM
                
        elif model is not None:
            if args.w_bit is None:
                raise ValueError("--quantize is set but --w-bit is None")
            # a_bit can be None (for weight-only quant) 
            q_config = {
                "q_group_size": args.q_group_size,
                "mode": args.quant_mode,   # Pass the run mode in
            }
            replace_quant_linear(
                model=model,
                w_bit=args.w_bit,
                a_bit=args.a_bit,
                q_config=q_config,
                use_zero_point=args.zero_point,
                init_only=False,
                nvfp=args.nvfp,
            )
            print("[Main] Quantization applied.")
        else:
            print("[Main] Quantization requested, but no action (eval/prompt) requires the model.")

    if args.tasks:
        if args.eval_lib == "lighteval":
            print(f"[Main] Using 'lighteval' for task: {args.tasks}")
            if args.backend == "hf":
                run_lighteval_evaluation(
                    model=model,
                    model_path=args.model_path,
                    task_name=args.tasks,
                    limit=args.limit if args.limit > 0 else None,
                    num_fewshot=args.num_fewshot,
                    batch_size=args.batch_size
                )
            elif args.backend == "vllm":
                ## vllm backend
                lighteval_main(model_path_vllm, args.tasks)
        else:
            # lm-eval path
            print(f"[Main] Using legacy 'lm-eval' library for tasks: {args.tasks}")
            run_lm_eval(
                args=args,
                model=model,
                tokenizer=tokenizer,
                tasks=args.tasks,
                batch_size=args.batch_size,
                num_fewshot=args.num_fewshot,
                limit=args.limit if args.limit > 0 else None,
            )
    # --- 4) (Optional) Run Prompting / Determinism Test ---
    if args.use_prompt:
        print(f"[Main Runner] Starting prompt generation / determinism test...")
        run_determinism_test(
            args=args, # Pass args for prompt, num_runs, max_new_tokens
            model=model,
            tokenizer=tokenizer
        )
    

if __name__ == "__main__":
    main()