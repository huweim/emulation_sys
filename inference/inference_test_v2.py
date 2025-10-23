import torch
import argparse
import time
from collections import defaultdict
import textwrap
from tqdm import tqdm

import torch
import torch.nn as nn

from .quant.pre_quant import replace_quant_linear 

from .lighteval_runner import run_lighteval_evaluation


def initialize_backend(model_path: str, dtype: torch.dtype): # Added dtype param
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



@torch.no_grad()
def generate(prompt: str, model, tokenizer, max_new_tokens=512, do_sample=False):
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def run_lm_eval(args, model, tokenizer, tasks: str, batch_size: int, num_fewshot: int, limit: int | None):
    """
    tasks e.g.: "arc_easy,hellaswag,winogrande"
    """
    from lm_eval import evaluator
    from lm_eval.models.huggingface import HFLM
    from lm_eval.utils import make_table

    if tasks in ['wikitext', 'c4', 'ptb']:
    # https://github.com/IST-DASLab/gptq/blob/2d65066eeb06a5c9ff5184d8cebdf33662c67faf/llama.py#L206
        from .utils.dataload_utils import get_loaders
        model.seqlen = 2048
        _, testenc = get_loaders(tasks, model=args.model_path, seqlen=model.seqlen)
        
        testenc = testenc.input_ids.to(model.device)
        nsamples = testenc.numel() // model.seqlen
        # nsamples = 10
        model = model.eval()
        nlls = []
        for i in tqdm(range(nsamples), desc="evaluating..."):
            batch = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)].to(
                model.device
            )
            with torch.no_grad():
                lm_logits = model(batch).logits
            shift_logits = lm_logits[:, :-1, :].contiguous().float()
            shift_labels = testenc[
                :, (i * model.seqlen) : ((i + 1) * model.seqlen)
            ][:, 1:]
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            neg_log_likelihood = loss.float() * model.seqlen
            nlls.append(neg_log_likelihood)

        ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
        print(f'PPL on {tasks}: {ppl.item()}')
        return {"results": {tasks: {"ppl": float(ppl)}}}

    else:
        hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)

        task_list = [t.strip() for t in tasks.split(",") if t.strip()]
        print(f"\n[LM-Eval] Running tasks: {task_list}  (fewshot={num_fewshot}, batch_size={batch_size}, limit={limit})")

        results = evaluator.simple_evaluate(
            model=hflm,
            tasks=task_list,
            batch_size=batch_size,
            num_fewshot=num_fewshot,
            limit=limit,               # Can be None: full eval; or set an int for quick run
        )

        print(make_table(results))
        # Also return structured results (convenient for saving to JSON)
        return results

def summarize_results(num_runs: int, baseline_output: str, inconsistencies: dict):
    """
    (This function is preserved as requested)
    Prints the final summary of the experiment.
    """
    inconsistent_runs_count = sum(len(runs) for runs in inconsistencies.values())
    consistent_runs_count = num_runs - inconsistent_runs_count
    inconsistency_rate = (inconsistent_runs_count / num_runs) * 100 if num_runs > 0 else 0

    print("\n" + "="*80)
    print(" " * 28 + "FINAL SUMMARY REPORT")
    print("="*80)
    print(f"Total Runs Executed: {num_runs}")
    print(f"✅ Consistent Runs (same as Run 1): {consistent_runs_count}")
    print(f"❌ Inconsistent Runs (different from Run 1): {inconsistent_runs_count} ({inconsistency_rate:.2f}%)")
    print("-" * 80)

    print("--- Baseline Output (Generated in Run 1) ---")
    # Use textwrap.indent for nice formatting
    indented_baseline = textwrap.indent(baseline_output, '    ')
    print(indented_baseline)
    print("-" * 80)

    if not inconsistencies:
        print("\n🎉 All runs produced identical, deterministic results!")
    else:
        print(f"\n--- Found {len(inconsistencies)} Inconsistent Variation(s) ---")
        for i, (text, runs) in enumerate(inconsistencies.items()):
            print(f"\n[ Variation {i+1} ] - Occurred in {len(runs)} runs: {runs}")
            indented_variation = textwrap.indent(text, '    ')
            print(indented_variation)
    
    print("="*80)


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
    parser.add_argument("--batch_size", type=int, default=1, help="Evaluation batch size (default: 1)") # Changed default to 1
    parser.add_argument("--num_fewshot", type=int, default=0, help="Number of few-shot examples")
    parser.add_argument("--limit", type=int, default=100, help="Limit eval samples (0 for no limit)") # Changed default

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
    parser.add_argument("--q-group-size", type=int, default=32, help="group size along in_features for weight per-group quant")
    parser.add_argument("--zero-point", action="store_true", help="use asymmetric (with zero-point) quant; default symmetric")
    parser.add_argument("--nvfp", action="store_true", help="use nvfp quant")

    args = parser.parse_args()

    # --- 1) Load model and tokenizer ---
    model_dtype = torch.bfloat16 if args.dtype == 'bf16' else (torch.float16 if args.dtype == 'fp16' else torch.float32)
    model, tokenizer = initialize_backend(args.model_path, model_dtype)

    # --- 2) (Optional) Apply Quantization ---
    # The `model` object is modified in-place
    if args.quantize:
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
        print("[Main] Running full precision (no quantization).")

    if args.tasks:
        if args.eval_lib == "lighteval":
            print(f"[Main] Using 'lighteval' for task: {args.tasks}")
            # --- This is the new, correct call to the refactored library ---
            run_lighteval_evaluation(
                model=model,
                model_path=args.model_path,
                task_name=args.tasks,
                limit=args.limit if args.limit > 0 else None,
                num_fewshot=args.num_fewshot,
                batch_size=args.batch_size
            )
            ## vllm backend
            # from .eval.inference import main as lighteval_main
            # lighteval_main(args.model_path, args.tasks)
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
        # print("\n" + "=" * 80)
        # print("[Single Generation]")
        # out = generate(args.prompt, model, tokenizer, max_new_tokens=args.max_new_tokens, do_sample=False)
        # print(out)
        # print("=" * 80)

        baseline_output = None
        inconsistencies = defaultdict(list)
        print("\n" + "="*80)
        print(f"Starting Determinism Test: {args.num_runs} runs (HF backend).")
        print("="*80)

        mismatch = 0
        for i in tqdm(range(1, args.num_runs + 1), desc="Running Inference"):
            print(f"\n--- Running Iteration {i}/{args.num_runs} ---")
            current_output = generate(args.prompt, model, tokenizer)
        #     if i == 1:
        #         baseline_output = current_output
        #     else:
        #         if current_output != baseline_output:
        #             inconsistencies[current_output].append(i)
        #             mismatch += 1
        #     print(f"Total Mismatch Count: {mismatch}")
            print('OUTPUT:', current_output)

        # TODO: compare outputs and summarize
        # summarize_results(args.num_runs, baseline_output, inconsistencies)


if __name__ == "__main__":
    main()