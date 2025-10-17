import torch
import argparse
import time
from collections import defaultdict
import textwrap
from tqdm import tqdm

import torch
import torch.nn as nn

from .quant.pre_quant import replace_quant_linear 

def initialize_backend(model_path: str):
    from transformers import AutoTokenizer, AutoModelForCausalLM

    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        raise RuntimeError("This script requires a CUDA-enabled GPU.")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
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
    tasks 例如: "arc_easy,hellaswag,winogrande"
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
        # nsamples = 30
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
            limit=limit,              # 可为 None：全量评测；或设一个整数快速跑
        )

        print(make_table(results))
        # 也把结构化结果返回（你想存 JSON 也方便）
        return results

def summarize_results(num_runs: int, baseline_output: str, inconsistencies: dict):
    """
    打印最终的实验结果总结。
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
    # 使用 textwrap.indent 美化输出
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
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)

    # 两种使用方式：prompt 或 tasks（二选一或都给；给了 tasks 就跑评测，其它情况下仅做生成）
    parser.add_argument("--tasks", type=str, default=None, help='Comma-separated task names for lm-eval, e.g. "arc_easy,hellaswag"')
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None, help="lm-eval limit per task for quick tests")

    parser.add_argument("--use_prompt", action="store_true", help="generate based on single prompt (if no tasks specified, default False)")
    parser.add_argument("--num-runs", type=int, default=1, help="Number of inference runs to perform.")
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
    parser.add_argument("--quant-mode", type=str, default="pseudo", choices=["pseudo", "real"],
                        help="Use 'pseudo' (fake-quant + F.linear) or 'real' (int GEMM API placeholder)")
    parser.add_argument("--w-bit", type=int, default=4, help="weight bitwidth, e.g., 4 for W4")
    parser.add_argument("--a-bit", type=int, default=4, help="activation bitwidth, e.g., 4 for A4")
    parser.add_argument("--q-group-size", type=int, default=32, help="group size along in_features for weight per-group quant")
    parser.add_argument("--zero-point", action="store_true", help="use asymmetric (with zero-point) quant; default symmetric")
    parser.add_argument("--nvfp", action="store_true", help="use nvfp quant")

    args = parser.parse_args()

    # 1) load once
    model, tokenizer = initialize_backend(args.model_path)

    # 2) (可选) 量化替换
    if args.quantize:
        if args.w_bit is None:
            raise ValueError("--quantize is set but --w-bit is None")
        # a_bit 可以为 None（只做 W 伪量化）
        q_config = {
            "q_group_size": args.q_group_size,
            "mode": args.quant_mode,   # ★ 新增：把运行模式传进去
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
    if args.tasks:
        run_lm_eval(
            args=args,
            model=model,
            tokenizer=tokenizer,
            tasks=args.tasks,
            batch_size=args.batch_size,
            num_fewshot=args.num_fewshot,
            limit=args.limit,
        )
        # 3B) 仍然支持直接单条 prompt 生成（可单独用，也可在评测后顺手来一条）
    if args.use_prompt:
        print("\n" + "=" * 80)
        print("[Single Generation]")
        out = generate(args.prompt, model, tokenizer, max_new_tokens=args.max_new_tokens, do_sample=False)
        print(out)
        print("=" * 80)

        # 3) determinism test
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
            print(current_output)

        # summarize_results(args.num_runs, baseline_output, inconsistencies)


if __name__ == "__main__":
    main()