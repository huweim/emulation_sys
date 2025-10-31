import torch
import argparse
import time
from collections import defaultdict
import textwrap
from tqdm import tqdm

def initialize_backend(backend: str, model_path: str):
    """
    根据选择的后端，初始化并返回模型和分词器/采样参数。
    模型只会被加载一次。
    """
    print(f"--- Initializing Backend: {backend.upper()} ---")
    print(f"--- Loading Model: {model_path} (this may take a while) ---")
    
    if backend == 'hf':
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
        print("--- Model Loaded Successfully ---")
        return model, tokenizer, None # HF doesn't use sampling_params object
        
    elif backend == 'vllm':
        from vllm import LLM, SamplingParams
        
        llm = LLM(model=model_path, tensor_parallel_size=1, dtype="bfloat16")
        
        # 定义确定性采样参数
        sampling_params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            top_k=-1,
            max_tokens=1024,
        )
        print("--- Model Loaded Successfully ---")
        return llm, None, sampling_params # vLLM uses llm engine and sampling_params
        
    else:
        raise ValueError("Invalid backend selected.")


def generate(backend: str, prompt: str, model, tokenizer, sampling_params):
    """
    执行单次推理生成。
    """
    if backend == 'hf':
        device = model.device
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        generation_kwargs = {
            "max_new_tokens": 1024,
            "do_sample": False,
            # "temperature": 0.0,
        }
        outputs = model.generate(**inputs, **generation_kwargs)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
        
    elif backend == 'vllm':
        outputs = model.generate([prompt], sampling_params)
        return outputs[0].prompt + outputs[0].outputs[0].text


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
    parser.add_argument("--model_path", type=str, default="meta-llama/Llama-3-8B-Instruct")
    parser.add_argument("--backend", type=str, required=True, choices=['hf', 'vllm'])
    parser.add_argument("--num-runs", type=int, default=10, help="Number of inference runs to perform.")
    parser.add_argument("--prompt", type=str, 
        default=(
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
        ),        help="The prompt to use for inference."
    )

    args = parser.parse_args()

    # 1. 初始化，只加载一次模型
    model, tokenizer, sampling_params = initialize_backend(args.backend, args.model_path)

    # Warm-up run for vLLM to JIT kernels
    if args.backend == 'vllm':
        print("\n" + "="*80)
        print("--- Performing 1 warm-up run (vLLM) to JIT kernels... ---")
        _ = generate(args.backend, "Warm-up run", model, tokenizer, sampling_params)
        print("--- Warm-up complete. Starting determinism test. ---")
    
    baseline_output = None
    # 使用 defaultdict(list) 更方便地收集不一致的轮次
    inconsistencies = defaultdict(list)
    
    print("\n" + "="*80)
    print(f"Starting Determinism Test: {args.num_runs} runs on '{args.backend.upper()}' backend.")
    print("="*80)
    
    nums_output_mismatch = 0
    # 2. 循环运行推理
    for i in tqdm(range(1, args.num_runs + 1), desc="Running Inference"):
        print(f"\n--- Running Iteration {i}/{args.num_runs} ---")
        current_output = generate(args.backend, args.prompt, model, tokenizer, sampling_params)
        
        # 3. 对比结果
        if i == 1:
            baseline_output = current_output
        else:
            if current_output != baseline_output:
                inconsistencies[current_output].append(i)
                nums_output_mismatch += 1

        print(f"Total Mismatch Count: {nums_output_mismatch}")

    # 4. 总结并输出报告
    summarize_results(args.num_runs, baseline_output, inconsistencies)

if __name__ == "__main__":
    main()