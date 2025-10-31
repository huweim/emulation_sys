# prompt_runner.py
import torch
import torch.nn as nn
from collections import defaultdict
import textwrap
from tqdm import tqdm

@torch.no_grad()
def generate_prompt(
    prompt: str,
    model: nn.Module,
    tokenizer,
    max_new_tokens: int = 512,
    do_sample: bool = False # Keep greedy for determinism test
):
    """
    Generates text for a single prompt using the HF model's generate method.

    Args:
        prompt (str): The input prompt string.
        model (nn.Module): The loaded Hugging Face model.
        tokenizer: The loaded Hugging Face tokenizer.
        max_new_tokens (int): Maximum number of new tokens to generate.
        do_sample (bool): Whether to use sampling (should be False for determinism).

    Returns:
        str: The generated text including the prompt.
    """
    device = model.device # Get device from model
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Ensure greedy decoding if do_sample is False
    gen_kwargs = {"max_new_tokens": max_new_tokens}
    if not do_sample:
        gen_kwargs["do_sample"] = False
        gen_kwargs["temperature"] = 1.0 # Technically irrelevant if do_sample=False
        gen_kwargs["top_p"] = 1.0       # but good practice to set neutral values
        gen_kwargs["top_k"] = 50        # Same as above

    outputs = model.generate(**inputs, **gen_kwargs)
    
    # Decode the full output sequence
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def summarize_results(num_runs: int, baseline_output: str, inconsistencies: dict):
    """
    Prints the final summary report for the determinism test.

    Args:
        num_runs (int): Total number of runs executed.
        baseline_output (str): The output from the first run.
        inconsistencies (dict): Dictionary mapping inconsistent outputs to run numbers.
    """
    inconsistent_runs_count = sum(len(runs) for runs in inconsistencies.values())
    consistent_runs_count = num_runs - inconsistent_runs_count
    inconsistency_rate = (inconsistent_runs_count / num_runs) * 100 if num_runs > 0 else 0

    print("\n" + "="*80)
    print(" " * 28 + "DETERMINISM TEST SUMMARY")
    print("="*80)
    print(f"Backend Used: HF (Transformers)") # Specify backend
    print(f"Total Runs Executed: {num_runs}")
    print(f"✅ Consistent Runs (same as Run 1): {consistent_runs_count}")
    print(f"❌ Inconsistent Runs (different from Run 1): {inconsistent_runs_count} ({inconsistency_rate:.2f}%)")
    print("-" * 80)

    print("--- Baseline Output (Generated in Run 1) ---")
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

def run_determinism_test(
    args, # Pass args for num_runs, prompt, max_new_tokens
    model: nn.Module,
    tokenizer
):
    """
    Runs the prompt generation N times and checks for inconsistencies.

    Args:
        args: Command line arguments containing num_runs, prompt, max_new_tokens.
        model (nn.Module): The loaded (potentially quantized) HF model.
        tokenizer: The loaded HF tokenizer.
    """
    baseline_output = None
    inconsistencies = defaultdict(list)
    
    print("\n" + "="*80)
    print(f"Starting Determinism Test: {args.num_runs} runs (HF backend).")
    print("="*80)
    
    mismatch_count = 0
    for i in tqdm(range(1, args.num_runs + 1), desc="Determinism Test (HF)"):
        # print(f"\n--- Running Iteration {i}/{args.num_runs} ---") # Can be noisy
        current_output = generate_prompt(
            prompt=args.prompt,
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=args.max_new_tokens,
            do_sample=False # Force greedy for determinism test
        )
        
        if i == 1:
            baseline_output = current_output
            print(f"[Run 1] Baseline established.")
        else:
            if current_output != baseline_output:
                inconsistencies[current_output].append(i)
                mismatch_count += 1
                print(f"[Run {i}] ❌ MISMATCH detected!")
            # else:
            #     print(f"[Run {i}] ✅ Consistent.") # Can be noisy

    # Summarize results at the end
    summarize_results(args.num_runs, baseline_output, inconsistencies)