#!/usr/bin/env python3
"""
Inference Prompt - Simple text generation with NVFP4 quantization support.

Usage:
    # No quantization
    python -m inference.inference_prompt \
        --model_path /mnt/model/llama-2-7b-hf \
        --prompt "Once upon a time"

    # Pseudo quantization
    python -m inference.inference_prompt \
        --model_path /mnt/model/llama-2-7b-hf \
        --prompt "Once upon a time" \
        --quant-mode pseudo

    # Real quantization (requires RTX 5090)
    python -m inference.inference_prompt \
        --model_path /mnt/model/llama-2-7b-hf \
        --prompt "Once upon a time" \
        --quant-mode real

    # Emulation mode (works on any GPU, e.g., A100)
    python -m inference.inference_prompt \
        --model_path /mnt/model/llama-2-7b-hf \
        --prompt "Once upon a time" \
        --quant-mode emulation
"""

import torch
import argparse
import time
from transformers import AutoTokenizer, AutoModelForCausalLM

from .quant.pre_quant import replace_quant_linear


def load_model(model_path: str, dtype: torch.dtype):
    """Load model and tokenizer."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        raise RuntimeError("This script requires a CUDA-enabled GPU.")
    
    print(f"[Load] Loading model from {model_path}...")
    print(f"[Load] Device: {device}, Dtype: {dtype}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto"
    )
    model.to(device)
    model.eval()
    
    print(f"[Load] Model loaded successfully!")
    return model, tokenizer, device


def generate_text(model, tokenizer, device, prompt: str, max_new_tokens: int, temperature: float = 0.0):
    """Generate text from prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Determine generation params based on temperature
    do_sample = temperature > 0
    top_p = 1.0 if not do_sample else 0.95
    
    with torch.no_grad():
        start_time = time.time()
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else None,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
            top_p=top_p,
        )
        end_time = time.time()
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generation_time = end_time - start_time
    tokens_per_sec = max_new_tokens / generation_time
    
    return generated_text, generation_time, tokens_per_sec


def main():
    parser = argparse.ArgumentParser(
        description="Simple text generation with NVFP4 quantization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic generation
  python -m inference.inference_prompt --model_path /mnt/model/llama-2-7b-hf \\
      --prompt "Once upon a time"

  # With pseudo quantization
  python -m inference.inference_prompt --model_path /mnt/model/llama-2-7b-hf \\
      --prompt "Once upon a time" --quant-mode pseudo

  # With real quantization (requires RTX 5090)
  python -m inference.inference_prompt --model_path /mnt/model/llama-2-7b-hf \\
      --prompt "Once upon a time" --quant-mode real

  # With emulation mode (works on A100/H100)
  python -m inference.inference_prompt --model_path /mnt/model/llama-2-7b-hf \\
      --prompt "Once upon a time" --quant-mode emulation
        """
    )
    
    # Model settings
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the model checkpoint")
    parser.add_argument("--dtype", type=str, default="fp16", 
                        choices=["fp16", "bf16", "fp32"],
                        help="Model dtype (default: fp16)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    # Generation settings
    parser.add_argument("--prompt", type=str, 
                        default=(
                            "The rapid advancement of artificial intelligence technology has fundamentally "
                            "transformed numerous industries and aspects of daily life. From autonomous vehicles "
                            "navigating complex urban environments to medical diagnostic systems detecting diseases "
                            "with unprecedented accuracy, AI systems demonstrate remarkable capabilities. However, "
                            "these technological breakthroughs also present significant challenges related to ethics, "
                            "privacy, and societal impact that require careful consideration."
                        ),
                        help="Input prompt for generation (about 128 tokens)")
    parser.add_argument("--max_new_tokens", type=int, default=100,
                        help="Number of tokens to generate (default: 100)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (default: 0.0 for greedy decoding, no randomness)")
    
    # Quantization settings
    parser.add_argument("--quant-mode", type=str, default=None,
                        choices=["pseudo", "real", "emulation"],
                        help="Quantization mode: pseudo, real (requires RTX 5090), or emulation (works on A100)")
    parser.add_argument("--w-bit", type=int, default=4,
                        help="Weight bitwidth (default: 4)")
    parser.add_argument("--a-bit", type=int, default=4,
                        help="Activation bitwidth (default: 4)")
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    
    # Convert dtype string to torch dtype
    dtype_map = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32
    }
    model_dtype = dtype_map[args.dtype]
    
    # Load model
    model, tokenizer, device = load_model(args.model_path, model_dtype)
    
    # Apply quantization if requested
    if args.quant_mode:
        print(f"[Quant] Applying {args.w_bit}-bit weight, {args.a_bit}-bit activation quantization")
        print(f"[Quant] Mode: {args.quant_mode}")
        
        if args.quant_mode == "real":
            print("[Quant] Note: 'real' mode requires RTX 5090 (sm_120a)")
        elif args.quant_mode == "emulation":
            print("[Quant] Note: 'emulation' mode works on any GPU (A100, H100, etc.)")
        
        q_config = {
            "q_group_size": 16,
            "mode": args.quant_mode,
        }
        
        replace_quant_linear(
            model=model,
            w_bit=args.w_bit,
            a_bit=args.a_bit,
            q_config=q_config,
            use_zero_point=False,
            init_only=False,
            nvfp=True,
            fp8=False,
        )
        print("[Quant] Quantization applied successfully!")
    
    # Print generation settings
    prompt_tokens = len(tokenizer.encode(args.prompt))
    print("\n" + "=" * 60)
    print("Generation Settings:")
    print(f"  Prompt tokens: ~{prompt_tokens}")
    print(f"  Max new tokens: {args.max_new_tokens}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Quantization: {args.quant_mode if args.quant_mode else 'None'}")
    print("=" * 60 + "\n")
    
    # Generate
    print("[Generate] Starting generation...\n")
    generated_text, gen_time, tps = generate_text(
        model, tokenizer, device,
        args.prompt, args.max_new_tokens, args.temperature
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("GENERATED TEXT:")
    print("=" * 60)
    print(generated_text)
    print("=" * 60)
    print(f"\nGeneration Statistics:")
    print(f"  Time: {gen_time:.2f}s")
    print(f"  Tokens: {args.max_new_tokens}")
    print(f"  Speed: {tps:.2f} tokens/sec")
    print("=" * 60)


if __name__ == "__main__":
    main()
