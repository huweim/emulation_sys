#!/usr/bin/env python3
"""
Deterministic prompt-generation helper for token-level qualitative comparison.

This script supports either:
1. a literal prompt string via --prompt, or
2. prompt extraction from a Hugging Face dataset via --dataset-name.

It prints the exact prompt, prompt/source metadata, and the generated
continuation so the output can be used directly for token-level comparison
figures across real / pseudo / emulation runs.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .quant.pre_quant import replace_quant_linear
from .utils.model_loading import load_tokenizer_and_model


COMMON_TEXT_FIELDS = ("text", "sentence", "content", "article", "document", "passage")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deterministic generation for token-level comparison figures."
    )
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp16",
        choices=["fp16", "bf16", "fp32"],
        help="Model dtype",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    parser.add_argument("--prompt", type=str, default=None, help="Literal input prompt")
    parser.add_argument("--dataset-name", type=str, default=None, help="HF dataset name, e.g. wikitext")
    parser.add_argument("--dataset-config", type=str, default=None, help="HF dataset config name")
    parser.add_argument("--dataset-split", type=str, default="test", help="HF dataset split")
    parser.add_argument(
        "--dataset-field",
        type=str,
        default=None,
        help="Text field name. If omitted, auto-detect common text fields.",
    )
    parser.add_argument(
        "--dataset-index",
        type=int,
        default=0,
        help="Starting example index when searching dataset prompts",
    )
    parser.add_argument(
        "--search-limit",
        type=int,
        default=1000,
        help="Max number of starting positions to scan for a suitable prompt",
    )
    parser.add_argument(
        "--min-prompt-tokens",
        type=int,
        default=100,
        help="Minimum prompt length in tokens when sampling from a dataset",
    )
    parser.add_argument(
        "--max-prompt-tokens",
        type=int,
        default=200,
        help="Maximum prompt length in tokens when sampling from a dataset",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Number of generated tokens",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature. Use 0.0 for deterministic greedy decoding.",
    )

    parser.add_argument(
        "--quant-mode",
        type=str,
        default=None,
        choices=["pseudo", "real", "emulation"],
        help="Quantization mode",
    )
    parser.add_argument("--w-bit", type=int, default=4, help="Weight bitwidth")
    parser.add_argument("--a-bit", type=int, default=4, help="Activation bitwidth")
    parser.add_argument(
        "--use-triton-emu",
        action="store_true",
        help="Use Triton-accelerated emulation path when --quant-mode emulation",
    )
    parser.add_argument(
        "--fp64-override",
        action="store_true",
        help="Load the local FP64 modeling override from ./inference/models",
    )

    parser.add_argument(
        "--save-json",
        type=str,
        default=None,
        help="Optional JSON output path for prompt/generation metadata",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to datasets.load_dataset when needed",
    )
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[name]


def detect_text_field(dataset, requested_field: str | None) -> str:
    if requested_field is not None:
        if requested_field not in dataset.column_names:
            raise ValueError(
                f"Requested dataset field '{requested_field}' not found. "
                f"Available fields: {dataset.column_names}"
            )
        return requested_field

    for field in COMMON_TEXT_FIELDS:
        if field in dataset.column_names:
            return field

    raise ValueError(
        "Could not auto-detect a text field. "
        f"Available fields: {dataset.column_names}. Please pass --dataset-field."
    )


def normalize_text(value) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        pieces = [item.strip() for item in value if isinstance(item, str) and item.strip()]
        return "\n".join(pieces).strip()
    return ""


def build_dataset_prompt(args: argparse.Namespace, tokenizer) -> tuple[str, dict]:
    from datasets import load_dataset

    load_kwargs = {
        "path": args.dataset_name,
        "split": args.dataset_split,
    }
    if args.dataset_config is not None:
        load_kwargs["name"] = args.dataset_config
    if args.trust_remote_code:
        load_kwargs["trust_remote_code"] = True

    dataset = load_dataset(**load_kwargs)
    text_field = detect_text_field(dataset, args.dataset_field)

    if args.dataset_index < 0 or args.dataset_index >= len(dataset):
        raise IndexError(f"--dataset-index {args.dataset_index} out of range for split size {len(dataset)}")

    max_start = min(len(dataset), args.dataset_index + args.search_limit)
    for start_idx in range(args.dataset_index, max_start):
        pieces: list[str] = []
        end_idx = start_idx

        while end_idx < len(dataset):
            piece = normalize_text(dataset[end_idx][text_field])
            end_idx += 1
            if not piece:
                continue
            pieces.append(piece)

            candidate = "\n\n".join(pieces).strip()
            token_ids = tokenizer(candidate, add_special_tokens=False).input_ids
            if len(token_ids) >= args.min_prompt_tokens:
                trimmed_ids = token_ids[: args.max_prompt_tokens]
                prompt = tokenizer.decode(trimmed_ids, skip_special_tokens=True)
                final_ids = tokenizer(prompt, add_special_tokens=False).input_ids
                if args.min_prompt_tokens <= len(final_ids) <= args.max_prompt_tokens:
                    meta = {
                        "source_type": "dataset",
                        "dataset_name": args.dataset_name,
                        "dataset_config": args.dataset_config,
                        "dataset_split": args.dataset_split,
                        "dataset_field": text_field,
                        "dataset_start_index": start_idx,
                        "dataset_end_index_exclusive": end_idx,
                    }
                    return prompt, meta
                break

    raise RuntimeError(
        "Could not find a dataset-backed prompt within the requested token range. "
        f"Tried starting indices [{args.dataset_index}, {max_start})."
    )


def resolve_prompt(args: argparse.Namespace, tokenizer) -> tuple[str, dict]:
    has_prompt = args.prompt is not None
    has_dataset = args.dataset_name is not None
    if has_prompt == has_dataset:
        raise ValueError("Provide exactly one of --prompt or --dataset-name.")

    if has_prompt:
        return args.prompt, {"source_type": "literal"}
    return build_dataset_prompt(args, tokenizer)


def load_model(model_path: str, dtype: torch.dtype, use_local_fp64_override: bool):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        raise RuntimeError("This script requires a CUDA-enabled GPU.")

    print(f"[Load] Loading model from {model_path}...")
    print(f"[Load] Device: {device}, Dtype: {dtype}")

    effective_attn_implementation = "eager" if use_local_fp64_override else None
    tokenizer, model, used_override_path = load_tokenizer_and_model(
        model_path,
        dtype,
        attn_implementation=effective_attn_implementation,
        use_local_fp64_override=use_local_fp64_override,
        require_local_fp64_override=use_local_fp64_override,
    )
    model.to(device)
    model.eval()

    print("[Load] Model loaded successfully!")
    attn_impl = getattr(getattr(model, "config", None), "_attn_implementation", "auto")
    if use_local_fp64_override:
        print("[Attention] fp64 override enabled -> forcing attn_implementation=eager")
    print(f"[Attention] attn_implementation={attn_impl}")
    if used_override_path:
        print(f"[Override] Using local modeling override: {used_override_path}")
    return model, tokenizer, device, used_override_path


def apply_quantization_if_needed(model, args: argparse.Namespace) -> None:
    if not args.quant_mode:
        return

    print(f"[Quant] Applying {args.w_bit}-bit weight, {args.a_bit}-bit activation quantization")
    print(f"[Quant] Mode: {args.quant_mode}")
    if args.quant_mode == "real":
        print("[Quant] Note: 'real' mode requires RTX 5090 (sm_120a)")
    elif args.quant_mode == "emulation":
        print("[Quant] Note: 'emulation' mode works on any GPU (A100, H100, etc.)")
        if args.use_triton_emu:
            print("[Quant] Triton emulation acceleration: enabled")

    q_config = {
        "q_group_size": 16,
        "mode": args.quant_mode,
        "use_triton": bool(args.use_triton_emu),
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


def generate(model, tokenizer, device: str, prompt: str, max_new_tokens: int, temperature: float):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    do_sample = temperature > 0.0
    top_p = 0.95 if do_sample else 1.0

    with torch.no_grad():
        start_time = time.time()
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else None,
            do_sample=do_sample,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )
        end_time = time.time()

    prompt_len = inputs["input_ids"].shape[1]
    full_ids = outputs[0].detach().cpu()
    generated_ids = full_ids[prompt_len:]
    return {
        "prompt_input_ids": inputs["input_ids"][0].detach().cpu(),
        "full_output_ids": full_ids,
        "generated_ids": generated_ids,
        "prompt_text": prompt,
        "full_text": tokenizer.decode(full_ids, skip_special_tokens=True),
        "generated_text": tokenizer.decode(generated_ids, skip_special_tokens=True),
        "prompt_tokens": int(prompt_len),
        "generated_tokens": int(generated_ids.shape[0]),
        "generation_time_sec": end_time - start_time,
    }


def maybe_save_json(path_str: str | None, payload: dict) -> None:
    if path_str is None:
        return

    path = Path(path_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = dict(payload)
    for key in ("prompt_input_ids", "full_output_ids", "generated_ids"):
        serializable[key] = payload[key].tolist()
    path.write_text(json.dumps(serializable, indent=2, ensure_ascii=False) + "\n")
    print(f"[Save] JSON written to {path}")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    model_dtype = dtype_from_name(args.dtype)
    model, tokenizer, device, used_override_path = load_model(
        args.model_path,
        model_dtype,
        use_local_fp64_override=bool(args.fp64_override),
    )
    apply_quantization_if_needed(model, args)

    prompt, prompt_meta = resolve_prompt(args, tokenizer)
    result = generate(
        model,
        tokenizer,
        device,
        prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    payload = {
        "model_path": args.model_path,
        "dtype": args.dtype,
        "quant_mode": args.quant_mode,
        "fp64_override": bool(args.fp64_override),
        "local_override": used_override_path,
        "temperature": args.temperature,
        "prompt_meta": prompt_meta,
        **result,
    }

    print("\n" + "=" * 72)
    print("Prompt Eval Settings:")
    print(f"  Source: {prompt_meta['source_type']}")
    if prompt_meta["source_type"] == "dataset":
        print(
            "  Dataset: %s / %s / %s"
            % (
                prompt_meta["dataset_name"],
                prompt_meta["dataset_config"] or "-",
                prompt_meta["dataset_split"],
            )
        )
        print(
            "  Dataset slice: [%d, %d)"
            % (
                prompt_meta["dataset_start_index"],
                prompt_meta["dataset_end_index_exclusive"],
            )
        )
        print(f"  Dataset field: {prompt_meta['dataset_field']}")
    print(f"  Prompt tokens: {payload['prompt_tokens']}")
    print(f"  Generated tokens: {payload['generated_tokens']}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Quantization: {args.quant_mode if args.quant_mode else 'None'}")
    print(f"  FP64 override: {args.fp64_override}")
    print(f"  Local override: {used_override_path if used_override_path else 'None'}")
    print("=" * 72)

    print("\nPROMPT:")
    print("-" * 72)
    print(payload["prompt_text"])
    print("-" * 72)

    print("\nGENERATED CONTINUATION:")
    print("-" * 72)
    print(payload["generated_text"])
    print("-" * 72)

    print("\nFULL TEXT:")
    print("-" * 72)
    print(payload["full_text"])
    print("-" * 72)

    print("\nGeneration Statistics:")
    print(f"  Time: {payload['generation_time_sec']:.2f}s")
    print(f"  Speed: {payload['generated_tokens'] / payload['generation_time_sec']:.2f} tokens/sec")

    maybe_save_json(args.save_json, payload)


if __name__ == "__main__":
    main()
