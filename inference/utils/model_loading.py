from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def _module_name_for_arch(arch: str) -> str:
    arch_lower = arch.lower()
    if "llama" in arch_lower:
        return "transformers.models.llama.modeling_llama_wmhu_override"
    if "qwen2" in arch_lower:
        return "transformers.models.qwen2.modeling_qwen2_wmhu_override"
    if "qwen3" in arch_lower:
        return "transformers.models.qwen3.modeling_qwen3_wmhu_override"
    raise ValueError(
        f"Unsupported architecture '{arch}' for local modeling override. "
        "Expected a Llama/Qwen2/Qwen3 causal LM."
    )


def _local_override_filename_for_arch(arch: str) -> str | None:
    arch_lower = arch.lower()
    if "llama" in arch_lower:
        return "modeling_llama_fp64_lmhead_attn.py"
    if "qwen2" in arch_lower:
        return "modeling_qwen2_fp64_lmhead_attn.py"
    if "qwen3" in arch_lower:
        return "modeling_qwen3_fp64_lmhead_attn.py"
    return None


def resolve_local_fp64_override_path(model_path: str) -> str | None:
    config = AutoConfig.from_pretrained(model_path)
    if not getattr(config, "architectures", None):
        return None

    filename = _local_override_filename_for_arch(config.architectures[0])
    if filename is None:
        return None

    candidate = Path(__file__).resolve().parents[1] / "models" / filename
    if candidate.exists():
        return str(candidate)
    return None


def _load_override_class(model_path: str, modeling_override_path: str):
    config = AutoConfig.from_pretrained(model_path)
    if not getattr(config, "architectures", None):
        raise ValueError("Model config does not expose `architectures`; cannot select override class.")

    arch = config.architectures[0]
    module_name = _module_name_for_arch(arch)
    override_path = Path(modeling_override_path).expanduser().resolve()

    spec = importlib.util.spec_from_file_location(module_name, override_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for override module: {override_path}")

    module = importlib.util.module_from_spec(spec)
    # Register before execution so import-time decorators like auto_docstring
    # can resolve classes back to a real module/file via sys.modules.
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise

    if not hasattr(module, arch):
        raise AttributeError(
            f"Override module {override_path} does not define expected class '{arch}'."
        )
    return getattr(module, arch)


def load_tokenizer_and_model(
    model_path: str,
    dtype: torch.dtype,
    *,
    attn_implementation: str | None = None,
    modeling_override_path: str | None = None,
    use_local_fp64_override: bool = False,
    require_local_fp64_override: bool = False,
):
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    common_kwargs = {
        "torch_dtype": dtype,
        "device_map": "auto",
    }
    if attn_implementation is not None:
        common_kwargs["attn_implementation"] = attn_implementation

    used_override_path = modeling_override_path
    if used_override_path is None and use_local_fp64_override:
        used_override_path = resolve_local_fp64_override_path(model_path)

    if require_local_fp64_override and used_override_path is None:
        config = AutoConfig.from_pretrained(model_path)
        arch = config.architectures[0] if getattr(config, "architectures", None) else "unknown"
        filename = _local_override_filename_for_arch(arch)
        expected = (
            str(Path(__file__).resolve().parents[1] / "models" / filename)
            if filename is not None
            else "./inference/models/<unsupported-arch override>.py"
        )
        raise FileNotFoundError(
            "FP64 local override requested, but no local override file was found for "
            f"architecture '{arch}'. Expected: {expected}"
        )

    if used_override_path:
        model_cls = _load_override_class(model_path, used_override_path)
        model = model_cls.from_pretrained(model_path, **common_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, **common_kwargs)

    return tokenizer, model, used_override_path
