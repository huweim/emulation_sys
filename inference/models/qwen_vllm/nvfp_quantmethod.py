# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any, Optional

import torch
import torch.nn.functional as F

from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
)

# --- [NEW] Direct imports for computation ---
# We no longer import QuantLinear from nvfp_quantizer.
# Instead, we import the ops it uses.
# Ensure 'nvfp' package is in your PYTHONPATH for the vLLM process.
import nvfp.ops as ops
import nvfp.pseudo_quant as pseudo_quant


class NVFPvLLMMethod(UnquantizedLinearMethod):
    """
    vLLM Quant Method for NVFP (replaces FakeQuantLinearMethod).
    
    This method *inlines* the computation logic from nvfp_quantizer.QuantLinear
    (both 'pseudo' and 'real' modes) directly into the 'apply' method,
    making it compatible with vLLM's forward pass and torch.compile.
    """

    def __init__(
        self,
        quant_config: "NVFPQuantConfig" # Receive the whole config
    ) -> None:
        super().__init__()
        self.cfg = quant_config # Store the config

        # Define constants for 'real' mode, copied from nvfp_quantizer.py
        self.FLOAT4_E2M1_MAX = 6.0
        self.FLOAT8_E4M3_MAX = 448.0

    def apply(
        self,
        layer: torch.nn.Module, # This is the vLLM layer (e.g., QKVParallelLinear)
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        
        # Get the original weight from the vLLM layer
        # Note: vLLM layers (e.g., QKVParallelLinear) typically have
        # weight shape [In_features, Out_features_tp]
        weight = layer.weight
        
        # 1. Select mode from config
        mode = self.cfg.quant_mode

        mode = 'real'

        if mode == "pseudo":
            # --- 'pseudo' mode logic ---
            # Replicates QuantLinear.forward(mode="pseudo")

            # 1a. Pseudo-quantize weight
            # This must be done on every forward pass, as torch.compile
            # cannot easily cache the result of this operation.
            Wq = pseudo_quant.nvfp4_pseudo_quantize(weight).to(weight.dtype)

            # 1b. Pseudo-quantize activation
            if self.cfg.a_bit is not None and self.cfg.a_bit < 16:
                x_q = pseudo_quant.nvfp4_pseudo_quantize(x)
            else:
                x_q = x

            # 1c. Perform computation.
            # vLLM layers use x @ W (not F.linear which is x @ W.T)
            # x shape: [..., In_features]
            # Wq shape (from vLLM layer): [In_features, Out_features_tp]
            output = torch.matmul(x_q.to(Wq.dtype), Wq.T).to(x.dtype)


        elif mode == "real":
            # --- 'real' mode logic ---
            # Replicates QuantLinear.forward(mode="real")
            
            # 2a. Quantize activations
            # Add epsilon to prevent division by zero if x_amax is 0
            x_amax = torch.abs(x).max().to(torch.float32)
            x_global_scale = (self.FLOAT8_E4M3_MAX * self.FLOAT4_E2M1_MAX) / (x_amax + 1e-6)
            x_fp4, scale_x_fp4 = ops.scaled_fp4_quant(x, x_global_scale)

            # 2b. Quantize weights (also done on every pass for torch.compile)
            w_amax = torch.abs(weight).max().to(torch.float32)
            w_global_scale = (self.FLOAT8_E4M3_MAX * self.FLOAT4_E2M1_MAX) / (w_amax + 1e-6)
            w_fp4, scale_w_fp4 = ops.scaled_fp4_quant(weight, w_global_scale)
            
            # 2c. Calculate alpha
            alpha = 1.0 / (x_global_scale * w_global_scale)

            # 2d. Perform computation
            # x_fp4 shape: [..., K(In)]
            # w_fp4 shape: [K(In), M(Out)]
            output = ops.cutlass_scaled_fp4_mm(
                x_fp4, w_fp4, scale_x_fp4, scale_w_fp4, alpha, x.dtype
            )
        
        else:
            # 'emulation' mode is not supported by vLLM backend
             raise NotImplementedError(
                 f"Quant mode '{mode}' is not implemented for vLLM backend. "
                 f"Use --backend hf for 'emulation' mode."
             )

        # Add bias (if it exists)
        if bias is not None:
            output += bias

        return output


@register_quantization_config("nvfp_quant")
class NVFPQuantConfig(QuantizationConfig):
    """
    NVFP quantization config.
    
    This config is injected by the custom model (e.g., Qwen2ForCausalLM_mxfp)
    and read by vLLM's Linear layers. It passes all necessary parameters
    (w_bit, a_bit, group_size, and quant_mode) to the NVFPvLLMMethod.
    """

    def __init__(
        self,
        w_bit: int = 4,
        a_bit: int = 4,
        group_size: int = 16,
        quant_mode: str = "pseudo", # [NEW] Added quant_mode
        ant_config: dict = None,    # [LEGACY] Kept for compatibility
    ) -> None:
        super().__init__()
        self.w_bit = w_bit
        self.a_bit = a_bit
        self.group_size = group_size
        self.quant_mode = quant_mode # [NEW] Store the mode
        self.ant_config = ant_config or {
            "weight_mxfp_mode": "base",
            "input_mxfp_mode": "base",
            "ant_mode": "float",
            "weight_sub_group_size": None,
            "weight_sub_group_mode": None,
            "input_sub_group_size": None,
            "input_sub_group_mode": None,
        }

        # vLLM backend cannot run 'emulation' mode as it requires HF-side logic
        if self.quant_mode not in ("pseudo", "real"):
            raise ValueError(
                f"vLLM backend received invalid quant_mode '{self.quant_mode}'. "
                f"Must be 'pseudo' or 'real'. Use --backend hf for 'emulation'.")
        
        print(f"[NVFPQuantConfig] Initialized with w_bit={w_bit}, a_bit={a_bit}, mode='{self.quant_mode}'")


    def get_name(self) -> str:
        return "nvfp_quant"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return -1 # Supports all

    @staticmethod
    def get_config_filenames() -> list[str]:
        # No separate config file needed, params are passed from python
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "NVFPQuantConfig":
        """
        Parses config from vllm_config.quant_config dict
        (which is set in qwen_mxfp.py)
        """
        return NVFPQuantConfig(
            w_bit=config.get("w_bit", 4),
            a_bit=config.get("a_bit", 4),
            group_size=config.get("group_size", 16),
            quant_mode=config.get("quant_mode", "pseudo"), # [NEW] Read quant_mode
            ant_config=config.get("ant_config"),
        )

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> Optional[NVFPvLLMMethod]:
        
        if isinstance(layer, LinearBase):
            # Pass the *entire config* (self) to the method
            return NVFPvLLMMethod(quant_config=self)
        
        return None