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

from inference.quant.nvfp_quantizer import QuantLinear


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

    def apply(
        self,
        layer: torch.nn.Module, # This is the vLLM layer (e.g., QKVParallelLinear)
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 1. On the first call, Linear → NVFP_Linear
        if not hasattr(layer, "_nvfp_linear"):
            # layer.weight -> [out, in]
            layer.in_features = layer.weight.shape[1]
            layer.out_features = layer.weight.shape[0]

            layer._nvfp_linear = QuantLinear.from_linear(
                lin=layer,
                w_bit=self.cfg.w_bit,
                a_bit=self.cfg.a_bit,
                group_size=self.cfg.group_size,
                mode=self.cfg.quant_mode,
                use_zero_point=self.cfg.use_zero_point,
            )
        
        nvfp_linear = layer._nvfp_linear

        # 2. NVFP_Linear.forward already handles weight/activation quantization internally
        out = nvfp_linear(x)

        return out


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
        quant_mode: str = "pseudo", 
        use_zero_point: bool = False,
    ) -> None:
        super().__init__()
        self.w_bit = w_bit
        self.a_bit = a_bit
        self.group_size = group_size
        self.quant_mode = quant_mode 
        self.use_zero_point = use_zero_point

        # vLLM backend cannot run 'emulation' mode as it requires HF-side logic
        if self.quant_mode not in ("pseudo", "real"):
            raise ValueError(
                f"vLLM backend received invalid quant_mode '{self.quant_mode}'. "
                f"Must be 'pseudo' or 'real'. Use --backend hf for 'emulation'.")
        
        print(f"[NVFPQuantConfig] Initialized with w_bit={w_bit}, a_bit={a_bit}, mode={self.quant_mode}, group_size={group_size}")


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
            quant_mode=config.get("quant_mode", "pseudo"), 
            use_zero_point=config.get("use_zero_point", False),
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