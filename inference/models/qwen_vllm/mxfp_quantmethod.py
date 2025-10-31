# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests register custom quantization config.

See https://github.com/vllm-project/vllm/issues/11926 for more details.

Run `pytest tests/quantization/test_register_quantization_config.py`.
"""
from typing import Any, Optional

import torch
import torch.nn.functional as F

from vllm.model_executor.layers.linear import LinearBase  # noqa: E501
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import (  # noqa: E501
    QuantizationConfig)

from mxq.quantize.qmodule_mxfp import MXFP_Linear

class FakeQuantLinearMethod(UnquantizedLinearMethod):
    """
    MXFP fake-quantization linear method.
    Reuses the exact quantization pipeline from MXFP_Linear.
    """
 
    def __init__(
        self,
        w_bit: int = 16,
        a_bit: int = 16,
        group_size: int = 32,
        ant_config: dict = None,
    ) -> None:
        super().__init__()
        self.w_bit = w_bit 
        self.a_bit = a_bit 
        self.group_size  = group_size 
        self.ant_config  = ant_config or {}
 
    def apply(
        self,
        layer: torch.nn.Module, 
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 1. 第一次调用时，把普通 Linear → MXFP_Linear 
        if not hasattr(layer, "_mxfp_linear"):
            layer._mxfp_linear = MXFP_Linear.from_linear( 
                linear=layer,
                w_bit=self.w_bit,
                a_bit=self.a_bit,
                group_size=self.group_size, 
                layer_id=getattr(layer, "layer_id", 0),
                layer_name=getattr(layer, "layer_name", "unknown"),
                ant_config=self.ant_config, 
            )
 
        mxfp_linear = layer._mxfp_linear 
 
        # 2. MXFP_Linear.forward  内部已经做了权重/输入量化 
        out = mxfp_linear(x).to(torch.bfloat16)

        return out 
    
@register_quantization_config("mxfp_quant")
class MXFPQuantConfig(QuantizationConfig):
    """
    MXFP quantization config.
    Replaces the old per-token dynamic fake-quant with MXFP.
    """
 
    def __init__(
        self,
        w_bit: int = 16,
        a_bit: int = 16,
        group_size: int = 32,
        ant_config: dict = None,
    ) -> None:
        super().__init__()
        self.w_bit = w_bit 
        self.a_bit = a_bit 
        self.group_size  = group_size 
        # 如果用户没给，给一个默认字典 
        self.ant_config  = ant_config or {
            "weight_mxfp_mode": "base",
            "input_mxfp_mode": "base",
            "ant_mode": "float",
            "weight_sub_group_size": None,
            "weight_sub_group_mode": None,
            "input_sub_group_size": None,
            "input_sub_group_mode": None,
        }
 
    def get_name(self) -> str:
        return "mxfp_quant"
 
    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.float16, torch.bfloat16] 
 
    @classmethod 
    def get_min_capability(cls) -> int:
        return -1 
 
    @staticmethod 
    def get_config_filenames() -> list[str]:
        # 可以按需改成 mxfp_quant.json  等 
        return []
 
    @classmethod 
    def from_config(cls, config: dict[str, Any]) -> "MXFPQuantConfig":
        """
        从模型目录下的 quantization_config 解析字段。
        示例：
        {
          "quant_method": "mxfp_quant",
          "w_bit": 4,
          "a_bit": 8,
          "group_size": 32,
          "ant_config": {...}
        }
        """
        return MXFPQuantConfig(
            w_bit=config.get("w_bit",  16),
            a_bit=config.get("a_bit",  16),
            group_size=config.get("group_size",  32),
            ant_config=config.get("ant_config"), 
        )
 
    def get_quant_method(
        self,
        layer: torch.nn.Module, 
        prefix: str,
    ) -> Optional[FakeQuantLinearMethod]:
        if isinstance(layer, LinearBase):
            return FakeQuantLinearMethod(
                w_bit=self.w_bit,
                a_bit=self.a_bit,
                group_size=self.group_size, 
                ant_config=self.ant_config, 
            )
        return None 