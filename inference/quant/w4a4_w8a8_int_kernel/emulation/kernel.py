"""
Deterministic emulation kernel APIs for W4A4 / W8A8 int GEMM.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from .core import MMAEngine

Mode = Literal["w4a4", "w8a8"]


@dataclass
class EmulationConfig:
    mode: Mode
    packed_w8_weight: bool = True


class EmulationKernel:
    """Simple callable emulation kernel wrapper."""
    
    def __init__(self, config: EmulationConfig):
        self.config = config
    
    def __call__(
        self,
        act: torch.Tensor,
        wgt: torch.Tensor,
        ascales: torch.Tensor,
        wscales: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.forward(act, wgt, ascales, wscales, bias=bias)
    
    def forward(
        self,
        act: torch.Tensor,
        wgt: torch.Tensor,
        ascales: torch.Tensor,
        wscales: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.config.mode == "w4a4":
            return MMAEngine.emulation_gemm_w4a4(act, wgt, ascales, wscales, bias=bias)
        return MMAEngine.emulation_gemm_w8a8(
            act,
            wgt,
            ascales,
            wscales,
            bias=bias,
            packed_wgt=self.config.packed_w8_weight,
        )
    
    @classmethod
    def for_w4a4(cls) -> "EmulationKernel":
        return cls(EmulationConfig(mode="w4a4"))
    
    @classmethod
    def for_w8a8(cls, packed_w8_weight: bool = True) -> "EmulationKernel":
        return cls(EmulationConfig(mode="w8a8", packed_w8_weight=packed_w8_weight))


def emulated_gemm_w4a4(
    act: torch.Tensor,
    wgt: torch.Tensor,
    ascales: torch.Tensor,
    wscales: torch.Tensor,
    bias: torch.Tensor | None = None,
    return_debug: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict]:
    return MMAEngine.emulation_gemm_w4a4(
        act,
        wgt,
        ascales,
        wscales,
        bias=bias,
        return_debug=return_debug,
    )


def emulated_gemm_w8a8(
    act: torch.Tensor,
    wgt: torch.Tensor,
    ascales: torch.Tensor,
    wscales: torch.Tensor,
    bias: torch.Tensor | None = None,
    packed_wgt: bool = True,
    return_debug: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict]:
    return MMAEngine.emulation_gemm_w8a8(
        act,
        wgt,
        ascales,
        wscales,
        bias=bias,
        packed_wgt=packed_wgt,
        return_debug=return_debug,
    )
