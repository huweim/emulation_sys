"""
Hardware Configuration for NVFP4 Emulation
"""
from dataclasses import dataclass
from typing import Optional

from .rounding import RoundStrategy


@dataclass
class HardwareConfig:
    """
    Hardware-specific configuration for NVFP4 emulation.
    
    Attributes:
        w_stage3: Bit width for Stage 3 (intra-block 4-to-1 reduction)
        w_stage4: Bit width for Stage 4 (inter-block accumulation)
        stage3_rounding: Rounding strategy for Stage 3
        stage4_rounding: Rounding strategy for Stage 4
        name: Optional name for this configuration
        description: Optional description
    """
    w_stage3: int = 34
    w_stage4: int = 28
    stage3_rounding: RoundStrategy = RoundStrategy.RZ
    stage4_rounding: RoundStrategy = RoundStrategy.RZ
    name: str = "custom"
    description: str = ""
    
    def __post_init__(self):
        """Validate configuration."""
        if self.w_stage3 < 20 or self.w_stage3 > 50:
            raise ValueError(f"w_stage3 should be in [20, 50], got {self.w_stage3}")
        if self.w_stage4 < 20 or self.w_stage4 > 50:
            raise ValueError(f"w_stage4 should be in [20, 50], got {self.w_stage4}")
    
    @classmethod
    def for_rtx_5090(cls) -> "HardwareConfig":
        """
        Optimal configuration for NVIDIA RTX 5090.
        
        Verified with 10,000+ test iterations across various matrix sizes
        and data distributions.
        """
        return cls(
            w_stage3=34,
            w_stage4=28,
            stage3_rounding=RoundStrategy.RZ,
            stage4_rounding=RoundStrategy.RZ,
            name="RTX_5090",
            description="NVIDIA RTX 5090 with NVFP4 input, FP16 output"
        )
    
    @classmethod
    def from_dict(cls, d: dict) -> "HardwareConfig":
        """Create from dictionary."""
        # Convert string rounding to enum
        if "stage3_rounding" in d and isinstance(d["stage3_rounding"], str):
            d["stage3_rounding"] = RoundStrategy[d["stage3_rounding"]]
        if "stage4_rounding" in d and isinstance(d["stage4_rounding"], str):
            d["stage4_rounding"] = RoundStrategy[d["stage4_rounding"]]
        return cls(**d)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "w_stage3": self.w_stage3,
            "w_stage4": self.w_stage4,
            "stage3_rounding": self.stage3_rounding.name,
            "stage4_rounding": self.stage4_rounding.name,
            "name": self.name,
            "description": self.description,
        }


# Predefined configurations
CONFIG_RTX_5090 = HardwareConfig.for_rtx_5090()
