"""
Utility classes: NVFP4Utils and DataGenerator
Fully aligned with verify_acc_modeling.py
"""
import torch
import random


class NVFP4Utils:
    """负责底层映射与解包"""
    _TABLE_CACHE = {}

    @staticmethod
    def get_fp4_e2m1_table(device="cuda"):
        key = str(device)
        if key not in NVFP4Utils._TABLE_CACHE:
            pos_vals = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
            neg_vals = [-0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
            NVFP4Utils._TABLE_CACHE[key] = torch.tensor(pos_vals + neg_vals, device=device, dtype=torch.float16)
        return NVFP4Utils._TABLE_CACHE[key]

    @staticmethod
    def unpack_nvfp4_to_fp16(packed_uint8, original_shape):
        device = packed_uint8.device
        table = NVFP4Utils.get_fp4_e2m1_table(device)
        low = packed_uint8 & 0x0F
        high = (packed_uint8 >> 4) & 0x0F
        unpacked = torch.stack([low, high], dim=-1).view(original_shape)
        return table[unpacked.long()]


class DataGenerator:
    """
    生成各种分布的测试数据
    与 verify_acc_modeling.py 完全一致
    """
    @staticmethod
    def get_random_tensor(shape, dist_type, device="cuda", dtype=torch.float16):
        if dist_type == "normal":
            return torch.randn(shape, device=device, dtype=dtype)
        elif dist_type == "uniform":
            return (torch.rand(shape, device=device, dtype=dtype) * 2 - 1)
        elif dist_type == "large":
            return torch.randn(shape, device=device, dtype=dtype) * 100.0
        elif dist_type == "small":
            return torch.randn(shape, device=device, dtype=dtype) * 0.001
        elif dist_type == "outliers":
            t = torch.randn(shape, device=device, dtype=dtype)
            mask = torch.rand(shape, device=device) < 0.01
            t[mask] *= 50.0
            return t
        elif dist_type == "mixed_rows":
            t = torch.randn(shape, device=device, dtype=dtype)
            scale = torch.exp(torch.randn(shape[0], 1, device=device) * 2)
            return t * scale.to(dtype)
        elif dist_type == "abs_large":
             return torch.abs(torch.randn(shape, device=device, dtype=dtype)) * 100.0
        else:
            raise ValueError(f"Unknown distribution type: {dist_type}")


class DataGenerator_Abs:
    """
    DEBUG: 只生成正数，便于分析
    """
    @staticmethod
    def get_random_tensor(shape, dist_type, device="cuda", dtype=torch.float16):
        if dist_type == "normal":
            return torch.abs(torch.randn(shape, device=device, dtype=dtype))
        elif dist_type == "uniform":
            return torch.rand(shape, device=device, dtype=dtype)  # 0 to 1, positive only
        elif dist_type == "large":
            return torch.abs(torch.randn(shape, device=device, dtype=dtype)) * 100.0
        elif dist_type == "small":
            return torch.abs(torch.randn(shape, device=device, dtype=dtype)) * 0.001
        elif dist_type == "outliers":
            t = torch.abs(torch.randn(shape, device=device, dtype=dtype))
            mask = torch.rand(shape, device=device) < 0.01
            t[mask] *= 50.0
            return t
        elif dist_type == "mixed_rows":
            t = torch.abs(torch.randn(shape, device=device, dtype=dtype))
            scale = torch.exp(torch.randn(shape[0], 1, device=device) * 2)
            return t * scale.to(dtype)
        elif dist_type == "abs_large":
             return torch.abs(torch.randn(shape, device=device, dtype=dtype)) * 100.0
        else:
            raise ValueError(f"Unknown distribution type: {dist_type}")
