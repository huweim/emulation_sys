import torch

FP8_E4M3_MAX = 448.0
FP8_E4M3_MIN = -448.0

@torch.no_grad()
def fp8_pseudo_quantize_groupwise(x: torch.Tensor, group_size: int = 32, eps: float = 1e-8) -> torch.Tensor:
    """
    Group-wise pseudo quantization to FP8 e4m3fn, then dequantize back to fp32.
    - 沿最后一维做分组，组内共享一个 scale。
    - 量化：q = clamp_to_fp8(x / s)，反量化：y = q.to(float32) * s
    - 要求最后一维可被 group_size 整除。
    """
    assert x.dim() >= 2, "输入至少为 2D(按最后一维分组)"
    K = x.size(-1)
    assert K % group_size == 0, "最后一维长度必须能被 group_size 整除"

    out_dtype = x.dtype
    x2d = x.reshape(-1, K)  # 合并前缀维度 -> [M, K]
    M = x2d.size(0)
    x_groups = x2d.view(M, K // group_size, group_size)
    amax = x_groups.abs().amax(dim=-1)  # [M, K/G]
    scale = (amax / FP8_E4M3_MAX).clamp_min(eps)  # [M, K/G]

    #扩一下方便乘
    scale_broadcast = scale.unsqueeze(-1).expand(M, K // group_size, group_size).reshape(M, K)

    # 量化到 FP8，再反量化回 float，再 cast 回原始 dtype
    q_fp8 = (x2d / scale_broadcast).to(torch.float8_e4m3fn)              # [M, K] (FP8)
    y_f32 = q_fp8.to(torch.float32) * scale_broadcast                     # [M, K] (f32)
    # y = y_f32.to(out_dtype).reshape(x.shape)                              # 恢复原形状与 dtype
    y = y_f32.reshape(x.shape)#反到32位回去做乘法
    return y.to(out_dtype)