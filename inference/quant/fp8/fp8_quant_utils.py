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

@torch.no_grad()
def fp8_quantize_blockwise(
    x: torch.Tensor,
    x_q: torch.Tensor,
    x_s: torch.Tensor,
    A_M_GRANULARITY: int = 1,
    A_K_GRANULARITY: int = 128, 
    B_K_GRANULARITY: int = 128,
    B_N_GRANULARITY: int = 128,
    weight: bool = False
    ) -> torch.Tensor:
    
    if(weight):
        # 量化 B (粒度 128 x 128)
        N = x.shape[0]
        K = x.shape[1]
        for k_block_idx in range(K // B_K_GRANULARITY):
            k_start = k_block_idx * B_K_GRANULARITY
            k_end = (k_block_idx + 1) * B_K_GRANULARITY
            for n_block_idx in range(N // B_N_GRANULARITY):
                n_start = n_block_idx * B_N_GRANULARITY
                n_end = (n_block_idx + 1) * B_N_GRANULARITY
                
                b_chunk = x[n_start:n_end, k_start:k_end]
                abs_max = b_chunk.abs().max()
                scale = abs_max / FP8_E4M3_MAX
                scale = torch.clamp(scale, min=1e-8)
                
                x_s[n_block_idx, k_block_idx] = scale
                x_q[n_start:n_end, k_start:k_end] = torch.clamp(
                    b_chunk / scale, min=FP8_E4M3_MIN, max=FP8_E4M3_MAX
                ).to(torch.float8_e4m3fn)
    else:
        # 太慢了，换成kernel了
        # 1 x 128
        M = x.shape[0]
        K = x.shape[1]
        # print(f"M:{M},K:{K}")

        for m_block_idx in range(M // A_M_GRANULARITY):
            m_start = m_block_idx * A_M_GRANULARITY
            m_end = (m_block_idx + 1) * A_M_GRANULARITY

            for k_block_idx in range(K // A_K_GRANULARITY):
                k_start = k_block_idx * A_K_GRANULARITY
                k_end = (k_block_idx + 1) * A_K_GRANULARITY

                a_chunk = x[m_start:m_end, k_start:k_end]
                abs_max = a_chunk.abs().max()
                scale = abs_max / FP8_E4M3_MAX
                scale = torch.clamp(scale, min=1e-8) 

                x_s[m_block_idx, k_block_idx] = scale
                x_q[m_start:m_end, k_start:k_end] = torch.clamp( 
                    a_chunk / scale, min=FP8_E4M3_MIN, max=FP8_E4M3_MAX
                ).to(torch.float8_e4m3fn)

        

        