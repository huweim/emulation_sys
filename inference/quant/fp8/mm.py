import torch
import scaled_fp8_ops as ops

@torch.no_grad()
def fp8_mm(x: torch.Tensor, s_x:torch.Tensor,
           w: torch.Tensor, s_w:torch.Tensor,
           group_size: int = 32, bias: float = None) -> torch.Tensor:
    '''
    -deq a
    -deq w
    -gemm
    '''
    x_deq = torch.empty_like(x, dtype=torch.bfloat16)
    w_deq = torch.empty_like(w, dtype=torch.bfloat16)

    # [M, K](fp8), [M, K/G](f32) -> [M, K](bf16)
    # print(x.shape)
    ops.groupwise_dequant_fp8_bf16(
        x, s_x, x_deq, group_size
    )

    # [N, K](fp8), [N, K/G](f32) -> [N, K](bf16)
    ops.groupwise_dequant_fp8_bf16(
        w, s_w, w_deq, group_size
    )

    # [M, K] @ [K, N] -> [M, N]
    output_bf16 = torch.matmul(x_deq, w_deq.t())
    # output_bf16 = torch.matmul(x, w.t())

    if bias is not None:
        output_bf16 += bias.to(torch.bfloat16)

    return output_bf16