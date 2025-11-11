import torch
import torch.nn as nn
import torch.nn.functional as F
import scaled_fp8_ops as ops
import fp8_quant_utils

class QuantLinear(nn.Module):
    def __init__(
        self,
        lin: nn.Linear,
        w_bit: int,
        a_bit: int,
        group_size: int = 32,
        granularity_m: int = 1,
        granularity_n: int = 128,
        granularity_k: int = 128,
        use_zero_point: bool = False,
        mode: str = "pseudo",
    ):
        super().__init__()
        assert mode in ("pseudo", "real"), "mode must be 'pseudo' or 'real'"

        self.in_features = lin.in_features
        self.out_features = lin.out_features
        if lin.bias is not None:
            self.register_buffer("bias", lin.bias.detach().clone())
        else:
            self.bias = None
        self.w_bit = w_bit
        self.a_bit = a_bit
        self.dtype = lin.weight.dtype
        self.granularity_m= granularity_m
        self.granularity_n= granularity_n
        self.granularity_k= granularity_k
        self.use_zero_point = use_zero_point
        self.mode = mode
        self.FLOAT8_E4M3_MAX = 448.0
        self.FLOAT8_E4M3_MIN = -448.0

        with torch.no_grad():
            W = lin.weight.detach()  # [out, in]
            if self.mode == "pseudo":
                Wq = fp8_quant_utils.fp8_pseudo_quantize_groupwise(W)
                self.register_buffer("qweight_fp", Wq)
            else:  # "real"
                N = W.shape[0]
                K = W.shape[1]
                assert (N % self.granularity_n) == 0, "n必须能被 granularity_n 整除"
                assert (K % self.granularity_k) == 0, "k必须能被 granularity_k 整除"
                W_q = torch.empty_like(W, dtype=torch.float8_e4m3fn)
                s_w = torch.empty(N // self.granularity_n, K // self.granularity_k, device=W.device, dtype=torch.float32)
                fp8_quant_utils.fp8_quantize_blockwise(W, W_q, s_w, weight=True)
                self.register_buffer("w_fp8", W_q)
                self.register_buffer("w_scale", s_w)
                self.qweight_fp = None  # real 路径不使用

        if self.bias is not None and isinstance(self.bias, torch.Tensor):
            self.bias = self.bias.float()

    @classmethod
    def from_linear(
        cls, lin, w_bit, a_bit, group_size=32, use_zero_point=False, mode="pseudo"
    ):
        return cls(lin, w_bit, a_bit, group_size, use_zero_point, mode)

    def forward(self, x):
        original_shape_prefix = x.shape[:-1]
        if self.mode == "pseudo":
            if self.a_bit is not None and self.a_bit < 16:
                x_q = fp8_quant_utils.fp8_pseudo_quantize_groupwise(x)
                x_q = x_q.to(dtype=self.dtype, device=x_q.device)
                x_q = x_q.clone().detach().contiguous()
            else:
                x_q = x.to(self.dtype)
            return F.linear(x_q, self.qweight_fp, self.bias).to(self.dtype)
        else:
            x_reshaped = x.reshape(-1, x.shape[-1])
            M, K = x_reshaped.shape
            assert (M % self.granularity_m) == 0, "M 必须能被 granularity_m 整除"
            assert (K % self.granularity_k) == 0, "k 必须能被 granularity_k 整除"

            x_q = torch.empty_like(x_reshaped, dtype=torch.float8_e4m3fn, device=x.device)
            s_x = torch.empty(M // self.granularity_m, K // self.granularity_k, dtype=torch.float32, device=x.device)
            output_bf16 = torch.empty((M, self.out_features), dtype=torch.bfloat16, device=x.device)

            ops.per_token_group_quant_fp8(
                x_reshaped, x_q, s_x, self.granularity_k, 1e-8, self.FLOAT8_E4M3_MIN, self.FLOAT8_E4M3_MAX, False
            )
            
            ops.cutlass_scaled_mm_blockwise_sm120_fp8(
                output_bf16,
                x_q,
                self.w_fp8,
                s_x.T.contiguous(),
                self.w_scale
            )

            output = output_bf16.reshape(*original_shape_prefix, self.out_features).to(self.dtype)
            # if torch.isnan(output).any():
            #     print("output nan!")
            return output

