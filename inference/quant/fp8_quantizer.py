import torch
import torch.nn as nn
import torch.nn.functional as F
# import nvfp.ops as ops
# import nvfp.pseudo_quant as pseudo_quant
import scaled_fp8_ops as ops
import fp8_pseudo_quantize
import mm



class QuantLinear(nn.Module):
    def __init__(
        self,
        lin: nn.Linear,
        w_bit: int,
        a_bit: int,
        group_size: int = 32,
        use_zero_point: bool = False,
        mode: str = "pseudo",  # "pseudo" or "real"
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
        self.group_size = group_size
        self.use_zero_point = use_zero_point
        self.mode = mode
        self.FLOAT8_E4M3_MAX = 448.0
        self.FLOAT8_E4M3_MIN = -448.0

        with torch.no_grad():
            W = lin.weight.detach()  # [out, in]
            if self.mode == "pseudo":
                # print(W.dtype)# llama-2-7b-hf -> float16
                Wq = fp8_pseudo_quantize.fp8_pseudo_quantize_groupwise(W)
                self.register_buffer("qweight_fp", Wq)
            else:  # "real"
                N, K = W.shape
                assert (K % self.group_size) == 0, "必须能被 group_size 整除"

                # 输出张量：W_q(fp8) 与 s_w(fp32)；scale shape 为 [K, N/G]
                W_q = torch.empty_like(W, dtype=torch.float8_e4m3fn)
                s_w = torch.empty(N, K // self.group_size, device=W.device, dtype=torch.float32)

                # per_token_group_quant_fp8(input, output_q, output_s, group_size, eps, fp8_min, fp8_max, scale_ue8m0)
                ops.per_token_group_quant_fp8(
                    W, W_q, s_w, self.group_size, 1e-8, self.FLOAT8_E4M3_MIN, self.FLOAT8_E4M3_MAX, False
                )

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
        # x: [*, in_features]
        original_shape_prefix = x.shape[:-1]
        
        if self.mode == "pseudo":
            if self.a_bit is not None and self.a_bit < 16:
                x_q = fp8_pseudo_quantize.fp8_pseudo_quantize_groupwise(x)
                x_q = x_q.to(dtype=self.dtype, device=x_q.device)
                x_q = x_q.clone().detach().contiguous()
            else:
                x_q = x.to(self.dtype)
            return F.linear(x_q, self.qweight_fp, self.bias).to(self.dtype)
        else:
            x = x.reshape(-1, x.shape[-1])
            M, N = x.shape
            assert (N % self.group_size) == 0, "in_features 必须能被 group_size 整除"

            x_q=torch.empty_like(x, dtype=torch.float8_e4m3fn, device=x.device)
            s_x=torch.empty(M, N // self.group_size, dtype=torch.float32, device=x.device)
            # per_token_group_quant_fp8(input, output_q, output_s, group_size, eps, fp8_min, fp8_max, scale_ue8m0)
            ops.per_token_group_quant_fp8(
                x, x_q, s_x, self.group_size, 1e-8, self.FLOAT8_E4M3_MIN, self.FLOAT8_E4M3_MAX, False
            )
            
            #todo:fp8 mm
            output_bf16 = mm.fp8_mm(x_q, s_x, self.w_fp8, self.w_scale, self.group_size, self.bias)

            output = output_bf16.reshape(*original_shape_prefix, self.w_fp8.shape[0])
            return output.to(self.dtype)
        
