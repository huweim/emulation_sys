import torch
import torch.nn as nn
import torch.nn.functional as F
import nvfp.ops as ops
import nvfp.pseudo_quant as pseudo_quant


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

        with torch.no_grad():
            W = lin.weight.detach()  # [out, in]
            if self.mode == "pseudo":
                Wq = pseudo_quant.nvfp4_pseudo_quantize(W).to(self.dtype)
                self.register_buffer("qweight_fp", Wq)
            else:  # "real"
                self.FLOAT4_E2M1_MAX = 6.0
                self.FLOAT8_E4M3_MAX = 448.0
                w_amax = torch.abs(W).max().to(torch.float32)
                w_global_scale = self.FLOAT8_E4M3_MAX * self.FLOAT4_E2M1_MAX / w_amax
                w_fp4, scale_w_fp4 = ops.scaled_fp4_quant(W, w_global_scale)
                self.register_buffer("w_fp4", w_fp4)
                self.register_buffer("w_scale_fp4", scale_w_fp4)
                self.w_global_scale = w_global_scale
                self.qweight_fp = None  # not used in real mode

        if self.bias is not None and isinstance(self.bias, torch.Tensor):
            self.bias = self.bias.to(self.dtype)

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
                x_q = pseudo_quant.nvfp4_pseudo_quantize(x)
                x_q = x_q.to(dtype=self.dtype, device=x_q.device)
                x_q = x_q.clone().detach().contiguous()
            else:
                x_q = x.to(self.dtype)
            return F.linear(x_q, self.qweight_fp, self.bias)
        else:  # "real"
            x_amax = torch.abs(x).max().to(torch.float32)
            x_global_scale = self.FLOAT8_E4M3_MAX * self.FLOAT4_E2M1_MAX / x_amax
            x_fp4, scale_x_fp4 = ops.scaled_fp4_quant(x, x_global_scale)

            alpha = 1.0 / (x_global_scale * self.w_global_scale)

            output = ops.cutlass_scaled_fp4_mm(
                x_fp4, self.w_fp4, scale_x_fp4, self.w_scale_fp4, alpha, self.dtype
            )

            # reshape output to original batch shape
            output = output.view(*original_shape_prefix, self.out_features)

            if self.bias is not None:
                output += self.bias

            return output.to(self.dtype)
