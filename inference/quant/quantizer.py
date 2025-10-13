import torch    
import torch.nn as nn
import torch.nn.functional as F

def _calc_qparams(x, n_bit, symmetric=True, eps=1e-6, min_val=None, max_val=None, group_size=128):
    org_shape = x.shape
    # 按行分组：每行的 in_features 维按 group_size 切
    assert x.shape[-1] % group_size == 0, "in_features must be divisible by group_size"
    x_g = _group_view_lastdim(x, group_size)  # [out, G, group]
    # 对每个 [out, G, group] 的最后一维 group 求 qparams

    if min_val is None or max_val is None:
        min_val = x_g.amin(dim=-1, keepdim=True)
        max_val = x_g.amax(dim=-1, keepdim=True)
    max_val = torch.max(max_val, min_val + eps)
    if symmetric:
        max_abs = torch.max(-min_val, max_val)
        qmax = 2 ** (n_bit - 1) - 1
        scale = (max_abs / qmax).clamp_min(eps) 
        zp = torch.zeros_like(scale)
    else:
        qmax = 2**n_bit - 1
        qmin = 0
        scale = (max_val - min_val) / max(qmax - qmin, 1)
        scale = scale.clamp_min(eps)
        zp = torch.round(qmin - min_val / scale).clamp(qmin, qmax)

    scale = scale.expand(*scale.shape[:-1], group_size).reshape(org_shape)
    zp = zp.expand(*zp.shape[:-1], group_size).reshape(org_shape)
    return scale, zp


def _fake_quant(x, n_bit, scale, zp=None, symmetric=True, qmin=None, qmax=None):
    if symmetric:
        qmin = -(2 ** (n_bit - 1))
        qmax = 2 ** (n_bit - 1) - 1
        zp = 0 if zp is None else zp
    else:
        qmin = 0 if qmin is None else qmin
        qmax = 2**n_bit - 1 if qmax is None else qmax
        zp = 0 if zp is None else zp
    x_int = torch.round(x / scale + zp).clamp(qmin, qmax)
    x_deq = (x_int - zp) * scale
    return x_deq

def _quantize_to_int(x, n_bit, group_size=128, symmetric=True):
    """
    Return integer tensor (kept in int8 for convenience) + per-element scale (broadcasted)
    NOTE: we DO NOT bit-pack nibble here; kernel can do packing later.
    """
    scale, zp = _calc_qparams(x, n_bit, symmetric=symmetric, group_size=group_size)
    if symmetric:
        qmin, qmax, zp = -(2 ** (n_bit - 1)), 2 ** (n_bit - 1) - 1, 0
        x_int = torch.round(x / scale).clamp(qmin, qmax)
    else:
        qmin, qmax = 0, 2 ** n_bit - 1
        x_int = torch.round(x / scale + zp).clamp(qmin, qmax)

    # store as int8 (range is still respected for 4/8-bit), no packing yet
    x_int = x_int.to(torch.int8)
    return x_int, scale, (None if symmetric else zp)


def _group_view_lastdim(x, group_size):
    # 把最后一维切成 (num_groups, group_size)
    assert x.shape[-1] % group_size == 0
    new_shape = x.shape[:-1] + (x.shape[-1] // group_size, group_size)
    return x.reshape(*new_shape)

class QuantLinear(nn.Module):
    """
    A Linear that can run in two modes:
      - 'pseudo': W/A fake-quant and immediately dequantize -> F.linear
      - 'real'  : W/A integer quant + (placeholder) low-bit GEMM API
    Both modes share the same scale computation logic.
    """
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
        self.group_size = group_size
        self.use_zero_point = use_zero_point
        self.mode = mode

        # 做一次“权重量化→反量化”，存成浮点（fake-quant）
        with torch.no_grad():   
            W = lin.weight.detach()  # [out, in]
            if self.mode == "pseudo":
                if self.use_zero_point:
                    scale_w, zp_w = _calc_qparams(W, self.w_bit, symmetric=False, group_size=self.group_size)
                    Wq = _fake_quant(W, self.w_bit, scale_w, zp_w, symmetric=False)
                else:
                    scale_w, zp_w = _calc_qparams(W, self.w_bit, symmetric=True, group_size=self.group_size)
                    Wq = _fake_quant(W, self.w_bit, scale_w, None, symmetric=True)
                self.register_buffer("qweight_fp", Wq.to(W.dtype))
            else:  # "real"
                # store int weights + scale (no packing here)
                if self.use_zero_point:
                    raise NotImplementedError("real mode currently supports symmetric only")
                w_int, w_scale, _ = _quantize_to_int(W, self.w_bit, group_size=self.group_size, symmetric=True)
                self.register_buffer("w_int", w_int)             # int8
                self.register_buffer("w_scale", w_scale)         # float, broadcastable
                self.w_zp = None
                self.qweight_fp = None  # not used in real mode
                # make sure bias dtype aligns with weight dtype
        
        if self.bias is not None and isinstance(self.bias, torch.Tensor):
            self.bias = self.bias.to(W.dtype)


    @classmethod
    def from_linear(cls, lin, w_bit, a_bit, group_size=32, use_zero_point=False, mode="pseudo"):
        return cls(lin, w_bit, a_bit, group_size, use_zero_point, mode)

    def forward(self, x):
        # x: [*, in_features]
        if self.mode == "pseudo":
            if self.a_bit is not None and self.a_bit < 16:
                # 动态逐张量伪量化（也可换成逐通道/逐组，这里先保守做法）
                if self.use_zero_point:
                    scale, zp = _calc_qparams(x, self.a_bit, symmetric=False, group_size=self.group_size)
                    x_q = _fake_quant(x, self.a_bit, scale, zp, symmetric=False)
                else:
                    scale, zp = _calc_qparams(x, self.a_bit, symmetric=True, group_size=self.group_size)
                    x_q = _fake_quant(x, self.a_bit, scale, None, symmetric=True)
                x_q = x_q.reshape_as(x)
            else:
                x_q = x
            return F.linear(x_q, self.qweight_fp, self.bias)
        else:  # "real"
            # TODO: REAL quantization implementation
            if self.use_zero_point:
                raise NotImplementedError("real mode currently supports symmetric only")
            x_int, a_scale, _ = _quantize_to_int(x, self.a_bit, group_size=self.group_size, symmetric=True)

            # TODO: call lowbit GEMM placeholder (replace later with CUDA kernel)
            return lowbit_gemm_kernel