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

def _group_view_lastdim(x, group_size):
    # 把最后一维切成 (num_groups, group_size)
    assert x.shape[-1] % group_size == 0
    new_shape = x.shape[:-1] + (x.shape[-1] // group_size, group_size)
    return x.reshape(*new_shape)

class QuantLinear(nn.Module):
    """
    伪量化 Linear：
      - 权重：按 out_features 行，沿 in_features 做 group-wise 量化（一次离线，存成浮点的“量化后权重”）
      - 激活：逐张量动态伪量化（每次 forward 基于当次输入 min/max 计算）
    """
    def __init__(self, lin: nn.Linear, w_bit: int, a_bit: int, group_size: int = 32, use_zero_point: bool = False):
        super().__init__()
        self.in_features = lin.in_features
        self.out_features = lin.out_features
        self.bias = None
        if lin.bias is not None:
            self.register_buffer("bias", lin.bias.detach().clone())
        self.w_bit = w_bit
        self.a_bit = a_bit
        self.group_size = group_size
        self.use_zero_point = use_zero_point
        # 做一次“权重量化→反量化”，存成浮点（fake-quant）
        with torch.no_grad():
            W = lin.weight.detach()  # [out, in]
            if self.use_zero_point:
                scale, zp = _calc_qparams(W, self.w_bit, symmetric=False, group_size=self.group_size)
                Wq = _fake_quant(W, self.w_bit, scale, zp, symmetric=False)
            else:
                scale, zp = _calc_qparams(W, self.w_bit, symmetric=True, group_size=self.group_size)
                Wq = _fake_quant(W, self.w_bit, scale, None, symmetric=True)
                # mse = nn.MSELoss()
            self.register_buffer("qweight", Wq.reshape_as(W).to(W.dtype))
            if self.bias is not None and isinstance(self.bias, torch.Tensor):
                self.bias = self.bias.to(W.dtype)

    @classmethod
    def from_linear(cls, lin, w_bit, a_bit, group_size=32, use_zero_point=False):
        qlin = cls(lin, w_bit, a_bit, group_size, use_zero_point)
        return qlin

    def forward(self, x):
        # x: [*, in_features]
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
        out = F.linear(x_q, self.qweight, self.bias)
        return out
