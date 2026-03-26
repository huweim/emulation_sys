import torch
import w4a4_w8a8_int as k


def main():
    x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)

    # W4A4 path
    act_q, act_scales = k.quantize_w4a4_act(x)
    w_q, w_scales = k.quantize_w4a4_wgt(w)
    out_w4a4 = k.gemm_w4a4(act_q, w_q, act_scales, w_scales)

    print("[W4A4] torch ref:")
    print(x @ w.T)
    print("[W4A4] kernel out:")
    print(out_w4a4)

    # W8A8 path
    act_q, act_scales = k.quantize_w8a8_act(x, fuse_glu=False)
    w8 = k.quantize_w8a8_wgt(w)
    w_q, w_scales = w8[0], w8[1]
    out_w8a8 = k.gemm_w8a8(act_q, w_q, act_scales, w_scales, bias=None)

    print("[W8A8] kernel out:")
    print(out_w8a8)


if __name__ == "__main__":
    main()
