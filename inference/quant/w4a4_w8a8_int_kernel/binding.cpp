#include <torch/extension.h>

#include <c10/util/Optional.h>

#include "interop/torch.h"
#include "kernels/zgemm/zgemm.h"

#include <vector>

namespace {

void check_float16_or_bfloat16(const torch::Tensor &t, const char *name) {
    TORCH_CHECK(t.dtype() == torch::kFloat16 || t.dtype() == torch::kBFloat16, name,
                " must be float16 or bfloat16, got ", t.dtype());
}

void check_int8(const torch::Tensor &t, const char *name) {
    TORCH_CHECK(t.dtype() == torch::kInt8, name, " must be int8, got ", t.dtype());
}

// For W4A4 kernels:
// - act: [M, K/2] int8
// - wgt: [N, K/2] int8
// - ascales: [K/64, M] (fp16/bf16)
// - wscales: [K/64, N] (fp16/bf16)
torch::Tensor gemm_w4a4_impl(torch::Tensor act,
                             torch::Tensor wgt,
                             torch::Tensor ascales,
                             torch::Tensor wscales,
                             c10::optional<torch::Tensor> bias,
                             bool act_unsigned,
                             bool fuse_silu) {
    act = act.contiguous();
    wgt = wgt.contiguous();
    ascales = ascales.contiguous();
    wscales = wscales.contiguous();
    if (bias.has_value()) {
        bias = bias.value().contiguous();
    }

    TORCH_CHECK(act.dim() == 2, "act must be 2D [M, K/2]");
    TORCH_CHECK(wgt.dim() == 2, "wgt must be 2D [N, K/2]");
    TORCH_CHECK(ascales.dim() == 2, "ascales must be 2D [K/64, M]");
    TORCH_CHECK(wscales.dim() == 2, "wscales must be 2D [K/64, N]");

    check_int8(act, "act");
    check_int8(wgt, "wgt");
    check_float16_or_bfloat16(ascales, "ascales");
    check_float16_or_bfloat16(wscales, "wscales");

    TORCH_CHECK(ascales.scalar_type() == wscales.scalar_type(),
                "ascales and wscales dtype must match (fp16/bf16)");

    const int64_t M = act.size(0);
    const int64_t K2 = act.size(1);
    const int64_t N = wgt.size(0);
    const int64_t K = K2 * 2;

    TORCH_CHECK(wgt.size(1) == K2, "wgt.shape[1] must match act.shape[1]");
    TORCH_CHECK(K % 64 == 0, "K must be multiple of 64 for zgemm w4a4 kernels");
    TORCH_CHECK(M % 256 == 0, "M must be multiple of 256 for zgemm w4a4 gemm");
    TORCH_CHECK(N % 128 == 0, "N must be multiple of 128 for zgemm w4a4 gemm");

    TORCH_CHECK(ascales.size(0) == K / 64 && ascales.size(1) == M, "ascales must be [K/64, M]");
    TORCH_CHECK(wscales.size(0) == K / 64 && wscales.size(1) == N, "wscales must be [K/64, N]");

    if (bias.has_value()) {
        TORCH_CHECK(bias.value().dim() == 1 && bias.value().size(0) == N, "bias must be 1D [N]");
        TORCH_CHECK(bias.value().scalar_type() == ascales.scalar_type(), "bias dtype must match ascales/wscales");
    }

    auto out = torch::empty({M, N}, ascales.options());

    TorchOpContext ctx;

    Tensor actT = from_torch(act);
    Tensor wgtT = from_torch(wgt);
    Tensor outT = from_torch(out);
    Tensor ascalesT = from_torch(ascales);
    Tensor wscalesT = from_torch(wscales);

    Tensor qout; // invalid
    Tensor oscales; // invalid
    Tensor poolout; // invalid

    Tensor lora_act_in;
    Tensor lora_up;
    Tensor lora_down;
    Tensor lora_act_out;

    Tensor norm_q;
    Tensor norm_k;
    Tensor rotary_emb;

    Tensor smooth_factor;

    Tensor out_vk;
    Tensor out_linearattn;

    Tensor wcscales;

    Tensor out_q;
    Tensor out_k;
    Tensor out_v;

    Tensor biasT;
    if (bias.has_value()) {
        biasT = from_torch(bias.value());
    }

    std::vector<float> lora_scales; // ignored when LoRA tensors are invalid

    // NVFP4 path is not supported in this lightweight wrapper.
    const bool fp4 = false;
    const float alpha = 1.0f;
    const int attn_tokens = 0;

    nunchaku::kernels::gemm_w4a4(actT,
                                 wgtT,
                                 outT,
                                 qout,
                                 ascalesT,
                                 wscalesT,
                                 oscales,
                                 poolout,
                                 lora_act_in,
                                 lora_up,
                                 lora_down,
                                 lora_act_out,
                                 norm_q,
                                 norm_k,
                                 rotary_emb,
                                 biasT,
                                 smooth_factor,
                                 out_vk,
                                 out_linearattn,
                                 act_unsigned,
                                 lora_scales,
                                 fuse_silu,
                                 fp4,
                                 alpha,
                                 wcscales,
                                 out_q,
                                 out_k,
                                 out_v,
                                 attn_tokens);

    return out;
}

} // namespace

std::tuple<torch::Tensor, torch::Tensor> quantize_w4a4_act(torch::Tensor input) {
    input = input.contiguous();
    TORCH_CHECK(input.dim() == 2, "input must be 2D [M, K]");

    check_float16_or_bfloat16(input, "input");
    TORCH_CHECK(input.size(1) % 64 == 0, "K must be multiple of 64 for w4a4 quantize");

    const int64_t M = input.size(0);
    const int64_t K = input.size(1);
    TORCH_CHECK(M % 32 == 0, "M must be multiple of 32 for w4a4 quantize");

    auto out_act = torch::empty({M, K / 2}, input.options().dtype(torch::kInt8));
    auto out_scales = torch::empty({K / 64, M}, input.options()); // fp16/bf16

    TorchOpContext ctx;

    Tensor inT = from_torch(input);
    Tensor outActT = from_torch(out_act);
    Tensor outScalesT = from_torch(out_scales);

    nunchaku::kernels::quantize_w4a4_act(inT, outActT, outScalesT);

    return {out_act, out_scales};
}

std::tuple<torch::Tensor, torch::Tensor> quantize_w4a4_wgt(torch::Tensor input) {
    input = input.contiguous();
    TORCH_CHECK(input.dim() == 2, "input must be 2D [N, K]");

    check_float16_or_bfloat16(input, "input");
    TORCH_CHECK(input.size(1) % 64 == 0, "K must be multiple of 64 for w4a4 weight quantize");

    const int64_t N = input.size(0);
    const int64_t K = input.size(1);
    TORCH_CHECK(N % 128 == 0, "N must be multiple of 128 for w4a4 weight quantize");

    auto out_wgt = torch::empty({N, K / 2}, input.options().dtype(torch::kInt8));
    auto out_scales = torch::empty({K / 64, N}, input.options()); // fp16/bf16

    TorchOpContext ctx;

    Tensor inT = from_torch(input);
    Tensor outWgtT = from_torch(out_wgt);
    Tensor outScalesT = from_torch(out_scales);

    nunchaku::kernels::quantize_w4a4_wgt(inT, outWgtT, outScalesT);

    return {out_wgt, out_scales};
}

torch::Tensor gemm_w4a4(torch::Tensor act,
                          torch::Tensor wgt,
                          torch::Tensor ascales,
                          torch::Tensor wscales,
                          bool act_unsigned,
                          bool fuse_silu) {
    return gemm_w4a4_impl(act, wgt, ascales, wscales, c10::nullopt, act_unsigned, fuse_silu);
}

torch::Tensor gemm_w4a4_bias(torch::Tensor act,
                               torch::Tensor wgt,
                               torch::Tensor ascales,
                               torch::Tensor wscales,
                               torch::Tensor bias,
                               bool act_unsigned,
                               bool fuse_silu) {
    bias = bias.contiguous();
    return gemm_w4a4_impl(act, wgt, ascales, wscales, bias, act_unsigned, fuse_silu);
}

// For W8A8 (int8) kernels:
// - input act (quantize): [M, K] bf16 -> output act: [M, K] int8, ascales: [M] bf16
// - gemm: act [M, K] int8, wgt [N, K] int8, ascales [M] bf16, wscales [N] bf16, out [M, N] bf16
std::tuple<torch::Tensor, torch::Tensor> quantize_w8a8_act(torch::Tensor input, bool fuse_glu) {
    input = input.contiguous();
    TORCH_CHECK(input.dim() == 2, "input must be 2D [M, K]");
    TORCH_CHECK(input.dtype() == torch::kBFloat16, "w8a8 quantize currently supports BF16 input only");
    TORCH_CHECK(input.size(1) % 32 == 0, "K must be multiple of 32 for w8a8 quantize");

    const int64_t M = input.size(0);
    const int64_t K = input.size(1);
    TORCH_CHECK(M % 32 == 0, "M must be multiple of 32 for w8a8 quantize");

    const int64_t outK = fuse_glu ? (K / 2) : K;

    auto out_act = torch::empty({M, outK}, input.options().dtype(torch::kInt8));
    auto out_scales = torch::empty({M}, input.options()); // bf16

    TorchOpContext ctx;
    Tensor inT = from_torch(input);
    Tensor outActT = from_torch(out_act);
    Tensor outScalesT = from_torch(out_scales);

    nunchaku::kernels::quantize_w8a8_act(inT, outActT, outScalesT, fuse_glu);

    return {out_act, out_scales};
}

torch::Tensor gemm_w8a8_impl(torch::Tensor act,
                              torch::Tensor wgt,
                              torch::Tensor ascales,
                              torch::Tensor wscales,
                              bool with_bias,
                              torch::Tensor bias) {
    act = act.contiguous();
    wgt = wgt.contiguous();
    ascales = ascales.contiguous();
    wscales = wscales.contiguous();
    if (with_bias) {
        bias = bias.contiguous();
    }

    TORCH_CHECK(act.dim() == 2, "act must be [M, K]");
    TORCH_CHECK(wgt.dim() == 2, "wgt must be [N, K]");
    TORCH_CHECK(ascales.dim() == 1, "ascales must be [M]");
    TORCH_CHECK(wscales.dim() == 1, "wscales must be [N]");

    check_int8(act, "act");
    check_int8(wgt, "wgt");
    TORCH_CHECK(ascales.dtype() == torch::kBFloat16, "ascales must be BF16");
    TORCH_CHECK(wscales.dtype() == torch::kBFloat16, "wscales must be BF16");

    const int64_t M = act.size(0);
    const int64_t K = act.size(1);
    const int64_t N = wgt.size(0);

    TORCH_CHECK(wgt.size(1) == K, "wgt.shape[1] must match act.shape[1]");
    TORCH_CHECK(ascales.size(0) == M, "ascales must be [M]");
    TORCH_CHECK(wscales.size(0) == N, "wscales must be [N]");
    TORCH_CHECK(M % 256 == 0, "M must be multiple of 256 for w8a8 gemm");
    TORCH_CHECK(N % 128 == 0, "N must be multiple of 128 for w8a8 gemm");

    if (with_bias) {
        TORCH_CHECK(bias.dim() == 1 && bias.size(0) == N, "bias must be [N]");
        TORCH_CHECK(bias.dtype() == torch::kBFloat16, "bias must be BF16");
    }

    auto out = torch::empty({M, N}, ascales.options());

    TorchOpContext ctx;
    Tensor actT = from_torch(act);
    Tensor wgtT = from_torch(wgt);
    Tensor outT = from_torch(out);
    Tensor ascalesT = from_torch(ascales);
    Tensor wscalesT = from_torch(wscales);

    Tensor biasT; // invalid by default
    if (with_bias) {
        biasT = from_torch(bias);
    }

    nunchaku::kernels::gemm_w8a8(actT, wgtT, outT, ascalesT, wscalesT, biasT);
    return out;
}

torch::Tensor gemm_w8a8(torch::Tensor act,
                         torch::Tensor wgt,
                         torch::Tensor ascales,
                         torch::Tensor wscales) {
    return gemm_w8a8_impl(act, wgt, ascales, wscales, false, torch::Tensor());
}

torch::Tensor gemm_w8a8_bias(torch::Tensor act,
                               torch::Tensor wgt,
                               torch::Tensor ascales,
                               torch::Tensor wscales,
                               torch::Tensor bias) {
    return gemm_w8a8_impl(act, wgt, ascales, wscales, true, bias);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quantize_w4a4_act", &quantize_w4a4_act, "Quantize activations to W4A4 packed int4 (int8 storage) + scales");
    m.def("quantize_w4a4_wgt", &quantize_w4a4_wgt, "Quantize weights to W4A4 packed int4 (int8 storage) + scales");

    m.def("gemm_w4a4", &gemm_w4a4, "W4A4 int4 GEMM (no bias)");
    m.def("gemm_w4a4_bias", &gemm_w4a4_bias, "W4A4 int4 GEMM (with bias)");

    m.def("quantize_w8a8_act", &quantize_w8a8_act, "Quantize activations to W8A8 packed int8 + scales");
    m.def("gemm_w8a8", &gemm_w8a8, "W8A8 int8 GEMM (no bias)");
    m.def("gemm_w8a8_bias", &gemm_w8a8_bias, "W8A8 int8 GEMM (with bias)");
}

