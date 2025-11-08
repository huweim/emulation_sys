#include <torch/extension.h>

void per_token_group_quant_fp8(const torch::Tensor& input,
                               torch::Tensor& output_q,
                               torch::Tensor& output_s,
                               int64_t group_size,
                               double eps,
                               double fp8_min,
                               double fp8_max,
                               bool scale_ue8m0);
                               
void groupwise_dequant_fp8_bf16(
    const torch::Tensor& input_q,
    const torch::Tensor& scales,
    torch::Tensor& output_dequant,
    int64_t group_size
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // m.def("static_scaled_fp8_quant", &static_scaled_fp8_quant);
  // m.def("dynamic_scaled_fp8_quant", &dynamic_scaled_fp8_quant);
  // m.def("dynamic_per_token_scaled_fp8_quant", &dynamic_per_token_scaled_fp8_quant);
  m.def("per_token_group_quant_fp8", &per_token_group_quant_fp8);
  // m.def("silu_and_mul_quant", &silu_and_mul_quant);
  // m.def("cutlass_scaled_fp8_mm", &cutlass_scaled_fp8_mm);
  m.def("groupwise_dequant_fp8_bf16", &groupwise_dequant_fp8_bf16);
  // m.def("groupwise_gemm_fp8", &groupwise_gemm_fp8);
}