#include <torch/extension.h>

void per_token_group_quant_fp8(const torch::Tensor& input,
                               torch::Tensor& output_q,
                               torch::Tensor& output_s,
                               int64_t group_size,
                               double eps,
                               double fp8_min,
                               double fp8_max,
                               bool scale_ue8m0);
                               

namespace vllm {
  void cutlass_scaled_mm_blockwise_sm120_fp8(
  torch::Tensor &out,                                       
  torch::Tensor const &a,                           
  torch::Tensor const &b,                                   
  torch::Tensor const &a_scales,                                           
  torch::Tensor const &b_scales);
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("per_token_group_quant_fp8", &per_token_group_quant_fp8);
  m.def("cutlass_scaled_mm_blockwise_sm120_fp8", &vllm::cutlass_scaled_mm_blockwise_sm120_fp8);
}