#include <torch/extension.h>

#include <gemm.h>
#include <quant.h>

namespace {
torch::Tensor matmul(const torch::Tensor &A, const torch::Tensor &B) {
  torch::checkAllContiguous("matmul", {{A, "A", 0}, {B, "B", 1}});
  torch::checkDeviceType("matmul", {A, B}, at::DeviceType::CUDA);
  torch::checkAllSameGPU("matmul", {{A, "A", 0}, {B, "B", 1}});
  TORCH_CHECK(A.scalar_type() == at::ScalarType::Byte,
              "A must be uint8-packed int4");
  TORCH_CHECK(B.scalar_type() == at::ScalarType::Byte,
              "B must be uint8-packed int4");
  TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D tensors");
  TORCH_CHECK(A.size(1) == B.size(1),
              "Packed K mismatch: A.size(1) must equal B.size(1)");
  uint32_t M = static_cast<uint32_t>(A.size(0));
  uint32_t N = static_cast<uint32_t>(B.size(0));
  uint32_t K = static_cast<uint32_t>(A.size(1)) * kElementsPerVector;
  auto C =
      torch::empty({M, N}, torch::dtype(torch::kInt32).device(A.device()));
  matmul_host(A.data_ptr<Int4Storage>(), B.data_ptr<Int4Storage>(), M, N, K,
              C.data_ptr<int32_t>());
  return C;
}
torch::Tensor sym_quant(const torch::Tensor &x, const torch::Tensor &scale) {
  torch::checkAllContiguous("sym_quant", {{x, "x", 0}, {scale, "scale", 1}});
  torch::checkDeviceType("sym_quant", {x, scale}, at::DeviceType::CUDA);
  torch::checkSameGPU("sym_quant", {x, "x", 0}, {scale, "scale", 1});
  TORCH_CHECK(x.scalar_type() == at::ScalarType::Half ||
                  x.scalar_type() == at::ScalarType::BFloat16,
              "x must be float16 or bfloat16");
  TORCH_CHECK(scale.scalar_type() == x.scalar_type(),
              "scale dtype must match x dtype");
  TORCH_CHECK(x.dim() == 2, "x must be 2D (rows x cols)");
  TORCH_CHECK(scale.dim() == 2 && scale.size(1) == 1,
              "scale must be 2D (rows x 1)");
  torch::checkSize("sym_quant", torch::TensorArg{scale, "scale", 1}, 0,
                   x.size(0));
  uint32_t rows = static_cast<uint32_t>(x.size(0));
  uint32_t colsSrc = static_cast<uint32_t>(x.size(1));
  uint32_t colsDst = cdiv(colsSrc, kElementsPerVector);
  auto q = torch::empty({rows, colsDst},
                        torch::dtype(torch::kUInt8).device(x.device()));

  if (x.scalar_type() == at::ScalarType::Half) {
    sym_quant_host(reinterpret_cast<const half *>(x.data_ptr()),
                   reinterpret_cast<const half *>(scale.data_ptr()), rows,
                   colsSrc, colsDst, q.data_ptr<Int4Storage>());
  } else {
    sym_quant_host_bf16(reinterpret_cast<const __nv_bfloat16 *>(x.data_ptr()),
                        reinterpret_cast<const __nv_bfloat16 *>(scale.data_ptr()),
                        rows, colsSrc, colsDst, q.data_ptr<Int4Storage>());
  }
  return q;
}
torch::Tensor sym_dequant(const torch::Tensor &q, const torch::Tensor &scale_row,
                          const torch::Tensor &scale_col, int bits) {
  torch::checkAllContiguous("sym_dequant",
                            {{q, "q", 0},
                             {scale_row, "scale_row", 1},
                             {scale_col, "scale_col", 2}});
  torch::checkDeviceType("sym_dequant", {q, scale_row, scale_col},
                         at::DeviceType::CUDA);
  torch::checkAllSameGPU("sym_dequant",
                         {{q, "q", 0},
                          {scale_row, "scale_row", 1},
                          {scale_col, "scale_col", 2}});
  TORCH_CHECK(bits == 32,
              "Only bits=32 supported (input q must be int32 GEMM output)");
  TORCH_CHECK(q.scalar_type() == at::ScalarType::Int, "q must be int32");
  TORCH_CHECK(scale_row.scalar_type() == at::ScalarType::Half ||
                  scale_row.scalar_type() == at::ScalarType::BFloat16,
              "scale_row must be float16 or bfloat16");
  TORCH_CHECK(scale_col.scalar_type() == scale_row.scalar_type(),
              "scale_col dtype must match scale_row dtype");
  TORCH_CHECK(q.dim() == 2, "q must be 2D (rows x cols)");
  TORCH_CHECK(scale_row.dim() == 2 && scale_row.size(1) == 1,
              "scale_row must be 2D (rows x 1)");
  TORCH_CHECK(scale_col.dim() == 2 && scale_col.size(1) == 1,
              "scale_col must be 2D (cols x 1)");
  uint32_t rows = static_cast<uint32_t>(q.size(0));
  uint32_t cols = static_cast<uint32_t>(q.size(1));
  torch::checkSize("sym_dequant",
                   torch::TensorArg{scale_row, "scale_row", 1}, 0, rows);
  torch::checkSize("sym_dequant",
                   torch::TensorArg{scale_col, "scale_col", 2}, 0, cols);

  torch::Tensor x;
  if (scale_row.scalar_type() == at::ScalarType::Half) {
    x = torch::empty(q.sizes(),
                     torch::dtype(torch::kHalf).device(q.device()));
    sym_dequant_host(q.data_ptr<int32_t>(),
                     reinterpret_cast<const half *>(scale_row.data_ptr()),
                     reinterpret_cast<const half *>(scale_col.data_ptr()), rows,
                     cols, reinterpret_cast<half *>(x.data_ptr()));
  } else {
    x = torch::empty(q.sizes(),
                     torch::dtype(torch::kBFloat16).device(q.device()));
    sym_dequant_host_bf16(
        q.data_ptr<int32_t>(),
        reinterpret_cast<const __nv_bfloat16 *>(scale_row.data_ptr()),
        reinterpret_cast<const __nv_bfloat16 *>(scale_col.data_ptr()), rows, cols,
        reinterpret_cast<__nv_bfloat16 *>(x.data_ptr()));
  }
  return x;
}
} // namespace
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("matmul", &matmul,
        "Packed int4 GEMM: (A[M,Kp], B[N,Kp]) -> C[M,N] int32");
  m.def("sym_quant", &sym_quant,
        "Symmetric per-row quant: (x[M,K], scale[M,1]) -> q[M,ceil(K/8)] uint8");
  m.def("sym_dequant", &sym_dequant,
        "Dequantize int32 output: (q[M,N], scale_row[M,1], scale_col[N,1], bits) -> x[M,N] fp16");
}

