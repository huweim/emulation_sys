#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <torch/all.h>

#include <dtype_bfloat16.cuh>
#include <quant_utils.cuh>

// fp8 gemm写好之后就不需要了
__global__ void groupwise_dequant_fp8_to_bf16_kernel(
    const uint8_t *__restrict__ input_q, // FP8 输入 [num_rows, num_cols]
    const float
        *__restrict__ scales, // FP32 scales [num_rows, num_cols / group_size]
    __nv_bfloat16
        *__restrict__ output_dequant, // BF16 输出 [num_rows, num_cols]
    const int num_rows, const int num_cols, const int group_size) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const int num_threads = blockDim.x;

  // 向量化大小 (8 个元素)
  constexpr int VEC_SIZE = 8;
  const int scales_per_row = num_cols / group_size;

  // 指向当前行
  const uint8_t *row_input_q = input_q + (int64_t)row * num_cols;
  const float *row_scales = scales + (int64_t)row * scales_per_row;
  __nv_bfloat16 *row_output_dequant = output_dequant + (int64_t)row * num_cols;

  // 线程循环，每次处理 8 个元素
  for (int col_start = tid * VEC_SIZE; col_start < num_cols;
       col_start += num_threads * VEC_SIZE) {

    // 1. 找到对应的 scale
    // 因为 group_size 是8的倍数，一个 VEC_SIZE 向量内的所有元素共享同一个 scale
    const int group_idx = col_start / group_size;
    const float scale = row_scales[group_idx];

    // 2. 加载FP8(8 个元素)
    const uint2 q_vec =
        *(reinterpret_cast<const uint2 *>(&row_input_q[col_start]));

    // 3. 反量化
    const vllm::bf16_8_t dequant_vec =
        vllm::fp8::scaled_vec_conversion<vllm::bf16_8_t, uint2>(
            q_vec, scale, __NV_E4M3 // 假设为 E4M3
        );

    // 4. 存储BF16
    *(reinterpret_cast<vllm::bf16_8_t *>(&row_output_dequant[col_start])) =
        dequant_vec;
  }
}

void groupwise_dequant_fp8_bf16(
    const torch::Tensor &input_q,  // [num_rows, num_cols]
    const torch::Tensor &scales,   // [num_rows, num_cols / G]
    torch::Tensor &output_dequant, // [num_rows, num_cols]
    int64_t group_size) {
  TORCH_CHECK(input_q.is_contiguous());
  TORCH_CHECK(scales.is_contiguous());
  TORCH_CHECK(output_dequant.is_contiguous());
  TORCH_CHECK(input_q.scalar_type() == torch::kFloat8_e4m3fn);
  TORCH_CHECK(scales.scalar_type() == torch::kFloat32);
  TORCH_CHECK(output_dequant.scalar_type() == torch::kBFloat16);
  TORCH_CHECK(input_q.dim() == 2);
  TORCH_CHECK(scales.dim() == 2);
  TORCH_CHECK(output_dequant.dim() == 2);

  const int num_rows = input_q.size(0);
  const int num_cols = input_q.size(1);

  TORCH_CHECK(num_cols % group_size == 0, "num_cols 必须被 group_size 整除");
  TORCH_CHECK(num_cols % 8 == 0, "num_cols 必须是 8 的倍数");
  TORCH_CHECK(group_size % 8 == 0, "group_size 必须是 8 的倍数");
  TORCH_CHECK(scales.size(0) == num_rows, "scales 行数不匹配");
  TORCH_CHECK(scales.size(1) == num_cols / group_size, "scales 列数不匹配");

  dim3 grid(num_rows);
  dim3 block(128);

  groupwise_dequant_fp8_to_bf16_kernel<<<grid, block>>>(
      (const uint8_t *)input_q.data_ptr(), (const float *)scales.data_ptr(),
      (__nv_bfloat16 *)output_dequant.data_ptr(), num_rows, num_cols,
      (int)group_size);
}