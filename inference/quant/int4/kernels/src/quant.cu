#include <quant.h>

__global__ void sym_quantize_f16_i4_kernel(const half *__restrict__ x,
                                          const half *__restrict__ scale,
                                          uint32_t rows, uint32_t colsSrc,
                                          uint32_t colsDst,
                                          Int4Storage *__restrict__ q) {
  uint32_t row = threadIdx.y + blockIdx.y * blockDim.y;
  uint32_t colDst = threadIdx.x + blockIdx.x * blockDim.x;
  if (row >= rows || colDst * kElementsPerVector >= colsSrc) {
    return;
  }
  Int4Storage storage;
  memset(&storage, 0, sizeof(storage));
  uint32_t id = colDst * kElementsPerVector + row * colsSrc;
#pragma unroll
  for (int i = 0; i < (int)kElementsPerVector; ++i) {
    bool safe = (colDst * kElementsPerVector + i) < colsSrc;
    if (safe) {
      half data = __hdiv(x[id + i], scale[row]);
      int qval = clamp(__half2int_rn(data), qmin, qmax);
      Int4Subbyte{reinterpret_cast<cutlass::int4b_t *>(&storage), i}.set(qval);
    }
  }

  q[colDst + row * colsDst] = storage;
}

void sym_quant_host(const half *x, const half *scale, uint32_t rows,
                    uint32_t colsSrc, uint32_t colsDst, Int4Storage *q) {
  dim3 block{std::min<uint32_t>(colsDst, 32), std::min<uint32_t>(rows, 16)};
  dim3 grid{cdiv(colsDst, block.x), cdiv(rows, block.y)};
  sym_quantize_f16_i4_kernel<<<grid, block>>>(x, scale, rows, colsSrc, colsDst,
                                             q);
}

__global__ void sym_quantize_bf16_i4_kernel(const __nv_bfloat16 *__restrict__ x,
                                           const __nv_bfloat16 *__restrict__ scale,
                                           uint32_t rows, uint32_t colsSrc,
                                           uint32_t colsDst,
                                           Int4Storage *__restrict__ q) {
  uint32_t row = threadIdx.y + blockIdx.y * blockDim.y;
  uint32_t colDst = threadIdx.x + blockIdx.x * blockDim.x;
  if (row >= rows || colDst * kElementsPerVector >= colsSrc) {
    return;
  }
  Int4Storage storage;
  memset(&storage, 0, sizeof(storage));
  uint32_t id = colDst * kElementsPerVector + row * colsSrc;
#pragma unroll
  for (int i = 0; i < (int)kElementsPerVector; ++i) {
    bool safe = (colDst * kElementsPerVector + i) < colsSrc;
    if (safe) {
      float xf = __bfloat162float(x[id + i]);
      float sf = __bfloat162float(scale[row]);
      float data = xf / sf;
      int qval = clamp(__float2int_rn(data), qmin, qmax);
      Int4Subbyte{reinterpret_cast<cutlass::int4b_t *>(&storage), i}.set(qval);
    }
  }
  q[colDst + row * colsDst] = storage;
}

void sym_quant_host_bf16(const __nv_bfloat16 *x, const __nv_bfloat16 *scale,
                         uint32_t rows, uint32_t colsSrc, uint32_t colsDst,
                         Int4Storage *q) {
  dim3 block{std::min<uint32_t>(colsDst, 32), std::min<uint32_t>(rows, 16)};
  dim3 grid{cdiv(colsDst, block.x), cdiv(rows, block.y)};
  sym_quantize_bf16_i4_kernel<<<grid, block>>>(x, scale, rows, colsSrc, colsDst,
                                              q);
}

__global__ void sym_dequantize_i32_f16_kernel(
    const int32_t *__restrict__ q, const half *__restrict__ scale_row,
    const half *__restrict__ scale_col, uint32_t rows, uint32_t cols,
    half *__restrict__ x) {
  uint32_t row = threadIdx.y + blockIdx.y * blockDim.y;
  uint32_t col = threadIdx.x + blockIdx.x * blockDim.x;

  if (col >= cols || row >= rows) {
    return;
  }
  half q_half = __int2half_rn(q[col + row * cols]);
  x[col + row * cols] = scale_row[row] * scale_col[col] * q_half;
}

void sym_dequant_host(const int32_t *q, const half *scale_row,
                      const half *scale_col, uint32_t rows, uint32_t cols,
                      half *x) {
  dim3 block{std::min<uint32_t>(cols, 16), std::min<uint32_t>(rows, 16)};
  dim3 grid{cdiv(cols, block.x), cdiv(rows, block.y)};
  sym_dequantize_i32_f16_kernel<<<grid, block>>>(q, scale_row, scale_col, rows,
                                                cols, x);
}

__global__ void sym_dequantize_i32_bf16_kernel(
    const int32_t *__restrict__ q, const __nv_bfloat16 *__restrict__ scale_row,
    const __nv_bfloat16 *__restrict__ scale_col, uint32_t rows, uint32_t cols,
    __nv_bfloat16 *__restrict__ x) {
  uint32_t row = threadIdx.y + blockIdx.y * blockDim.y;
  uint32_t col = threadIdx.x + blockIdx.x * blockDim.x;

  if (col >= cols || row >= rows) {
    return;
  }

  float sr = __bfloat162float(scale_row[row]);
  float sc = __bfloat162float(scale_col[col]);
  // Match fp16 path: net effect is sr * sc * q (the fp16 kernel does /10 then *10).
  float xElement = static_cast<float>(q[col + row * cols]);
  float out = sr * sc * xElement;
  x[col + row * cols] = __float2bfloat16_rn(out);
}

void sym_dequant_host_bf16(const int32_t *q, const __nv_bfloat16 *scale_row,
                           const __nv_bfloat16 *scale_col, uint32_t rows,
                           uint32_t cols, __nv_bfloat16 *x) {
  dim3 block{std::min<uint32_t>(cols, 16), std::min<uint32_t>(rows, 16)};
  dim3 grid{cdiv(cols, block.x), cdiv(rows, block.y)};
  sym_dequantize_i32_bf16_kernel<<<grid, block>>>(q, scale_row, scale_col, rows,
                                                 cols, x);
}

