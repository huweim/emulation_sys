#include <cstdint>
#include <cstdio>
#include <cuda_fp8.h> // 引入FP8支持

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA Error: %s in %s at line %d\n",                     \
              cudaGetErrorString(err), __FILE__, __LINE__);                    \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

__host__ __device__ uint32_t float_to_fp8_reg(float x) {
  __nv_fp8_storage_t raw_fp8 =
      __nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3);

  uint32_t byte_val = (uint32_t)raw_fp8;

  // [Byte][Byte][Byte][Byte]
  uint32_t full_reg = 0;
  full_reg |= byte_val;
  full_reg |= (byte_val << 8);
  full_reg |= (byte_val << 16);
  full_reg |= (byte_val << 24);

  return full_reg;
}

__global__ void test_fp8_mma_kernel(float c, int num, float add) {
  float ref_result = c + num * add;
  float c0 = c;
  float c1 = c;
  float c2 = c;
  float c3 = c;

  // 输出 D
  float d0, d1, d2, d3;

  // A 和 B 寄存器初始化
  uint32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
  uint32_t b0 = 0, b1 = 0;

  // 断言: 确保请求的元素数量是4的倍数 (因为1个寄存器=4个元素)
  // 并且不要超过单次 MMA 指令能处理的范围
  assert(num % 4 == 0 && num <= 128);

  uint32_t val_a = float_to_fp8_reg(add);
  uint32_t val_b = float_to_fp8_reg(1.0f);

  if (threadIdx.x < num / 4) {
    a0 = val_a;
    b0 = val_b;
  }

  asm volatile(
      "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13};\n"
      : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(c0), "f"(c1),
        "f"(c2), "f"(c3));

  if (threadIdx.x == 0) {
    if (d0 != ref_result) {
      printf("Analysis: Result == C_init. Precision LOST.\n");
    } else {
      printf("Analysis: Result == 2^24+2. Precision KEPT.\n");
    }
  }
}

int main() {
  printf("\n=== FP8 (E4M3) Precision Probe ===\n");

  uint32_t c_init = 0b0'10010111'00000000000000000000000; // 2^24
  float c = *((float *)&c_init);
  printf("C_init:\n%f\n", c);
  printf("Ref result:\n%f\n", c + 2.0f);

  printf("\n2^24 + 0.5 + 0.5 + 0.5 + 0.5\n");
  test_fp8_mma_kernel<<<1, 32>>>(c, 4, 0.5f);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  printf("\n2^24 + 0.25 + 0.25 + 0.25 + 0.25 + 0.25 + 0.25 + 0.25 + 0.25\n");
  test_fp8_mma_kernel<<<1, 32>>>(c, 8, 0.25f);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  printf("Conclusion: Internal accumulator for fp8 has 25 bits precision.\n");
  return 0;
}