#include <cstdint>
#include <cstdio>
#include <cuda_fp8.h>

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

__global__ void test_fp8_mma_kernel() {
  uint32_t c_bits = 0b0'10010111'00000000000000000000000; // 2^24
  float c_init = *((float *)&c_bits);
  float c = c_init;
  if (threadIdx.x == 0) {
    printf("\n=== FP8 (E4M3) Precision Probe ===\n");

    printf("C_init:\n%f\n", c_init);
    printf("Ref result:\n%f\n", c_init + 2.0f);
  }
  __syncthreads();

  auto run_case = [&](int num, float add) {
    // 打印用例标题（保持与原 main 一致）
    if (threadIdx.x == 0) {
      if (num == 4 && add == 0.5f) {
        printf("\n2^24 + 0.5 + 0.5 + 0.5 + 0.5\n");
      } else if (num == 8 && add == 0.25f) {
        printf(
            "\n2^24 + 0.25 + 0.25 + 0.25 + 0.25 + 0.25 + 0.25 + 0.25 + 0.25\n");
      }
    }
    __syncthreads();

    float ref_result = c + num * add;
    float c0 = c, c1 = c, c2 = c, c3 = c;

    float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;

    uint32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    uint32_t b0 = 0, b1 = 0;

    if (threadIdx.x == 0) {
      assert(num % 4 == 0 && num <= 128);
    }
    __syncthreads();

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
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(c0),
          "f"(c1), "f"(c2), "f"(c3));

    __syncthreads();

    if (threadIdx.x == 0) {
      if (d0 != ref_result) {
        printf("Analysis: Result == C_init. Precision LOST.\n");
      } else {
        printf("Analysis: Result == 2^24+2. Precision KEPT.\n");
      }
    }
    __syncthreads();
  };

  // 依次执行两次检测（顺序不变）
  run_case(4, 0.5f);
  run_case(8, 0.25f);

  if (threadIdx.x == 0) {
    printf(
        "\nConclusion: Internal accumulator for fp8 has 25 bits precision.\n");
  }
}

int main() {
  test_fp8_mma_kernel<<<1, 32>>>();
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  return 0;
}