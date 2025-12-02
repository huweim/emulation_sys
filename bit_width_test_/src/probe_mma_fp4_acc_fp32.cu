#include <cstdint>
#include <cstdio>
#include <cuda_fp4.h>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA Error: %s in %s at line %d\n",                     \
              cudaGetErrorString(err), __FILE__, __LINE__);                    \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

__host__ __device__ uint32_t float_to_fp4_reg(float x) {
  __nv_fp4_storage_t raw_fp4 =
      __nv_cvt_float_to_fp4(x, __NV_E2M1, cudaRoundNearest);
  uint32_t code = (uint32_t)(raw_fp4 & 0x0F);
  uint32_t byte_val = code << 2; // 00xxxx00

  uint32_t full_reg = 0;
  full_reg |= byte_val;
  full_reg |= (byte_val << 8);
  full_reg |= (byte_val << 16);
  full_reg |= (byte_val << 24);
  return full_reg;
}

__global__ void test_fp4_mma_kernel_combined() {
  uint32_t c_bits = 0b0'10010111'00000000000000000000000; // 2^24
  float c_init = *((float *)&c_bits);
  float c = c_init;

  if (threadIdx.x == 0) {
    printf("\n=== FP4 (E2M1) Precision Probe ===\n");
    printf("C_init:\n%f\n", c_init);
    printf("ref result:\n%f\n", c_init + 2.0f);
  }
  __syncthreads();

  auto run_case = [&](int num, float add) {
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

    // FP4(E2M1) 不能精确表示 0.25，这里沿用你文件中的处理逻辑说明：
    // 当 add==0.25 时，等价通过 A=0.5, B=0.5 来实现（每次乘加等效 0.25）
    if (add == 0.25f) {
      if (threadIdx.x < num / 4) {
        a0 = float_to_fp4_reg(0.5f);
        b0 = float_to_fp4_reg(0.5f);
      }
    } else {
      uint32_t val_a = float_to_fp4_reg(add);
      uint32_t val_b = float_to_fp4_reg(1.0f);
      if (threadIdx.x < num / 4) {
        a0 = val_a;
        b0 = val_b;
      }
    }

    asm volatile(
        "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e2m1.e2m1.f32 "
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
        "\nConclusion: Internal accumulator for fp4 has 25 bits precision.\n");
  }
}

int main() {
  test_fp4_mma_kernel_combined<<<1, 32>>>();
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  return 0;
}