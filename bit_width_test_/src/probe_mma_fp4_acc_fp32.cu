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
  uint32_t byte_val = code << 2;

  // 这样寄存器里就是 00xxxx00 00xxxx00 00xxxx00 00xxxx00
  // 对应 8 个 FP4 元素
  uint32_t full_reg = 0;
  full_reg |= byte_val;
  full_reg |= (byte_val << 8);
  full_reg |= (byte_val << 16);
  full_reg |= (byte_val << 24);

  return full_reg;
}

__global__ void test_fp4_mma_kernel(float c, int num, float add) {
  float ref_result = c + num * add;
  float c0 = c;
  float c1 = c;
  float c2 = c;
  float c3 = c;

  // 输出
  float d0, d1, d2, d3;

  // a是16*32的，每个线程是16个元素，即64 bits，应该是两个寄存器
  // 但mma要求是4个，也就是说可能按照f8的精度来传了
  // 打包细节:fp4放在8位的中间4位，比如00011100表示的是0111即6
  // e2m1的bias是1，但是0001是非规格化数，0.M*2^(1-bias)，即0.5
  uint32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0, b0 = 0, b1 = 0;

  // fragment划分:https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-fragment-mma-16832
  assert(num % 4 == 0 && num <= 128);
  assert(add > 0.125);
  if (num == 0.25) { // 0.25无法直接表示
    add = 0.5;
    if (threadIdx.x < num / 4) {
      a0 = float_to_fp4_reg(add);
      b0 = float_to_fp4_reg(add);
    }
  } else {
    if (threadIdx.x < num / 4) {
      a0 = float_to_fp4_reg(add);
      b0 = float_to_fp4_reg(1.0f);
    }
  }

  // 5. 执行内联汇编
  asm volatile(
      "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e2m1.e2m1.f32 "
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
  printf("\n=== FP4 (E2M1) Precision Probe ===\n");
  uint32_t c_init = 0b0'10010111'00000000000000000000000;
  float c = *((float *)&c_init);
  printf("C_init:\n%f\n", c);
  printf("ref result:\n%f\n", c + 2.0f);
  printf("\n2^24 + 0.5 + 0.5 + 0.5 + 0.5\n");
  test_fp4_mma_kernel<<<1, 32>>>(c, 4, 0.5);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  printf("\n2^24 + 0.25 + 0.25 + 0.25 + 0.25 + 0.25 + 0.25 + 0.25 + 0.25\n");
  test_fp4_mma_kernel<<<1, 32>>>(c, 8, 0.25);

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  printf("Conclusion: Internal accumulator for fp4 has 25 bits precision.\n");

  return 0;
}