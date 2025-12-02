#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

/// RUN: nvcc -O3 -std=c++17 -arch=sm_80  probe_mma_fp16_acc_manual.cu -o probe_mma_fp16_acc_manual
/// RUN: nvcc -O3 -std=c++17 -arch=sm_120  probe_mma_fp16_acc_manual.cu -o probe_mma_fp16_acc_manual

// 错误检查宏
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA Error: %s in %s at line %d\n",                     \
              cudaGetErrorString(err), __FILE__, __LINE__);                    \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// 辅助函数：将两个 half 打包成一个 uint32_t (对应 .f16x2 寄存器)
__device__ __host__ uint32_t pack_half2(float a, float b) {
    __half2 h2 = __floats2half2_rn(a, b);
    return *reinterpret_cast<uint32_t*>(&h2);
}

__global__ void test_fp16_acc_kernel() {
  // 1. 准备 C (Accumulator)
  // 测试点: 2048.0 (2^11)。
  // FP16 的尾数是 10bit，在 2048 时 ULP 为 2.0。
  // 如果累加器是纯 FP16，2048 + 1.0 应该无法改变结果 (Stagnation)。

  float c_val = 0.f;
  if (threadIdx.x == 0) {
      c_val = 1024.0f; 
  }
  uint32_t c_packed = pack_half2(c_val, 0);
  
  // C/D 矩阵需要 2 个 32位寄存器
  uint32_t c0 = c_packed;
  uint32_t c1 = c_packed;
  uint32_t d0 = 0, d1 = 0;
  
//   uint32_t a_val = pack_half2(0.5f, 0.5f);
//   uint32_t a_val = pack_half2(0.75f, 0.25f);
//   uint32_t a_val = pack_half2(0.875f, 0.125f);

  // Bit 4: 2^-4 = 0.0625
  // uint32_t a_val = pack_half2(0.9375f, 0.0625f);
  
  // Bit 5: 2^-5 = 0.03125
  // uint32_t a_val = pack_half2(0.96875f, 0.03125f);
  
  // Bit 6: 2^-6 = 0.015625
  // uint32_t a_val = pack_half2(0.984375f, 0.015625f);
  
  // Bit 7: 2^-7 = 0.0078125
  // uint32_t a_val = pack_half2(0.9921875f, 0.0078125f);
  
  // Bit 8: 2^-8 = 0.00390625
  // uint32_t a_val = pack_half2(0.99609375f, 0.00390625f);

  // Bit 9: 2^-9 = 0.001953125
  // uint32_t a_val = pack_half2(0.998046875f, 0.001953125f);

  // Bit 10: 2^-10 = 0.0009765625
  // uint32_t a_val = pack_half2(0.9990234375f, 0.0009765625f);

  // Bit 11: 2^-11 = 0.00048828125 (FP16 输入精度的极限)
  uint32_t a_val = pack_half2(0.99951171875f, 0.00048828125f);

  uint32_t b_val = pack_half2(1.0f, 1.0f);

  uint32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
  uint32_t b0 = 0, b1 = 0;

  if (threadIdx.x == 0) {
    // A 需要 4 个寄存器
    a0 = a_val;
    // B 需要 2 个寄存器
    b0 = b_val;
  }

  // 3. 执行 FP16 MMA PTX 指令
  // 格式: mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16
  // D: {d0, d1}
  // A: {a0, a1, a2, a3}
  // B: {b0, b1}
  // C: {c0, c1}
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
      "{%0, %1}, "         // D
      "{%2, %3, %4, %5}, " // A
      "{%6, %7}, "         // B
      "{%8, %9};\n"        // C
      : "=r"(d0), "=r"(d1)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), 
        "r"(b0), "r"(b1), 
        "r"(c0), "r"(c1)
  );

  // 4. 打印结果 (Thread 0)
  if (threadIdx.x == 0) {
    // 解包结果
    __half2* d0_h2 = (__half2*)&d0;
    __half2* d1_h2 = (__half2*)&d1;
    
    float res0 = __half2float(d0_h2->x);
    float res1 = __half2float(d0_h2->y);
    
    printf("\n--- FP16 MMA Accumulation Probe ---\n");
    printf("Instruction: mma.sync...f16.f16.f16.f16 (All FP16)\n");
    printf("C_init:      %.1f\n", c_val);
    printf("A * B Sum:   1.0\n");
    printf("Result d0.x: %.1f\n", res0);
    
    if (res0 == 1024.0f) {
        printf("Analysis: Result == C_init. Precision LOST.\n");
        printf("Conclusion: Accumulator behaves like FP16 (or output truncation happened).\n");
    } else if (res0 == 1025.0f) {
        printf("Analysis: Result == 1025.0. Precision KEPT.\n");
        printf("Conclusion: Internal accumulator has >11 bits precision (likely FP32).\n");
    } else {
        printf("Analysis: Unexpected result.\n");
    }
  }
}

int main() {
  test_fp16_acc_kernel<<<1, 32>>>();

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
    return -1;
  }

  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("Kernel execution failed: %s\n", cudaGetErrorString(err));
    return -1;
  }

  return 0;
}