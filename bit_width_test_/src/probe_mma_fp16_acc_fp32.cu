/// RUN: nvcc -O3 -std=c++17 -arch=sm_80  probe_mma_fp16_acc_fp32.cu -o probe_mma_fp16_acc_fp32
/// RUN: nvcc -O3 -std=c++17 -arch=sm_120 probe_mma_fp16_acc_fp32.cu -o probe_mma_fp16_acc_fp32

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <iostream>
#include <iomanip>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define CUDA_CHECK(call) do {                                            \
  cudaError_t err = (call);                                              \
  if (err != cudaSuccess) {                                              \
    fprintf(stderr, "CUDA Error: %s at %s:%d\n",                         \
            cudaGetErrorString(err), __FILE__, __LINE__);                \
    std::exit(EXIT_FAILURE);                                             \
  }                                                                      \
} while (0)

// 把两个 float 打成 .f16x2（对应 mma 的 32-bit 寄存器）
static inline __device__ uint32_t pack_half2(float a, float b) {
  __half2 h2 = __floats2half2_rn(a, b);
  return *reinterpret_cast<uint32_t*>(&h2);
}

// 一次 PTX mma 指令：f16 x f16 -> f32 累加到 f32（D写回f32）
// 仅 lane 0 提供非零 A/B；其它 lane 的 A/B=0；所有 lane 的 C 初始化为 c_init
__global__ void probe_mma_fp16_acc_fp32_once(float c_init,
                                             float a_hi, float a_lo,
                                             float* out_lane0_d0) {
  // D/C 四个 f32 scalars（每 lane）
  float d0=0.f, d1=0.f, d2=0.f, d3=0.f;
  float c0=c_init, c1=c_init, c2=c_init, c3=c_init;

  // A(4 regs, 每个 .f16x2) / B(2 regs, 每个 .f16x2)
  uint32_t a0=0, a1=0, a2=0, a3=0;
  uint32_t b0=0, b1=0;

  // 只有 lane 0 注入 {a_hi, a_lo} 和 {1,1}，其余 lane 全 0
  if (threadIdx.x == 0) {
    a0 = pack_half2(a_hi, a_lo);      // A 前两个元素：(1-eps, eps)
    b0 = pack_half2(1.0f, 1.0f);      // B 前两个元素：(1, 1)
    // a1/a2/a3, b1 默认 0
  }

  // PTX: mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0, %1, %2, %3}, "
      "{%4, %5, %6, %7}, "
      "{%8, %9}, "
      "{%10, %11, %12, %13};\n"
      : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
        "r"(b0), "r"(b1),
        "f"(c0), "f"(c1), "f"(c2), "f"(c3)
  );

  if (threadIdx.x == 0) {
    *out_lane0_d0 = d0;   // 读 lane0 的 d0 作为观测值
  }
}

int main() {
  // --------- 配置（保持与原 WMMA 版本一致）---------
  const int PROBE_START_BIT = 24;  // 先从 guard bit 起扫
  const int PROBE_END_BIT   = 39;  // 23 + 16

  // c_init = 2^23（8388608.0f），保证“+1.0”应当可见
  const uint32_t c_init_bin = 0b0'10010110'00000000000000000000000u;
  const float c_init = *reinterpret_cast<const float*>(&c_init_bin);

  const double expect_pass = double(c_init) + 1.0;
  const double expect_fail = double(c_init);

  float* d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));

  std::cout << "--- PTX MMA (m16n8k16 f16xf16->f32) Accumulator Width Probe ---\n";
  std::cout << "c_init = 2^23 = " << std::fixed << std::setprecision(1) << c_init << "\n";
  std::cout << "Host keeps add1+add2=1.0; only lane0 provides non-zero A/B.\n";
  std::cout << "--------------------------------------------------------------------------\n";
  std::cout << std::left
            << std::setw(12) << "Bit Probed" << " | "
            << std::setw(18) << "Addend2"    << " | "
            << std::setw(20) << "TC Result"  << " | "
            << "Analysis\n";
  std::cout << "--------------------------------------------------------------------------\n";

  for (int bit = PROBE_START_BIT; bit <= PROBE_END_BIT; ++bit) {
    // 原逻辑：addend2 = 2^{-(bit-23)}, addend1 = 1 - addend2
    const double add2 = std::ldexp(1.0, -(bit - 23));
    const double add1 = 1.0 - add2;

    // 发射一次 MMA
    probe_mma_fp16_acc_fp32_once<<<1, 32>>>(
        c_init,
        static_cast<float>(add1),
        static_cast<float>(add2),
        d_out);
    CUDA_CHECK(cudaDeviceSynchronize());

    float tc = 0.f;
    CUDA_CHECK(cudaMemcpy(&tc, d_out, sizeof(float), cudaMemcpyDeviceToHost));

    const char* verdict = (tc == static_cast<float>(expect_pass)) ? "PASS" : "FAIL";
    std::cout << std::left
              << std::setw(12) << bit << " | "
              << std::scientific << std::setprecision(8) << std::setw(18) << add2 << " | "
              << std::fixed      << std::setprecision(8) << std::setw(20) << tc   << " | "
              << verdict << "\n";

    if (verdict[0] == 'F') {
      std::cout << "\nFlip at bit " << bit
                << " -> effective mantissa likely " << (bit - 1) << " bits.\n";
      break;
    }
  }

  std::cout << "--------------------------------------------------------------------------\n";
  CUDA_CHECK(cudaFree(d_out));
  return 0;
}