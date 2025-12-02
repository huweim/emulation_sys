/// RUN: nvcc -O3 -std=c++17 -arch=sm_80  probe_mma_fp16_acc_fp16_ptx.cu -o probe_mma_fp16_acc_fp16_ptx
/// RUN: nvcc -O3 -std=c++17 -arch=sm_120 probe_mma_fp16_acc_fp16_ptx.cu -o probe_mma_fp16_acc_fp16_ptx

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

// pack 两个 float 成 .f16x2（mma 的 32-bit 寄存器载体）
static inline __device__ uint32_t pack_half2(float a, float b) {
  __half2 h2 = __floats2half2_rn(a, b);
  return *reinterpret_cast<uint32_t*>(&h2);
}

// 单次 PTX mma：f16xf16 -> f16 累加（C/D 都是 f16）
// lane0 注入 A/B；所有 lane 的 C 都初始化为 c_init（按 f16x2）
__global__ void probe_mma_fp16_acc_fp16_once(float c_init_f32,
                                             float a_hi_f32, float a_lo_f32,
                                             float* out_lane0_d0_as_f32)
{
  // C/D 用 .f16x2 打包在 32b 寄存器里
  uint32_t c0 = pack_half2(c_init_f32, c_init_f32);
  uint32_t c1 = pack_half2(c_init_f32, c_init_f32);
  uint32_t d0 = 0, d1 = 0;

  // A(4 regs), B(2 regs) 也用 .f16x2 寄存器载体
  uint32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
  uint32_t b0 = 0, b1 = 0;

  if (threadIdx.x == 0) {
    a0 = pack_half2(a_hi_f32, a_lo_f32);  // (1-eps, eps)
    b0 = pack_half2(1.0f, 1.0f);          // (1, 1)
  }

  // mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
      "{%0, %1}, "
      "{%2, %3, %4, %5}, "
      "{%6, %7}, "
      "{%8, %9};\n"
      : "=r"(d0), "=r"(d1)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
        "r"(b0), "r"(b1),
        "r"(c0), "r"(c1)
  );

  // 只观测 lane0 的 d0 低半（对应 fragment 的一个元素）
  if (threadIdx.x == 0) {
    __half2 d0h2 = *reinterpret_cast<__half2*>(&d0);
    float out0 = __half2float(__low2half(d0h2));
    *out_lane0_d0_as_f32 = out0;
  }
}

int main() {
  const int EPS_BIT_START = 11;
  const int EPS_BIT_END   = 26;                  

  const float c_init = std::ldexp(1.0f, 10); // 10 is the mantissa bit width of FP16 (S1E5M10)
  const float expect_pass = c_init + 1.0f;
  const float expect_fail = c_init;


  float* d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));

  std::cout << "--- PTX MMA (m16n8k16 f16xf16->f16) Accumulator Width Probe ---\n";
  std::cout << "c_init (f16) = 2^10"
            << " = " << std::fixed << std::setprecision(1) << c_init
            << "  (ULP ~ 1.0 around this magnitude)\n";
  std::cout << "Host sets (a_hi, a_lo) = (1 - eps, eps), eps = 2^{-(bit-10)}; only lane0 non-zero.\n";
  std::cout << "If a_hi rounds to 1.0 or a_lo rounds to 0.0 in f16, the probe is INVALID (inputs invisible under f16 quantization).\n";
  std::cout << "---------------------------------------------------------------------------------------------------------------------------------\n";
  std::cout << std::left
            << std::setw(10) << "Bit"
            << " | " << std::setw(12) << "a_hi[f16]"
            << " | " << std::setw(12) << "a_lo[f16]"
            << " | " << std::setw(18) << "TC result (f32 view)"
            << " | Analysis\n";
  std::cout << "---------------------------------------------------------------------------------------------------------------------------------\n";

  for (int bit = EPS_BIT_START; bit <= EPS_BIT_END; ++bit) {
    // eps = 2^{-(bit-10)}
    const double eps = std::ldexp(1.0, -(bit - 10));
    const double a_lo = eps;
    const double a_hi = 1.0 - eps;

    // 打印 “经 FP16 量化后”的实际输入（便于观察在 1.0 / 0.0 的量化边界）
    __half h_hi = __float2half(static_cast<float>(a_hi));
    __half h_lo = __float2half(static_cast<float>(a_lo));
    const float a_hi_f16 = __half2float(h_hi);
    const float a_lo_f16 = __half2float(h_lo);

    // 发射一次 PTX mma（f16 累加/输出）
    probe_mma_fp16_acc_fp16_once<<<1, 32>>>(
        c_init,
        static_cast<float>(a_hi), static_cast<float>(a_lo),
        d_out);
    CUDA_CHECK(cudaDeviceSynchronize());

    float tc = 0.f;
    CUDA_CHECK(cudaMemcpy(&tc, d_out, sizeof(float), cudaMemcpyDeviceToHost));

    // 判定逻辑：
    // 1) 若输入已在 f16 侧“不可见”（a_hi->1 或 a_lo->0），标记 INVALID
    // 2) 否则正常比较 PASS/FAIL
    bool invalid = false;
    if ((a_hi < 1.0 && a_hi_f16 == 1.0f) || (a_lo > 0.0 && a_lo_f16 == 0.0f)) {
      invalid = true;
    }

    const char* verdict =
        invalid ? "INVALID (inputs rounded)"
      : (tc == expect_pass) ? "PASS (+1 captured)"
      : (tc == expect_fail) ? "FAIL (stagnated)"
      : "???";

    std::cout << std::left
              << std::setw(10) << bit
              << " | " << std::scientific << std::setprecision(8) << std::setw(18) << eps
              << " | " << std::fixed << std::setprecision(7) << std::setw(12) << a_hi_f16
              << " | " << std::fixed << std::setprecision(7) << std::setw(12) << a_lo_f16
              << " | " << std::fixed << std::setprecision(7) << std::setw(18) << tc
              << " | " << verdict;

    if (a_hi_f16 == 1.0f && a_hi < 1.0) std::cout << "  (a_hi rounded to 1.0)";
    if (a_lo_f16 == 0.0f && a_lo > 0.0) std::cout << "  (a_lo rounded to 0.0)";
    std::cout << "\n";

    // 只有“真实 FAIL（非 INVALID）”时，才可据此推断有效位宽翻转点
    if (!invalid && tc == expect_fail) {
      std::cout << "\nFlip at bit " << bit
                << " -> observed stagnation with f16 accumulator/output.\n";
      break;
    }
  }

  std::cout << "---------------------------------------------------------------------------------------------------------------------------------\n";
  CUDA_CHECK(cudaFree(d_out));
  return 0;
}