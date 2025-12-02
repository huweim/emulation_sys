/// RUN: nvcc -O3 -std=c++17 -arch=sm_120a probe_mma_nvfp4_acc_fp32.cu -o probe_mma_nvfp4_acc_fp32
/// NOTE:
///   - SM120a, PTX path: kind::mxf4nvf4 + block_scale + scale_vec::2X
///   - This is the **NVFP4** route (packed FP4 per byte: **two nibbles per byte**, 8 FP4 per .b32)
///   - Scale type = ue8m0 (block scale), we set every byte to 1.0 (0x7F)

#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA Error: %s in %s at line %d\n",                     \
              cudaGetErrorString(err), __FILE__, __LINE__);                    \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// Pack 8 FP4 nibbles into one .b32 (two nibbles per byte: low[3:0], high[7:4])
__host__ __device__ static inline
uint32_t pack_fp4_2perbyte(uint8_t n0, uint8_t n1, uint8_t n2, uint8_t n3,
                           uint8_t n4, uint8_t n5, uint8_t n6, uint8_t n7) {
  return  ((uint32_t)( (n0 & 0xF) | ((n1 & 0xF) << 4) )      ) |
          ((uint32_t)( (n2 & 0xF) | ((n3 & 0xF) << 4) ) << 8 ) |
          ((uint32_t)( (n4 & 0xF) | ((n5 & 0xF) << 4) ) << 16) |
          ((uint32_t)( (n6 & 0xF) | ((n7 & 0xF) << 4) ) << 24);
}

__global__ void probe_mma_nvfp4_acc_fp32_kernel() {
  // C init: 2^24 (“large number”)
  const uint32_t bin_2p24 = 0b0'10010111'00000000000000000000000;
  float c0 = *reinterpret_cast<const float*>(&bin_2p24);
  float c1 = c0, c2 = c0, c3 = c0;

  if (threadIdx.x == 0) {
    printf("c0=%f\n", c0);
  }

  // Outputs
  float d0=0.f, d1=0.f, d2=0.f, d3=0.f;

  // A/B fragments (NVFP4: **two FP4 per byte**; here we只在 lane0 最小赋值以命中 lane0,d0)
  uint32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
  uint32_t b0 = 0, b1 = 0;

  if (threadIdx.x == 0) {
    // FP4(E2M1)常用 nibble：0x1 ≈ +0.5, 0x2 ≈ +1.0
    // 让 (lane0,d0) 接收到两项 MAC：0.5*1.0 + 0.5*1.0 = +1.0
    // 其它位置清零，避免串扰。具体两个位置的选择依赖 fragment 映射；
    // 本最小例中把它们放在 a0/b0 的第一个字节（两个 nibble）与第二个字节（两个 nibble）。
    a0 = pack_fp4_2perbyte(
      0x1, 0x1,   // byte0: two 0.5
      0x0, 0x0,   // byte1: zeros
      0x0, 0x0,   // byte2: zeros
      0x0, 0x0    // byte3: zeros
    );
    a1 = a2 = a3 = 0;

    // B 放置对应的 1.0
    b0 = pack_fp4_2perbyte(
      0x2, 0x2,   // byte0: two 1.0
      0x0, 0x0,
      0x0, 0x0,
      0x0, 0x0
    );
    b1 = 0;

    // 说明：
    //   这构成 “C = 2^24；一次 MMA 贡献 +1.0” 的最小可观测样例。
    //   若内部累加器在最终写回 f32 之前能保留该进位，则 d0 = 2^24 + 1.0。
  }

  // Block scales: ue8m0 的 1.0 = 0x7F；用 scale_vec::2X，这里把四个字节都写成 0x7F，
  // 任意 byte-id 选择都等于 1.0
  const uint32_t scaleA = 0x7F7F7F7Fu;
  const uint32_t scaleB = 0x7F7F7F7Fu;
  // 选择器（byte-id, thread-id）设为 0
  uint16_t selA_byte = 0, selA_lane = 0;
  uint16_t selB_byte = 0, selB_lane = 0;

  // PTX: NVFP4 path (mxf4nvf4), K=64, two FP4 per byte, block_scale + scale_vec::2X
  asm volatile(
    "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X."
    "m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 \n\t"
    "{%0,  %1,  %2,  %3},\n\t"           // D (4 x f32)
    "{%4,  %5,  %6,  %7},\n\t"           // A (4 x b32)
    "{%8,  %9},\n\t"           // B (4 x b32)  <-- 注意：K=64 需要 4 个 B 寄存器
    "{%10, %11, %12, %13},\n\t"          // C (4 x f32)
    "{%14}, {%15, %16},\n\t"             // scaleA (ue8m0), {byte-id-a, thread-id-a}
    "{%17}, {%18, %19};\n\t"             // scaleB (ue8m0), {byte-id-b, thread-id-b}
    : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
    : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
      "r"(b0), "r"(b1),
      "f"(c0), "f"(c1), "f"(c2), "f"(c3),
      "r"(scaleA), "h"(selA_byte), "h"(selA_lane),
      "r"(scaleB), "h"(selB_byte), "h"(selB_lane)
  );

  if (threadIdx.x == 0) {
    printf("NVFP4 scaled MMA (m16n8k64, e2m1×e2m1 -> f32, ue8m0=1.0): d0=%f d1=%f\n", d0, d1);
  }
}

int main() {
  printf("NVFP4(E2M1, packed 2-per-byte, block_scale ue8m0=1.0) Probe via mxf4nvf4 (m16n8k64)\n");
  probe_mma_nvfp4_acc_fp32_kernel<<<1, 32>>>();
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  return 0;
}