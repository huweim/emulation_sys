/// RUN: nvcc -O3 -std=c++17 -arch=sm_120a probe_mma_mxfp4_acc_fp32.cu -o probe_mma_mxfp4_acc_fp32
/// NOTE:
///   - SM120 scaled FP4 path using PTX: kind::mxf8f6f4 + block_scale + scale_vec::1X
///   - Inputs are FP4(E2M1) packed per byte’s low nibble (one nibble per byte -> 4 nibbles per .b32 here)
///   - Scale type = ue8m0, scale = 1.0 (0x7F)

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

__global__ void probe_mma_mxfp4_acc_fp32_kernel() {
  // C init: 2^24 “large number” to test whether +1 gets preserved
  const uint32_t bin_2p24 = 0b0'10010111'00000000000000000000000;

  // const uint32_t bin_2p24 = 0b0'1001000'00000000000000000000000; // 2^25

  float c0 = *reinterpret_cast<const float*>(&bin_2p24);
  float c1 = c0, c2 = c0, c3 = c0;

  if (threadIdx.x == 0) {
    printf("c0=%f\n", c0);
  }

  // Outputs
  float d0=0.f, d1=0.f, d2=0.f, d3=0.f;

  // A/B fragments (FP4 E2M1 packed in nibble-at-byte-low)
  uint32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
  uint32_t b0 = 0, b1 = 0;

  if (threadIdx.x == 0) {
    /// NOTE: 2^24 + (1.0×0.5) + (1.0×0.5) in (lane0,d0) → expect d0 = 16777216.0 + 1.0 = 16777217.0
    /// NOTE: This demonstrates an at least 25-bit (24 bits + 1-bit hidden mantissa) internal accumulator (carry survives to the final round).

    // The individual 4-bit and the 6-bit floating point type elements must be packed in an 8-bit container
    // Thus, the FP4 in 8-bit container is xxS1E2M1xx, and the FP6 in 8-bit container is xxS1E2M3. `xx` means padding bit.
    a0 = 0b00'0001'00'00'0001'00'00'0000'00'00'0000'00; // [0.5, 0.5, 0, 0]
    a1 = 0; a2 = 0; a3 = 0;

    
    b0 = 0b00'0010'00'00'0010'00'00'0000'00'00'0000'00; // [1.0, 1.0, 0, 0]
    // b0 = 0b00'0001'00'00'0001'00'00'0000'00'00'0000'00; // [0.5, 0.5, 0, 0]
    b1 = 0;

    /// NOTE: 2^24 + 4 x (0.5×0.5) in (lane0,d0) → expect d0 = 16777216.0 + 1.0 = 16777217.0
    /// NOTE: However, it is still 16777216.0, It shows the accumulation bit width is exactly 25 bits.
    // a0 = 0b00'0001'00'00'0001'00'00'0001'00'00'0001'00; // [0.5, 0.5, 0.5, 0.5]
    // b0 = 0b00'0001'00'00'0001'00'00'0001'00'00'0001'00; // [0.5, 0.5, 0.5, 0.5]

  }

  // Scales: ue8m0 encoding for 1.0 is 0x7F (bias=127 -> 2^(127-127)=1.0)
  uint32_t scaleA = 0x7F;
  uint32_t scaleB = 0x7F;

  // Metadata selectors (block-scale): (bid, tid) = (0, 0) here
  uint16_t bidA = 0, tidA = 0;
  uint16_t bidB = 0, tidB = 0;

  // PTX: SM120 scaled FP4 MMA
  // kind::mxf8f6f4 + block_scale + scale_vec::1X, shape m16n8k32, row.col
  // Types: f32 accum/output, e2m1 × e2m1 inputs, scale stype = ue8m0
  asm volatile(
    "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X."
    "m16n8k32.row.col.f32.e2m1.e2m1.f32.ue8m0 \n\t"
    "{%0,  %1,  %2,  %3},\n\t"   // D (4 x f32)
    "{%4,  %5,  %6,  %7},\n\t"   // A (4 x b32)
    "{%8,  %9},\n\t"             // B (2 x b32)
    "{%10, %11, %12, %13},\n\t"  // C (4 x f32)
    "{%14},\n\t"                 // scaleA (ue8m0)
    "{%15, %16},\n\t"            // {bidA, tidA}
    "{%17},\n\t"                 // scaleB (ue8m0)
    "{%18, %19};\n\t"            // {bidB, tidB}
    : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
    : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
      "r"(b0), "r"(b1),
      "f"(c0), "f"(c1), "f"(c2), "f"(c3),
      "r"(scaleA), "h"(bidA), "h"(tidA),
      "r"(scaleB), "h"(bidB), "h"(tidB)
  );

  if (threadIdx.x == 0) {
    printf("NVFP4 scaled MMA (m16n8k32, e2m1×e2m1 -> f32, ue8m0=1.0): d0=%f d1=%f\n", d0, d1);
  }
}

int main() {
  printf("NVFP4 FP4(E2M1) Accumulator Probe [block_scale mxf8f6f4, ue8m0=1.0]\n");
  probe_mma_mxfp4_acc_fp32_kernel<<<1, 32>>>();
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  return 0;
}