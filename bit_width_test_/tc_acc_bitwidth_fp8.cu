#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>

#define CUDA_CHECK(call) do { \
  cudaError_t err = call; \
  if (err != cudaSuccess) { \
    fprintf(stderr, "CUDA Error: %s in %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
    exit(EXIT_FAILURE); \
  } \
} while (0)

// --- WGMMA Shape (m64n8k32) ---
#define M_T 64
#define N_T 8
#define K_T 32

using FP8_E4M3 = __nv_fp8_e4m3;
using FP8_E5M2 = __nv_fp8_e5m2;
using FP32 = float;

#define HOST_DEVICE __forceinline__ __host__ __device__
#define DEVICE __forceinline__ __device__

// ---- Minimal GMMA descriptor (no-swizzle) ----
union GmmaDescriptor {
  HOST_DEVICE constexpr GmmaDescriptor() noexcept : desc_(0) {}
  HOST_DEVICE constexpr GmmaDescriptor(uint64_t desc) noexcept : desc_(desc) {}
  uint64_t desc_;
  struct {
    uint16_t start_address_ : 14, : 2;
    uint16_t leading_byte_offset_ : 14, : 2;
    uint16_t stride_byte_offset_  : 14, : 2;
    uint8_t  : 1,  base_offset_   : 3, : 4;
    uint8_t  : 6,  layout_type_   : 2;
  } bitfield;
  HOST_DEVICE constexpr operator uint64_t() const noexcept { return desc_; }
};

template <class PointerType>
DEVICE GmmaDescriptor make_desc(PointerType smem_ptr) {
  GmmaDescriptor desc;
  uint32_t u = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  desc.bitfield.start_address_      = u >> 4;  // 16B granularity
  desc.bitfield.layout_type_        = 0;       // no swizzle
  desc.bitfield.leading_byte_offset_= 8;       // conservative minimal example
  desc.bitfield.stride_byte_offset_ = 16;
  return desc;
}

DEVICE void warpgroup_arrive()      { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
DEVICE void warpgroup_commit_batch(){ asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int N> DEVICE void warpgroup_wait() {
  asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
}

// ---- FP8 storage unions ----
union fp8_e4m3_u { __nv_fp8_storage_t s; __nv_fp8_e4m3 t; };
union fp8_e5m2_u { __nv_fp8_storage_t s; __nv_fp8_e5m2 t; };

// ===================================================================================
// KERNEL: WGMMA (H100) FP32 Accumulator Bit-Width Probe (m64n8k32, E5M2×E4M3→F32)
// ===================================================================================
__global__ void probe_tensorcore_wgmma_f32_kernel(
    const FP8_E5M2* __restrict__ g_A,      // M x K  (row-major content)
    const FP8_E4M3* __restrict__ g_B_col,  // K x N  (column-major content)
    FP32* __restrict__ C_out,              // linear buffer of size M*N
    int M, int N, int K, FP32 c_init)
{
    // Expect 1 warpgroup = 128 threads
    if (blockDim.x != 128) return;
    if (threadIdx.x >= 128) return;

    extern __shared__ uint8_t smem[];
    // Layout in shared (contiguous, no swizzle): A then B
    FP8_E5M2* s_A = reinterpret_cast<FP8_E5M2*>(smem);
    FP8_E4M3* s_B = reinterpret_cast<FP8_E4M3*>(smem + M * K * sizeof(FP8_E5M2));

    // Gmem -> Smem (coalesced linear copy)
    int num_A = M * K;
    int num_B = K * N;
    for (int i = threadIdx.x; i < num_A; i += 128) s_A[i] = g_A[i];
    for (int i = threadIdx.x; i < num_B; i += 128) s_B[i] = g_B_col[i];
    __syncthreads();

    // Descriptors (minimal, no-swizzle)
    GmmaDescriptor desc_a = make_desc(s_A);
    GmmaDescriptor desc_b = make_desc(s_B);

    // Each thread holds 4 accumulators (m64n8k32.f32.*.*)
    FP32 d[4] = {c_init, c_init, c_init, c_init};

    // D = A*B + D
    warpgroup_arrive();
    asm volatile(
        "{\n"
        " .reg .pred p;\n"
        " setp.ne.b32 p, %6, 0;\n" // scale_D != 0 -> use D = A*B + D
        " wgmma.mma_async.sync.aligned.m64n8k32.f32.e5m2.e4m3 "
        "   {%0, %1, %2, %3}, %4, %5, p, 1, 1;\n"
        "}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "l"(desc_a), "l"(desc_b), "r"(int32_t(1))
    );
    warpgroup_commit_batch();
    warpgroup_wait<0>();

    // Write back: one thread writes 4 slots linearly, covering exactly M*N (=512) entries
    int base = (threadIdx.x << 2); // *4
    if (base + 3 < M * N) {
      C_out[base + 0] = d[0];
      C_out[base + 1] = d[1];
      C_out[base + 2] = d[2];
      C_out[base + 3] = d[3];
    }
}

// ===================================================================================
// Host: FP32 accumulator bit-width probe (C_init + 1)
// ===================================================================================
int main() {
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  std::cout << "GPU: " << prop.name << " (sm_" << prop.major << prop.minor << ")\n";
  if (!(prop.major == 9)) {
    std::cerr << "This program targets Hopper (sm_90a) WGMMA.\n";
    return 1;
  }

  // Bit-width probe on FP32 accumulator (mantissa 23+guard etc.)
  const int PROBE_START_BIT = 12;
  const int PROBE_END_BIT   = 39;

  // --- c_init = 2^(PROBE_START_BIT - 1) ---
  const int E = PROBE_START_BIT - 1;

  // fp32 encoding: sign=0, exp=(127+E), mantissa=0
  uint32_t c_init_bin = static_cast<uint32_t>((127 + E) << 23);

//   uint32_t c_init_bin = 0b0'10010110'00000000000000000000000; // 2^23

  float c_init = *reinterpret_cast<float*>(&c_init_bin);
  double ideal_pass = (double)c_init + 1.0;
  double ideal_fail = (double)c_init;

    std::cout << "Using Shape: m64n8k32, full-tile output = 1 so every slot is probed.\n";
    std::cout << "C_init (f32) = " << std::fixed << std::setprecision(8)
            << c_init << " (= 2^" << E << ")\n";
    std::cout << "---------------------------------------------------------------------------------------------------------------\n";
    std::cout << std::left << std::setw(10) << "Bit"
            << "| " << std::setw(14) << "add1"
            << "| " << std::setw(14) << "add2"
            << "| " << std::setw(16) << "ideal_pass"
            << "| " << std::setw(16) << "TC_result"
            << "| " << std::setw(12) << "delta"
            << "| " << "Analysis\n";
    std::cout << "---------------------------------------------------------------------------------------------------------------\n";

  const int M = M_T, N = N_T, K = K_T; // 64 x 8 x 32

  // Host buffers
  FP8_E5M2 *h_A;
  FP8_E4M3 *h_B, *h_B_col;
  FP32     *h_C;
  CUDA_CHECK(cudaMallocHost(&h_A,     M*K*sizeof(FP8_E5M2)));
  CUDA_CHECK(cudaMallocHost(&h_B,     K*N*sizeof(FP8_E4M3)));   // row-major staging
  CUDA_CHECK(cudaMallocHost(&h_B_col, K*N*sizeof(FP8_E4M3)));   // col-major for kernel
  CUDA_CHECK(cudaMallocHost(&h_C,     M*N*sizeof(FP32)));

  // Device buffers
  FP8_E5M2 *d_A;
  FP8_E4M3 *d_B_col;
  FP32     *d_C;
  CUDA_CHECK(cudaMalloc(&d_A,     M*K*sizeof(FP8_E5M2)));
  CUDA_CHECK(cudaMalloc(&d_B_col, K*N*sizeof(FP8_E4M3)));
  CUDA_CHECK(cudaMalloc(&d_C,     M*N*sizeof(FP32)));

//   int fp32_mantissa = 23;

  for (int bit = PROBE_START_BIT; bit <= PROBE_END_BIT; ++bit) {
    double add2 = std::pow(2.0, -(bit - E));
    double add1 = 1.0 - add2;

    // double add2 = 1.25;
    // double add1 = 1.25;

    std::memset(h_A, 0, M*K*sizeof(FP8_E5M2));
    std::memset(h_B, 0, K*N*sizeof(FP8_E4M3));
    std::memset(h_C, 0, M*N*sizeof(FP32));

    // Fill A: for all rows r, set A[r,0] = a1, A[r,1] = a2  (others 0)
    fp8_e5m2_u ua1, ua2;
    ua1.s = __nv_cvt_float_to_fp8((float)add1, __NV_SATFINITE, __NV_E5M2);
    ua2.s = __nv_cvt_float_to_fp8((float)add2, __NV_SATFINITE, __NV_E5M2);
    for (int r = 0; r < M; ++r) {
      h_A[r*K + 0] = ua1.t;
      h_A[r*K + 1] = ua2.t;

      // h_A[r*K + 0]  = ua1.t;  // k0 = 0
      // h_A[r*K + 16] = ua2.t;  // k1 = 16，落在 K 的另一半
      // the rest of row r remains 0
    }

    // Fill B (row-major staging): for all cols c, set B[0,c]=1, B[1,c]=1 (others 0)
    fp8_e4m3_u uone;  uone.s = __nv_cvt_float_to_fp8(1.0f, __NV_SATFINITE, __NV_E4M3);
    for (int c = 0; c < N; ++c) {
      h_B[0*N + c] = uone.t;  // row 0
      h_B[1*N + c] = uone.t;  // row 1
      // h_B[16*N + c] = uone.t;  // k=16 对应的行
    }

    // Convert B to column-major (KxN) for kernel (matches prior working pattern)
    for (int r=0; r<K; ++r) {
      for (int c=0; c<N; ++c) {
        h_B_col[c*K + r] = h_B[r*N + c];
      }
    }

    CUDA_CHECK(cudaMemcpy(d_A,     h_A,     M*K*sizeof(FP8_E5M2), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B_col, h_B_col, K*N*sizeof(FP8_E4M3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_C, 0, M*N*sizeof(FP32)));

    // Dynamic shared memory footprint
    size_t smem_size = (size_t)M*K*sizeof(FP8_E5M2) + (size_t)K*N*sizeof(FP8_E4M3);

    // Launch: exactly one warpgroup (128 threads)
    probe_tensorcore_wgmma_f32_kernel<<<1, 128, smem_size>>>(
        d_A, d_B_col, d_C, M, N, K, c_init
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_C, d_C, M*N*sizeof(FP32), cudaMemcpyDeviceToHost));
    float tc_result = h_C[0];  // 全部位置都应是 c_init+1，这里取第一个即可

    double delta = static_cast<double>(tc_result) - ideal_pass;
    std::string analysis = (tc_result == static_cast<float>(ideal_pass))
                        ? "PASS" : "FAIL (Flipped!)";

    // 行内采用科学计数法展示 add1/add2；其余保持定点
    std::cout << std::left << std::setw(10) << bit
            << "| " << std::scientific << std::setprecision(6) << std::setw(14) << add1
            << "| " << std::scientific << std::setprecision(6) << std::setw(14) << add2
            << "| " << std::fixed      << std::setprecision(8) << std::setw(16) << ideal_pass
            << "| " << std::fixed      << std::setprecision(8) << std::setw(16) << tc_result
            << "| " << std::scientific << std::setprecision(3) << std::setw(12) << delta
            << "| " << analysis << "\n";

    if (analysis.find("FAIL") != std::string::npos) {
      int eff = bit - 1;
      std::cout << "\nFlip detected at bit " << bit
                << ". Effective mantissa width likely " << eff << " bits.\n";
      break;
    }
  }

  // Cleanup
  CUDA_CHECK(cudaFreeHost(h_A));
  CUDA_CHECK(cudaFreeHost(h_B));
  CUDA_CHECK(cudaFreeHost(h_B_col));
  CUDA_CHECK(cudaFreeHost(h_C));
  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B_col));
  CUDA_CHECK(cudaFree(d_C));
  return 0;
}
