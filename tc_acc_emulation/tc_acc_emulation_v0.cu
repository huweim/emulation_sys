#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdint>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h> 

#define CUDA_CHECK(call)                                                                 \
do {                                                                                     \
    cudaError_t err = call;                                                              \
    if (err != cudaSuccess) {                                                            \
        fprintf(stderr, "CUDA Error: %s in %s at line %d\n", cudaGetErrorString(err),    \
                __FILE__, __LINE__);                                                     \
        exit(EXIT_FAILURE);                                                              \
    }                                                                                    \
} while (0)

// ===================================================================================
// KERNEL 1: 手动位操作仿真内核
// ===================================================================================

struct DecomposedFloat {
    bool sign;
    int32_t exponent;
    uint32_t mantissa;
};

__device__ inline uint32_t float_as_int(float f) { return *((uint32_t*)&f); }
__device__ inline float int_as_float(uint32_t i) { return *((float*)&i); }

__device__ void decompose_fp32(float f, DecomposedFloat& out) {
    if (f == 0.0f) {
        out.sign = (signbit(f) != 0);
        out.exponent = -127;
        out.mantissa = 0;
        return;
    }
    uint32_t i = float_as_int(f);
    out.sign = (i >> 31) & 1;
    uint32_t biased_exp = (i >> 23) & 0xFF;
    out.mantissa = i & 0x7FFFFF;
    if (biased_exp != 0) {
        out.exponent = biased_exp - 127;
        out.mantissa |= (1 << 23);
    } else {
        out.exponent = -126;
    }
}

__device__ float recompose_fp32_from_wide_mantissa(bool sign, int32_t exponent, uint64_t mantissa) {
    if (mantissa == 0) return 0.0f;
    
    int msb_pos = (mantissa > 0) ? (63 - __clzll(mantissa)) : 0;
    exponent += msb_pos;
    
    int shift = msb_pos - 23;
    if (shift > 0) {
        mantissa >>= shift;
    }

    uint32_t final_mantissa = mantissa & 0x7FFFFF;
    uint32_t biased_exponent = exponent + 127;

    if (biased_exponent >= 255) return sign ? -1.0f/0.0f : 1.0f/0.0f;
    if (biased_exponent <= 0) return 0.0f;

    uint32_t final_float_int = ((uint32_t)sign << 31) | (biased_exponent << 23) | final_mantissa;
    return int_as_float(final_float_int);
}

// --- FIX: Corrected function signature to accept M, N, K ---
__global__ void gemm_manual_bitwise_kernel(const half* A, const half* B, float* C, int M, int N, int K, float c_init) {
    if (blockIdx.x > 0 || blockIdx.y > 0 || threadIdx.x > 0) return;
    
    const int ACCUMULATOR_MANTISSA_WIDTH = 25;
    
    float numbers_to_sum[17];
    numbers_to_sum[0] = c_init;
    for (int k = 0; k < 16; ++k) {
        numbers_to_sum[k + 1] = __half2float(A[k]) * __half2float(B[k * N]);
    }

    DecomposedFloat decomposed[17];
    int32_t max_exp = -200;
    for (int i = 0; i < 17; ++i) {
        decompose_fp32(numbers_to_sum[i], decomposed[i]);
        if (decomposed[i].mantissa != 0) {
            max_exp = max(max_exp, decomposed[i].exponent);
        }
    }

    uint64_t sum_mantissa = 0;
    for (int i = 0; i < 17; ++i) {
        if (decomposed[i].mantissa == 0) continue;
        uint64_t mant = decomposed[i].mantissa;
        int shift = max_exp - decomposed[i].exponent;
        if (shift < 64) {
            mant >>= shift;
        } else {
            mant = 0;
        }
        sum_mantissa += mant;
    }
    
    C[0] = recompose_fp32_from_wide_mantissa(false, max_exp, sum_mantissa);
}

// ===================================================================================
// KERNEL 2: Tensor Core Kernel (硬件基准)
// ===================================================================================
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
__global__ void probe_tensorcore_inner_acc_kernel(const half* A, const half* B_col, float* C_out, int M, int N, int K, float c_init) {
    if (blockIdx.x == 0 && threadIdx.x < 32) {
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major> fr_A;
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::col_major> fr_B;
        nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> fr_C;
        
        nvcuda::wmma::fill_fragment(fr_C, c_init);
        nvcuda::wmma::load_matrix_sync(fr_A, A, K);
        nvcuda::wmma::load_matrix_sync(fr_B, B_col, K);
        nvcuda::wmma::mma_sync(fr_C, fr_A, fr_B, fr_C); 
        nvcuda::wmma::store_matrix_sync(C_out, fr_C, N, nvcuda::wmma::mem_row_major);
    }
}

// ===================================================================================
// Main Host Code
// ===================================================================================
int main() {
    uint32_t c_init_bin  = 0b0'10010110'00000000000000000000000; // 8388608.0f
    uint32_t addend1_bin = 0b0'01111110'11100000000000000000000; // 0.875f
    uint32_t addend2_bin = 0b0'01111100'00000000000000000000000; // 0.125f

    float c_init  = *((float*)&c_init_bin);
    float addend1 = *((float*)&addend1_bin);
    float addend2 = *((float*)&addend2_bin);

    std::cout << "--- Probing Inner-MMA Accumulation: C_init + (addend1 + addend2) ---" << std::endl;
    std::cout << std::fixed << std::setprecision(8);
    std::cout << "C_init:    " << c_init << std::endl;
    std::cout << "Addend 1:  " << addend1 << std::endl;
    std::cout << "Addend 2:  " << addend2 << std::endl;
    double ideal_result = (double)c_init + (double)addend1 + (double)addend2;
    std::cout << "Ideal (FP64) Result: " << ideal_result << std::endl;

    const int M = 16, N = 16, K = 16;
    half *h_A, *h_B, *h_B_col;
    float *h_C_tc, *h_C_emu;
    CUDA_CHECK(cudaMallocHost(&h_A, M*K*sizeof(half))); CUDA_CHECK(cudaMallocHost(&h_B, K*N*sizeof(half))); CUDA_CHECK(cudaMallocHost(&h_B_col, K*N*sizeof(half)));
    CUDA_CHECK(cudaMallocHost(&h_C_tc, M*N*sizeof(float))); CUDA_CHECK(cudaMallocHost(&h_C_emu, M*N*sizeof(float)));
    
    memset(h_A, 0, M*K*sizeof(half)); memset(h_B, 0, K*N*sizeof(half));
    h_A[0] = __float2half(addend1); h_A[1] = __float2half(addend2);
    h_B[0] = __float2half(1.0f); h_B[K+1] = __float2half(1.0f);
    for(int r=0; r<K; ++r) for(int c=0; c<N; ++c) h_B_col[c*K+r] = h_B[r*N+c];

    half *d_A, *d_B_col;
    float *d_C_tc, *d_C_emu;
    CUDA_CHECK(cudaMalloc(&d_A, M*K*sizeof(half))); CUDA_CHECK(cudaMalloc(&d_B_col, K*N*sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_C_tc, M*N*sizeof(float))); CUDA_CHECK(cudaMalloc(&d_C_emu, M*N*sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, M*K*sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B_col, h_B_col, K*N*sizeof(half), cudaMemcpyHostToDevice));
    
    probe_tensorcore_inner_acc_kernel<<<1, 32>>>(d_A, d_B_col, d_C_tc, M, N, K, c_init);
    gemm_manual_bitwise_kernel<<<1, 1>>>(d_A, h_B, d_C_emu, M, N, K, c_init);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_C_tc, d_C_tc, M*N*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_C_emu, d_C_emu, M*N*sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "\n--- Results ---" << std::endl;
    std::cout << "Bitwise Emulation Result:    " << h_C_emu[0] << std::endl;
    std::cout << "Tensor Core Result (Golden): " << h_C_tc[0] << std::endl;
    
    std::cout << "\nCleaning up..." << std::endl;
    CUDA_CHECK(cudaFreeHost(h_A)); CUDA_CHECK(cudaFreeHost(h_B)); CUDA_CHECK(cudaFreeHost(h_B_col));
    CUDA_CHECK(cudaFreeHost(h_C_tc)); CUDA_CHECK(cudaFreeHost(h_C_emu));
    CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_B_col));
    CUDA_CHECK(cudaFree(d_C_tc)); CUDA_CHECK(cudaFree(d_C_emu));
    std::cout << "Cleanup complete." << std::endl;
    
    return 0;
}