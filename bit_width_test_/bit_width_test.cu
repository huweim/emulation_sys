#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h> // For WMMA (Tensor Core) API

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
// KERNEL 1: FMA Emulation for the 2-step accumulation
// ===================================================================================
__global__ void probe_fma_kernel_2step(float* C_out, float c_init, float addend1, float addend2) {
    if (threadIdx.x == 0) {
        float acc = c_init;
        // acc = fmaf(addend1, 1.0f, acc); // C_step1 = addend1 * 1.0 + c_init
        // acc = fmaf(addend2, 1.0f, acc); // C_step2 = addend2 * 1.0 + C_step1

        // round to zero
        acc = __fmaf_rz(addend1, 1.0f, acc); // C_step1 = addend1 * 1.0 + c_init
        acc = __fmaf_rz(addend2, 1.0f, acc); // C_step2 = addend2 * 1.0 + C_step1
        C_out[0] = acc;
    }
}

// ===================================================================================
// KERNEL 2: Tensor Core Kernel for the 2-step accumulation
// ===================================================================================
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// --- FIX: Added M, N, K to the function signature ---
__global__ void probe_tensorcore_kernel_2step(const half* A1, const half* B1_col, 
                                            const half* A2, const half* B2_col, 
                                            float* C_out, int M, int N, int K, float c_init) {
    if (blockIdx.x == 0 && threadIdx.x < 32) { // Only one warp executes
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major> fr_A;
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::col_major> fr_B;
        nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> fr_C;
        
        nvcuda::wmma::fill_fragment(fr_C, c_init);
        
        // Step 1: Accumulate the first "small number"
        nvcuda::wmma::load_matrix_sync(fr_A, A1, K); // ldm for row-major A is K
        nvcuda::wmma::load_matrix_sync(fr_B, B1_col, K); // ldm for col-major B is K
        nvcuda::wmma::mma_sync(fr_C, fr_A, fr_B, fr_C); 

        // Step 2: Accumulate the second "small number"
        nvcuda::wmma::load_matrix_sync(fr_A, A2, K);
        nvcuda::wmma::load_matrix_sync(fr_B, B2_col, K);
        nvcuda::wmma::mma_sync(fr_C, fr_A, fr_B, fr_C);
        
        nvcuda::wmma::store_matrix_sync(C_out, fr_C, N, nvcuda::wmma::mem_row_major);
    }
}

// ===================================================================================
// Main Host Code
// ===================================================================================
int main() {
    // --- [ USER-DEFINED TEST CASE ] ---
    uint32_t c_init_bin  = 0b0'10010110'00000000000000000000001; // 10010110 -> 150
    uint32_t addend1_bin = 0b0'01111110'00000000000000000000000; // 01111111 -> 127
    uint32_t addend2_bin = 0b0'01111110'00000000000000000000000; // 1.0f
    // --- [ END OF TEST CASE ] ---

    float c_init  = *((float*)&c_init_bin);
    float addend1 = *((float*)&addend1_bin);
    float addend2 = *((float*)&addend2_bin);

    std::cout << "--- Probing a 2-step Accumulation ---" << std::endl;
    std::cout << std::fixed << std::setprecision(8);
    std::cout << "C_init:   " << c_init << std::endl;
    std::cout << "Addend 1:  " << addend1 << std::endl;
    std::cout << "Addend 2:  " << addend2 << std::endl;
    float math_result = (c_init + addend1) + addend2;
    std::cout << "Expected (CPU float math): " << math_result << std::endl;

    const int M = 16, N = 16, K = 16;
    half *h_A1, *h_B1, *h_B1_col, *h_A2, *h_B2, *h_B2_col;
    float *h_C_fma, *h_C_tc;
    CUDA_CHECK(cudaMallocHost(&h_A1, M*K*sizeof(half))); CUDA_CHECK(cudaMallocHost(&h_B1, K*N*sizeof(half))); CUDA_CHECK(cudaMallocHost(&h_B1_col, K*N*sizeof(half)));
    CUDA_CHECK(cudaMallocHost(&h_A2, M*K*sizeof(half))); CUDA_CHECK(cudaMallocHost(&h_B2, K*N*sizeof(half))); CUDA_CHECK(cudaMallocHost(&h_B2_col, K*N*sizeof(half)));
    CUDA_CHECK(cudaMallocHost(&h_C_fma, M*N*sizeof(float))); CUDA_CHECK(cudaMallocHost(&h_C_tc, M*N*sizeof(float)));
    
    memset(h_A1, 0, M*K*sizeof(half)); memset(h_B1, 0, K*N*sizeof(half));
    h_A1[0] = __float2half(addend1); h_B1[0] = __float2half(1.0f);
    memset(h_A2, 0, M*K*sizeof(half)); memset(h_B2, 0, K*N*sizeof(half));
    h_A2[0] = __float2half(addend2); h_B2[0] = __float2half(1.0f);
    for(int r=0; r<K; ++r) for(int c=0; c<N; ++c) h_B1_col[c*K+r] = h_B1[r*N+c];
    for(int r=0; r<K; ++r) for(int c=0; c<N; ++c) h_B2_col[c*K+r] = h_B2[r*N+c];

    half *d_A1, *d_B1_col, *d_A2, *d_B2_col;
    float *d_C_fma, *d_C_tc;
    CUDA_CHECK(cudaMalloc(&d_A1, M*K*sizeof(half))); CUDA_CHECK(cudaMalloc(&d_B1_col, K*N*sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_A2, M*K*sizeof(half))); CUDA_CHECK(cudaMalloc(&d_B2_col, K*N*sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_C_fma, M*N*sizeof(float))); CUDA_CHECK(cudaMalloc(&d_C_tc, M*N*sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A1, h_A1, M*K*sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B1_col, h_B1_col, K*N*sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_A2, h_A2, M*K*sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B2_col, h_B2_col, K*N*sizeof(half), cudaMemcpyHostToDevice));

    // --- Kernel Launch ---
    probe_fma_kernel_2step<<<1, 1>>>(d_C_fma, c_init, addend1, addend2);
    // --- FIX: Pass M, N, K to the kernel launch ---
    probe_tensorcore_kernel_2step<<<1, 32>>>(d_A1, d_B1_col, d_A2, d_B2_col, d_C_tc, M, N, K, c_init);
    CUDA_CHECK(cudaDeviceSynchronize());

    // --- Get Results ---
    CUDA_CHECK(cudaMemcpy(h_C_fma, d_C_fma, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_C_tc, d_C_tc, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "\n--- Results ---" << std::endl;
    std::cout << "FMA Emulation Result (at C[0]): " << h_C_fma[0] << std::endl;
    std::cout << "Tensor Core Result (at C[0]):   " << h_C_tc[0] << std::endl;
    
    // --- Cleanup ---
    // ( ... all cudaFree and cudaFreeHost calls ... )
    return 0;
}