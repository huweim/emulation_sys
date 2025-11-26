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
// KERNEL: Tensor Core Kernel for inner-instruction accumulation
// ===================================================================================
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

__global__ void probe_tensorcore_inner_acc_kernel(const half* A, const half* B_col, 
                                                  float* C_out, int M, int N, int K, float c_init) {
    if (blockIdx.x == 0 && threadIdx.x < 32) { // Only one warp executes
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major> fr_A;
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::col_major> fr_B;
        nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> fr_C;
        
        // 用“大数”初始化累加器
        nvcuda::wmma::fill_fragment(fr_C, c_init);
        
        // 加载 A 和 B, A*B 的结果将是 (addend1 + addend2)
        nvcuda::wmma::load_matrix_sync(fr_A, A, K);
        nvcuda::wmma::load_matrix_sync(fr_B, B_col, K);

        // 执行一次 mma 指令: C_final = (A*B) + C_init
        nvcuda::wmma::mma_sync(fr_C, fr_A, fr_B, fr_C); 
        
        nvcuda::wmma::store_matrix_sync(C_out, fr_C, N, nvcuda::wmma::mem_row_major);
    }
}

// ===================================================================================
// Main Host Code
// ===================================================================================
int main() {
    // --- [ USER-DEFINED TEST CASE ] ---
    // 请在这里用二进制定义您想探测的三个数
    // C_final = (addend1 + addend2) + C_init
    
    // "大数" (累加器初始值 C_init)
    uint32_t c_init_bin  = 0b0'10010110'00000000000000000000000; // 8388608.0f (1.0 * 2^23)
    // 第一个“小数”
    uint32_t addend1_bin = 0b0'01111110'11000000000000000000000; // 0.5f
    // 第二个“小数”
    uint32_t addend2_bin = 0b0'01111100'00000000000000000000000; // 0.5f
    // --- [ END OF TEST CASE ] ---

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

    // --- Memory Allocation & Setup ---
    const int M = 16, N = 16, K = 16;
    half *h_A, *h_B, *h_B_col;
    float *h_C_tc;
    CUDA_CHECK(cudaMallocHost(&h_A, M*K*sizeof(half))); 
    CUDA_CHECK(cudaMallocHost(&h_B, K*N*sizeof(half))); 
    CUDA_CHECK(cudaMallocHost(&h_B_col, K*N*sizeof(half)));
    CUDA_CHECK(cudaMallocHost(&h_C_tc, M*N*sizeof(float)));
    
    // --- NEW: 使用您的新方法构造 A 和 B 矩阵 ---
    // 目标: 让 A*B 的点积在 C[0][0] 处等于 addend1 + addend2
    memset(h_A, 0, M*K*sizeof(half)); 
    memset(h_B, 0, K*N*sizeof(half));

    // A[0][0] = addend1
    // A[0][1] = addend2
    h_A[0 * K + 0] = __float2half(addend1);
    h_A[0 * K + 1] = __float2half(addend2);

    // B[0][0] = 1.0f
    // B[1][0] = 1.0f
    // 这样 A[0,:] 和 B[:,0] 的点积就是 addend1*1.0 + addend2*1.0
    h_B[0 * N + 0] = __float2half(1.0f);
    h_B[1 * N + 0] = __float2half(1.0f);
    
    // 为 Tensor Core 创建 B 的列主序版本
    for(int r=0; r<K; ++r) for(int c=0; c<N; ++c) h_B_col[c*K+r] = h_B[r*N+c];

    // --- Device Memory ---
    half *d_A, *d_B_col;
    float *d_C_tc;
    CUDA_CHECK(cudaMalloc(&d_A, M*K*sizeof(half))); 
    CUDA_CHECK(cudaMalloc(&d_B_col, K*N*sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_C_tc, M*N*sizeof(float)));

    // --- Copy Data ---
    CUDA_CHECK(cudaMemcpy(d_A, h_A, M*K*sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B_col, h_B_col, K*N*sizeof(half), cudaMemcpyHostToDevice));
    
    // --- Kernel Launch ---
    probe_tensorcore_inner_acc_kernel<<<1, 32>>>(d_A, d_B_col, d_C_tc, M, N, K, c_init);
    CUDA_CHECK(cudaDeviceSynchronize());

    // --- Get Results ---
    CUDA_CHECK(cudaMemcpy(h_C_tc, d_C_tc, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "\n--- Results ---" << std::endl;
    std::cout << "Tensor Core Result (at C[0]): " << h_C_tc[0] << std::endl;

    // --- Cleanup ---
    CUDA_CHECK(cudaFreeHost(h_A)); CUDA_CHECK(cudaFreeHost(h_B)); CUDA_CHECK(cudaFreeHost(h_B_col));
    CUDA_CHECK(cudaFreeHost(h_C_tc));
    CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_B_col));
    CUDA_CHECK(cudaFree(d_C_tc));
    
    return 0;
}