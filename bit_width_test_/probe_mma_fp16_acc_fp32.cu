#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>
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

// TOOD: Any corner case? The START_BIT may not start from 23/24. Generally, we will see the test from PASS to FAIL. 

__global__ void probe_tensorcore_inner_acc_kernel(const half* A, const half* B_col, 
                                                  float* C_out, int M, int N, int K, float c_init) {
    if (blockIdx.x == 0 && threadIdx.x < 32) { // Only one warp executes
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
    // --- [ 实验配置 ] ---
    const int PROBE_START_BIT = 24; // 从第24位 (Guard bit) 开始探测
    const int PROBE_END_BIT   = 39; // 探测到第39位 (23 + 16)
    
    // "大数" (累加器初始值 C_init)
    uint32_t c_init_bin  = 0b0'10010110'00000000000000000000000; // 8388608.0f (1.0 * 2^23)
    float c_init  = *((float*)&c_init_bin);
    double ideal_result_pass = (double)c_init + 1.0;
    double ideal_result_fail = (double)c_init;

    std::cout << "--- General Tensor Core Accumulator Width Prober ---" << std::endl;
    std::cout << "Probing by testing C_init + (addend1 + addend2), where addend1+addend2=1.0" << std::endl;
    std::cout << std::fixed << std::setprecision(8);
    std::cout << "C_init = " << c_init << std::endl;
    std::cout << "Expected PASS result (width sufficient): " << ideal_result_pass << std::endl;
    std::cout << "Expected FAIL result (width insufficient): " << ideal_result_fail << std::endl;
    std::cout << "--------------------------------------------------------------------------" << std::endl;
    std::cout << std::left << std::setw(12) << "Bit Probed"
              << "| " << std::setw(18) << "Addend2"
              << "| " << std::setw(20) << "Tensor Core Result"
              << "| " << "Analysis" << std::endl;
    std::cout << "--------------------------------------------------------------------------" << std::endl;

    // --- 内存分配与设置 ---
    const int M = 16, N = 16, K = 16;
    half *h_A, *h_B, *h_B_col;
    float *h_C_tc;
    CUDA_CHECK(cudaMallocHost(&h_A, M*K*sizeof(half))); CUDA_CHECK(cudaMallocHost(&h_B, K*N*sizeof(half))); CUDA_CHECK(cudaMallocHost(&h_B_col, K*N*sizeof(half)));
    CUDA_CHECK(cudaMallocHost(&h_C_tc, M*N*sizeof(float)));
    
    half *d_A, *d_B_col;
    float *d_C_tc;
    CUDA_CHECK(cudaMalloc(&d_A, M*K*sizeof(half))); CUDA_CHECK(cudaMalloc(&d_B_col, K*N*sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_C_tc, M*N*sizeof(float)));

    // --- 探测循环 ---
    for (int bit_to_probe = PROBE_START_BIT; bit_to_probe <= PROBE_END_BIT; ++bit_to_probe) {
        // 构造 addend1 和 addend2, 它们的和为1.0, 但计算需要 bit_to_probe 的精度
        double addend2 = pow(2.0, -(bit_to_probe - 23));
        double addend1 = 1.0 - addend2;

        memset(h_A, 0, M*K*sizeof(half)); 
        memset(h_B, 0, K*N*sizeof(half));
        h_A[0 * K + 0] = __double2half(addend1);
        h_A[0 * K + 1] = __double2half(addend2);
        h_B[0 * N + 0] = __float2half(1.0f);
        h_B[1 * N + 0] = __float2half(1.0f);
        for(int r=0; r<K; ++r) for(int c=0; c<N; ++c) h_B_col[c*K+r] = h_B[r*N+c];

        CUDA_CHECK(cudaMemcpy(d_A, h_A, M*K*sizeof(half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B_col, h_B_col, K*N*sizeof(half), cudaMemcpyHostToDevice));
        
        probe_tensorcore_inner_acc_kernel<<<1, 32>>>(d_A, d_B_col, d_C_tc, M, N, K, c_init);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_C_tc, d_C_tc, M*N*sizeof(float), cudaMemcpyDeviceToHost));
        float tc_result = h_C_tc[0];

        // --- 打印与分析 ---
        std::string analysis = (tc_result == ideal_result_pass) ? "PASS" : "FAIL (Flipped!)";
        std::cout << std::left << std::setw(12) << bit_to_probe
                  << "| " << std::scientific << std::setprecision(8) << std::setw(18) << addend2
                  << "| " << std::fixed << std::setprecision(8) << std::setw(20) << tc_result
                  << "| " << analysis << std::endl;
        
        // 如果已经翻转，可以提前退出循环
        if (analysis == "FAIL (Flipped!)") {
            int effective_width = bit_to_probe - 1;
            std::cout << "\nFlip detected at bit " << bit_to_probe 
                      << ". Effective mantissa width is likely " << effective_width << " bits." << std::endl;
            break; 
        }
    }
    std::cout << "--------------------------------------------------------------------------" << std::endl;

    // --- Cleanup ---
    // ...
    return 0;
}