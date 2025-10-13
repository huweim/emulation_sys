#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdint>
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
// KERNEL 1: 手动位操作仿真内核 (单次舍入模型)
// ===================================================================================
// 自定义舍入函数：将 double 按照 RZ 模式舍入到指定尾数位宽
__device__ double custom_round_rz(double val, int mantissa_width) {
    if (val == 0.0) return 0.0; // Return double 0.0
    
    int exponent;
    double mantissa_frac = frexp(val, &exponent); // mantissa_frac is in [0.5, 1.0)
    
    // 修正了-1的错误
    double scaled_mantissa = ldexp(mantissa_frac, mantissa_width);
    
    // RZ 舍入就是截断
    long long truncated_mantissa = (long long)scaled_mantissa;

    // 将截断后的整数尾数重新缩放回去
    double rounded_val = ldexp((double)truncated_mantissa, exponent - mantissa_width);

    // --- FIX: Return the high-precision double value ---
    return rounded_val;
}

__global__ void gemm_manual_emulation_kernel(const half* A, const half* B, float* C, int M, int N, int K, float c_init) {
    if (threadIdx.x > 0 || blockIdx.x > 0) return;
    const int ACCUMULATOR_MANTISSA_WIDTH = 26;

    // 1. 累加器是 double 类型，但其值在每一步都受限
    double running_sum = c_init;
    for (int k = 0; k < K; ++k) {
        double product = (double)__half2float(A[k]) * (double)__half2float(B[k * N]);
        running_sum += product;
        // if (threadIdx.x == 0 || blockIdx.x == 0) {
        //     printf("Debug: k=%d, A=%f, B=%f, product=%f, running_sum(before round)=%f\n", k, __half2float(A[k]), __half2float(B[k * N]), product, running_sum);
        // }
        // 2. 每一次加法后，都立即进行自定义舍入
        running_sum = custom_round_rz(running_sum, ACCUMULATOR_MANTISSA_WIDTH);

    }
    // 3. 最终结果再进行一次舍入，符合最终输出的精度要求
    running_sum = custom_round_rz(running_sum, 24);
    C[0] = (float)running_sum;
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
    // --- [ USER-DEFINED TEST CASE ] ---
    // C_final = (addend1 + addend2) + C_init
    uint32_t c_init_bin  = 0b0'10010110'00000000000000000000000; // 8388608.0f
    uint32_t addend1_bin = 0b0'01111110'10000000000000000000000; // 0.875f
    uint32_t addend2_bin = 0b0'01111101'00000000000000000000000; // 0.125f
    // --- [ END OF TEST CASE ] ---

    float c_init  = *((float*)&c_init_bin);
    float addend1 = *((float*)&addend1_bin);
    float addend2 = *((float*)&addend2_bin);

    std::cout << "--- Probing Inner-MMA Width with Carry Propagation ---" << std::endl;
    std::cout << std::fixed << std::setprecision(8);
    std::cout << "C_init:    " << c_init << std::endl;
    std::cout << "Addend 1:  " << addend1 << std::endl;
    std::cout << "Addend 2:  " << addend2 << std::endl;
    double ideal_result = (double)c_init + (double)addend1 + (double)addend2;
    std::cout << "Ideal (FP64) Result:      " << ideal_result << std::endl;
    
    const int M = 16, N = 16, K = 16;
    half *h_A, *h_B, *h_B_col;
    float *h_C_tc, *h_C_emu;
    CUDA_CHECK(cudaMallocHost(&h_A, M*K*sizeof(half))); CUDA_CHECK(cudaMallocHost(&h_B, K*N*sizeof(half))); CUDA_CHECK(cudaMallocHost(&h_B_col, K*N*sizeof(half)));
    CUDA_CHECK(cudaMallocHost(&h_C_tc, M*N*sizeof(float))); CUDA_CHECK(cudaMallocHost(&h_C_emu, M*N*sizeof(float)));
    
    memset(h_A, 0, M*K*sizeof(half)); memset(h_B, 0, K*N*sizeof(half));
    h_A[0 * K + 0] = __float2half(addend1);
    h_A[0 * K + 1] = __float2half(addend2);
    h_B[0 * N + 0] = __float2half(1.0f);
    // --- FIX: Correct indexing for B[1][0] ---
    h_B[1 * N + 0] = __float2half(1.0f); 
    for(int r=0; r<K; ++r) for(int c=0; c<N; ++c) h_B_col[c*K+r] = h_B[r*N+c];

    half *d_A, *d_B_col;
    float *d_C_tc, *d_C_emu;
    CUDA_CHECK(cudaMalloc(&d_A, M*K*sizeof(half))); CUDA_CHECK(cudaMalloc(&d_B_col, K*N*sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_C_tc, M*N*sizeof(float))); CUDA_CHECK(cudaMalloc(&d_C_emu, M*N*sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, M*K*sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B_col, h_B_col, K*N*sizeof(half), cudaMemcpyHostToDevice));
    
    probe_tensorcore_inner_acc_kernel<<<1, 32>>>(d_A, d_B_col, d_C_tc, M, N, K, c_init);
    gemm_manual_emulation_kernel<<<1, 1>>>(h_A, h_B, d_C_emu, M, N, K, c_init);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_C_tc, d_C_tc, M*N*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_C_emu, d_C_emu, M*N*sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "\n--- Results (Emulation with 25-bit mantissa) ---" << std::endl;
    std::cout << "Bitwise Emulation Result:    " << h_C_emu[0] << std::endl;
    std::cout << "Tensor Core Result (Golden): " << h_C_tc[0] << std::endl;
    
    // ... Cleanup ...
    return 0;
}