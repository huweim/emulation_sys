#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>

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
// KERNEL 1: FMA (float) Probe
// ===================================================================================
__global__ void probe_fmaf_kernel(float* C_out, float c_init, float addend) {
    if (threadIdx.x == 0) {
        C_out[0] = fmaf(addend, 1.0f, c_init);
    }
}

// ===================================================================================
// KERNEL 2: FMA (double) Probe
// ===================================================================================
__global__ void probe_fma_double_kernel(float* C_out, double c_init, double addend) {
    if (threadIdx.x == 0) {
        double result = fma(addend, 1.0, c_init);
        C_out[0] = (float)result;
    }
}

// ===================================================================================
// Main Host Code
// ===================================================================================
int main() {
    // --- 定义测试用例 (纯加法版本) ---
    uint32_t c_init_bin  = 0b0'10010110'00000000000000000000000; // 10010110 -> 150
    float c_init_f  = *((float*)&c_init_bin);
    double c_init_d = (double)c_init_f;

    std::vector<double> addends;
    std::vector<std::string> labels;
    
    // Case 1: 恰好在中间 (平局) -> 应该舍入到偶数 8388608.0
    addends.push_back(0.5);
    labels.push_back("0.5 (Tie)");

    // Case 2 onwards: 加上一个逐渐变小的 epsilon 来设置 Sticky bit
    for (int i = 3; i <= 54; ++i) {
        addends.push_back(0.5 + pow(2.0, -i));
        labels.push_back("0.5 + 2^-" + std::to_string(i));
    }
    
    // --- 内存分配 ---
    float *h_C_f32, *h_C_f64;
    float *d_C_f32, *d_C_f64;
    CUDA_CHECK(cudaMallocHost(&h_C_f32, sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_C_f64, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C_f32, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C_f64, sizeof(float)));

    std::cout << "--- Probing FMA Accumulator Width via Sticky Bit (Addition Only) ---" << std::endl;
    std::cout << "Operation: " << std::fixed << std::setprecision(2) << c_init_d << " + (0.5 + epsilon)" << std::endl;
    std::cout << "Tie-breaking result (rounds to even): 8388608.00" << std::endl;
    std::cout << "Sticky bit forces round-up to:      8388609.00" << std::endl;
    std::cout << "-----------------------------------------------------------------------" << std::endl;
    std::cout << std::left << std::setw(20) << "Addend" 
              << "| " << std::setw(24) << "fmaf (float) Result"
              << "| " << std::setw(26) << "fma (double) Result" << std::endl;
    std::cout << "-----------------------------------------------------------------------" << std::endl;

    for (size_t i = 0; i < addends.size(); ++i) {
        double addend_d = addends[i];
        float  addend_f = (float)addend_d;
        
        // --- 启动内核 ---
        probe_fmaf_kernel<<<1, 1>>>(d_C_f32, c_init_f, addend_f);
        probe_fma_double_kernel<<<1, 1>>>(d_C_f64, c_init_d, addend_d);
        CUDA_CHECK(cudaDeviceSynchronize());

        // --- 获取结果 ---
        CUDA_CHECK(cudaMemcpy(h_C_f32, d_C_f32, sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_C_f64, d_C_f64, sizeof(float), cudaMemcpyDeviceToHost));
        
        std::cout << std::left << std::setw(20) << labels[i] << "| ";
        
        // 分析 fmaf 结果
        std::cout << std::fixed << std::setprecision(2) << std::setw(14) << h_C_f32[0];
        if (h_C_f32[0] < 8388609.0f) { std::cout << " (Flipped!)"; } else { std::cout << "           "; }
        std::cout << "| ";

        // 分析 fma 结果
        std::cout << std::setw(16) << h_C_f64[0];
        if (h_C_f64[0] < 8388609.0f) { std::cout << " (Flipped!)"; } else { std::cout << "           "; }
        std::cout << std::endl;
    }
    std::cout << "-----------------------------------------------------------------------" << std::endl;

    // ... Cleanup ...
    CUDA_CHECK(cudaFree(d_C_f32));
    CUDA_CHECK(cudaFree(d_C_f64));
    CUDA_CHECK(cudaFreeHost(h_C_f32));
    CUDA_CHECK(cudaFreeHost(h_C_f64));
    
    return 0;
}