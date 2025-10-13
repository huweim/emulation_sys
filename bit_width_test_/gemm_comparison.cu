#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h> // For WMMA (Tensor Core) API
#include <cstdint>

// CUDA 错误检查宏
#define CUDA_CHECK(call)                                                                 \
do {                                                                                     \
    cudaError_t err = call;                                                              \
    if (err != cudaSuccess) {                                                            \
        fprintf(stderr, "CUDA Error: %s in %s at line %d\n", cudaGetErrorString(err),    \
                __FILE__, __LINE__);                                                     \
        exit(EXIT_FAILURE);                                                              \
    }                                                                                    \
} while (0)

float generate_extreme_float() {
    int mode = rand() % 100; // 生成 0-99 的随机数来决定模式

    // 69% 的概率生成范围在 [-2, 2] 的“正常”随机数
    if (mode < 69) {
        return ((float)rand() / (float)RAND_MAX) * 4.0f - 2.0f;
    }
    // 15% 的概率生成大数量级的数
    else if (mode < 84) { // 69 + 15 = 84
        return (((float)rand() / (float)RAND_MAX) - 0.5f) * 2000.0f; // Range [-1000, 1000]
    }
    // 15% 的概率生成小数量级的数 (但仍在 half 的正常范围内)
    else if (mode < 99) { // 84 + 15 = 99
        return (((float)rand() / (float)RAND_MAX) - 0.5f) * 1e-2f; // Range [-0.005, 0.005]
    }
    // 1% 的概率生成极小的、可能下溢或变为次正规数的数
    else { // mode == 99
        return (((float)rand() / (float)RAND_MAX) - 0.5f) * 1e-6f; // Range close to half's precision limit
    }
}

void compare_results(const float* h_C1, const float* h_C2, int M, int N) {
    int mismatches = 0;
    float max_abs_error = 0.0f;
    float max_rel_error = 0.0f;
    // 使用 double 累加求和，以保证精度
    double sum_abs_error = 0.0; 

    for (int i = 0; i < M * N; ++i) {
        float val1 = h_C1[i];
        float val2 = h_C2[i];

        // 计算一次绝对误差，以备后用
        float abs_error = fabsf(val1 - val2);
        
        // 累加绝对误差
        sum_abs_error += abs_error;

        // 根据容差计算不匹配元素的数量
        if (abs_error > 1e-5) {
            mismatches++;
        }

        // 更新最大绝对误差
        if (abs_error > max_abs_error) {
            max_abs_error = abs_error;
        }

        // 更新最大相对误差
        if (fabsf(val1) > 1e-7) { // 避免除以非常小的数
            float rel_error = abs_error / fabsf(val1);
            if (rel_error > max_rel_error) {
                max_rel_error = rel_error;
            }
        }
    }
    
    // 计算平均绝对误差
    double avg_abs_error = sum_abs_error / (double)(M * N);

    std::cout << "\n--- Comparison Results ---" << std::endl;
    std::cout << "Total elements: " << M * N << std::endl;
    std::cout << "Mismatched elements (tolerance > 1e-5): " << mismatches << std::endl;
    std::cout << std::scientific << std::setprecision(8);
    // --- NEW ---: 输出平均绝对误差
    std::cout << "Avg Absolute Error: " << avg_abs_error << std::endl;
    std::cout << "Max Absolute Error: " << max_abs_error << std::endl;
    std::cout << "Max Relative Error: " << max_rel_error << std::endl;
    std::cout << "--------------------------" << std::endl;
}

__global__ void gemm_fma_emulation_kernel_double(const half* A, const half* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        // --- MODIFIED ---: 将累加器从 float 改为 double
        double acc = 0.0;

        // 沿 K 维度进行严格的顺序累加
        for (int k = 0; k < K; ++k) {
            half a_val = A[row * K + k];
            half b_val = B[k * N + col];

            // --- MODIFIED ---: 将所有操作数提升到 double 并使用 double 版本的 fma
            // 注意函数名是 fma() 而不是 fmaf()
            acc = fma((double)__half2float(a_val), (double)__half2float(b_val), acc);
        }

        // --- MODIFIED ---: 将 double 结果转换回 float 并存储
        C[row * N + col] = (float)acc;
    }
}


// ===================================================================================
// KERNEL 1: FMA Emulation Kernel (修改为 FP32 输出)
// ===================================================================================
__global__ void gemm_fma_emulation_kernel(const half* A, const half* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {

        uint32_t init_val_binary = 0b01000000010000000111001111111111;
        float init_val_float = *((float*)&init_val_binary);
        float acc = init_val_float;
        
        // float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            half a_val = A[row * K + k];
            half b_val = B[k * N + col];
            acc = __fmaf_rn(__half2float(a_val), __half2float(b_val), acc);
        }
        C[row * N + col] = acc;
    }
}

// ===================================================================================
// KERNEL 2: Tensor Core Kernel (修改为 FP32 输出并简化)
// ===================================================================================
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

__global__ void gemm_tensorcore_kernel(const half* A, const half* B, float* C, int M, int N, int K) {
    int c_tile_row = blockIdx.y * WMMA_M;
    int c_tile_col = blockIdx.x * WMMA_N;

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major> fr_A;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::col_major> fr_B;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> fr_C;

    uint32_t init_val_binary = 0b01000000010000000111001111111111;
    float init_val_float = *((float*)&init_val_binary);
    nvcuda::wmma::fill_fragment(fr_C, init_val_float);
    
    // nvcuda::wmma::fill_fragment(fr_C, 0.0f);

    for (int k = 0; k < K; k += WMMA_K) {
        int a_row = c_tile_row;
        int a_col = k;
        int b_row = k;
        int b_col = c_tile_col;
        
        // 注意：B 矩阵现在需要是列主序
        nvcuda::wmma::load_matrix_sync(fr_A, A + a_row * K + a_col, K);
        nvcuda::wmma::load_matrix_sync(fr_B, B + b_row + b_col * K, K); // B is col-major, so ldm is K

        nvcuda::wmma::mma_sync(fr_C, fr_A, fr_B, fr_C);
    }

    // --- MODIFIED & SIMPLIFIED ---: 直接将 FP32 累加器 fragment 存储到 FP32 内存
    nvcuda::wmma::store_matrix_sync(C + c_tile_row * N + c_tile_col, fr_C, N, nvcuda::wmma::mem_row_major);
}


// ===================================================================================
// 主机端代码
// ===================================================================================
int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "GPU Name: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;

    const int M = 16, N = 16, K = 32;
    std::cout << "\nMatrix Dimensions: M=" << M << ", N=" << N << ", K=" << K << std::endl;

    // 分配 Host & Device 内存
    size_t A_size = M * K * sizeof(half);
    size_t B_size = K * N * sizeof(half);
    size_t C_size = M * N * sizeof(float); // --- MODIFIED ---: C is now float
    
    half *h_A, *h_B, *h_B_col_major; // h_B is row-major, h_B_col_major is for TC
    float *h_C_tc, *h_C_fma;         // C is float
    CUDA_CHECK(cudaMallocHost(&h_A, A_size));
    CUDA_CHECK(cudaMallocHost(&h_B, B_size));
    CUDA_CHECK(cudaMallocHost(&h_B_col_major, B_size));
    CUDA_CHECK(cudaMallocHost(&h_C_tc, C_size));
    CUDA_CHECK(cudaMallocHost(&h_C_fma, C_size));

    half *d_A, *d_B_row, *d_B_col;
    float *d_C_tc, *d_C_fma;
    CUDA_CHECK(cudaMalloc(&d_A, A_size));
    CUDA_CHECK(cudaMalloc(&d_B_row, B_size));
    CUDA_CHECK(cudaMalloc(&d_B_col, B_size));
    CUDA_CHECK(cudaMalloc(&d_C_tc, C_size));
    CUDA_CHECK(cudaMalloc(&d_C_fma, C_size));

    // 初始化输入数据
    // srand(time(0));
    // for (int i = 0; i < M * K; ++i) h_A[i] = __float2half((float)(rand() % 100) / 50.0f - 1.0f);
    // for (int i = 0; i < K * N; ++i) h_B[i] = __float2half((float)(rand() % 100) / 50.0f - 1.0f);

    // 极端情况
    srand(time(0)); 
    std::cout << "\nInitializing with new random data generator..." << std::endl;

    for (int i = 0; i < M * K; ++i) {
        h_A[i] = __float2half(generate_extreme_float());
        h_A[i] = 0.0f;
    }
    for (int i = 0; i < K * N; ++i) {
        h_B[i] = __float2half(generate_extreme_float());
        h_B[i] = 0.0f; 
    }

    // --- BUG FIX ---: 创建 B 的列主序版本
    for (int r = 0; r < K; ++r) {
        for (int c = 0; c < N; ++c) {
            h_B_col_major[c * K + r] = h_B[r * N + c];
        }
    }
    
    // 拷贝数据到 Device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, A_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B_row, h_B, B_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B_col, h_B_col_major, B_size, cudaMemcpyHostToDevice));
    
    // --- Kernel 启动 ---
    
    // FMA Kernel (使用行主序 B)
    dim3 fma_blockDim(16, 16);
    dim3 fma_gridDim((N + fma_blockDim.x - 1) / fma_blockDim.x, (M + fma_blockDim.y - 1) / fma_blockDim.y);
    std::cout << "\nLaunching FMA Emulation Kernel (using row-major B)..." << std::endl;
    gemm_fma_emulation_kernel<<<fma_gridDim, fma_blockDim>>>(d_A, d_B_row, d_C_fma, M, N, K);
    // gemm_fma_emulation_kernel_double<<<fma_gridDim, fma_blockDim>>>(d_A, d_B_row, d_C_fma, M, N, K);
    
    // Tensor Core Kernel (使用列主序 B)
    dim3 tc_gridDim(N / WMMA_N, M / WMMA_M);
    dim3 tc_blockDim(32); // 1 warp
    std::cout << "Launching Tensor Core Kernel (using column-major B)..." << std::endl;
    gemm_tensorcore_kernel<<<tc_gridDim, tc_blockDim>>>(d_A, d_B_col, d_C_tc, M, N, K);
    
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "Kernels executed successfully." << std::endl;

    // 结果拷贝与验证
    CUDA_CHECK(cudaMemcpy(h_C_tc, d_C_tc, C_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_C_fma, d_C_fma, C_size, cudaMemcpyDeviceToHost));

    compare_results(h_C_tc, h_C_fma, M, N);

    // 清理
    // ... (cudaFree and cudaFreeHost calls) ...

    return 0;
}