#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <ctime>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h> // For WMMA (Tensor Core) API

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
    int mode = rand() % 100;
    if (mode < 69) return ((float)rand() / (float)RAND_MAX) * 4.0f - 2.0f;
    else if (mode < 84) return (((float)rand() / (float)RAND_MAX) - 0.5f) * 2000.0f;
    else if (mode < 99) return (((float)rand() / (float)RAND_MAX) - 0.5f) * 1e-2f;
    else return (((float)rand() / (float)RAND_MAX) - 0.5f) * 1e-6f;
}

void compare_results(const char* name1, const float* h_C1, const char* name2, const float* h_C2, int M, int N) {
    int mismatches = 0;
    float max_abs_error = 0.0f;
    float max_rel_error = 0.0f;
    double sum_abs_error = 0.0; 

    for (int i = 0; i < M * N; ++i) {
        float val1 = h_C1[i];
        float val2 = h_C2[i];
        float abs_error = fabsf(val1 - val2);
        sum_abs_error += abs_error;
        if (abs_error > 1e-5) mismatches++;
        if (abs_error > max_abs_error) max_abs_error = abs_error;
        if (fabsf(val1) > 1e-7) {
            float rel_error = abs_error / fabsf(val1);
            if (rel_error > max_rel_error) max_rel_error = rel_error;
        }
    }
    
    double avg_abs_error = sum_abs_error / (double)(M * N);

    std::cout << "\n--- Comparing [" << name1 << "] vs [" << name2 << "] ---" << std::endl;
    std::cout << "Mismatched elements (tolerance > 1e-5): " << mismatches << " / " << M*N << std::endl;
    std::cout << std::scientific << std::setprecision(8);
    std::cout << "Avg Absolute Error: " << avg_abs_error << std::endl;
    std::cout << "Max Absolute Error: " << max_abs_error << std::endl;
    std::cout << "Max Relative Error: " << max_rel_error << std::endl;
    std::cout << "------------------------------------------------" << std::endl;
}

// ===================================================================================
// KERNEL 1: Tensor Core Kernel (硬件基准)
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
    nvcuda::wmma::fill_fragment(fr_C, 0.0f);
    for (int k = 0; k < K; k += WMMA_K) {
        nvcuda::wmma::load_matrix_sync(fr_A, A + c_tile_row * K + k, K);
        nvcuda::wmma::load_matrix_sync(fr_B, B + k + c_tile_col * K, K);
        nvcuda::wmma::mma_sync(fr_C, fr_A, fr_B, fr_C);
    }
    nvcuda::wmma::store_matrix_sync(C + c_tile_row * N + c_tile_col, fr_C, N, nvcuda::wmma::mem_row_major);
}

// ===================================================================================
// KERNEL 2: Sequential FMA Emulation Kernel (简单模型)
// ===================================================================================
__global__ void gemm_fma_sequential_kernel(const half* A, const half* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            acc = fmaf(__half2float(A[row * K + k]), __half2float(B[k * N + col]), acc);
        }
        C[row * N + col] = acc;
    }
}

// ===================================================================================
// KERNEL 3: Tree Reduction FMA Emulation Kernel (高级模型)
// ===================================================================================
__global__ void gemm_fma_tree_kernel(const half* A, const half* B, float* C, int M, int N, int K) {
    // 每个 Block 计算一个输出元素 C[row, col]
    int row = blockIdx.y;
    int col = blockIdx.x;
    
    // 使用 256 线程进行并行规约
    const int num_threads = 256;
    __shared__ float partial_sums[num_threads];

    // 1. 各线程并行计算部分点积和
    // 每个线程计算 K/num_threads 个乘积，并用 double 累加
    double pvt_acc = 0.0;
    for (int k = threadIdx.x; k < K; k += num_threads) {
        pvt_acc += (double)__half2float(A[row * K + k]) * (double)__half2float(B[k * N + col]);
    }
    partial_sums[threadIdx.x] = (float)pvt_acc;
    __syncthreads();

    // 2. 在 Shared Memory 中进行树状(并行)规约
    for (int stride = num_threads / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial_sums[threadIdx.x] += partial_sums[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // 3. 线程 0 将最终归约结果写回全局内存
    if (threadIdx.x == 0) {
        if (row < M && col < N) {
            C[row * N + col] = partial_sums[0];
        }
    }
}

// ===================================================================================
// 主机端代码
// ===================================================================================
int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "GPU Name: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;

    // --- MODIFIED ---: 使用新的实验 Shape
    const int M = 16, N = 16, K = 32;
    std::cout << "\nMatrix Dimensions: M=" << M << ", N=" << N << ", K=" << K << std::endl;

    size_t A_size = M * K * sizeof(half);
    size_t B_size = K * N * sizeof(half);
    size_t C_size = M * N * sizeof(float);
    
    half *h_A, *h_B, *h_B_col_major;
    float *h_C_tc, *h_C_seq, *h_C_tree;
    CUDA_CHECK(cudaMallocHost(&h_A, A_size));
    CUDA_CHECK(cudaMallocHost(&h_B, B_size));
    CUDA_CHECK(cudaMallocHost(&h_B_col_major, B_size));
    CUDA_CHECK(cudaMallocHost(&h_C_tc, C_size));
    CUDA_CHECK(cudaMallocHost(&h_C_seq, C_size));
    CUDA_CHECK(cudaMallocHost(&h_C_tree, C_size));

    half *d_A, *d_B_row, *d_B_col;
    float *d_C_tc, *d_C_seq, *d_C_tree;
    CUDA_CHECK(cudaMalloc(&d_A, A_size));
    CUDA_CHECK(cudaMalloc(&d_B_row, B_size));
    CUDA_CHECK(cudaMalloc(&d_B_col, B_size));
    CUDA_CHECK(cudaMalloc(&d_C_tc, C_size));
    CUDA_CHECK(cudaMalloc(&d_C_seq, C_size));
    CUDA_CHECK(cudaMalloc(&d_C_tree, C_size));

    srand(time(0)); 
    std::cout << "\nInitializing with extreme random data generator..." << std::endl;
    for (int i = 0; i < M * K; ++i) h_A[i] = __half2float(generate_extreme_float());
    for (int i = 0; i < K * N; ++i) h_B[i] = __half2float(generate_extreme_float());
    for (int r = 0; r < K; ++r) for (int c = 0; c < N; ++c) h_B_col_major[c * K + r] = h_B[r * N + c];
    
    CUDA_CHECK(cudaMemcpy(d_A, h_A, A_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B_row, h_B, B_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B_col, h_B_col_major, B_size, cudaMemcpyHostToDevice));
    
    // --- Kernel 启动 ---
    
    // Kernel 1: Tensor Core
    dim3 tc_gridDim(N / WMMA_N, M / WMMA_M);
    dim3 tc_blockDim(32);
    std::cout << "\nLaunching [Tensor Core Kernel]..." << std::endl;
    gemm_tensorcore_kernel<<<tc_gridDim, tc_blockDim>>>(d_A, d_B_col, d_C_tc, M, N, K);

    // Kernel 2: Sequential FMA
    dim3 seq_gridDim(N / 16, M / 16); // Assuming 16x16 block
    dim3 seq_blockDim(16, 16);
    std::cout << "Launching [Sequential FMA Kernel]..." << std::endl;
    gemm_fma_sequential_kernel<<<seq_gridDim, seq_blockDim>>>(d_A, d_B_row, d_C_seq, M, N, K);
    
    // Kernel 3: Tree Reduction FMA
    // 每个 Block 计算一个输出点，因此 Grid 维度和输出矩阵一样
    dim3 tree_gridDim(N, M);
    dim3 tree_blockDim(256); // 使用 256 线程进行并行规约
    std::cout << "Launching [Tree Reduction FMA Kernel]..." << std::endl;
    gemm_fma_tree_kernel<<<tree_gridDim, tree_blockDim>>>(d_A, d_B_row, d_C_tree, M, N, K);
    
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "All kernels executed successfully." << std::endl;

    // --- 结果拷贝与验证 ---
    CUDA_CHECK(cudaMemcpy(h_C_tc, d_C_tc, C_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_C_seq, d_C_seq, C_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_C_tree, d_C_tree, C_size, cudaMemcpyDeviceToHost));

    compare_results("Tensor Core", h_C_tc, "Sequential FMA", h_C_seq, M, N);
    compare_results("Tensor Core", h_C_tc, "Tree FMA", h_C_tree, M, N);
    compare_results("Sequential FMA", h_C_seq, "Tree FMA", h_C_tree, M, N);

    // --- 清理 ---
    std::cout << "\nCleaning up..." << std::endl;
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B_row));
    CUDA_CHECK(cudaFree(d_B_col));
    CUDA_CHECK(cudaFree(d_C_tc));
    CUDA_CHECK(cudaFree(d_C_seq));
    CUDA_CHECK(cudaFree(d_C_tree));
    CUDA_CHECK(cudaFreeHost(h_A));
    CUDA_CHECK(cudaFreeHost(h_B));
    CUDA_CHECK(cudaFreeHost(h_B_col_major));
    CUDA_CHECK(cudaFreeHost(h_C_tc));
    CUDA_CHECK(cudaFreeHost(h_C_seq));
    CUDA_CHECK(cudaFreeHost(h_C_tree));
    std::cout << "Cleanup complete." << std::endl;

    return 0;
}