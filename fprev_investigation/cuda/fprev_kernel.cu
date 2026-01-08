#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <mma.h>  // For WMMA (Tensor Core) API

#include <cmath>
#include <cstdint>
#include <cstdio>

#include "../include/fprev_kernel.h"

#define CUDA_CHECK(call)                                    \
  do {                                                      \
    cudaError_t err = call;                                 \
    if (err != cudaSuccess) {                               \
      fprintf(stderr, "CUDA Error: %s in %s at line %d\n",  \
              cudaGetErrorString(err), __FILE__, __LINE__); \
      exit(EXIT_FAILURE);                                   \
    }                                                       \
  } while (0)

// Large value M for swamping effect (8190.0f is the minimum value on RTX 5090)
#define M_VALUE 8190.0f

// CUDA kernel to test FMA accumulation sequence
// Creates array A where A[i] = M, A[j] = -M, others = 1.0
// Accumulates using FMA to reveal accumulation order
__global__ void fma_sequence_kernel(float* result, int i, int j, int n) {
  if (threadIdx.x == 0) {
    // Initialize accumulator
    float sum = 0.0f;

    // Create test array and accumulate using FMA
    for (int k = 0; k < n; k++) {
      float value;
      if (k == i) {
        value = M_VALUE;
      } else if (k == j) {
        value = -M_VALUE;
      } else {
        value = 1.0f;
      }

      // Use FMA for accumulation: sum = sum + value * 1.0
      sum = fmaf(value, 1.0f, sum);
    }

    result[0] = sum;
  }
}

// Batch kernel for testing multiple (i,j) pairs
__global__ void fma_sequence_batch_kernel(float* results, const int* i_indices,
                                          const int* j_indices, int num_pairs,
                                          int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_pairs) {
    int i = i_indices[idx];
    int j = j_indices[idx];

    // Initialize accumulator
    float sum = 0.0f;

    // Create test array and accumulate using FMA
    for (int k = 0; k < n; k++) {
      float value;
      if (k == i) {
        value = M_VALUE;
      } else if (k == j) {
        value = -M_VALUE;
      } else {
        value = 1.0f;
      }

      // Use FMA for accumulation
      sum = fmaf(value, 1.0f, sum);
    }

    results[idx] = sum;
  }
}

// Host function to test single (i,j) pair
float fma_sequence_test(int i, int j, int n) {
  float* d_result;
  CUDA_CHECK(cudaMalloc(&d_result, sizeof(float)));

  // Launch kernel
  fma_sequence_kernel<<<1, 1>>>(d_result, i, j, n);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Copy result back
  float h_result;
  CUDA_CHECK(
      cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_result));
  return h_result;
}

// Host function to test multiple (i,j) pairs
void fma_sequence_batch_test(const int* i_indices, const int* j_indices,
                             float* results, int num_pairs, int n) {
  // Allocate device memory
  int *d_i_indices, *d_j_indices;
  float* d_results;

  CUDA_CHECK(cudaMalloc(&d_i_indices, num_pairs * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_j_indices, num_pairs * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_results, num_pairs * sizeof(float)));

  // Copy data to device
  CUDA_CHECK(cudaMemcpy(d_i_indices, i_indices, num_pairs * sizeof(int),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_j_indices, j_indices, num_pairs * sizeof(int),
                        cudaMemcpyHostToDevice));

  // Launch kernel
  int blockSize = 256;
  int gridSize = (num_pairs + blockSize - 1) / blockSize;
  fma_sequence_batch_kernel<<<gridSize, blockSize>>>(d_results, d_i_indices,
                                                     d_j_indices, num_pairs, n);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Copy results back
  CUDA_CHECK(cudaMemcpy(results, d_results, num_pairs * sizeof(float),
                        cudaMemcpyDeviceToHost));

  // Cleanup
  CUDA_CHECK(cudaFree(d_i_indices));
  CUDA_CHECK(cudaFree(d_j_indices));
  CUDA_CHECK(cudaFree(d_results));
}

// Initialize CUDA resources
void fprev_init() {
  // CUDA initialization if needed
  cudaSetDevice(0);
}

// Fixed WMMA-based kernel for GEMV using Tensor Cores
// Treats GEMV as 1×n * n×1 matrix multiplication using 16×16 tiles
// Note: WMMA requires half precision for input matrices, but using float for
// accumulator
__global__ void gemv_wmma_kernel(const half* input, const half* weights,
                                 float* result, int n) {
  // Use 16x16x16 WMMA tile
  constexpr int WMMA_M = 16;
  constexpr int WMMA_N = 16;
  constexpr int WMMA_K = 16;

  // Only first warp participates
  if (threadIdx.x < 32) {
    // Declare fragments
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half,
                           nvcuda::wmma::row_major>
        frag_a;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half,
                           nvcuda::wmma::col_major>
        frag_b;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K,
                           float>
        frag_c;

    // Initialize accumulator to zero
    nvcuda::wmma::fill_fragment(frag_c, 0.0f);

    // Shared memory to hold padded input/weights (16 elements each)
    __shared__ half s_input[WMMA_K];
    __shared__ half s_weights[WMMA_K];

    // Zero-initialize shared memory (only first n elements matter)
    for (int i = threadIdx.x; i < WMMA_K; i += 32) {
      s_input[i] = __float2half(0.0f);
      s_weights[i] = __float2half(0.0f);
    }
    __syncthreads();

    // Load actual input and weights (n <= 16 guaranteed)
    if (threadIdx.x < n) {
      s_input[threadIdx.x] = input[threadIdx.x];
      s_weights[threadIdx.x] = weights[threadIdx.x];
    }
    __syncthreads();

    // Load into fragments:
    nvcuda::wmma::load_matrix_sync(frag_a, s_input,
                                   WMMA_K);  // row-major: first row = s_input
    nvcuda::wmma::load_matrix_sync(frag_b, s_weights,
                                   WMMA_K);  // col-major: first col = s_weights

    // Single WMMA operation
    nvcuda::wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);

    // Store result (only [0][0] is non-zero)
    __shared__ float s_result[WMMA_M * WMMA_N];
    nvcuda::wmma::store_matrix_sync(s_result, frag_c, WMMA_N,
                                    nvcuda::wmma::mem_row_major);

    if (threadIdx.x == 0) {
      result[0] = s_result[0];  // dot product result
    }
  }
}

float gemv_sequence_test(int i, int j, int n) {
  half* d_input = nullptr;
  half* d_weights = nullptr;
  float* d_result = nullptr;
  float h_result = 0.0f;

  CUDA_CHECK(cudaMalloc(&d_input, n * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&d_weights, n * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&d_result, sizeof(float)));

  // Host vectors (use half precision for WMMA)
  half* h_input = new half[n];
  half* h_weights = new half[n];

  for (int k = 0; k < n; ++k) {
    if (k == i) {
      h_input[k] = __float2half(M_VALUE);
      h_weights[k] = __float2half(M_VALUE);
    } else if (k == j) {
      h_input[k] = __float2half(-M_VALUE);
      h_weights[k] = __float2half(M_VALUE);
    } else {
      h_input[k] = __float2half(1.0f);
      h_weights[k] = __float2half(1.0f);
    }
  }

  CUDA_CHECK(
      cudaMemcpy(d_input, h_input, n * sizeof(half), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_weights, h_weights, n * sizeof(half),
                        cudaMemcpyHostToDevice));

  // Use WMMA kernel with proper block configuration
  gemv_wmma_kernel<<<1, 32>>>(d_input, d_weights, d_result, n);

  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(
      cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost));

  // Cleanup
  delete[] h_input;
  delete[] h_weights;
  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_weights));
  CUDA_CHECK(cudaFree(d_result));

  return h_result;
}

// Cleanup CUDA resources
void fprev_cleanup() { cudaDeviceReset(); }
