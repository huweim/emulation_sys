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

// Large value M for swamping effect (2^127 for float32)
#define M_VALUE 5e4f

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
  // WMMA dimensions
  constexpr int WMMA_M = 16;
  constexpr int WMMA_N = 16;
  constexpr int WMMA_K = 16;

  // Ensure all threads in warp participate - no conditional execution
  // Use only the first warp (threads 0-31) for WMMA operations
  if (threadIdx.x < 32) {
    // Initialize WMMA fragments (half for inputs, float for accumulator)
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half,
                           nvcuda::wmma::row_major>
        frag_input;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half,
                           nvcuda::wmma::col_major>
        frag_weights;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K,
                           float>
        frag_result;

    // Initialize accumulator to zero
    nvcuda::wmma::fill_fragment(frag_result, 0.0f);

    // Process in chunks of WMMA_K
    for (int k = 0; k < n; k += WMMA_K) {
      // Use properly aligned shared memory for WMMA
      __align__(32) __shared__ half padded_input[WMMA_M * WMMA_K];
      __align__(32) __shared__ half padded_weights[WMMA_K * WMMA_N];

      // Initialize shared memory to zero
      for (int idx = threadIdx.x; idx < WMMA_M * WMMA_K; idx += 32) {
        padded_input[idx] = __float2half(0.0f);
      }
      for (int idx = threadIdx.x; idx < WMMA_K * WMMA_N; idx += 32) {
        padded_weights[idx] = __float2half(0.0f);
      }
      __syncthreads();

      // Load actual data (only first row/col, rest stays zero)
      int k_chunk = min(WMMA_K, n - k);
      for (int idx = threadIdx.x; idx < k_chunk; idx += 32) {
        // First row of input matrix (1×n) - row-major layout
        if (k + idx < n) {
          padded_input[idx] = input[k + idx];
        }
        // First column of weight matrix (n×1) - column-major layout
        if (k + idx < n) {
          padded_weights[idx] = weights[k + idx];
        }
      }
      __syncthreads();

      // Load WMMA fragments from shared memory with correct layout
      nvcuda::wmma::load_matrix_sync(frag_input, padded_input, WMMA_K);
      nvcuda::wmma::load_matrix_sync(frag_weights, padded_weights,
                                     WMMA_K);  // Use WMMA_K for both

      // Perform matrix multiplication
      nvcuda::wmma::mma_sync(frag_result, frag_input, frag_weights,
                             frag_result);
    }

    // Store result (only the [0,0] element is needed)
    __align__(32) __shared__ float result_matrix[WMMA_M * WMMA_N];
    nvcuda::wmma::store_matrix_sync(result_matrix, frag_result, WMMA_N,
                                    nvcuda::wmma::mem_row_major);

    // Only thread 0 writes the final result
    if (threadIdx.x == 0) {
      result[0] = result_matrix[0];  // Top-left element
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
