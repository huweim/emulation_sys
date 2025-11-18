#ifndef FPREV_KERNEL_H
#define FPREV_KERNEL_H

#include <cuda_runtime.h>

#include <cstdint>

// Function to test FMA accumulation sequence
// Creates array A where A[i] = M, A[j] = -M, others = 1.0
// Returns the sum result which reveals accumulation order
extern "C" {
// Test FMA accumulation for given mask positions i,j in sequence length n
float fma_sequence_test(int i, int j, int n);

// Batch version for testing multiple (i,j) pairs
void fma_sequence_batch_test(const int* i_indices, const int* j_indices,
                             float* results, int num_pairs, int n);

// Initialize CUDA resources
void fprev_init();

// Cleanup CUDA resources
void fprev_cleanup();

// Test GEMV accumulation for given mask positions i,j in sequence length n
float gemv_sequence_test(int i, int j, int n);
}

#endif  // FPREV_KERNEL_H
