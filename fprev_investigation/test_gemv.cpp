#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <chrono>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "../include/fprev_kernel.h"

// Enhanced CUDA error checking macro
#define CUDA_CHECK(call)                                                \
  do {                                                                  \
    cudaError_t err = call;                                             \
    if (err != cudaSuccess) {                                           \
      std::cerr << "CUDA Error in " << #call << " (" << __FILE__ << ":" \
                << __LINE__ << "): " << cudaGetErrorString(err)         \
                << " (error code: " << err << ")" << std::endl;         \
      throw std::runtime_error("CUDA operation failed");                \
    }                                                                   \
  } while (0)

// Function to check CUDA device capabilities
void check_cuda_device() {
  int device_count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&device_count));

  if (device_count == 0) {
    throw std::runtime_error("No CUDA devices found");
  }

  std::cout << "Found " << device_count << " CUDA device(s)" << std::endl;

  for (int i = 0; i < device_count; i++) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, i));

    std::cout << "Device " << i << ": " << prop.name << std::endl;
    std::cout << "  Compute capability: " << prop.major << "." << prop.minor
              << std::endl;
    std::cout << "  Total global memory: "
              << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
    std::cout << "  Warp size: " << prop.warpSize << std::endl;
    std::cout << "  Max threads per block: " << prop.maxThreadsPerBlock
              << std::endl;
    std::cout << "  Max threads per multiprocessor: "
              << prop.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "  Number of SMs: " << prop.multiProcessorCount << std::endl;

    // Check for Tensor Core support
    if (prop.major >= 7) {
      std::cout << "  Tensor Cores: Supported" << std::endl;
    } else if (prop.major == 6 && prop.minor >= 1) {
      std::cout << "  Tensor Cores: Partially supported (Volta)" << std::endl;
    } else {
      std::cout << "  Tensor Cores: Not supported" << std::endl;
    }
    std::cout << std::endl;
  }

  // Set to device 0
  CUDA_CHECK(cudaSetDevice(0));
  std::cout << "Using CUDA device 0" << std::endl;
}

// Function to test basic CUDA functionality
void test_basic_cuda() {
  std::cout << "\n=== Testing Basic CUDA Functionality ===" << std::endl;

  const int N = 1024;
  float *d_data, *h_data;

  // Allocate memory
  CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(float)));
  h_data = new float[N];

  // Initialize host data
  for (int i = 0; i < N; i++) {
    h_data[i] = static_cast<float>(i);
  }

  // Copy to device
  CUDA_CHECK(
      cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice));

  // Copy back
  float* h_result = new float[N];
  CUDA_CHECK(
      cudaMemcpy(h_result, d_data, N * sizeof(float), cudaMemcpyDeviceToHost));

  // Verify
  bool correct = true;
  for (int i = 0; i < N; i++) {
    if (std::abs(h_result[i] - h_data[i]) > 1e-6) {
      correct = false;
      break;
    }
  }

  std::cout << "Basic memory copy test: " << (correct ? "PASSED" : "FAILED")
            << std::endl;

  // Cleanup
  delete[] h_data;
  delete[] h_result;
  CUDA_CHECK(cudaFree(d_data));
}

// Function to test FMA sequence (simpler, should work)
void test_fma_sequence() {
  std::cout << "\n=== Testing FMA Sequence ===" << std::endl;

  try {
    // Test parameters
    int i = 0, j = 1, n = 10;

    std::cout << "Testing FMA sequence with i=" << i << ", j=" << j
              << ", n=" << n << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    float result = fma_sequence_test(i, j, n);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "FMA result: " << std::scientific << std::setprecision(6)
              << result << std::endl;
    std::cout << "Execution time: " << duration.count() << " microseconds"
              << std::endl;
    std::cout << "FMA test: PASSED" << std::endl;

  } catch (const std::exception& e) {
    std::cout << "FMA test: FAILED - " << e.what() << std::endl;
  }
}

// Function to test GEMV sequence with detailed error reporting
void test_gemv_sequence() {
  std::cout << "\n=== Testing GEMV Sequence ===" << std::endl;

  try {
    // Test parameters - start with small values
    std::vector<std::tuple<int, int, int>> test_cases = {
        {0, 4, 16},  // Small test case
    };

    for (const auto& [i, j, n] : test_cases) {
      std::cout << "\nTesting GEMV sequence with i=" << i << ", j=" << j
                << ", n=" << n << std::endl;

      // Check if n is compatible with WMMA (should be multiple of 16 for
      // optimal performance)
      if (n % 16 != 0) {
        std::cout << "Warning: n=" << n
                  << " is not a multiple of 16, WMMA may have padding overhead"
                  << std::endl;
      }

      // Reset CUDA error state before test
      cudaGetLastError();

      auto start = std::chrono::high_resolution_clock::now();
      float result = gemv_sequence_test(i, j, n);
      auto end = std::chrono::high_resolution_clock::now();

      // Check for CUDA errors
      cudaError_t err = cudaGetLastError();
      if (err != cudaSuccess) {
        std::cerr << "CUDA kernel launch error: " << cudaGetErrorString(err)
                  << " (error code: " << err << ")" << std::endl;
        throw std::runtime_error("CUDA kernel launch failed");
      }

      auto duration =
          std::chrono::duration_cast<std::chrono::microseconds>(end - start);

      std::cout << "GEMV result: " << std::scientific << std::setprecision(6)
                << result << std::endl;
      std::cout << "Execution time: " << duration.count() << " microseconds"
                << std::endl;
      std::cout << "GEMV test case (" << i << "," << j << "," << n
                << "): PASSED" << std::endl;
    }

  } catch (const std::exception& e) {
    std::cout << "GEMV test: FAILED - " << e.what() << std::endl;

    // Additional debugging information
    std::cout << "\n=== Additional Debugging Information ===" << std::endl;

    // Check current device
    int current_device;
    CUDA_CHECK(cudaGetDevice(&current_device));
    std::cout << "Current CUDA device: " << current_device << std::endl;

    // Check device properties for WMMA support
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, current_device));
    std::cout << "Device compute capability: " << prop.major << "."
              << prop.minor << std::endl;

    if (prop.major < 7) {
      std::cout << "Warning: Device may not support WMMA operations properly"
                << std::endl;
    }

    // Check memory usage
    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    std::cout << "GPU memory: " << free_mem / (1024 * 1024) << " MB free / "
              << total_mem / (1024 * 1024) << " MB total" << std::endl;
  }
}

int main() {
  std::cout << "=== FPrev Investigation GEMV Test Suite ===" << std::endl;
  std::cout << "This test suite will help diagnose CUDA kernel launch issues"
            << std::endl;

  try {
    // Initialize CUDA
    std::cout << "\n=== Initializing CUDA ===" << std::endl;
    fprev_init();
    std::cout << "CUDA initialization: PASSED" << std::endl;

    // Check device capabilities
    check_cuda_device();

    // Test basic functionality
    test_basic_cuda();

    // Test FMA
    test_fma_sequence();

    // Test GEMV
    test_gemv_sequence();

    std::cout << "\n=== Test Suite Summary ===" << std::endl;
    std::cout << "All tests completed. Check above for any FAILED cases."
              << std::endl;

  } catch (const std::exception& e) {
    std::cerr << "Test suite failed with exception: " << e.what() << std::endl;
    return 1;
  }

  try {
    // Cleanup
    std::cout << "\n=== Cleaning Up ===" << std::endl;
    fprev_cleanup();
    std::cout << "CUDA cleanup: PASSED" << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "Cleanup failed: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
