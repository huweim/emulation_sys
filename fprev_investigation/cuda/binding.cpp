#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <vector>

#include "../include/fprev_kernel.h"

namespace py = pybind11;

// Wrapper for single FMA sequence test
float fma_sequence_test_wrapper(int i, int j, int n) {
  return fma_sequence_test(i, j, n);
}

// Wrapper for GEMV sequence test
float gemv_sequence_test_wrapper(int i, int j, int n) {
  return gemv_sequence_test(i, j, n);
}

// Wrapper for batch FMA sequence test
py::array_t<float> fma_sequence_batch_test_wrapper(py::array_t<int> i_indices,
                                                   py::array_t<int> j_indices,
                                                   int n) {
  // Get buffer info
  py::buffer_info i_buf = i_indices.request();
  py::buffer_info j_buf = j_indices.request();

  if (i_buf.size != j_buf.size) {
    throw std::runtime_error("Input arrays must have the same size");
  }

  int num_pairs = i_buf.size;

  // Prepare results array
  py::array_t<float> results(num_pairs);
  py::buffer_info res_buf = results.request();

  // Call CUDA function
  fma_sequence_batch_test(static_cast<int*>(i_buf.ptr),
                          static_cast<int*>(j_buf.ptr),
                          static_cast<float*>(res_buf.ptr), num_pairs, n);

  return results;
}

// Helper function to create test arrays for FPRev algorithm
py::array_t<float> create_test_array(int i, int j, int n) {
  py::array_t<float> array(n);
  py::buffer_info buf = array.request();
  float* ptr = static_cast<float*>(buf.ptr);

  // Large value M for swamping effect (2^127 for float32)
  const float M_VALUE = 1.701411834604692317316873037158841056e38f;

  for (int k = 0; k < n; k++) {
    if (k == i) {
      ptr[k] = M_VALUE;
    } else if (k == j) {
      ptr[k] = -M_VALUE;
    } else {
      ptr[k] = 1.0f;
    }
  }

  return array;
}

// SUMIMPL function that uses CUDA FMA for the FPRev algorithm
float sumimpl_fma(py::array_t<float> input_array) {
  py::buffer_info buf = input_array.request();
  float* data = static_cast<float*>(buf.ptr);
  int n = buf.size;

  // Use CUDA to sum with FMA
  float* d_data;
  float* d_result;
  cudaMalloc(&d_data, n * sizeof(float));
  cudaMalloc(&d_result, sizeof(float));

  cudaMemcpy(d_data, data, n * sizeof(float), cudaMemcpyHostToDevice);

  // Simple kernel to sum with FMA
  // For now, we'll use a simple CPU implementation
  // TODO: Implement proper CUDA kernel for this
  float sum = 0.0f;
  for (int i = 0; i < n; i++) {
    sum = fmaf(data[i], 1.0f, sum);
  }

  cudaFree(d_data);
  cudaFree(d_result);

  return sum;
}

PYBIND11_MODULE(fprev_cuda, m) {
  m.doc() = "FMA Sequence Investigation using CUDA and FPRev Algorithm";

  // Core FMA testing functions
  m.def(
      "fma_sequence_test", &fma_sequence_test_wrapper,
      "Test FMA accumulation for given mask positions i,j in sequence length n",
      py::arg("i"), py::arg("j"), py::arg("n"));

  m.def("fma_sequence_batch_test", &fma_sequence_batch_test_wrapper,
        "Batch test FMA accumulation for multiple (i,j) pairs",
        py::arg("i_indices"), py::arg("j_indices"), py::arg("n"));

  // GEMV and GEMM testing functions
  m.def("gemv_sequence_test", &gemv_sequence_test_wrapper,
        "Test GEMV accumulation for given mask positions i,j in sequence "
        "length n",
        py::arg("i"), py::arg("j"), py::arg("n"));

  // Helper functions for FPRev algorithm
  m.def("create_test_array", &create_test_array,
        "Create test array A where A[i] = M, A[j] = -M, others = 1.0",
        py::arg("i"), py::arg("j"), py::arg("n"));

  m.def("sumimpl_fma", &sumimpl_fma,
        "SUMIMPL function using FMA for FPRev algorithm",
        py::arg("input_array"));

  // CUDA management functions
  m.def("init", &fprev_init, "Initialize CUDA resources");
  m.def("cleanup", &fprev_cleanup, "Cleanup CUDA resources");
}
