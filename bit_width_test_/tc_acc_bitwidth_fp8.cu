#include <iostream>
#include <iomanip>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp8.h>

#define CUDA_CHECK(call) do { \
  cudaError_t err = call; \
  if (err != cudaSuccess) { \
    fprintf(stderr, "CUDA Error: %s in %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
    exit(EXIT_FAILURE); \
  } \
} while (0)

__global__ void tensor_core_mma_16x8x32_mixed_fp8() {
    // 准备 A(4×.b32) 与 B(2×.b32)。这两个寄存器里按“每 .b32 打包 4 个 fp8”理解。
    // 这里我们放全 0，等价于 A*B = 0
    uint32_t a[4] = {0u, 0u, 0u, 0u}; // A -> e4m3（概念上）
    uint32_t b[2] = {0u, 0u};         // B -> e5m2（概念上）

    // 准备 C、D（4×.f32）
    uint32_t bits = 0b01000000010000000111001111111111; // 只是一个非零浮点做验证
    float c0 = *reinterpret_cast<float*>(&bits);
    float c1 = c0, c2 = c0, c3 = c0;

    float d0, d1, d2, d3;

    // 关键：混合 FP8 变体 + 向量个数 D{4}, A{4}, B{2}, C{4}
    asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
      "{%0,%1,%2,%3}, "
      "{%4,%5,%6,%7}, "
      "{%8,%9}, "
      "{%10,%11,%12,%13};\n"
      : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)                // D 输出（4×f32）
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),           // A 输入（4×.b32）
        "r"(b[0]), "r"(b[1]),                                 // B 输入（2×.b32）
        "f"(c0), "f"(c1), "f"(c2), "f"(c3)                    // C 输入（4×f32）
    );

    if (threadIdx.x == 0) {
        printf("\n--- PTX MMA Result (A*B=0, mixed FP8 e4m3×e5m2) ---\n");
        printf("C (Input):  %f, %f, %f, %f\n", c0, c1, c2, c3);
        printf("D (Output): %f, %f, %f, %f\n", d0, d1, d2, d3);
        if (d0 == c0 && d1 == c1 && d2 == c2 && d3 == c3) {
            printf("Result: D == C. Accumulation OK.\n");
        } else {
            printf("Result: D != C. Check operand packing/types.\n");
        }
    }
}

int main() {
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    int sm = prop.major * 10 + prop.minor;
    if (sm < 89) {
        std::cerr << "Need Ada (sm_89+) for this FP8 MMA instruction. Found sm_"
                  << prop.major << prop.minor << "\n";
        return EXIT_FAILURE;
    }
    std::cout << "GPU: " << prop.name << " (sm_" << prop.major << prop.minor << ")\n";
    std::cout << "Launching minimal mixed-FP8 PTX kernel...\n";

    tensor_core_mma_16x8x32_mixed_fp8<<<1, 32>>>();
    CUDA_CHECK(cudaDeviceSynchronize());

    std::cout << "Kernel launch complete.\n";
    return 0;
}
