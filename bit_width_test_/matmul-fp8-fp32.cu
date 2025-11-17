#include <cassert>
#include <chrono>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include <time.h>
#include <type_traits>
#include <vector>

#include <iomanip>  // std::setw / std::setprecision
#include <cmath>
#include <cstring> 

/// RUN: nvcc -arch=sm_90a -std=c++17 matmul-fp8-fp32.cu -o test && ./test

typedef __nv_fp8_e5m2 e5m2;

// 帮助把 __nv_cvt_float_to_fp8 返回的 storage 转成 e5m2
union fp8_e5m2_u {
  __nv_fp8_storage_t s;
  e5m2               t;
};

static constexpr int BLOCKM = 128;
static constexpr int BLOCKN = 128;
static constexpr int BLOCKK = 32;
static constexpr int WGMMAM = 64;
static constexpr int WGMMAN = 128;
static constexpr int WGMMAK = 32;
static_assert(BLOCKK == WGMMAK, "WGMMA K should be the same as BLOCK K.");
static constexpr int WARP_SIZE = 32;
static constexpr int WARP_GROUP_SIZE = 4;
static constexpr int WARP_GROUP_NUMBER = 1;
static constexpr int WARP_NUMBER = WARP_GROUP_NUMBER * WARP_GROUP_SIZE;

#define HOST_DEVICE __forceinline__ __host__ __device__
#define DEVICE __forceinline__ __device__

int STAGES = 1;
int ITERS = 20;

template <class T> struct sizeof_bits {
  static constexpr int value = sizeof(T) * 8;
};

template <class T> static constexpr int sizeof_bits_v = sizeof_bits<T>::value;

// ---------------------------
// GMMA descriptor
// ---------------------------
union GmmaDescriptor {

  HOST_DEVICE constexpr GmmaDescriptor() noexcept : desc_(0) {}
  HOST_DEVICE constexpr GmmaDescriptor(uint64_t desc) noexcept : desc_(desc) {}
  HOST_DEVICE constexpr GmmaDescriptor(GmmaDescriptor const &t) noexcept
      : desc_(t.desc_) {}
  HOST_DEVICE constexpr GmmaDescriptor(GmmaDescriptor &&t) noexcept
      : desc_(t.desc_) {}

  HOST_DEVICE constexpr GmmaDescriptor &
  operator=(GmmaDescriptor const &t) noexcept {
    desc_ = t.desc_;
    return *this;
  }

  HOST_DEVICE constexpr GmmaDescriptor &operator=(GmmaDescriptor &&t) noexcept {
    desc_ = t.desc_;
    return *this;
  }

  uint64_t desc_;
  uint32_t reg32_[2];
  uint16_t reg16_[4];

  struct {
    // start_address, bit [0,14), 4LSB not included
    uint16_t start_address_ : 14, : 2; 
    // leading dimension byte offset, bit [16,30), 4LSB not included
    uint16_t leading_byte_offset_ : 14, : 2; 
    // stride dimension byte offset, bit [32,46), 4LSB not included
    uint16_t stride_byte_offset_ : 14, : 2;
    // base_offset, bit [49,52)
    uint8_t : 1,
        base_offset_ : 3, : 4; 
    // layout type, bit [62,64)
    uint8_t : 6, layout_type_ : 2; 
  } bitfield;

  HOST_DEVICE constexpr operator uint64_t() const noexcept { return desc_; }
};

// ---------------------------
// MatmulKernel: FP8 × FP8 -> FP32 (accumulator / output)
// ---------------------------
struct MatmulKernel {
  using ADType = e5m2;
  using BDType = e5m2;
  using CDType = float;      // <--- 现在用 FP32 作为 accumulator / C_dtype
  using ACCType = CDType;    // accumulator 元素类型

  static_assert(std::is_same<ADType, BDType>::value);
  static constexpr int AElements = BLOCKM * BLOCKK;
  static constexpr int BElements = BLOCKN * BLOCKK;

  static constexpr int LD_SMEM_VEC_BYTES =
      16; /// shared memory vector load maximum size is 16 bytes (4 banks)
  static constexpr int LD_SMEM_VEC_ELEMS_A = LD_SMEM_VEC_BYTES / sizeof(ADType);
  static constexpr int LD_SMEM_VEC_ELEMS_B = LD_SMEM_VEC_BYTES / sizeof(BDType);

  /// fixed by WGMMA shared memory layout
  static constexpr int WGMMA_SMEM_BLOCK_ROW = 8;
  static constexpr int WGMMA_SMEM_BLOCK_COL = 128 / sizeof_bits_v<ADType>;
  static_assert(LD_SMEM_VEC_BYTES == WGMMA_SMEM_BLOCK_COL);

  /// fixed by WGMMA fragment layout: accumulator C/D
  static constexpr int WGMMA_FRGA_ELEMNTS_C =
      WGMMAM * WGMMAN / (WARP_NUMBER * WARP_SIZE);  // 64 对于 m64n128k32

  using WGMMA_ACCUM_ITEM_TYPE = ACCType; // float
  static constexpr int WGMMA_FRAG_ITEM_BYTES = sizeof(WGMMA_ACCUM_ITEM_TYPE);
  static constexpr int WGMMA_FRAG_ELEMENTS_PER_ITEM =
      WGMMA_FRAG_ITEM_BYTES / sizeof(CDType);  // =1

  static constexpr int WGMMA_LANE_ROW_ELEMENT_STRIDE = 8;
  static constexpr int WGMMA_LANE_COL_ELEMENT_STRIDE = 8;
  static constexpr int WGMMA_LANES_PER_ROW =
      WGMMA_LANE_ROW_ELEMENT_STRIDE / WGMMA_FRAG_ELEMENTS_PER_ITEM; /// 8
  static constexpr int WGMMA_LANES_PER_COL =
      WARP_SIZE / WGMMA_LANES_PER_ROW; /// 4

  static constexpr int WGMMA_WARPS_PER_ROW = 1;
  static constexpr int WGMMA_WARPS_PER_COL =
      WARP_GROUP_SIZE / WGMMA_WARPS_PER_ROW; /// 4
  static constexpr int WGMMA_WARP_REPEATS_PER_ROW =
      WGMMAN / WGMMA_WARPS_PER_ROW / WGMMA_LANE_ROW_ELEMENT_STRIDE; /// 128 / 8 = 16
  static constexpr int WGMMA_WARP_REPEATS_PER_COL =
      WGMMAM / WGMMA_WARPS_PER_COL / WGMMA_LANES_PER_COL; /// 64 / 4 / 4 = 4
  static constexpr int WGMMA_WARP_ROW_ELEMENT_STRIDE =
      WGMMAN / WGMMA_WARP_REPEATS_PER_ROW / WGMMA_WARPS_PER_ROW; /// 8
  static constexpr int WGMMA_WARP_COL_ELEMENT_STRIDE =
      WGMMAM / WGMMA_WARP_REPEATS_PER_COL / WGMMA_WARPS_PER_COL; /// 4

  /// swizzle decisions
  /// matrix A swizzle: row-major
  static constexpr int LoadWarpsPerRowA = 2;
  static constexpr int LoadWarpsPerColA = WARP_NUMBER / LoadWarpsPerRowA;
  static constexpr int LoadElementsPerRowPerWarpA = BLOCKK / LoadWarpsPerRowA;
  static constexpr int LoadThreadsPerRowPerWarpA =
      LoadElementsPerRowPerWarpA / LD_SMEM_VEC_ELEMS_A;
  static constexpr int LoadElementsPerColPerWarpA =
      WARP_SIZE / LoadThreadsPerRowPerWarpA;
  static constexpr int LoadThreadsPerColPerWarpA =
      WARP_SIZE / LoadThreadsPerRowPerWarpA;
  static constexpr int LoadRepeatsPerRowPerWarpA =
      BLOCKK / LoadWarpsPerRowA / LoadElementsPerRowPerWarpA;
  static constexpr int LoadRepeatsPerColPerWarpA =
      BLOCKM / LoadWarpsPerColA / LoadElementsPerColPerWarpA;

  /// matrix B swizzle: column-major
  static constexpr int LoadWarpsPerColB = 2;
  static constexpr int LoadWarpsPerRowB = WARP_NUMBER / LoadWarpsPerColB;
  static constexpr int LoadElementsPerColPerWarpB = BLOCKK / LoadWarpsPerColB;
  static constexpr int LoadThreadsPerColPerWarpB =
      LoadElementsPerColPerWarpB / LD_SMEM_VEC_ELEMS_B;
  static constexpr int LoadElementsPerRowPerWarpB =
      WARP_SIZE / LoadThreadsPerColPerWarpB;
  static constexpr int LoadThreadsPerRowPerWarpB =
      WARP_SIZE / LoadThreadsPerColPerWarpB;
  static constexpr int LoadRepeatsPerColPerWarpB =
      BLOCKK / LoadWarpsPerColB / LoadElementsPerColPerWarpB;
  static constexpr int LoadRepeatsPerRowPerWarpB =
      BLOCKN / LoadWarpsPerRowB / LoadElementsPerRowPerWarpB;

  /// compute swizzle
  static constexpr int ComputeWarpsPerCol = 2;
  static constexpr int ComputeWarpsPerRow = WARP_NUMBER / ComputeWarpsPerCol;

  /// matrix C swizzle: row-major
  /// warpgroup organization
  static constexpr int WarpGroupsPerRow = 1;
  static constexpr int WarpGroupsPerCol =
      WARP_GROUP_NUMBER / WarpGroupsPerRow; /// 1
  static constexpr int WarpGroupRepeatsPerRowC =
      BLOCKN / WarpGroupsPerRow / WGMMAN; /// 1
  static constexpr int WarpGroupRepeatsPerColC =
      BLOCKM / WarpGroupsPerCol / WGMMAM; /// 2

  /// constructor
  DEVICE MatmulKernel(int M, int N, int K, uint8_t *smem, CDType c_init) {
    GlobalM = M;
    GlobalN = N;
    GlobalK = K;
    smem_   = smem;

    SA_ = reinterpret_cast<e5m2 *>(smem_);
    SB_ = SA_ + AElements;

    accum_item_ = reinterpret_cast<WGMMA_ACCUM_ITEM_TYPE *>(accum_);

    // 初始化所有 accumulator 元素为 C_init（FP32），用于做 D = A*B + C_init
    int total_accum_elems =
        WarpGroupRepeatsPerRowC * WarpGroupRepeatsPerColC * WGMMA_FRGA_ELEMNTS_C;
#pragma unroll
    for (int i = 0; i < total_accum_elems; ++i) {
      accum_[i] = c_init;
    }
  }

  /// PTX wrapper: cp.async
  template <class PointerType>
  DEVICE void async_copy(void *smem, PointerType global) {
    uint32_t smem_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem));

    asm volatile(
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2, %3;\n" ::"r"(
            smem_ptr),
        "l"(global), "n"(LD_SMEM_VEC_BYTES), "r"(LD_SMEM_VEC_BYTES));
  }

  DEVICE void async_copy_fence() {
    asm volatile("cp.async.commit_group;\n" ::);
  }

  template <int Group> DEVICE void async_copy_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(Group));
  }

  /// load functions
  DEVICE void ldGlobalSharedA(ADType *globalA, int ko) {
    for (int i = 0; i < LoadRepeatsPerColPerWarpA; ++i) {
      int globalRowIdx = globalRowElementIdxA(i);
      int sharedRowIdx = sharedRowElementIdxA(i);
      for (int j = 0; j < LoadRepeatsPerRowPerWarpA; ++j) {
        int globalColIdx = globalColElementIdxA(ko, j);
        int sharedColIdx = sharedColElementIdxA(j);
        int globalElementIdx = globalElementIdxA(globalRowIdx, globalColIdx);
        int sharedElementIdx = sharedElementIdxA(sharedRowIdx, sharedColIdx);
        void *ptr = (void *)(SA_ + sharedElementIdx);
        async_copy(ptr, &globalA[globalElementIdx]);
      }
    }
  }

  DEVICE void ldGlobalSharedB(BDType *globalB, int ko) {
    for (int i = 0; i < LoadRepeatsPerRowPerWarpB; ++i) {
      int globalColIdx = globalColElementIdxB(i);
      int sharedColIdx = sharedColElementIdxB(i);
      for (int j = 0; j < LoadRepeatsPerColPerWarpB; ++j) {
        int globalRowIdx = globalRowElementIdxB(ko, j);
        int sharedRowIdx = sharedRowElementIdxB(j);
        int globalElementIdx = globalElementIdxB(globalRowIdx, globalColIdx);
        int sharedElementIdx = sharedElementIdxB(sharedRowIdx, sharedColIdx);
        void *ptr = (void *)(SB_ + sharedElementIdx);
        async_copy(ptr, &globalB[globalElementIdx]);
      }
    }
  }

  /// compute functions
  /// call wgmma.mma_async: m64n128k32.f32.e5m2.e5m2
  DEVICE void MMA(uint64_t const &desc_a, uint64_t const &desc_b,
                  float &d00, float &d01, float &d02, float &d03, float &d04,
                  float &d05, float &d06, float &d07, float &d08, float &d09,
                  float &d10, float &d11, float &d12, float &d13, float &d14,
                  float &d15, float &d16, float &d17, float &d18, float &d19,
                  float &d20, float &d21, float &d22, float &d23, float &d24,
                  float &d25, float &d26, float &d27, float &d28, float &d29,
                  float &d30, float &d31, float &d32, float &d33, float &d34,
                  float &d35, float &d36, float &d37, float &d38, float &d39,
                  float &d40, float &d41, float &d42, float &d43, float &d44,
                  float &d45, float &d46, float &d47, float &d48, float &d49,
                  float &d50, float &d51, float &d52, float &d53, float &d54,
                  float &d55, float &d56, float &d57, float &d58, float &d59,
                  float &d60, float &d61, float &d62, float &d63) {
    int scale_D = 1; // use D = A*B + D

    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %66, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n128k32.f32.e5m2.e5m2 "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
        " %8,  %9,  %10, %11, %12, %13, %14, %15, "
        " %16, %17, %18, %19, %20, %21, %22, %23, "
        " %24, %25, %26, %27, %28, %29, %30, %31, "
        " %32, %33, %34, %35, %36, %37, %38, %39, "
        " %40, %41, %42, %43, %44, %45, %46, %47, "
        " %48, %49, %50, %51, %52, %53, %54, %55, "
        " %56, %57, %58, %59, %60, %61, %62, %63}, "
        " %64,"
        " %65,"
        " p , 1, 1;\n"
        "}\n"
        : "+f"(d00), "+f"(d01), "+f"(d02), "+f"(d03), "+f"(d04), "+f"(d05),
          "+f"(d06), "+f"(d07), "+f"(d08), "+f"(d09), "+f"(d10), "+f"(d11),
          "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17),
          "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23),
          "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29),
          "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35),
          "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41),
          "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47),
          "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53),
          "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59),
          "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63)
        : "l"(desc_a), "l"(desc_b), "r"(int32_t(scale_D)));
  }

  /// make shared memory descriptor
  template <class PointerType>
  DEVICE GmmaDescriptor make_desc_a(PointerType smem_ptr) {
    GmmaDescriptor desc;
    uint32_t uint_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    desc.bitfield.start_address_ = uint_ptr >> 4;
    desc.bitfield.layout_type_ = 0;            /// no swizzle
    desc.bitfield.leading_byte_offset_ = 8;    /// 8 bytes
    desc.bitfield.stride_byte_offset_ = 16;    /// 16 bytes
    return desc;
  }

  template <class PointerType>
  DEVICE GmmaDescriptor make_desc_b(PointerType smem_ptr) {
    GmmaDescriptor desc;
    uint32_t uint_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    desc.bitfield.start_address_ = uint_ptr >> 4;
    desc.bitfield.layout_type_ = 0;            /// no swizzle
    desc.bitfield.leading_byte_offset_ = 8;    /// 8 bytes
    desc.bitfield.stride_byte_offset_ = 16;    /// 16 bytes
    return desc;
  }

  DEVICE
  void warpgroup_arrive() {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
  }

  DEVICE
  void warpgroup_commit_batch() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
  }

  template <int N> DEVICE void warpgroup_wait() {
    static_assert(N >= 0 && N <= 7, "WGMMA wait: N must be in range [0, 7]");
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
  }

  /// naive gemm without pipeline (single wgmma, no multi-stage pipeline)
  DEVICE void gemm_no_pipeline(ADType *globalA, BDType *globalB,
                               CDType *globalC) {
    for (int ko = 0; ko < GlobalK / BLOCKK; ++ko) {
      ldGlobalSharedA(globalA, ko);
      ldGlobalSharedB(globalB, ko);
      async_copy_fence();
      async_copy_wait<0>();
      __syncthreads();
      for (int i = 0; i < WarpGroupRepeatsPerColC; ++i) { // 2
        ADType *smemA = SA_ + i * WGMMAM * WGMMAK;
        for (int j = 0; j < WarpGroupRepeatsPerRowC; ++j) { // 1
          BDType *smemB = SB_ + j * WGMMAN * WGMMAK;
          GmmaDescriptor desc_a = make_desc_a(smemA);
          GmmaDescriptor desc_b = make_desc_b(smemB);
          WGMMA_ACCUM_ITEM_TYPE *curr_accum_item =
              accum_item_ + (i * WarpGroupRepeatsPerRowC + j) *
                                WGMMA_FRGA_ELEMNTS_C; // 每 tile 64 个 FP32

          warpgroup_arrive();
          MMA(desc_a, desc_b,
              curr_accum_item[0],  curr_accum_item[1],  curr_accum_item[2],
              curr_accum_item[3],  curr_accum_item[4],  curr_accum_item[5],
              curr_accum_item[6],  curr_accum_item[7],  curr_accum_item[8],
              curr_accum_item[9],  curr_accum_item[10], curr_accum_item[11],
              curr_accum_item[12], curr_accum_item[13], curr_accum_item[14],
              curr_accum_item[15], curr_accum_item[16], curr_accum_item[17],
              curr_accum_item[18], curr_accum_item[19], curr_accum_item[20],
              curr_accum_item[21], curr_accum_item[22], curr_accum_item[23],
              curr_accum_item[24], curr_accum_item[25], curr_accum_item[26],
              curr_accum_item[27], curr_accum_item[28], curr_accum_item[29],
              curr_accum_item[30], curr_accum_item[31], curr_accum_item[32],
              curr_accum_item[33], curr_accum_item[34], curr_accum_item[35],
              curr_accum_item[36], curr_accum_item[37], curr_accum_item[38],
              curr_accum_item[39], curr_accum_item[40], curr_accum_item[41],
              curr_accum_item[42], curr_accum_item[43], curr_accum_item[44],
              curr_accum_item[45], curr_accum_item[46], curr_accum_item[47],
              curr_accum_item[48], curr_accum_item[49], curr_accum_item[50],
              curr_accum_item[51], curr_accum_item[52], curr_accum_item[53],
              curr_accum_item[54], curr_accum_item[55], curr_accum_item[56],
              curr_accum_item[57], curr_accum_item[58], curr_accum_item[59],
              curr_accum_item[60], curr_accum_item[61], curr_accum_item[62],
              curr_accum_item[63]);
          warpgroup_commit_batch();
          warpgroup_wait<0>();
        }
      }
    }
    stAccumGlobalC(globalC);
  }

  /// store functions
  /// 对于 bit-width probe，只需要一个槽就够了：把第一个 accumulator 写到 C[0]
  DEVICE void stAccumGlobalC(CDType *globalC) {
    if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) {
      globalC[0] = accum_[0];
    }
  }

  /// (保留原 naive_gemm，虽然目前没用)
  DEVICE void naive_gemm(ADType *globalA, BDType *globalB, CDType *globalC) {
    ADType fragA[8 * 8];
    BDType fragB[16 * 8];
    CDType accum_local[8 * 16] = {CDType(0.0f)};

    int sharedRowIdx = warpId() / 2 * 64 + lane() / 4 * 8;
    int sharedColIdx = warpId() % 2 * 64 + lane() % 4 * 16;
    for (int ko = 0; ko < GlobalK / BLOCKK; ++ko) {
      ldGlobalSharedA(globalA, ko);
      ldGlobalSharedB(globalB, ko);
      async_copy_fence();
      async_copy_wait<0>();
      __syncthreads();
      for (int ki = 0; ki < BLOCKK / 8; ++ki) {
        for (int i = 0; i < 8; ++i) {
          for (int kii = 0; kii < 8; ++kii) {
            int SA_row = sharedRowIdx + i;
            int SA_col = ki * 8 + kii;
            fragA[i * 8 + kii] = SA_[sharedElementIdxA(SA_row, SA_col)];
          }
        }
        for (int j = 0; j < 16; ++j) {
          for (int kii = 0; kii < 8; ++kii) {
            int SB_row = ki * 8 + kii;
            int SB_col = sharedColIdx + j;
            fragB[j * 8 + kii] = SB_[sharedElementIdxB(SB_row, SB_col)];
          }
        }
        for (int kii = 0; kii < 8; ++kii) {
          for (int i = 0; i < 8; ++i) {
            for (int j = 0; j < 16; ++j) {
              accum_local[i * 16 + j] +=
                  float(fragA[i * 8 + kii]) * float(fragB[j * 8 + kii]);
            }
          }
        }
      }
    }
    int globalRowIdx = blockY() * BLOCKM + sharedRowIdx;
    int globalColIdx = blockX() * BLOCKN + sharedColIdx;
    for (int i = 0; i < 8; ++i) {
      for (int j = 0; j < 16; ++j) {
        globalC[(globalRowIdx + i) * GlobalN + (globalColIdx + j)] =
            accum_local[i * 16 + j];
      }
    }
  }

  DEVICE void run(ADType *globalA, BDType *globalB, CDType *globalC) {
    // naive_gemm(globalA, globalB, globalC);
    gemm_no_pipeline(globalA, globalB, globalC);
  }

  /// index related calculations
  DEVICE int blockX() { return blockIdx.x; }
  DEVICE int blockY() { return blockIdx.y; }
  DEVICE int warpId() { return threadIdx.x / WARP_SIZE; }
  DEVICE int warpGroupId() { return warpId() / WARP_GROUP_SIZE; }
  DEVICE int lane() { return threadIdx.x % WARP_SIZE; }
  DEVICE int warpIdInGroup() { return warpId() % WARP_GROUP_SIZE; }

  /// load A from global to shared swizzle
  /// global: row-major A load
  DEVICE int globalBlockRowElementIdxA() { return blockY() * BLOCKM; }
  DEVICE int globalBlockColElementIdxA(int ko) { return ko * BLOCKK; }
  DEVICE int globalWarpRowElementIdxA() {
    return warpId() / LoadWarpsPerRowA * LoadElementsPerColPerWarpA;
  }
  DEVICE int globalWarpColElementIdxA() {
    return warpId() % LoadWarpsPerRowA * LoadElementsPerRowPerWarpA;
  }
  DEVICE int globalLaneRowElementIdxA() {
    return lane() / LoadThreadsPerRowPerWarpA;
  }
  DEVICE int globalLaneColElementIdxA() {
    return lane() % LoadThreadsPerRowPerWarpA * LD_SMEM_VEC_ELEMS_A;
  }
  DEVICE int globalRowElementIdxA(int warpRepeatRow) {
    return globalBlockRowElementIdxA() + globalWarpRowElementIdxA() +
           warpRepeatRow * LoadElementsPerColPerWarpA * LoadWarpsPerColA +
           globalLaneRowElementIdxA();
  }
  DEVICE int globalColElementIdxA(int ko, int warpRepeatCol) {
    return globalBlockColElementIdxA(ko) + globalWarpColElementIdxA() +
           warpRepeatCol * LoadElementsPerRowPerWarpA * LoadWarpsPerRowA +
           globalLaneColElementIdxA();
  }
  DEVICE int globalElementIdxA(int globalRowIdx, int globalColIdx) {
    return globalRowIdx * GlobalK + globalColIdx;
  }

  /// shared: row-major A store
  DEVICE int sharedRowElementIdxA(int warpRepeatRow) {
    return globalWarpRowElementIdxA() +
           warpRepeatRow * LoadElementsPerColPerWarpA * LoadWarpsPerColA +
           globalLaneRowElementIdxA();
  }
  DEVICE int sharedColElementIdxA(int warpRepeatCol) {
    return globalWarpColElementIdxA() +
           warpRepeatCol * LoadElementsPerRowPerWarpA * LoadWarpsPerRowA +
           globalLaneColElementIdxA();
  }

  /// WGMMA layout for A
  DEVICE int sharedElementIdxA(int sharedRowIdx, int sharedColIdx) {
    return sharedRowIdx / WGMMA_SMEM_BLOCK_ROW * WGMMA_SMEM_BLOCK_ROW * BLOCKK +
           sharedColIdx / WGMMA_SMEM_BLOCK_COL * WGMMA_SMEM_BLOCK_ROW *
               WGMMA_SMEM_BLOCK_COL +
           sharedRowIdx % WGMMA_SMEM_BLOCK_ROW * WGMMA_SMEM_BLOCK_COL +
           sharedColIdx % WGMMA_SMEM_BLOCK_COL;
  }

  /// load B from global to shared swizzle
  /// global: col-major B load
  DEVICE int globalBlockColElementIdxB() { return blockX() * BLOCKN; }
  DEVICE int globalBlockRowElementIdxB(int ko) { return ko * BLOCKK; }
  DEVICE int globalWarpColElementIdxB() {
    return warpId() / LoadWarpsPerColB * LoadElementsPerRowPerWarpB;
  }
  DEVICE int globalWarpRowElementIdxB() {
    return warpId() % LoadWarpsPerColB * LoadElementsPerColPerWarpB;
  }
  DEVICE int globalLaneColElementIdxB() {
    return lane() / LoadThreadsPerColPerWarpB;
  }
  DEVICE int globalLaneRowElementIdxB() {
    return lane() % LoadThreadsPerColPerWarpB * LD_SMEM_VEC_ELEMS_B;
  }
  DEVICE int globalColElementIdxB(int warpRepeatCol) {
    return globalBlockColElementIdxB() + globalWarpColElementIdxB() +
           warpRepeatCol * LoadElementsPerRowPerWarpB * LoadWarpsPerRowB +
           globalLaneColElementIdxB();
  }
  DEVICE int globalRowElementIdxB(int ko, int warpRepeatRow) {
    return globalBlockRowElementIdxB(ko) + globalWarpRowElementIdxB() +
           warpRepeatRow * LoadElementsPerColPerWarpB * LoadWarpsPerColB +
           globalLaneRowElementIdxB();
  }
  DEVICE int globalElementIdxB(int globalRowIdx, int globalColIdx) {
    return globalRowIdx + globalColIdx * GlobalK;
  }

  /// shared: col-major B store
  DEVICE int sharedColElementIdxB(int warpRepeatCol) {
    return globalWarpColElementIdxB() +
           warpRepeatCol * LoadElementsPerRowPerWarpB * LoadWarpsPerRowB +
           globalLaneColElementIdxB();
  }
  DEVICE int sharedRowElementIdxB(int warpRepeatRow) {
    return globalWarpRowElementIdxB() +
           warpRepeatRow * LoadElementsPerColPerWarpB * LoadWarpsPerColB +
           globalLaneRowElementIdxB();
  }

  /// WGMMA layout for B
  DEVICE int sharedElementIdxB(int sharedRowIdx, int sharedColIdx) {
    return sharedRowIdx % WGMMA_SMEM_BLOCK_COL +
           sharedColIdx % WGMMA_SMEM_BLOCK_ROW * WGMMA_SMEM_BLOCK_COL +
           sharedRowIdx / WGMMA_SMEM_BLOCK_COL * WGMMA_SMEM_BLOCK_ROW *
               WGMMA_SMEM_BLOCK_COL +
           sharedColIdx / WGMMA_SMEM_BLOCK_ROW * WGMMA_SMEM_BLOCK_ROW * BLOCKK;
  }

  /// global C indexing helpers (目前 stAccumGlobalC 没用到，但保留)
  DEVICE int globalBlockRowElementIdxC() { return blockY() * BLOCKM; }
  DEVICE int globalBlockColElementIdxC() { return blockX() * BLOCKN; }
  DEVICE int globalWarpGroupRowElementIdxC(int warpGroupRepeatRow) {
    return warpGroupId() / WarpGroupsPerRow * WarpGroupRepeatsPerColC * WGMMAM +
           warpGroupRepeatRow * WGMMAM;
  }
  DEVICE int globalWarpGroupColElementIdxC(int warpGroupRepeatCol) {
    return warpGroupId() % WarpGroupsPerRow * WarpGroupRepeatsPerRowC * WGMMAN +
           warpGroupRepeatCol * WGMMAN;
  }
  DEVICE int globalWarpRowElementIdxC(int warpRepeatRow) {
    return warpIdInGroup() / WGMMA_WARPS_PER_ROW * WGMMA_WARP_REPEATS_PER_COL *
               WGMMA_WARP_COL_ELEMENT_STRIDE +
           warpRepeatRow * WGMMA_WARP_COL_ELEMENT_STRIDE;
  }
  DEVICE int globalWarpColElementIdxC(int warpRepeatCol) {
    return warpIdInGroup() % WGMMA_WARPS_PER_ROW * WGMMA_WARP_REPEATS_PER_ROW *
               WGMMA_WARP_ROW_ELEMENT_STRIDE +
           warpRepeatCol * WGMMA_WARP_ROW_ELEMENT_STRIDE;
  }
  DEVICE int globalLaneRowElementIdxC() { return lane() / WGMMA_LANES_PER_ROW; }
  DEVICE int globalLaneColElementIdxC(int i) {
    return lane() % WGMMA_LANES_PER_ROW * WGMMA_FRAG_ELEMENTS_PER_ITEM + i;
  }
  DEVICE int globalRowElementIdxC(int warpGroupRepeatRow, int warpRepeatRow) {
    return globalBlockRowElementIdxC() +
           globalWarpGroupRowElementIdxC(warpGroupRepeatRow) +
           globalWarpRowElementIdxC(warpRepeatRow) + globalLaneRowElementIdxC();
  }
  DEVICE int globalColElementIdxC(int warpGroupRepeatCol, int warpRepeatCol,
                                  int i) {
    return globalBlockColElementIdxC() +
           globalWarpGroupColElementIdxC(warpGroupRepeatCol) +
           globalWarpColElementIdxC(warpRepeatCol) +
           globalLaneColElementIdxC(i);
  }
  DEVICE int globalElementIdxC(int globalRowIdx, int globalColIdx) {
    return globalRowIdx * GlobalN + globalColIdx;
  }

private:
  uint8_t *smem_;
  ADType *SA_;
  BDType *SB_;
  int GlobalM;
  int GlobalN;
  int GlobalK;
  CDType accum_[WarpGroupRepeatsPerRowC * WarpGroupRepeatsPerColC *
                WGMMA_FRGA_ELEMNTS_C] = {CDType(0.0f)}; /// 1 * 2 * 64 = 128
  WGMMA_ACCUM_ITEM_TYPE *accum_item_;
};

// ---------------------------
// Kernel
// ---------------------------
__global__ void matmul_fp8(e5m2 *A, e5m2 *B, float *C,
                           int M, int N, int K,
                           float c_init) {
  extern __shared__ uint8_t smem[];
  MatmulKernel kernel(M, N, K, smem, c_init);
  kernel.run(A, B, C);
}

#define MAX(a, b) (a) > (b) ? (a) : (b)

/**
 * Panic wrapper for unwinding CUDA runtime errors
 */
#define CUDA_CHECK(status)                                                     \
  {                                                                            \
    cudaError_t error = status;                                                \
    if (error != cudaSuccess) {                                                \
      std::cerr << "Got bad cuda status: " << cudaGetErrorString(error)        \
                << " at line: " << __LINE__ << std::endl;                      \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

// ===================================================================================
// Host: FP32 accumulator bit-width probe (C_init + 1)
// ===================================================================================
int main(int argc, char *argv[]) {
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  std::cout << "GPU: " << prop.name << " (sm_" << prop.major << prop.minor
            << ")\n";
  if (prop.major != 9) {
    std::cerr << "This program targets Hopper (sm_90) WGMMA.\n";
    return 1;
  }

  // --------------------
  // 配置：探测 FP32 累加器的“有效位宽”
  // --------------------
  
  const int PROBE_START_BIT = 12; // suggest use 12 or 13
  const int PROBE_END_BIT   = 20;

  // C_init = 2^E（精确编码成 FP32）
  const int E = PROBE_START_BIT - 1;
  
  uint32_t c_init_bin = static_cast<uint32_t>((127 + E) << 23);
  float c_init = *reinterpret_cast<float*>(&c_init_bin);

  // union {
  //   uint32_t u;
  //   float    f;
  // } c_init_u;
  // c_init_u.u = static_cast<uint32_t>((127 + E) << 23); // sign=0, exp=127+E, mantissa=0
  // float c_init = c_init_u.f;

  double ideal_pass = static_cast<double>(c_init) + 1.0;
  double ideal_fail = static_cast<double>(c_init);     // 没用上，先留着

  const int M = BLOCKM;  // 128
  const int N = BLOCKN;  // 128
  const int K = BLOCKK;  // 32

  std::cout << "Using Shape: m" << WGMMAM << "n" << WGMMAN
            << "k" << WGMMAK << " within a block "
            << M << "x" << N << "x" << K << " (one warpgroup)\n";
  std::cout << "C_init (FP32) = " << std::fixed << std::setprecision(8)
            << c_init << " (= 2^" << E << " ideally)\n";
  std::cout << "---------------------------------------------------------------------------------------------------------------\n";
  std::cout << std::left << std::setw(10) << "Bit"
            << "| " << std::setw(14) << "add1_fp8"
            << "| " << std::setw(14) << "add2_fp8"
            << "| " << std::setw(16) << "ideal_pass"
            << "| " << std::setw(16) << "TC_result"
            << "| " << std::setw(12) << "delta"
            << "| " << "Analysis\n";
  std::cout << "---------------------------------------------------------------------------------------------------------------\n";

  // --------------------
  // Host / Device buffer 分配
  // --------------------
  e5m2 *hA = nullptr;
  e5m2 *hB = nullptr; // B 直接用列主序 (KxN)
  float *hC = nullptr;

  CUDA_CHECK(cudaMallocHost(&hA, M * K * sizeof(e5m2)));
  CUDA_CHECK(cudaMallocHost(&hB, K * N * sizeof(e5m2)));
  CUDA_CHECK(cudaMallocHost(&hC, M * N * sizeof(float)));

  e5m2 *dA = nullptr;
  e5m2 *dB = nullptr;
  float *dC = nullptr;

  CUDA_CHECK(cudaMalloc(&dA, M * K * sizeof(e5m2)));
  CUDA_CHECK(cudaMalloc(&dB, K * N * sizeof(e5m2)));
  CUDA_CHECK(cudaMalloc(&dC, M * N * sizeof(float)));

  // 预先构造 B 中的 “1.0” (FP8 e5m2)
  fp8_e5m2_u u_one;
  u_one.s = __nv_cvt_float_to_fp8(1.0f, __NV_SATFINITE, __NV_E5M2);

  // 线程组织：一个 CTA=一个 warpgroup，覆盖一个 128x128 block
  dim3 dimBlock(WARP_SIZE * WARP_NUMBER, 1, 1); // 128
  dim3 dimGrid(1, 1, 1);

  // 动态 shared memory：A(128x32) + B(128x32)
  int smem_size =
      (BLOCKM + BLOCKN) * BLOCKK * sizeof(e5m2); 
  if (smem_size >= (48 << 10)) {
    CUDA_CHECK(cudaFuncSetAttribute(
        matmul_fp8, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
  }

  // --------------------
  // 探测循环：bit = PROBE_START_BIT..PROBE_END_BIT
  // --------------------
  for (int bit = PROBE_START_BIT; bit <= PROBE_END_BIT; ++bit) {
    double add2 = std::pow(2.0, -(bit - E));  // 2^{-(bit-E)}
    double add1 = 1.0 - add2;                 // add1+add2 = 1

    // 清零 host buffer
    std::memset(hA, 0, M * K * sizeof(e5m2));
    std::memset(hB, 0, K * N * sizeof(e5m2));
    std::memset(hC, 0, M * N * sizeof(float));

    // 把 add1/add2 转成 FP8 e5m2
    fp8_e5m2_u ua1, ua2;
    ua1.s = __nv_cvt_float_to_fp8(static_cast<float>(add1),
                                  __NV_SATFINITE, __NV_E5M2);
    ua2.s = __nv_cvt_float_to_fp8(static_cast<float>(add2),
                                  __NV_SATFINITE, __NV_E5M2);

    // 量化后的 add1/add2（真正喂进 Tensor Core 的值）
    float add1_fp8 = static_cast<float>(ua1.t);
    float add2_fp8 = static_cast<float>(ua2.t);

    // ---- 填 A: 每行只有 k=0,1 非零 ----
    // A 是 [M,K] row-major，host index: A[r*K + k]
    for (int r = 0; r < M; ++r) {
      hA[r * K + 0] = ua1.t;  // A[r,0] = add1
      hA[r * K + 1] = ua2.t;  // A[r,1] = add2
    }

    // ---- 填 B: B[k,c]，列主序 [K,N]，host index: B[c*K + k] ----
    // 对所有列 c，令 B[0,c]=1, B[1,c]=1，其余行为 0
    for (int c = 0; c < N; ++c) {
      hB[c * K + 0] = u_one.t;  // k=0 行
      hB[c * K + 1] = u_one.t;  // k=1 行
    }

    CUDA_CHECK(cudaMemcpy(dA, hA, M * K * sizeof(e5m2),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, K * N * sizeof(e5m2),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dC, hC, M * N * sizeof(float),
                          cudaMemcpyHostToDevice));

    // Launch kernel
    matmul_fp8<<<dimGrid, dimBlock, smem_size>>>(dA, dB, dC, M, N, K, c_init);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(hC, dC, M * N * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // 我们在 kernel 里只写了 C[0] = 第一个 accumulator
    float tc_result = hC[0];

    double delta  = static_cast<double>(tc_result) - ideal_pass;
    std::string analysis =
        (tc_result == static_cast<float>(ideal_pass))
            ? "PASS"
            : "FAIL (Flipped!)";

    std::cout << std::left << std::setw(10) << bit
              << "| " << std::scientific << std::setprecision(6)
              << std::setw(14) << add1_fp8
              << "| " << std::scientific << std::setprecision(6)
              << std::setw(14) << add2_fp8
              << "| " << std::fixed << std::setprecision(8)
              << std::setw(16) << ideal_pass
              << "| " << std::fixed << std::setprecision(8)
              << std::setw(16) << tc_result
              << "| " << std::scientific << std::setprecision(3)
              << std::setw(12) << delta
              << "| " << analysis << "\n";

    if (analysis.find("FAIL") != std::string::npos) {
      int eff = bit - 1;
      std::cout << "\nFlip detected at bit " << bit
                << ". Effective FP32 accumulator mantissa width (upper bound) ~ "
                << eff << " bits.\n";
      break;
    }
  }

  // 清理内存
  CUDA_CHECK(cudaFreeHost(hA));
  CUDA_CHECK(cudaFreeHost(hB));
  CUDA_CHECK(cudaFreeHost(hC));
  CUDA_CHECK(cudaFree(dA));
  CUDA_CHECK(cudaFree(dB));
  CUDA_CHECK(cudaFree(dC));

  return 0;
}