# ===================================================================================
# Makefile for CUDA GEMM Emulation Project (Final)
# ===================================================================================

# Compiler & Flags
NVCC := nvcc
# Default architecture for older tests (e.g., RTX 4090 / A100)
DEFAULT_ARCH := sm_80
# DEFAULT_ARCH := sm_89
# DEFAULT_ARCH := sm_90a
# DEFAULT_ARCH := sm_120
# Specific architecture for H100/Hopper tests (WGMMA requires sm_90a)
H100_ARCH := sm_90a

# Common Flags
NVCCFLAGS := -O3 -std=c++17
LDFLAGS   := -lcudart

# Path settings (adjust if needed)
CUTLASS_DIR := /cephfs/wmhu/emulation_workspace/cutlass
INCLUDES    := -I$(CUTLASS_DIR)/include

# Output Directory
BINDIR := bin

# --- Targets Definition ---

# 1. Precision Comparison (TC vs FMA)
TARGET_2WAY := $(BINDIR)/gemm_comparison
SOURCE_2WAY := gemm_comparison.cu

# 2. 3-Way Comparison (TC vs Seq FMA vs Tree FMA)
TARGET_3WAY := $(BINDIR)/gemm_3way_comparison
SOURCE_3WAY := gemm_3way_comparison.cu

# 3. Basic Bit Width Tests (Legacy)
TARGET_BIT      := $(BINDIR)/bit_width_test
SOURCE_BIT      := bit_width_test.cu

TARGET_FMABIT   := $(BINDIR)/probe_fma_fp32
SOURCE_FMABIT   := probe_fma_fp32.cu

TARGET_TCBIT    := $(BINDIR)/probe_mma_fp16_acc_fp32
SOURCE_TCBIT    := probe_mma_fp16_acc_fp32.cu

# 4. H100 FP8 Probing (The Core Result)
# Renamed to reflect its purpose: probing FP8 input -> FP32/FP16 accumulation
TARGET_H100_FP8 := $(BINDIR)/probe_mma_sm90a_fp8_acc_fp32
SOURCE_H100_FP8 := probe_mma_sm90a_fp8_acc_fp32.cu

TARGET_H100_FP8_FP16ACC := $(BINDIR)/probe_mma_sm90a_fp8_acc_fp16
SOURCE_H100_FP8_FP16ACC := probe_mma_sm90a_fp8_acc_fp16.cu


# --- Rules ---

# Default: Build everything
# all: $(BINDIR) $(TARGET_2WAY) $(TARGET_3WAY) $(TARGET_H100_FP8) $(TARGET_FMABIT) $(TARGET_TCBIT)
all: $(BINDIR) $(TARGET_TCBIT)

# Create bin directory
$(BINDIR):
	@mkdir -p $@

# --- Compilation Rules ---

# Standard targets (using DEFAULT_ARCH)
$(TARGET_2WAY): $(SOURCE_2WAY)
	@echo "--- Compiling $< [$(DEFAULT_ARCH)] ---"
	$(NVCC) $(NVCCFLAGS) -arch=$(DEFAULT_ARCH) $(INCLUDES) -o $@ $< $(LDFLAGS)

$(TARGET_3WAY): $(SOURCE_3WAY)
	@echo "--- Compiling $< [$(DEFAULT_ARCH)] ---"
	$(NVCC) $(NVCCFLAGS) -arch=$(DEFAULT_ARCH) $(INCLUDES) -o $@ $< $(LDFLAGS)

$(TARGET_BIT): $(SOURCE_BIT)
	@echo "--- Compiling $< [$(DEFAULT_ARCH)] ---"
	$(NVCC) $(NVCCFLAGS) -arch=$(DEFAULT_ARCH) $(INCLUDES) -o $@ $< $(LDFLAGS)

$(TARGET_FMABIT): $(SOURCE_FMABIT)
	@echo "--- Compiling $< [$(DEFAULT_ARCH)] ---"
	$(NVCC) $(NVCCFLAGS) -arch=$(DEFAULT_ARCH) $(INCLUDES) -o $@ $< $(LDFLAGS)

$(TARGET_TCBIT): $(SOURCE_TCBIT)
	@echo "--- Compiling $< [$(DEFAULT_ARCH)] ---"
	$(NVCC) $(NVCCFLAGS) -arch=$(DEFAULT_ARCH) $(INCLUDES) -o $@ $< $(LDFLAGS)

# H100 Specific targets (FORCE sm_90a)
$(TARGET_H100_FP8): $(SOURCE_H100_FP8)
	@echo "--- Compiling $< [$(H100_ARCH)] ---"
	$(NVCC) $(NVCCFLAGS) -arch=$(H100_ARCH) $(INCLUDES) -o $@ $< $(LDFLAGS)

$(TARGET_H100_FP8_FP16ACC): $(SOURCE_H100_FP8_FP16ACC)
	@echo "--- Compiling $< [$(H100_ARCH)] ---"
	$(NVCC) $(NVCCFLAGS) -arch=$(H100_ARCH) $(INCLUDES) -o $@ $< $(LDFLAGS)

# --- Utilities ---

# Generate SASS for analysis
sass: $(TARGET_3WAY) $(TARGET_H100_FP8)
	@echo "--- Generating SASS ---"
	cuobjdump -sass $(TARGET_3WAY) > $(TARGET_3WAY).sass
	cuobjdump -sass $(TARGET_H100_FP8) > $(TARGET_H100_FP8).sass
	@echo "--- SASS files generated in $(BINDIR) ---"

clean:
	@echo "--- Cleaning up ---"
	rm -rf $(BINDIR)

.PHONY: all clean sass