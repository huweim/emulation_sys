/// RUN: make sm80     # A100 (sm_80): build universal probes
/// RUN: make sm89     # RTX 4090 (sm_89): build universal probes
/// RUN: make sm90     # H100 (sm_90 + sm_90a): universal + WGMMA probes
/// RUN: make sm120    # RTX 5090 (sm_120): build universal probes
/// RUN: make clean    # remove bin/

# ===================================================================================
# Makefile for Tensor Core Bit-Width Probing
# ===================================================================================

NVCC      := nvcc
BINDIR    := bin
SRCDIR    := src
NVCCFLAGS := -O3 -std=c++17
LDFLAGS   := -lcudart

# CUTLASS path (Adjust as needed)
CUTLASS_DIR := /cephfs/wmhu/emulation_workspace/cutlass
INCLUDES    := -I$(CUTLASS_DIR)/include -I./include

# --- Source Files Definitions ---

# 1) Universal Probes (SM80+): WMMA / MMA，不依赖 WGMMA
SRC_UNIVERSAL := \
    $(SRCDIR)/probe_fma_fp32.cu \
    $(SRCDIR)/probe_wmma_fp16_acc_fp32.cu \
    $(SRCDIR)/probe_mma_fp16_acc_fp16.cu

# 2) Hopper-specific Probes (SM90a): WGMMA inline PTX
#    使用你目录中的实际文件名（probe_wgmma_sm90a_*）
SRC_HOPPER := \
    $(SRCDIR)/probe_wgmma_sm90a_fp8_acc_fp32.cu \
    $(SRCDIR)/probe_wgmma_sm90a_fp8_acc_fp16.cu

# --- Helper: map src -> bin ---
GET_BIN_NAME = $(patsubst $(SRCDIR)/%.cu,$(BINDIR)/%,$(1))

# --- Rules ---

.PHONY: all clean sm80 a100 sm89 4090 sm90 h100 sm120 5090 hopper_specific

# 默认：先编 Ada (sm_89)
all: sm89

$(BINDIR):
	@mkdir -p $@

# =============== Architecture Targets ===============

# A100 (Ampere)
sm80 a100: ARCH_FLAG := -arch=sm_80
sm80 a100: $(BINDIR) $(call GET_BIN_NAME,$(SRC_UNIVERSAL))
	@echo ">>> Built Universal Probes for A100 (sm_80)"

# RTX 4090 (Ada)
sm89 4090: ARCH_FLAG := -arch=sm_89
sm89 4090: $(BINDIR) $(call GET_BIN_NAME,$(SRC_UNIVERSAL))
	@echo ">>> Built Universal Probes for RTX 4090 (sm_89)"

# H100 (Hopper): Universal + Hopper-specific(WGMMA, sm_90a)
sm90 h100: ARCH_FLAG := -arch=sm_90
sm90 h100: $(BINDIR) $(call GET_BIN_NAME,$(SRC_UNIVERSAL)) hopper_specific
	@echo ">>> Built All Probes for H100 (sm_90 + sm_90a WGMMA)"

# RTX 5090 (Blackwell Consumer, SM120): 仅编 Universal（mma.sync 路线）
sm120 5090: ARCH_FLAG := -arch=sm_120
sm120 5090: $(BINDIR) $(call GET_BIN_NAME,$(SRC_UNIVERSAL))
	@echo ">>> Built Universal Probes for RTX 5090 (sm_120)"

# =============== Compile Recipes ===============

# 先放 Hopper-specific（更专用的模式优先匹配）
$(BINDIR)/probe_wgmma_sm90a_%: $(SRCDIR)/probe_wgmma_sm90a_%.cu
	@echo "--- Compiling Hopper Probe $< [sm_90a] ---"
	$(NVCC) $(NVCCFLAGS) -arch=sm_90a $(INCLUDES) -o $@ $< $(LDFLAGS)

# Universal sources（跟随各自 ARCH_FLAG）
$(BINDIR)/%: $(SRCDIR)/%.cu
	@echo "--- Compiling $< [$(ARCH_FLAG)] ---"
	$(NVCC) $(NVCCFLAGS) $(ARCH_FLAG) $(INCLUDES) -o $@ $< $(LDFLAGS)

# 聚合构建：所有 Hopper 专用探针
hopper_specific: $(call GET_BIN_NAME,$(SRC_HOPPER))

clean:
	rm -rf $(BINDIR)