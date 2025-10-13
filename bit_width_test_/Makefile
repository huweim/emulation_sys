# ===================================================================================
# Makefile for CUDA GEMM Emulation Project (v3)
# 支持为多个目标生成 SASS
# ===================================================================================

NVCC := nvcc
ARCH := sm_120
NVCCFLAGS := -O3 -std=c++17 -arch=$(ARCH)
LDFLAGS := -lcudart

# 新增：输出目录
BINDIR := bin

# 目标1: 2-way comparison
TARGET_2WAY := $(BINDIR)/gemm_comparison
SOURCE_2WAY := gemm_comparison.cu

# 目标2: 3-way comparison
TARGET_3WAY := $(BINDIR)/gemm_3way_comparison
SOURCE_3WAY := gemm_3way_comparison.cu

TARGET_BIT := $(BINDIR)/bit_width_test
SOURCE_BIT := bit_width_test.cu

TARGET_FMABIT := $(BINDIR)/fma_acc_bitwidth
SOURCE_FMABIT := fma_acc_bitwidth.cu

TARGET_TCBIT := $(BINDIR)/tc_acc_bitwidth
SOURCE_TCBIT := tc_acc_bitwidth.cu

# 默认规则
all: $(BINDIR) $(TARGET_2WAY) $(TARGET_3WAY) $(TARGET_BIT) $(TARGET_FMABIT) $(TARGET_TCBIT)

# 新增：创建 bin/
$(BINDIR):
	@mkdir -p $@

# 编译规则
$(TARGET_2WAY): $(SOURCE_2WAY)
	@echo "--- Compiling $< -> $@"
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(LDFLAGS)
	@echo "--- Finished building $@ ---"

$(TARGET_3WAY): $(SOURCE_3WAY)
	@echo "--- Compiling $< -> $@"
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(LDFLAGS)
	@echo "--- Finished building $@ ---"

$(TARGET_BIT): $(SOURCE_BIT)
	@echo "--- Compiling $< -> $@"
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(LDFLAGS)
	@echo "--- Finished building $@ ---"

$(TARGET_FMABIT): $(SOURCE_FMABIT)
	@echo "--- Compiling $< -> $@"
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(LDFLAGS)
	@echo "--- Finished building $@ ---"

$(TARGET_TCBIT): $(SOURCE_TCBIT)
	@echo "--- Compiling $< -> $@"
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(LDFLAGS)
	@echo "--- Finished building $@ ---"

# SASS 生成（输出到 bin/）
sass: $(BINDIR)/gemm_comparison.sass \
      $(BINDIR)/gemm_3way_comparison.sass \
      $(BINDIR)/bit_width_test.sass \
      $(BINDIR)/fma_acc_bitwidth.sass \
      $(BINDIR)/tc_acc_bitwidth.sass
	@echo "--- All SASS files generated ---"

# 从 bin/ 下的可执行生成 bin/ 下的 .sass
$(BINDIR)/%.sass: $(BINDIR)/%
	@echo "--- Generating SASS for $< ---"
	cuobjdump -sass $< > $@
	@echo "--- SASS code saved to $@ ---"

# 清理：整个 bin/
clean:
	@echo "--- Cleaning up built files ---"
	rm -rf $(BINDIR)

.PHONY: all clean sass
