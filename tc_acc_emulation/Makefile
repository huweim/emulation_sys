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

TARGET_TCACCEMU_THREE_ADD := $(BINDIR)/tc_acc_emulation_double_three_add
SOURCE_TCACCEMU_THREE_ADD := tc_acc_emulation_double_three_add.cu

TARGET_TCACCEMU_MULTI_ADD := $(BINDIR)/tc_acc_emulation_double_multi_add
SOURCE_TCACCEMU_MULTI_ADD := tc_acc_emulation_double_multi_add.cu

all: $(BINDIR) $(TARGET_TCACCEMU_MULTI_ADD) $(TARGET_TCACCEMU_THREE_ADD)

# 新增：创建 bin/
$(BINDIR):
	@mkdir -p $@

$(TARGET_TCACCEMU_MULTI_ADD): $(SOURCE_TCACCEMU_MULTI_ADD)
	@echo "--- Compiling $< -> $@"
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(LDFLAGS)
	@echo "--- Finished building $@ ---"

$(TARGET_TCACCEMU_THREE_ADD): $(SOURCE_TCACCEMU_THREE_ADD)
	@echo "--- Compiling $< -> $@"
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(LDFLAGS)
	@echo "--- Finished building $@ ---"

# SASS 到 bin/
sass: $(BINDIR)/tc_acc_emulation_double_multi_add.sass \
      $(BINDIR)/tc_acc_emulation_double_three_add.sass
	@echo "--- All SASS files generated ---"

$(BINDIR)/%.sass: $(BINDIR)/%
	@echo "--- Generating SASS for $< ---"
	cuobjdump -sass $< > $@
	@echo "--- SASS code saved to $@ ---"

clean:
	@echo "--- Cleaning up built files ---"
	rm -rf $(BINDIR)

.PHONY: all clean sass
