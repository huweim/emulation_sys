# Makefile for FPrev Investigation Test Suite

# Compiler and flags
CXX = g++
NVCC = nvcc

# CUDA path (adjust if needed)
CUDA_PATH ?= /usr/local/cuda
CUDA_INCLUDE_PATH = $(CUDA_PATH)/include
CUDA_LIB_PATH = $(CUDA_PATH)/lib64

# Compiler flags
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra
NVCC_FLAGS = -std=c++14 -O3 --use_fast_math -arch=sm_89

# Include paths
INCLUDES = -I./include -I$(CUDA_INCLUDE_PATH)

# Library paths and libraries
LIBPATHS = -L$(CUDA_LIB_PATH)
LIBS = -lcuda -lcudart

# Source files
CUDA_SOURCES = cuda/fprev_kernel.cu
TEST_SOURCES = test_gemv.cpp

# Object files
CUDA_OBJECTS = $(CUDA_SOURCES:.cu=.o)
TEST_OBJECTS = $(TEST_SOURCES:.cpp=.o)

# Target executable
TARGET = test_gemv

# Default target
all: $(TARGET)

# Link the final executable
$(TARGET): $(CUDA_OBJECTS) $(TEST_OBJECTS)
	$(CXX) $(LIBPATHS) -o $@ $^ $(LIBS)

# Compile CUDA source files
%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

# Compile C++ source files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Clean build artifacts
clean:
	rm -f $(CUDA_OBJECTS) $(TEST_OBJECTS) $(TARGET)

# Run the test
test: $(TARGET)
	./$(TARGET)

# Check CUDA installation
check-cuda:
	@echo "Checking CUDA installation..."
	@which nvcc > /dev/null || (echo "nvcc not found in PATH" && exit 1)
	@nvcc --version
	@echo "CUDA installation OK"

# Help target
help:
	@echo "Available targets:"
	@echo "  all        - Build the test executable"
	@echo "  clean      - Remove build artifacts"
	@echo "  test       - Build and run the test"
	@echo "  check-cuda - Check CUDA installation"
	@echo "  help       - Show this help message"

.PHONY: all clean test check-cuda help
