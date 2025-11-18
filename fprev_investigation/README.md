# FPRev Investigation

A CUDA-based tool for investigating accumulation patterns using the FPRev algorithm.

## Installation

```bash
cd fprev_investigation
python setup.py install
```

## Usage

### Command Line Interface

The tool provides a simple CLI to investigate sequences and generate graphs:

```bash
# Investigate GEMV accumulation with default length (64)
python python/main.py --method gemv

# Investigate FMA accumulation with custom length
python python/main.py --method fma --n 128

# Specify output path for the graph
python python/main.py --method gemv --n 64 --output my_graph.png
```

### Options

- `--n`: Sequence length to investigate (default: 64)
- `--method`: Accumulation method (`fma` or `gemv`, default: `gemv`)
- `--output`: Output path for the generated graph (default: auto-generated)

### C++ Testing

For low-level testing, you can build and run the C++ test suite:

```bash
# Build the test executable
make

# Run the test
make test

# Clean build artifacts
make clean
```

## Output

The tool generates visualization graphs showing:
- Accumulation tree structure
- Pattern analysis
- Depth and node statistics

Graphs are saved as PNG files with auto-generated names like `fprev_graph_gemv_n64.png`.

## Project Structure

- `cuda/`: CUDA kernels and C++ bindings
- `include/`: Header files
- `python/`: Python interface and main script
- `Makefile`: Build system for C++ tests
- `setup.py`: Python package installation
