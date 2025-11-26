# Numerical Emulation System for Tensor Cores

A rigorous toolkit for reverse-engineering, probing, and emulating the numerical behaviors of NVIDIA Tensor Cores.

This project aims to bridge the gap between "bit-perfect" software emulation and hardware execution by revealing the hidden properties of Tensor Cores: **Accumulation Bit-Width**, **Rounding Modes**, and **Accumulation Order**.

## Key Findings (So Far)

Through systematic probing, we have reverse-engineered the following behaviors:

| **Architecture**             | **Input Type** | **Accumulator Type** | **Instruction / API** | **Observed Mantissa Width** | **Rounding Mode**                 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| A100 (sm_80), 4090 (sm_89), A6000 (sm_89) | FP16           | FP32                 | `wmma`                | **24 bits** | **RZ** (Round-to-Zero / Truncate) |
| H100 (sm_90a), 5090 (sm_120)            | FP16           | FP32                 | `wmma`                | **25 bits**  | **RZ** (Round-to-Zero / Truncate) |
| H100 (sm_90a)            | FP8 (E5M2)     | FP32                 | `wgmma`               | **13 bits**                 | **RZ** (Round-to-Zero / Truncate) |

**Core Conclusion**: Tensor Core accumulation is best modeled as a **Multi-term Summation** followed by a **Single Final Rounding (RZ)** step.

---

## Installation & Build

**Prerequisites:**
* NVIDIA GPU (Ampere or newer recommended; Hopper required for FP8 tests)
* CUDA Toolkit 11.8+ (12.0+ recommended for FP8)

```shell
# Clone the repository
git clone [https://github.com/huweim/emulation_sys.git](https://github.com/huweim/emulation_sys.git)
cd emulation_sys

# Build all probing tools
# Note: This will compile targets for both sm_89 (Ada) and sm_90a (Hopper).
# Ensure your nvcc supports these architectures.
make

# Clean build artifacts
make clean
```

## Run Probing Experiments

### 1. Probe Accumulator Bit-Width (FP16)

Run this on A100 or RTX 3090/4090 to detect the effective mantissa width of the FP32 accumulator during FP16 GEMM.

Shell

```
# Uses "Carry Propagation" method to detect if the accumulator can hold 
# intermediate results beyond 24 bits.
./bin/probe_mma_fp16_acc_fp32
```

### 2. Probe Accumulator Bit-Width (FP8 on H100)


**Requires H100 GPU.** This test uses the `wgmma` PTX instruction to probe the accumulator behavior when using FP8 (E5M2/E4M3) inputs.

Shell

```
# Probes the accumulation path: FP8 x FP8 -> FP32 Accumulator
# Verifies if H100 uses a wider-than-standard-FP32 accumulator.
./bin/probe_h100_fp8
```


## Roadmap and TODO


### Phase 1: Reverse Engineering (Hardware Probing)

- [x] **Rounding Mode**: Confirmed as **RZ (Round-to-Zero/Truncate)** for `mma` instructions, distinct from the standard RN (Round-to-Nearest) of `fma` instructions.
- [x] **FP16 Bit-Width**: Confirmed 24-bit for Ampere/Ada, 25-bit for Hopper using `wmma`.
- [x] **H100 FP8 Bit-Width**: Developed `wgmma` probe for H100. Confirmed effective mantissa width is 13 bits.
- [x] **5090 FP8/FP4 Bit-Width**: Developed `mma` probe for 5090. Confirmed effective mantissa width is 25 bits.
- [ ] **INT8/INT4 Bit-Width**: Develop probes for A100 to test integer accumulation behaviors and potential overflows/saturations for INT8 and INT4 data types.
- [ ] **Blackwell (B200) Support**: Prepare inline PTX (`mma`) for probing native FP4/FP8 quantization and accumulation behaviors once hardware is available.

- [ ] PTX Assembly: Achieve 100% coverage of Inline PTX versions for all probes.
  + For every probe implemented via the C++ `wmma/wgmma API`, implement a corresponding version using `PTX mma` instructions. This ensures ISA-level verification and bypasses potential compiler/API abstractions.

### Phase 2: Emulation Kernel (Software Modeling)


- [ ] **Finalize "Bit-wise" Emulator**: Implement a generalized CUDA kernel that uses integer bit-wise operations to strictly enforce the "Wide Accumulation -> Truncate to N bits -> Add -> Truncate" model.
  - *Status*: Concept proven, implementation in progress.
- [ ] **Python/PyTorch Integration**: Wrap the C++ emulator kernel into a PyTorch custom op (`torch.ops.emulation.gemm`) for easy integration into LLM evaluation pipelines.

### Phase 3: Fidelity Evaluation

- [ ] **Accuracy Benchmarking**: Run PPL (Perplexity) tests on Llama-2/3 models using our emulator. Compare the PPL curve against:
  - Ideal FP16 (Baseline)
  - Real Pseudo-Quantization (Current SOTA)
  - **Our Emulation Result** (Target: Match Real Hardware PPL exactly)