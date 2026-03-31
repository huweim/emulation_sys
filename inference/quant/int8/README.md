# INT8 Quant (Standalone, No FP8 Dependency)

This directory provides a standalone INT8 path similar to `inference/quant/int4`:

- symmetric per-row INT8 quantization (`sym_int8_quant`)
- INT8 GEMM + dequant (`int8_linear`)
- pseudo quantization and emulation path for alignment checks

No separate install/build step is required.

## Usage

From repo root (`/state/partition/zhzhang/emulation_sys`):

```bash
python -u -m inference.quant.int8.example
```

## Verification (real vs pseudo/emul)

```bash
python -u -m inference.quant.int8.test_refactor
```

Optional env vars:

- `INT8_TEST_ITERS` (default: `200`)
- `INT8_TEST_OUT_DTYPE` (`float16` or `bfloat16`, default: `float16`)
- `INT8_TEST_CLIP_RATIO` (default: `1.0`)
- `INT8_TEST_MODEL` (`pseudo` or `emul`, default: `emul`)
- `INT8_TEST_PRINT_SAMPLES` (default: `5`)
- `INT8_TEST_MAX_MISMATCH_PRINT` (default: `20`)

Example:

```bash
INT8_TEST_ITERS=100000 INT8_TEST_MODEL=emul python -u -m inference.quant.int8.test_refactor
```
