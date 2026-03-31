#pragma once

#include <common.h>

void sym_quant_host(const half *x, const half *scale, uint32_t rows,
                    uint32_t colsSrc, uint32_t colsDst, Int4Storage *q);

void sym_quant_host_bf16(const __nv_bfloat16 *x, const __nv_bfloat16 *scale,
                         uint32_t rows, uint32_t colsSrc, uint32_t colsDst,
                         Int4Storage *q);

void sym_dequant_host(const int32_t *q, const half *scale_row,
                      const half *scale_col, uint32_t rows, uint32_t cols,
                      half *x);

void sym_dequant_host_bf16(const int32_t *q, const __nv_bfloat16 *scale_row,
                           const __nv_bfloat16 *scale_col, uint32_t rows,
                           uint32_t cols, __nv_bfloat16 *x);

