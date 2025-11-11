import torch
import scaled_fp8_ops as ops 
import time
import fp8_quant_utils

# --- 0. 配置 ---
FLOAT8_E4M3_MAX = 448.0
FLOAT8_E4M3_MIN = -448.0

# 内核的块粒度
A_K_GRANULARITY = 128
B_K_GRANULARITY = 128
B_N_GRANULARITY = 128

# 尺寸
M = 2048
# K = 4096#正确
K = 1024
N = 4096

# 计时配置
WARMUP_ITERS = 5
N_ITERS = 20

# 检查维度是否有效
assert K % A_K_GRANULARITY == 0, f"K 必须是 {A_K_GRANULARITY} 的倍数"
assert K % B_K_GRANULARITY == 0, f"K 必须是 {B_K_GRANULARITY} 的倍数"
assert N % B_N_GRANULARITY == 0, f"N 必须是 {B_N_GRANULARITY} 的倍数"

print(f"运行 GEMM: M={M}, K={K}, N={N}")
print(f"A 缩放粒度: (1, {A_K_GRANULARITY})")
print(f"B 缩放粒度: ({B_K_GRANULARITY}, {B_N_GRANULARITY})")
print(f"预热迭代: {WARMUP_ITERS}, 计时迭代: {N_ITERS}")
print("-" * 30)

# --- 1. 创建输入张量 ---
a_bf16 = torch.randn((M, K), dtype=torch.bfloat16, device="cuda")
b_bf16 = torch.randn((K, N), dtype=torch.bfloat16, device="cuda")
torch.cuda.synchronize()

# ===================================================================
# 方法 1: 伪量化 (fp8_quant_utils_groupwise) + BF16 GEMM
# ===================================================================

# 预热
for _ in range(WARMUP_ITERS):
    a_pseudo_q = fp8_quant_utils.fp8_quant_utils_groupwise(a_bf16)
    b_pseudo_q = fp8_quant_utils.fp8_quant_utils_groupwise(b_bf16)
    # print(a_pseudo_q.dtype)
    out_pseudo_temp = a_pseudo_q @ b_pseudo_q
torch.cuda.synchronize()

# 计时
start_time = time.time()
for _ in range(N_ITERS):
    a_pseudo_q = fp8_quant_utils.fp8_quant_utils_groupwise(a_bf16)
    b_pseudo_q = fp8_quant_utils.fp8_quant_utils_groupwise(b_bf16)
    out_pseudo_temp = a_pseudo_q @ b_pseudo_q
torch.cuda.synchronize()
end_time = time.time()

# 保存最后一次运行的结果
out_pseudo = out_pseudo_temp
print(f"pseudo quant & bf16 gemm: {(end_time - start_time) / N_ITERS * 1000:.4f} ms")


# ===================================================================
# 方法 2 & 3 的共同准备工作：执行逐块量化
# ===================================================================
# A
a_q_fp8 = torch.empty((M, K), dtype=torch.float8_e4m3fn, device="cuda")
a_s_f32 = torch.empty((M, K // A_K_GRANULARITY), dtype=torch.float32, device="cuda")
# B
b_q_fp8 = torch.empty((K, N), dtype=torch.float8_e4m3fn, device="cuda")
b_s_f32 = torch.empty((K // B_K_GRANULARITY, N // B_N_GRANULARITY), dtype=torch.float32, device="cuda")

# 量化 A (粒度 1 x 128)
for m_idx in range(M):
    for k_block_idx in range(K // A_K_GRANULARITY):
        k_start = k_block_idx * A_K_GRANULARITY
        k_end = (k_block_idx + 1) * A_K_GRANULARITY
        
        a_chunk = a_bf16[m_idx, k_start:k_end]
        abs_max = a_chunk.abs().max()
        scale = abs_max / FLOAT8_E4M3_MAX
        scale = torch.clamp(scale, min=1e-8) 
        
        a_s_f32[m_idx, k_block_idx] = scale
        a_q_fp8[m_idx, k_start:k_end] = torch.clamp(
            a_chunk / scale, min=FLOAT8_E4M3_MIN, max=FLOAT8_E4M3_MAX
        ).to(torch.float8_e4m3fn)

# 量化 B (粒度 128 x 128)
for k_block_idx in range(K // B_K_GRANULARITY):
    k_start = k_block_idx * B_K_GRANULARITY
    k_end = (k_block_idx + 1) * B_K_GRANULARITY
    for n_block_idx in range(N // B_N_GRANULARITY):
        n_start = n_block_idx * B_N_GRANULARITY
        n_end = (n_block_idx + 1) * B_N_GRANULARITY
        
        b_chunk = b_bf16[k_start:k_end, n_start:n_end]
        abs_max = b_chunk.abs().max()
        scale = abs_max / FLOAT8_E4M3_MAX
        scale = torch.clamp(scale, min=1e-8)
        
        b_s_f32[k_block_idx, n_block_idx] = scale
        b_q_fp8[k_start:k_end, n_start:n_end] = torch.clamp(
            b_chunk / scale, min=FLOAT8_E4M3_MIN, max=FLOAT8_E4M3_MAX
        ).to(torch.float8_e4m3fn)


# 准备 CUTLASS 内核所需的转置输入
b_q_fp8_t = b_q_fp8.T.contiguous()
b_s_f32_t = b_s_f32.T.contiguous()
a_s_f32_t = a_s_f32.T.contiguous()
out_cutlass = torch.zeros((M, N), dtype=torch.bfloat16, device="cuda")
torch.cuda.synchronize()

# ===================================================================
# 方法 2: CUTLASS FP8 内核 (cutlass_scaled_mm_blockwise_sm120_fp8)
# ===================================================================

# 预热
for _ in range(WARMUP_ITERS):
    ops.cutlass_scaled_mm_blockwise_sm120_fp8(
        out_cutlass, a_q_fp8, b_q_fp8_t, a_s_f32_t, b_s_f32_t
    )
torch.cuda.synchronize()

# 计时
start_time = time.time()
for _ in range(N_ITERS):
    ops.cutlass_scaled_mm_blockwise_sm120_fp8(
        out_cutlass,
        a_q_fp8, 
        b_q_fp8_t, 
        a_s_f32_t, 
        b_s_f32_t
    )
torch.cuda.synchronize()
end_time = time.time()
print(f"real quant & cutlass fp8 gemm: {(end_time - start_time) / N_ITERS * 1000:.4f} ms")


# ===================================================================
# 方法 3 的准备工作：执行逐块反量化
# ===================================================================
a_dequant_f32 = torch.empty((M, K), dtype=torch.float32, device="cuda")
b_dequant_f32 = torch.empty((K, N), dtype=torch.float32, device="cuda")

# 反量化 A
for m_idx in range(M):
    for k_block_idx in range(K // A_K_GRANULARITY):
        ks = k_block_idx * A_K_GRANULARITY
        ke = (k_block_idx + 1) * A_K_GRANULARITY
        scale = a_s_f32[m_idx, k_block_idx].to(torch.float32)
        a_dequant_f32[m_idx, ks:ke] = a_q_fp8[m_idx, ks:ke].to(torch.float32) * scale

# 反量化 B
for k_block_idx in range(K // B_K_GRANULARITY):
    ks = k_block_idx * B_K_GRANULARITY
    ke = (k_block_idx + 1) * B_K_GRANULARITY
    for n_block_idx in range(N // B_N_GRANULARITY):
        ns = n_block_idx * B_N_GRANULARITY
        ne = (n_block_idx + 1) * B_N_GRANULARITY
        scale = b_s_f32[k_block_idx, n_block_idx].to(torch.float32)
        b_dequant_f32[ks:ke, ns:ne] = b_q_fp8[ks:ke, ns:ne].to(torch.float32) * scale
# print("逐块反量化完成。")
torch.cuda.synchronize()

# ===================================================================
# 方法 3: 参考 FP32 GEMM (来自反量化的数据)
# ===================================================================
# print("\n--- 3. 计时: 参考 FP32 GEMM (来自反量化) ---")

# 预热
for _ in range(WARMUP_ITERS):
    out_ref_temp = (a_dequant_f32 @ b_dequant_f32).to(torch.bfloat16)
torch.cuda.synchronize()

# 计时
start_time = time.time()
for _ in range(N_ITERS):
    out_ref_temp = (a_dequant_f32 @ b_dequant_f32).to(torch.bfloat16)
torch.cuda.synchronize()
end_time = time.time()

# 保存最后一次运行的结果
out_ref = out_ref_temp
print(f"real quant & fp32 gemm: {(end_time - start_time) / N_ITERS * 1000:.4f} 毫秒")


# ===================================================================
# 结果比较
# ===================================================================
print("\n" + "=" * 30)
print(" 结果比较 ")
print("=" * 30)

print(f"pseudo [{out_pseudo.shape}, {out_pseudo.dtype}]:")
print(out_pseudo[:4, :4])

print(f"\nCUTLASS [{out_cutlass.shape}, {out_cutlass.dtype}]:")
print(out_cutlass[:4, :4])

print(f"\nref [{out_ref.shape}, {out_ref.dtype}]:")
print(out_ref[:4, :4])


def calculate_diff(a, b, name_a, name_b):
    a_f32 = a.to(torch.float32)
    b_f32 = b.to(torch.float32)
    
    # 绝对差异
    abs_diff = (a_f32 - b_f32).abs()
    max_abs_diff = abs_diff.max()
    
    # # 相对差异
    # # (A - B) / (|A| + epsilon)
    # rel_diff_mean = (abs_diff / (a_f32.abs() + 1e-6)).mean() 
    
    print(f"\n比较: {name_a} vs {name_b}")
    print(f"  最大绝对差异: {max_abs_diff.item():.6e}")
    # print(f"  平均相对差异: {rel_diff_mean.item():.6e}")


calculate_diff(out_pseudo, out_cutlass, "pseudo", "CUTLASS")
calculate_diff(out_pseudo, out_ref, "pseudo", "ref")
calculate_diff(out_cutlass, out_ref, "CUTLASS", "ref")
