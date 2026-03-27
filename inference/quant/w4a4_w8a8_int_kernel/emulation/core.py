"""
Core emulation for W4A4 / W8A8 int GEMM kernels.
"""
from __future__ import annotations

import torch


class IntKernelUtils:
    """Low-level unpack/depack helpers for int kernels."""

    BLOCK_M = 256
    NUM_WARPS = 8
    WARP_M = 32
    WARP_N = 128
    GROUP_K_W4A4 = 64

    @staticmethod
    def _sign_extend_4bit(x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.int16)
        return ((x ^ 0x8) - 0x8).to(torch.int8)
    
    @staticmethod
    def unpack_int4_pairs(packed: torch.Tensor, original_k: int) -> torch.Tensor:
        """
        Backward-compatible alias for activation unpack.
        """
        return IntKernelUtils.unpack_int4_pairs_act(packed, original_k)

    @staticmethod
    def unpack_int4_pairs_act(packed: torch.Tensor, original_k: int) -> torch.Tensor:
        """
        Unpack W4A4 activation with exact nunchaku act layout.

        act storage is [M/BLOCK_M, K/64, NUM_WARPS, WARP_M_TILES, WARP_SIZE] of packed_act_t(uint4),
        flattened to [M, K/2] int8. Each byte stores two signed int4 values (low/high nibble).

        Compared with weight, activation path has an extra ldmatrix-based lane remap. We invert that
        remap directly at byte/nibble level per (32x64) warp tile.
        """
        if packed.dtype != torch.int8:
            raise TypeError(f"packed activation must be int8, got {packed.dtype}")

        M = packed.shape[0]
        K = original_k
        if packed.shape[1] != K // 2:
            raise ValueError(f"packed shape mismatch: expected second dim K/2={K//2}, got {packed.shape[1]}")
        if M % 32 != 0 or K % 64 != 0:
            raise ValueError("w4a4 activation layout requires M%32==0 and K%64==0")

        warp_m = IntKernelUtils.WARP_M
        group_k = IntKernelUtils.GROUP_K_W4A4
        num_warps = IntKernelUtils.NUM_WARPS

        m_warps = M // warp_m
        if m_warps % num_warps != 0:
            raise ValueError("w4a4 activation layout requires (M/32) % NUM_WARPS == 0")

        k_tiles = K // group_k
        num_blocks_m = m_warps // num_warps

        u = packed.view(torch.uint8)
        flat = u.reshape(-1)
        out_u4 = torch.empty((M, K), device=packed.device, dtype=torch.uint8)

        # local coordinates in one warp output (32x64 across two 16x64 tiles)
        r = torch.arange(32, device=packed.device, dtype=torch.int64).view(32, 1)
        c = torch.arange(64, device=packed.device, dtype=torch.int64).view(1, 64)

        # Inverse mapping from logical (r, c) -> byte matrix [32,32] + nibble select.
        # This captures quantize_w4a4_warp + ldmatrix remap and has been probe-validated.
        pr_local = (r % 8) * 2 + (r // 16) * 16 + ((c // 16) % 2)
        pc_local = ((c % 8) // 2) + ((c // 8) % 2) * 16 + ((c // 32) % 2) * 8 + ((r // 8) % 2) * 4
        nib_hi = (c & 1).bool()

        bytes_per_warp_output = 2 * 32 * 16  # WARP_M_TILES * WARP_SIZE * sizeof(uint4) = 1024

        for bm in range(num_blocks_m):
            bm_row_base = bm * IntKernelUtils.BLOCK_M
            for bk in range(k_tiles):
                for warp_id in range(num_warps):
                    chunk_idx = ((bm * k_tiles + bk) * num_warps + warp_id)
                    chunk_base = chunk_idx * bytes_per_warp_output

                    # physical bytes for one warp output: [32, 32]
                    chunk = flat[chunk_base: chunk_base + bytes_per_warp_output].view(32, 32)
                    bytes_ = chunk[pr_local, pc_local]
                    vals = torch.where(nib_hi, (bytes_ >> 4) & 0x0F, bytes_ & 0x0F)

                    row_out = bm_row_base + warp_id * warp_m + r
                    col_out = bk * group_k + c
                    out_u4[row_out, col_out] = vals

        return IntKernelUtils._sign_extend_4bit(out_u4)

    @staticmethod
    def unpack_int4_pairs_wgt(packed: torch.Tensor, original_k: int) -> torch.Tensor:
        """
        Unpack W4A4 weight with nunchaku layout.

        This follows deepcompressor/nunchaku weight packer inverse for bits=4,
        which matches zgemm w4a4 weight memory layout.
        """
        if packed.dtype != torch.int8:
            raise TypeError(f"packed weight must be int8, got {packed.dtype}")

        N = packed.shape[0]
        K = original_k
        if packed.shape[1] != K // 2:
            raise ValueError(f"packed shape mismatch: expected second dim K/2={K//2}, got {packed.shape[1]}")
        if N % 128 != 0 or K % 64 != 0:
            raise ValueError("w4a4 layout requires N%128==0 and K%64==0")

        # Packer constants for bits=4, warp_n=128
        num_n_packs, n_pack_size, num_n_lanes, reg_n = 8, 2, 8, 1
        num_k_packs, k_pack_size, num_k_lanes = 1, 2, 4
        mem_n, mem_k = 128, 64

        n_tiles = N // mem_n
        k_tiles = K // mem_k

        # Forward pack merged reg_k=8 int4 into int32, then viewed as 4 bytes.
        x = packed.view(torch.uint8).reshape(
            n_tiles,
            k_tiles,
            num_k_packs,
            num_n_packs,
            num_n_lanes,
            num_k_lanes,
            n_pack_size,
            k_pack_size,
            reg_n,
            4,  # bytes per int32
        )

        # Recover 8 int4 values from 4 bytes (little-endian nibble order).
        lo = x & 0x0F
        hi = (x >> 4) & 0x0F
        reg_vals = torch.stack([lo, hi], dim=-1).reshape(*x.shape[:-1], 8)

        # Inverse permute of forward (0,5,6,1,3,8,2,7,4,9)
        reg_vals = reg_vals.permute(0, 3, 6, 4, 8, 1, 2, 7, 5, 9).contiguous()

        q_u4 = reg_vals.reshape(N, K)
        return IntKernelUtils._sign_extend_4bit(q_u4)

    @staticmethod
    def unpack_nunchaku_w8_weight(packed_w: torch.Tensor, N: int, K: int) -> torch.Tensor:
        """
        Inverse of deepcompressor NunchakuWeightPacker(bits=8).pack_weight().
        packed_w is expected to be int8 with shape [N, K] from convert_to_nunchaku_w8x8y16_linear_weight.
        Returns unpacked logical qweight [N, K] int8.
        """
        if packed_w.dtype != torch.int8:
            raise TypeError(f"packed_w must be int8, got {packed_w.dtype}")
        if packed_w.shape[0] != N:
            raise ValueError(f"packed_w.shape[0] != N ({packed_w.shape[0]} vs {N})")
        if packed_w.shape[1] != K:
            raise ValueError(f"packed_w.shape[1] != K ({packed_w.shape[1]} vs {K})")

        # Packer constants for bits=8, warp_n=128
        num_n_packs, n_pack_size, num_n_lanes, reg_n = 8, 2, 8, 1
        num_k_packs, k_pack_size, num_k_lanes, reg_k = 1, 2, 4, 4
        mem_n, mem_k, num_k_unrolls = 128, 32, 2

        n_tiles = N // mem_n
        k_tiles = K // mem_k
        if N % mem_n != 0 or K % (mem_k * num_k_unrolls) != 0:
            raise ValueError("W8A8 packed layout requires N%128==0 and K%(32*2)==0")

        # Undo final int32->int8 view in little endian.
        # During packing: each last-dim reg_k(=4) values were packed to one int32.
        x = packed_w.view(torch.uint8).reshape(
            n_tiles,
            k_tiles,
            num_k_packs,
            num_n_packs,
            num_n_lanes,
            num_k_lanes,
            n_pack_size,
            k_pack_size,
            reg_n,
            reg_k,
        )

        # Inverse permute of (0,5,6,1,3,8,2,7,4,9)
        x = x.permute(0, 3, 6, 4, 8, 1, 2, 7, 5, 9).contiguous()

        # Collapse back to [N, K]
        x = x.reshape(N, K)
        return x.view(torch.int8)


class MMAEngine:
    """W4A4 / W8A8 emulation entrypoints."""
    
    @staticmethod
    def _sample_debug_tensor(t: torch.Tensor, max_rows: int = 2, max_cols: int = 8) -> list[list[float]]:
        rows = min(max_rows, t.shape[0])
        cols = min(max_cols, t.shape[1]) if t.ndim == 2 else min(max_cols, t.shape[-1])
        if t.ndim == 1:
            return [t[:cols].detach().float().cpu().tolist()]
        return t[:rows, :cols].detach().float().cpu().tolist()

    @staticmethod
    def emulation_gemm_w4a4(
        act: torch.Tensor,
        wgt: torch.Tensor,
        ascales: torch.Tensor,
        wscales: torch.Tensor,
        bias: torch.Tensor | None = None,
        return_debug: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        """
        Emulate w4a4 kernel mathematically:
        out[m,n] = sum_g sum_k_in_g (a_q * w_q) * ascales[g,m] * wscales[g,n] + bias[n]

        Inputs follow wrapper shapes:
          act: [M, K/2] int8 (packed int4)
          wgt: [N, K/2] int8 (packed int4)
          ascales: [K/64, M]
          wscales: [K/64, N]
        """
        if act.dtype != torch.int8 or wgt.dtype != torch.int8:
            raise TypeError("act/wgt must be int8")

        M, K2 = act.shape
        N, K2w = wgt.shape
        if K2 != K2w:
            raise ValueError("act and wgt K/2 mismatch")
        K = K2 * 2
        G = K // 64

        # act / wgt have different quantized memory layout in nunchaku
        a_q = IntKernelUtils.unpack_int4_pairs(act, K).to(torch.float32)
        w_q = IntKernelUtils.unpack_int4_pairs_wgt(wgt, K).to(torch.float32)

        # [M,G,64], [N,G,64]
        a_g = a_q.view(M, G, 64)
        w_g = w_q.view(N, G, 64)

        # integer partial per group: [M,N,G]
        int_partial = torch.einsum("mgk,ngk->mng", a_g, w_g)

        # scales to float32 and align dims
        sa = ascales.to(torch.float32).transpose(0, 1).unsqueeze(1)   # [M,1,G]
        sw = wscales.to(torch.float32).transpose(0, 1).unsqueeze(0)   # [1,N,G]
        out = (int_partial * sa * sw).sum(dim=-1)  # [M,N]

        if bias is not None:
            out = out + bias.to(torch.float32).view(1, N)

        out = out.to(ascales.dtype)
        if not return_debug:
            return out

        debug_info = {
            "mode": "w4a4",
            "shape": {"M": M, "N": N, "K": K, "G": G},
            "a_q_sample": MMAEngine._sample_debug_tensor(a_q),
            "w_q_sample": MMAEngine._sample_debug_tensor(w_q),
            "ascales_sample": MMAEngine._sample_debug_tensor(ascales.transpose(0, 1)),
            "wscales_sample": MMAEngine._sample_debug_tensor(wscales.transpose(0, 1)),
            "int_partial_sample": MMAEngine._sample_debug_tensor(int_partial[:, :, 0]),
        }
        return out, debug_info

    @staticmethod
    def emulation_gemm_w8a8(
        act: torch.Tensor,
        wgt: torch.Tensor,
        ascales: torch.Tensor,
        wscales: torch.Tensor,
        bias: torch.Tensor | None = None,
        packed_wgt: bool = True,
        return_debug: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        """
        Emulate w8a8 kernel.

        act: [M,K] int8
        wgt: [N,K] int8 (packed nunchaku layout by default)
        ascales: [M]
        wscales: [N]
        """
        if act.dtype != torch.int8 or wgt.dtype != torch.int8:
            raise TypeError("act/wgt must be int8")

        M, K = act.shape
        N, Kw = wgt.shape
        if Kw != K:
            raise ValueError("act and wgt K mismatch")

        if packed_wgt:
            w_logical = IntKernelUtils.unpack_nunchaku_w8_weight(wgt, N, K)
        else:
            w_logical = wgt

        a_f = act.to(torch.float32)
        w_f = w_logical.to(torch.float32)

        int_acc = a_f @ w_f.t()  # [M,N]
        out = int_acc * ascales.to(torch.float32).view(M, 1) * wscales.to(torch.float32).view(1, N)

        if bias is not None:
            out = out + bias.to(torch.float32).view(1, N)

        out = out.to(ascales.dtype)
        if not return_debug:
            return out
        
        debug_info = {
            "mode": "w8a8",
            "shape": {"M": M, "N": N, "K": K},
            "packed_wgt": packed_wgt,
            "act_q_sample": MMAEngine._sample_debug_tensor(act),
            "w_q_sample": MMAEngine._sample_debug_tensor(w_logical),
            "ascales_sample": MMAEngine._sample_debug_tensor(ascales),
            "wscales_sample": MMAEngine._sample_debug_tensor(wscales),
            "int_acc_sample": MMAEngine._sample_debug_tensor(int_acc),
        }
        return out, debug_info
