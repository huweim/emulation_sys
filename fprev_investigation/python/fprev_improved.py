import numpy as np
from typing import List, Tuple, Set, Dict, Optional, Any
from dataclasses import dataclass
import matplotlib.pyplot as plt
import networkx as nx
import torch
import math

# Try to import nvfp for nvfp quantization support
try:
    import nvfp.ops as ops
    import nvfp.pseudo_quant as pseudo_quant

    NVFP_AVAILABLE = True
except ImportError:
    NVFP_AVAILABLE = False
    ops = None
    pseudo_quant = None


def nvfp4_pseudo_quantize(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert x.is_cuda, "x must be a CUDA tensor"
    assert x.dtype in (
        torch.float,
        torch.float16,
        torch.bfloat16,
    ), f"x.dtype needs to be float, fp16 or bf16 but got {x.dtype}"
    assert x.ndim >= 1, f"x.ndim needs to be >= 1, but got {x.ndim}"
    assert (
        x.shape[-1] % 16 == 0
    ), f"last dim has to be multiple of 16, but got {x.shape[-1]}"
    org_shape = x.shape
    x = x.reshape(-1, org_shape[-1])
    fp4_weight, weight_scale_interleaved, weight_global_scale = (
        pseudo_quant.quantize_linear_weight_to_nvfp4(x)
    )
    """Dequantize the fp4 tensor back to high precision."""
    # Two fp4 values are packed into one uint8.
    assert fp4_weight.dtype == torch.float4_e2m1fn_x2
    m, packed_k = fp4_weight.shape
    k = packed_k * 2
    tensor_f32 = pseudo_quant.unpack_fp4_bytes(fp4_weight)
    tensor_f32 = tensor_f32.reshape(m, k)
    weight_scale_interleaved = weight_scale_interleaved.view(torch.float8_e4m3fn)
    weight_scale_interleaved = pseudo_quant.swizzled_to_linear_128_4(
        weight_scale_interleaved, m, k
    )

    return tensor_f32, weight_scale_interleaved, weight_global_scale


def torch_add_custom_precision(a, b, precision_bits):
    a_t = torch.as_tensor(a, dtype=torch.float64)
    b_t = torch.as_tensor(b, dtype=torch.float64)
    res = a_t + b_t

    mantissa, exponent = torch.frexp(res)
    scale = 2.0**precision_bits
    mantissa_rounded = torch.round(mantissa * scale) / scale
    final_res = torch.ldexp(mantissa_rounded, exponent)

    return final_res


def nvfp4_gemm_simulation(
    A_val: torch.Tensor,
    B_val: torch.Tensor,
    scale_A: torch.Tensor,
    scale_B: torch.Tensor,
    alpha: float,
    out_dtype: torch.dtype = torch.float16,
):
    """
    NVIDIA NVFP4 GEMM Simulation Kernel
    """

    M, K = A_val.shape
    N, _ = B_val.shape
    assert K % 64 == 0, "K must be multiple of 64"

    # A_int, B_int: int64
    A_int = (A_val * 2).to(torch.int64)
    B_int = (B_val * 2).to(torch.int64)

    sA_u8 = scale_A.view(torch.uint8).to(torch.int64)
    sB_u8 = scale_B.view(torch.uint8).to(torch.int64)

    def decode_e4m3(u8_tensor):
        # Mask 0x78 (01111000) >> 3 -> Exponent
        exp = (u8_tensor & 0x78) >> 3
        # Mask 0x07 | 0x08 -> Mantissa (1.MMM -> 1MMM)
        man = (u8_tensor & 0x07) | 0x08
        return man, exp

    sA_man, sA_exp = decode_e4m3(sA_u8)
    sB_man, sB_exp = decode_e4m3(sB_u8)

    n_blocks = K // 64

    # [M/N, Blocks, Group(4), Elements(16)]
    A_4d = A_int.reshape(M, n_blocks, 4, 16)
    B_4d = B_int.reshape(N, n_blocks, 4, 16)

    sA_man_3d = sA_man.reshape(M, n_blocks, 4)
    sA_exp_3d = sA_exp.reshape(M, n_blocks, 4)
    sB_man_3d = sB_man.reshape(N, n_blocks, 4)
    sB_exp_3d = sB_exp.reshape(N, n_blocks, 4)

    output_acc = torch.zeros((M, N), dtype=torch.float64, device=A_val.device)

    for b in range(n_blocks):
        a_blk = A_4d[:, b, :, :]  # [M, 4, 16] (int64)
        b_blk = B_4d[:, b, :, :]  # [N, 4, 16] (int64)

        # [M, 4, 16] @ [N, 4, 16] -> [M, N, 4]
        vec_sum_f = torch.einsum("mgk, ngk -> mng", a_blk.float(), b_blk.float())

        vec_sum = vec_sum_f.to(torch.int64)  # [M, N, 4] S10P2

        sa_m = sA_man_3d[:, b, :]
        sa_e = sA_exp_3d[:, b, :]
        sb_m = sB_man_3d[:, b, :]
        sb_e = sB_exp_3d[:, b, :]

        man_prod = sa_m.unsqueeze(1) * sb_m.unsqueeze(0)  # [M, N, 4]
        exp_sum = sa_e.unsqueeze(1) + sb_e.unsqueeze(0)  # [M, N, 4]
        val_unshifted = vec_sum * man_prod

        reserve_bits = 16
        exp_max, _ = exp_sum.max(dim=-1, keepdim=True)  # [M, N, 1]

        dE = exp_max - exp_sum  # [M, N, 4]
        val_boosted = val_unshifted << reserve_bits
        val_shifted = val_boosted >> dE

        block_res_int = val_shifted.sum(dim=-1)  # [M, N]

        exponent_real = exp_max.squeeze(-1).float() - 22.0 - reserve_bits

        factor = torch.pow(2.0, exponent_real)
        output_acc = torch_add_custom_precision(
            output_acc, block_res_int.double() * factor, 25  # maximum precision
        )

    final_output = output_acc * alpha

    return final_output.to(out_dtype)


@dataclass
class TreeNode:
    id: int
    children: List["TreeNode"]
    is_leaf: bool = False
    parent: Optional["TreeNode"] = None

    def __post_init__(self):
        for child in self.children:
            child.parent = self


class SummationTree:
    def __init__(self):
        self.nodes: Dict[int, TreeNode] = {}
        self.root: Optional[TreeNode] = None
        self.next_internal_id: int = 0  # internal nodes start from 0, but we'll offset

    def add_leaf(self) -> TreeNode:
        node_id = self.next_internal_id
        self.next_internal_id += 1
        node = TreeNode(id=node_id, children=[], is_leaf=True)
        self.nodes[node_id] = node
        return node

    def add_internal_node(self, children: List[TreeNode]) -> TreeNode:
        node_id = self.next_internal_id
        self.next_internal_id += 1
        node = TreeNode(id=node_id, children=children, is_leaf=False)
        self.nodes[node_id] = node
        return node

    def set_root(self, node: TreeNode):
        self.root = node

    def get_root_of(self, idx: int) -> TreeNode:
        """Find current root of the tree containing leaf idx (via parent links)"""
        node = self.nodes[idx]
        while node.parent is not None:
            node = node.parent
        return node

    def visualize(self, title: str = "Summation Tree", save_path: str = None):
        if self.root is None:
            print("No root to visualize")
            return

        G = nx.DiGraph()
        for node_id, node in self.nodes.items():
            label = f"{node_id}" if node.is_leaf else f"+"
            color = "lightblue" if node.is_leaf else "lightgreen"
            G.add_node(node_id, label=label, color=color)

        for node in self.nodes.values():
            for child in node.children:
                G.add_edge(node.id, child.id)

        pos = nx.nx_pydot.pydot_layout(G, prog="dot")
        labels = nx.get_node_attributes(G, "label")
        colors = [G.nodes[n]["color"] for n in G.nodes()]

        plt.figure(figsize=(12, 8))
        nx.draw(
            G,
            pos,
            labels=labels,
            node_color=colors,
            node_size=80,
            font_size=9,
            font_weight="bold",
            arrows=True,
            arrowsize=15,
        )
        plt.title(title)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Graph saved to: {save_path}")
        else:
            plt.savefig(f"{title}.png", dpi=300, bbox_inches="tight")
            print(f"Graph saved to: {title}.png")

        plt.close()  # Close the figure to free memory


class ImprovedFPRev:
    def __init__(self, cuda_module):
        self.cuda_module = cuda_module

    def sumimpl_fma(self, A: np.ndarray) -> float:
        return self.cuda_module.sumimpl_fma(A)

    def create_test_array(self, i: int, j: int, n: int) -> np.ndarray:
        array = np.ones(n, dtype=np.float32)
        M_VALUE = np.float32(
            1.701411834604692317316873037158841056e38
        )  # 2^127 for float32
        array[i] = M_VALUE
        array[j] = -M_VALUE
        return array

    def _compute_l_ij_nvfp(self, i: int, j: int, n: int) -> float:
        """
        Compute l_ij using NVFP quantization to test accumulation order.
        Creates n×n matrices A and B, applies NVFP quantization, and uses matrix multiplication.
        Similar to gemv_sequence_test in fprev_kernel.cu but with matrix multiplication.

        Args:
            i, j: Indices for the test values
            n: Matrix dimension
        """
        if not NVFP_AVAILABLE:
            raise ImportError("NVFP is not available. Please install nvfp package.")

        M_VALUE = 28672.0  # or 57344.0 ensure 1 is quantized to 1.0
        self.FLOAT4_E2M1_MAX = 6.0
        self.FLOAT8_E4M3_MAX = 448.0

        n *= 16
        A = torch.ones(n, n, dtype=torch.half, device="cuda")
        B = torch.ones(n, n, dtype=torch.half, device="cuda")

        for t in range(16):
            A[0, 16 * i + t] = -M_VALUE
            B[0, 16 * i + t] = M_VALUE
            A[0, 16 * j + t] = M_VALUE
            B[0, 16 * j + t] = M_VALUE

        # i_base_4 = i & (~3)
        # j_base_4 = j & (~3)

        # for t in range(4):
        #     if (j_base_4 + t) != i and (j_base_4 + t) != j:
        #         for k in range(16):
        #             A[0, 16 * (j_base_4 + t) + k] = 0

        # Real quantization using actual FP4 computation
        A_amax = torch.abs(A).max().to(torch.float32)
        A_global_scale = self.FLOAT8_E4M3_MAX * self.FLOAT4_E2M1_MAX / A_amax
        A_fp4, scale_A_fp4 = ops.scaled_fp4_quant(A, A_global_scale)

        B_amax = torch.abs(B).max().to(torch.float32)
        B_global_scale = self.FLOAT8_E4M3_MAX * self.FLOAT4_E2M1_MAX / B_amax
        B_fp4, scale_B_fp4 = ops.scaled_fp4_quant(B, B_global_scale)

        alpha = 1.0 / (A_global_scale * B_global_scale)
        output = ops.cutlass_scaled_fp4_mm(
            A_fp4, B_fp4, scale_A_fp4, scale_B_fp4, alpha, torch.float16
        )
        sum_val_real = output[0, 0].item() / 16.0

        sim_A_fp4, sim_A_scale, sim_A_global = nvfp4_pseudo_quantize(A)
        sim_B_fp4, sim_B_scale, sim_B_global = nvfp4_pseudo_quantize(B)
        sim_val = (
            nvfp4_gemm_simulation(
                sim_A_fp4,
                sim_B_fp4,
                sim_A_scale,
                sim_B_scale,
                (1.0 / sim_A_global / sim_B_global).item(),
                torch.float16,
            )[0, 0].item()
            / 16.0
        )
        # Pseudo quantization: quantize to FP4 and dequantize back to float32
        A_pseudo = pseudo_quant.nvfp4_pseudo_quantize(A)
        B_pseudo = pseudo_quant.nvfp4_pseudo_quantize(B)

        # Use standard matrix multiplication with pseudo-quantized tensors
        output = torch.matmul(A_pseudo.double(), B_pseudo.double().T).to(torch.float16)
        sum_val_pseudo = output[0, 0].item() / 16.0
        print(
            f"i={i}, j={j}, sum_real={sum_val_real}, sim_val={sim_val},sum_pseudo={sum_val_pseudo}"
        )

        return sum_val_real

    def compute_l_ij(self, i: int, j: int, n: int, method: str = "gemv") -> int:
        """
        Compute l_ij value using specified method and quantization mode.

        Args:
            i, j: Indices for the test values
            n: Array/matrix dimension
            method: "fma", "gemv", "nvfp_real", "nvfp_pseudo"
            quant_mode: "real" or "pseudo" (only used with nvfp methods)
        """
        if method == "fma":
            A = self.create_test_array(i, j, n)
            sum_val = self.sumimpl_fma(A)
        elif method == "gemv":
            sum_val = self.cuda_module.gemv_sequence_test(i, j, n)
        elif method == "nvfp":
            sum_val = self._compute_l_ij_nvfp(i, j, n)
        else:
            raise ValueError(f"Unknown method: {method}")
        val = int(n - sum_val)  # l_ij = n - SUMIMPL(A_ij)
        return val

    def print_all(self, n: int, method: str = "nvfp"):
        """
        Print all l_ij values for given n using specified method and quantization mode.

        Args:
            n: Array/matrix dimension
            method: "fma", "gemv", "nvfp", "nvfp_real", "nvfp_pseudo"
        """
        for i in range(n):
            for j in range(i + 1, n):
                self.compute_l_ij(i, j, n, method)

    def investigate_sequence(self, n: int, method: str = "gemv") -> SummationTree:
        self.n = n
        self.method = method
        self.tree = SummationTree()

        # Initialize all leaves
        for _ in range(n):
            self.tree.add_leaf()

        # Build tree recursively
        root, _ = self._build_subtree(set(range(n)))
        self.tree.set_root(root)
        return self.tree

    def _build_subtree(self, I: Set[int]) -> Tuple[TreeNode, int]:
        """Returns (root_node, size_of_subtree_in_leaves)"""
        if len(I) == 1:
            i = next(iter(I))
            return self.tree.nodes[i], 1

        i = min(I)
        Li = {}  # l_ij -> list of j
        for j in I:
            if j == i:
                continue
            l_ij = self.compute_l_ij(i, j, self.n, self.method)
            if l_ij not in Li:
                Li[l_ij] = []
            Li[l_ij].append(j)

        # Sort l values in increasing order
        sorted_l_vals = sorted(Li.keys())
        current_root = self.tree.nodes[i]

        for l in sorted_l_vals:
            Jl = set(Li[l])

            # Recursively build subtree for Jl
            sub_root, sub_size_actual = self._build_subtree(Jl)

            # Determine if sub_root is sibling or parent of current_root
            if len(Jl) == sub_size_actual:
                # Sibling: create new parent
                new_node = self.tree.add_internal_node([current_root, sub_root])
                current_root = new_node
            else:
                # Parent: attach current_root as child of sub_root
                sub_root.children.append(current_root)
                current_root = sub_root
                # size is sub_size_actual (which > len(Jl))

        return current_root, max(Li)

    def analyze_accumulation_pattern(self, tree: SummationTree) -> Dict[str, Any]:
        if tree.root is None:
            return {"error": "No root"}

        def get_depth(node: TreeNode) -> int:
            if not node.children:
                return 0
            return 1 + max(get_depth(c) for c in node.children)

        def get_branching(node: TreeNode, factors: List[int]):
            if node.children:
                factors.append(len(node.children))
                for c in node.children:
                    get_branching(c, factors)

        branching_factors = []
        get_branching(tree.root, branching_factors)
        depth = get_depth(tree.root)
        avg_bf = np.mean(branching_factors) if branching_factors else 0
        max_bf = max(branching_factors) if branching_factors else 0

        return {
            "depth": depth,
            "avg_branching_factor": avg_bf,
            "max_branching_factor": max_bf,
            "num_nodes": len(tree.nodes),
            "is_binary": max_bf <= 2,
            "accumulation_pattern": (
                "binary" if max_bf <= 2 else f"multi-way (max {max_bf})"
            ),
        }
