import numpy as np
from typing import List, Tuple, Set, Dict, Optional, Any
from dataclasses import dataclass
import matplotlib.pyplot as plt
import networkx as nx

# Try to import nvfp for nvfp quantization support
try:
    import torch
    import nvfp.ops as ops
    import nvfp.pseudo_quant as pseudo_quant

    NVFP_AVAILABLE = True
except ImportError:
    NVFP_AVAILABLE = False
    torch = None
    ops = None
    pseudo_quant = None


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

        M_VALUE = 57344.0  # ensure 1 is quantized to 1.0
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

        # Pseudo quantization: quantize to FP4 and dequantize back to float32
        A_pseudo = pseudo_quant.nvfp4_pseudo_quantize(A)
        B_pseudo = pseudo_quant.nvfp4_pseudo_quantize(B)

        # Use standard matrix multiplication with pseudo-quantized tensors
        output = torch.matmul(A_pseudo.double(), B_pseudo.double().T).to(torch.float16)
        sum_val_pseudo = output[0, 0].item() / 16.0
        print(f"i={i}, j={j}, sum_real={sum_val_real}, sum_pseudo={sum_val_pseudo}")

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
