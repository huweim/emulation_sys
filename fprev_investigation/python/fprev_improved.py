import numpy as np
from typing import List, Tuple, Set, Dict, Optional, Any
from dataclasses import dataclass
import matplotlib.pyplot as plt
import networkx as nx


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

    def compute_l_ij(self, i: int, j: int, n: int, method: str = "gemv") -> int:
        if method == "fma":
            A = self.create_test_array(i, j, n)
            sum_val = self.sumimpl_fma(A)
        elif method == "gemv":
            sum_val = self.cuda_module.gemv_sequence_test(i, j, n)
        else:
            raise ValueError(f"Unknown method: {method}")
        val = int(round(n - sum_val))  # l_ij = n - SUMIMPL(A_ij)
        return val

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
