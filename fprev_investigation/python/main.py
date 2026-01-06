"""
Simplified FPRev Investigation Tool

Input: method and n
Output: graph saved to disk
"""

import numpy as np
import argparse
import sys
import os
from matplotlib import pyplot as plt

# Add the parent directory to the path to import the CUDA module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import fprev_cuda

    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False

from fprev_improved import ImprovedFPRev, SummationTree


class SimpleFPRev:
    """Simplified FPRev investigator for method and n input, graph output"""

    def __init__(self, method: str = "fma"):
        """
        Initialize the investigator

        Args:
            method: Accumulation method ("fma", "gemv", "nvfp")
        """
        self.method = method

        assert CUDA_AVAILABLE, "CUDA module not available"
        fprev_cuda.init()
        self.fprev = ImprovedFPRev(fprev_cuda)

    def investigate_and_save(self, n: int, output_path: str = None):
        """
        Investigate sequence and save graph to disk

        Args:
            n: Length of sequence to investigate
            output_path: Path to save the graph (default: auto-generated)
        """
        if output_path is None:
            output_path = f"fprev_graph_{self.method}_n{n}.png"

        print(f"Investigating {self.method.upper()} sequence of length {n}")

        self.fprev.print_all(n)

        # Run FPRev algorithm
        # tree = self.fprev.investigate_sequence(n, method=self.method)

        # # Save visualization to disk
        # tree.visualize(
        #     f"{self.method.upper()} Accumulation Tree (n={n})", save_path=output_path
        # )

        # # Basic analysis
        # analysis = self.fprev.analyze_accumulation_pattern(tree)
        # print(
        #     f"Pattern: {analysis['accumulation_pattern']}, Depth: {analysis['depth']}, Nodes: {analysis['num_nodes']}"
        # )

    def cleanup(self):
        """Cleanup resources"""
        fprev_cuda.cleanup()


def main():
    """Main function for command line interface"""
    parser = argparse.ArgumentParser(
        description="Simple FPRev Investigation: input method and n, output graph to disk"
    )
    parser.add_argument(
        "--n", type=int, default=16, help="Sequence length to investigate"
    )
    parser.add_argument(
        "--method",
        choices=["fma", "gemv", "nvfp"],
        default="nvfp",
        help="Accumulation method to use (default: gemv)",
    )
    parser.add_argument(
        "--output", type=str, help="Output path for the graph (default: auto-generated)"
    )

    args = parser.parse_args()

    # Initialize investigator
    investigator = SimpleFPRev(method=args.method)

    try:
        # Investigate and save graph
        investigator.investigate_and_save(args.n, args.output)
    finally:
        # Cleanup
        investigator.cleanup()


if __name__ == "__main__":
    main()
