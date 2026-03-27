"""
Parameter search for optimal W_stage3 and W_stage4 values.
Added early exit support and rounding strategy search.
"""
import sys
import random
from typing import List, Tuple, Dict, Optional
import torch

from .utils import DataGenerator
from .core import MMAEngine, RoundStrategy


class SearchResult:
    """Result of testing one configuration."""
    def __init__(self, w_stage3: int, w_stage4: int, 
                 stage3_rounding: RoundStrategy, stage4_rounding: RoundStrategy,
                 matches: int, total: int):
        self.w_stage3 = w_stage3
        self.w_stage4 = w_stage4
        self.stage3_rounding = stage3_rounding
        self.stage4_rounding = stage4_rounding
        self.matches = matches
        self.total = total
        self.accuracy = matches / total if total > 0 else 0.0
    
    def __repr__(self):
        return (f"SearchResult(W3={self.w_stage3}, W4={self.w_stage4}, "
                f"R3={self.stage3_rounding.name}, R4={self.stage4_rounding.name}, "
                f"accuracy={self.accuracy*100:.2f}%)")


class ParameterSearch:
    """
    Search for optimal W_stage3 and W_stage4 parameters.
    Added early exit and rounding strategy support.
    """
    
    def __init__(
        self,
        num_iterations: int = 1000,
        dims_m: List[int] = None,
        dims_n: List[int] = None,
        dims_k: List[int] = None,
        distributions: List[str] = None,
        early_exit_mismatches: Optional[int] = None,
    ):
        """
        Args:
            num_iterations: Number of test cases per configuration
            early_exit_mismatches: Early exit if mismatches exceed this (None=disabled)
        """
        self.num_iterations = num_iterations
        self.dims_m = dims_m or [128, 256, 1024, 2048]
        self.dims_n = dims_n or [128, 256, 1024, 2048, 4096]
        self.dims_k = dims_k or [128, 256, 512, 1024]
        self.distributions = distributions or ["normal", "uniform", "large", "outliers", "mixed_rows", "abs_large"]
        self.early_exit_mismatches = early_exit_mismatches
    
    def search_grid(
        self,
        w3_range: range,
        w4_range: range,
        ops_module,
        get_global_scale_fn,
        stage3_roundings: List[RoundStrategy] = None,
        stage4_roundings: List[RoundStrategy] = None,
        verbose: bool = True
    ) -> List[SearchResult]:
        """
        Grid search over W_stage3 and W_stage4 ranges.
        """
        stage3_roundings = stage3_roundings or [RoundStrategy.RZ]
        stage4_roundings = stage4_roundings or [RoundStrategy.RZ]
        
        results = []
        total_configs = len(w3_range) * len(w4_range) * len(stage3_roundings) * len(stage4_roundings)
        config_idx = 0
        
        for r3 in stage3_roundings:
            for r4 in stage4_roundings:
                for w3 in w3_range:
                    for w4 in w4_range:
                        config_idx += 1
                        if verbose:
                            print(f"\n{'='*70}")
                            print(f"[{config_idx}/{total_configs}] Testing W3={w3}, W4={w4}, R3={r3.name}, R4={r4.name}")
                            print(f"{'='*70}")
                            if self.early_exit_mismatches:
                                print(f"(Early exit enabled: max {self.early_exit_mismatches} mismatches)")
                        
                        matches, total = self._test_config(
                            w3, w4, r3, r4, ops_module, get_global_scale_fn, verbose
                        )
                        
                        result = SearchResult(w3, w4, r3, r4, matches, total)
                        results.append(result)
                        
                        if verbose:
                            print(f"Result: {matches}/{total} matches ({result.accuracy*100:.2f}%)")
                            if matches == total:
                                print(f"*** PERFECT MATCH FOUND! ***")
        
        results.sort(key=lambda x: x.accuracy, reverse=True)
        return results
    
    def _test_config(
        self,
        w_stage3: int,
        w_stage4: int,
        stage3_rounding: RoundStrategy,
        stage4_rounding: RoundStrategy,
        ops_module,
        get_global_scale_fn,
        verbose: bool = True
    ) -> Tuple[int, int]:
        """Test a single configuration with optional early exit."""
        matches = 0
        total = 0
        mismatch_count = 0
        
        for i in range(self.num_iterations):
            M = random.choice(self.dims_m)
            N = random.choice(self.dims_n)
            K = random.choice(self.dims_k)
            dist_a = random.choice(self.distributions)
            dist_b = random.choice(self.distributions)
            
            if verbose:
                print(f"\rTest {i+1}/{self.num_iterations}: K={K:4}, A={dist_a:10}, B={dist_b:10} ... ", end="")
                sys.stdout.flush()
            
            # Generate data
            a = DataGenerator.get_random_tensor((M, K), dist_a)
            b = DataGenerator.get_random_tensor((N, K), dist_b)
            
            # Quantize
            a_gs = get_global_scale_fn(a)
            b_gs = get_global_scale_fn(b)
            alpha_val = 1.0 / (a_gs.item() * b_gs.item())
            alpha_tensor = torch.tensor([alpha_val], device="cuda", dtype=torch.float32)
            
            a_fp4, scale_a = ops_module.scaled_fp4_quant(a, a_gs)
            b_fp4, scale_b = ops_module.scaled_fp4_quant(b, b_gs)
            
            # Hardware execution
            real_output = ops_module.cutlass_scaled_fp4_mm(
                a_fp4, b_fp4, scale_a, scale_b, alpha_tensor, torch.float16
            )
            
            # Emulation with rounding strategies
            model_output = MMAEngine.emulation_scaled_fp4_mm(
                a_fp4, b_fp4, scale_a, scale_b, alpha_tensor, M, N, K,
                W_stage3=w_stage3, W_stage4=w_stage4,
                stage3_rounding=stage3_rounding,
                stage4_rounding=stage4_rounding
            )
            
            # Compare
            diff = (real_output.float() - model_output.float()).abs()
            match_mask = (real_output == model_output)
            diff[match_mask] = 0.0
            both_nan = real_output.isnan() & model_output.isnan()
            diff[both_nan] = 0.0
            diff[diff.isnan()] = float('inf')
            
            max_diff = diff.max().item()
            
            if max_diff == 0:
                matches += 1
            else:
                mismatch_count += 1
            total += 1
            
            # Early exit check
            if self.early_exit_mismatches is not None:
                if mismatch_count >= self.early_exit_mismatches:
                    if verbose:
                        print(f"EARLY EXIT: {mismatch_count} mismatches reached")
                    break
            
            if verbose:
                status = "SUCCESS" if max_diff == 0 else "MISMATCH"
                print(f"{status}", end="")
        
        if verbose:
            print()  # New line after progress
        
        return matches, total
