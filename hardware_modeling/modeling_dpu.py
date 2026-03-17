from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Optional, Any, Dict

# --- 0. 基础工具 (兼容性支持) ---
# 如果 Python 版本 >= 3.11，可以直接从 enum 导入 StrEnum
# 这里手写一个以支持旧版本，确保行为一致
class StrEnum(str, Enum):
    def __str__(self):
        return self.value
    def __repr__(self):
        return f"'{self.value}'"

# --- 1. 基础原语定义 (String Enums) ---

class NumericalDomain(StrEnum):
    INT = "int"               # Integer (INT8, INT4)
    FP  = "fp"                # Floating Point (FP16, BF16, FP32)
    MX  = "microscaling"      # Block Floating Point / Microscaling (MXFP4, NVFP4)
    FXP = "fixed_point"       # Fixed Point (DSP style)

class RoundingMode(StrEnum):
    RNE = "rne"               # Round to Nearest Even (Standard IEEE 754)
    RZ  = "rz"                # Round towards Zero (Truncate)
    RP  = "rp"                # Round towards Plus Infinity (Ceil)
    RM  = "rm"                # Round towards Minus Infinity (Floor)
    SR  = "stochastic"        # Stochastic Rounding (Training / Special HW)
    UNK = "unknown"           # [Black Box] 待探测

class AccumulationOrder(StrEnum):
    SERIAL  = "serial"        # 朴素串行累加 (Iterative)
    TREE    = "tree_reduction"# 树状归约 (常见于 Tensor Core)
    SPATIAL = "spatial_array" # 脉动阵列流动累加 (常见于 TPU/Ascend)
    BLOCK   = "block_parallel"# 分块并行 (Split-K)
    UNK     = "unknown"       # [Black Box] 待探测

class SaturationMode(StrEnum):
    WRAP = "wrap"             # 溢出回绕
    SAT  = "saturate"         # 饱和 (Clamp to max)

class ReductionStructure(StrEnum):
    TREE   = "tree"
    SERIAL = "serial"
    SPATIAL_ARRAY = "spatial_array"
    BLOCK_PARALLEL = "block_parallel"
    UNK = "unknown"

# --- 2. 精度描述原语 ---

@dataclass
class PrecisionPrimitive:
    """
    描述一个数值格式的物理属性。
    例如: E4M3 (FP8), INT8, E2M1 (NVFP4)
    """
    domain: NumericalDomain
    total_bits: int
    mantissa_bits: int
    exponent_bits: int
    bias: Optional[int] = None # 如果不填，默认为标准 bias (2^(E-1)-1)
    
    def __repr__(self):
        # 打印出来像: FP_E4M3_B8
        return f"{self.domain}_E{self.exponent_bits}M{self.mantissa_bits}B{self.total_bits}"

@dataclass
class InternalPrecisionPrimitive:
    """
    描述一个中间计算精度的物理属性。
    例如: INT32 累加器，FP16 乘法结果
    """
    internal_exponent_bits: int
    internal_mantissa_bits: int


# --- 3. 核心 Dot Product Unit (DPU) 建模 ---

@dataclass
class DotProductUnitAttribute:
    """
    Hardware-Aware Dot Product Unit (DPU) Model.
    它是 Computation Graph 中的核心计算节点，承载了所有关于数值行为的元数据。
    """
    name: str  # e.g., "NVIDIA_H100_TC_FP8", "Ascend_CUBE_INT8"
    
    # === Input Stage ===
    input_precision_a: PrecisionPrimitive
    input_precision_b: PrecisionPrimitive
    
    # [Black Box]: 隐式强转。宣称 INT4 但内部可能转为 INT8 计算。
    internal_cast_precision: Optional[PrecisionPrimitive] = None 

    # === Multiplication Stage (M) ===
    # 乘积的精度。通常情况下，我们假设乘法是 lossless，尤其考虑到，我们探索的对象是 low-bit GEMM.
    multiplication_precision: InternalPrecisionPrimitive = field(
        default_factory=lambda: InternalPrecisionPrimitive(10, 25)
    )
    
    # [Black Box]: Subnormal 处理 (Flush-to-Zero?)
    flush_subnormal_on_input: bool = True 

    # === Accumulation Stage (A) - 核心误差源 ===
    
    # 累加器精度。通常是 FP32，但有些 NPU 是 FP16 或 INT32
    accumulator_precision: InternalPrecisionPrimitive = field(
        default_factory=lambda: InternalPrecisionPrimitive(0, 25)
    )
    
    # [Black Box]: 累加拓扑。
    # 不同的加法树深度会导致截断误差显著不同。
    accumulation_topology: AccumulationOrder = AccumulationOrder.UNK
    
    # [Black Box]: 空间归约大小 (K-dimension spatial size)。
    # 决定了一次原子操作累加了多少个元素 (16, 32, 64?)
    spatial_k_size: int = 16 
    
    # [Black Box]: 中间截断位宽。
    # 极其重要的未公开特性。加法树中间是否为了省面积把 23bit 尾数截断成了 16bit？
    intermediate_truncation_bits: Optional[int] = None

    # === Post-Processing Stage ===
    
    output_rounding_mode: RoundingMode = RoundingMode.RNE
    output_saturation_mode: SaturationMode = SaturationMode.SAT

    def is_fully_specified(self) -> bool:
        """检查模型中是否还包含未知的黑盒属性"""
        return (self.accumulation_topology != AccumulationOrder.UNK) and \
               (self.output_rounding_mode != RoundingMode.UNK)

    def to_dict(self) -> Dict[str, Any]:
        """
        序列化为字典，方便导出为 JSON/YAML 配置。
        由于使用了 StrEnum，结果会非常干净。
        """
        # dataclasses.asdict 会递归地把 dataclass 转 dict，但 Enum 保持原样
        # 因为我们的 Enum 是 str 子类，JSON dump 时会自动处理
        return asdict(self)

@dataclass
class DotProductUnitReductionTree:
    """
    描述 DPU 内部 dot-product 的归约结构（数值行为层面的 abstraction）。
    我们不关心它是不是 Wallace/CSA，只关心：
      - 一次 dot 涉及多少个乘积（num_nodes）
      - 归约结构是什么（tree/serial/spatial）
      - 每一级是否有中间截断/舍入（数值行为核心）
    """

    # Step 1: 确定 nodes_num in Reduction Tree
    # First, does the smallest $K_{min}$ in MMA is equal to the number of nodes $N$ in reduction tree?
    # If $K_{min}$ > $N$, which means $K_{min}$ needs to be partitioned and reduction.
    # 因此，这个还是比较好验证的。实际上，我认为可以初步认为在 NVIDIA GPU 中 $K_{min}$ == N
    # 啊，也不好说，从 FPRev 的图片来看，并不一定是这样的，但还是可以沿用 FPRev 的思路来验证。
    num_nodes: Optional[int] = None   # None = unknown；num_nodes=1 => FMA

    # If num_nodes = 1, it equals to a FMA unit.
    # If num_nodes > 1, 我们需要确认粒度，这是我们第一个需要验证的 UNK information.   --->   We provide a API to build it.

    # Second, for a reduction tree, does every nodes has the same intermediate accumulation bit width? 
    # It is mostly true, from a hardware design view, but it can also be verified through some cases.
    identical_nodes: bool = True  # 是否所有节点都相同（同样的乘法精度和累加精度）

    # 如果只是考虑 GPU 的 Wallace Tree，那么比较简单；如果他是 Systolic Array，而一个 MAC 单元内部又是 Wallace Tree 支持 vector reduction，那这样的话最终的数值就和 data layout 高度相关，这样情况会变得更加复杂；所以最终可能只会考虑 GPU 的简单情况，或者说至少这里要有一定的已知信息，而不是能有有一个 API，直接 return 他的 topology.
    structure: ReductionStructure = ReductionStructure.UNK

    # [Black Box] 如果你认为 truncation 与树层级绑定，建议放 topology
    # simplest: 一个整体截断位宽（后续可升级为 per-level list）
    intermediate_truncation_bits: Optional[int] = None

    # [optional] 你后续要精细建模 Wallace/CSA 树时会用到；比如普通 MMA，实际上就是1级 Wallace Tree，而 NVFP4 则是2级 Wallace Tree.
    # 也不对，我们实际不关心他是不是 Wallace Tree，我们关心的是整个 Reduction Tree 的中间位宽，是不是每一级 Intermediate Bits 都一样（可以假设都一样，可以做一个验证）
    # Wallace Tree 是考虑到 latency 和 area 之后的一个 implementation choice，我们实际不关心这个，我们关心的是最终的数值行为。
    tree_arity: int = 2
    tree_depth: Optional[int] = None
    note: str = ""

    # NOTE: 想到，还得添加一个，operation to results
    # 比如正常来说，results 就是和其他的 res 相加；对于对于 NVFP4，等于 res 还有一个浮点乘法的过程
    # 可能有点混乱啊，只是写到这里了，所以先记录一下

# --- 4. 使用示例 ---

if __name__ == "__main__":
    import json

    # 实例化：模拟 B200 的 NVFP4 Tensor Core
    b200_nvfp4_tc = DotProductUnitAttribute(
        name="B200_NVFP4_TensorCore",
        
        # 输入：FP4
        input_precision_a=PrecisionPrimitive(NumericalDomain.FP, 4, 2, 1),
        input_precision_b=PrecisionPrimitive(NumericalDomain.FP, 4, 2, 1),
        
        # 乘法输出：FP16
        multiplication_precision=InternalPrecisionPrimitive(10, 25),
        
        # 累加器：FP32
        accumulator_precision=InternalPrecisionPrimitive(0, 25),
        
        # 待探测属性
        accumulation_topology=AccumulationOrder.UNK, 
        output_rounding_mode=RoundingMode.RNE,
        
        spatial_k_size=32 
    )

    print(f"=== Model: {b200_nvfp4_tc.name} ===")
    print(f"Is Fully Specified? {b200_nvfp4_tc.is_fully_specified()}")
    
    # 演示序列化效果
    print("\n=== JSON Export (Graph Node Definition) ===")
    print(json.dumps(b200_nvfp4_tc.to_dict(), indent=2))