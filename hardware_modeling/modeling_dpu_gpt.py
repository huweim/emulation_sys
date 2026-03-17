from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import math
import hashlib


# =========================
# 0) 基础枚举/规格
# =========================

class RoundingMode(str, Enum):
    RNE = "round_to_nearest_even"
    RTZ = "round_toward_zero"
    RTP = "round_toward_plus_inf"
    RTN = "round_toward_minus_inf"
    SR  = "stochastic_rounding"   # 可用于 training / 某些路径

class SaturationMode(str, Enum):
    NONE = "none"
    SAT  = "saturate"             # 超界 clamp 到 max/min

class NumericKind(str, Enum):
    FP = "fp"
    INT = "int"
    FXP = "fxp"  # fixed-point

@dataclass(frozen=True)
class DTypeSpec:
    """描述一种数值类型（可含 table/codebook 等）。"""
    kind: NumericKind
    bits: int
    name: str                      # e.g., "fp16", "fp8_e4m3", "nvfp4_e2m1", "int8"
    encoding: Optional[str] = None # e.g., "e4m3", "e5m2", "e2m1", "symmetric_int"
    # 可选：用于 table-based FP4 (e2m1) 之类
    codebook: Optional[Tuple[float, ...]] = None

@dataclass(frozen=True)
class QuantOpSpec:
    """描述一次量化/截断/舍入动作发生在何处，用什么规则做。"""
    target_dtype: DTypeSpec
    rounding: RoundingMode = RoundingMode.RNE
    saturation: SaturationMode = SaturationMode.NONE
    # 可选：scale/zero_point 等（例如 NVFP4 的 block scale / tensor scale）
    scale_ref: Optional[str] = None     # 指向外部 scale 的名字（如 "block_scale_fp8"）
    zero_point: Optional[int] = None    # for asymmetric int quant (如需要)


# =========================
# 1) Reduction Tree / Accumulation 结构（DPU 的核心）
# =========================

class TreeTopology(str, Enum):
    """用来表达 accumulation tree 的拓扑类型。"""
    LINEAR = "linear"          # (((a+b)+c)+d)...
    PAIRWISE = "pairwise"      # 分治二叉树
    CSA = "carry_save_tree"    # CSA/Wallace/Dadda 风格（可抽象）
    VENDOR_OPAQUE = "vendor_opaque"  # 未公开但确定（architecture-dependent）
    NONDETERMINISTIC = "nondeterministic"  # 真正的 run-to-run 不确定（一般尽量别用）

@dataclass
class ReductionTreeSpec:
    topology: TreeTopology
    arity: int = 2                      # 二叉/多叉 reduction
    depth_hint: Optional[int] = None    # 若未知可不填
    deterministic: bool = True          # 语义上是否 deterministic
    note: str = ""                      # 备注：来自测量/文档/推断

    # 允许“黑盒”树：你后续可以把 topology 从 VENDOR_OPAQUE -> PAIRWISE/CSA/...
    is_black_box: bool = False
    exploration_domain: Optional[List["ReductionTreeSpec"]] = None


@dataclass
class AccumulatorSpec:
    """描述累加寄存器/内部累加精度的规格。"""
    dtype: DTypeSpec
    # 累加过程中是否会出现“内部截断点”（例如 partial sum 存回更低精度寄存器）
    internal_truncations: List[QuantOpSpec] = field(default_factory=list)
    # 是否有 FMA pipeline 中间舍入点
    intermediate_rounding_points: List[str] = field(default_factory=list)  # e.g., ["after_mul", "after_partial_sum"]


# =========================
# 2) DPU Spec：把 Dot Product 当作可组合 primitive
# =========================

class BlackBoxField:
    """
    一个占位符：表示某个属性当前是黑盒，需要探索/测量/反推。
    exploration_domain 可以是离散候选列表（枚举），也可以由外部提供搜索策略。
    """
    def __init__(self, name: str, exploration_domain: Optional[List[Any]] = None, note: str = ""):
        self.name = name
        self.exploration_domain = exploration_domain or []
        self.note = note

    def __repr__(self) -> str:
        return f"BlackBoxField(name={self.name}, domain={len(self.exploration_domain)} candidates, note={self.note})"


@dataclass
class DotProductUnitSpec:
    """
    一个 Dot Product Unit（如 Tensor Core 的 MMA micro-op，或 CUBE 的 MAC 阵列子块）规格。
    它是 HACG 的“primitive building block”。
    """
    # --- 输入/输出规格 ---
    a_dtype: DTypeSpec
    b_dtype: DTypeSpec
    c_dtype: DTypeSpec                 # accumulator input (if any)
    out_dtype: DTypeSpec               # output dtype (可能等于 c_dtype 或更低)
    accumulator: AccumulatorSpec

    # --- 计算形态（tile/micro-kernel 语义，不必等同 ISA，但要能映射到图）---
    m: int                              # MMA tile shape or micro-tile
    n: int
    k: int                              # dot-product length per op

    # --- 关键“硬件语义点”（HACG nodes 需要编码的东西）---
    mul_quant: Optional[QuantOpSpec] = None        # 乘法后是否 quant/trunc
    add_quant: Optional[QuantOpSpec] = None        # 每次加法/partial sum 后是否 quant/trunc
    out_quant: Optional[QuantOpSpec] = None        # 写回输出前的 quant/trunc

    reduction_tree: Union[ReductionTreeSpec, BlackBoxField, None] = None

    # --- 数据流/并行结构（用于构建 HACG 的拓扑与映射）---
    lanes: int = 1                      # 并行 lanes 数（warp-level / array-level 的抽象）
    pipeline_stages: int = 1
    dataflow: Union[str, BlackBoxField] = "unknown"  # e.g., "systolic", "SIMT_tree", "mma_fragment"

    # --- scale/metadata（和你 microscaling primitive 思路是同构的）---
    # 比如 NVFP4: block_scale_fp8, tensor_scale_fp32 等，可以在这里声明符号
    scale_symbols: Dict[str, DTypeSpec] = field(default_factory=dict)
    metadata_symbols: Dict[str, Any] = field(default_factory=dict)

    # --- 可复现版本信息（便于“语义版本化”）---
    vendor: str = "unknown"
    arch: str = "unknown"               # e.g., "sm90", "sm100", "ascend910b"
    isa_hint: Optional[str] = None      # e.g., "wgmma", "mma.sync", "CUBE"
    note: str = ""

    def semantic_fingerprint(self) -> str:
        """
        给这个 DPU 的“语义”算一个指纹（用于 cache / 复现 / regression）。
        黑盒字段会影响指纹稳定性——你可以在 finalize 后再用。
        """
        payload = repr(self).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:16]

    # =========================
    # 2.1 合法性检查（你后续可以越写越严格）
    # =========================
    def validate(self) -> None:
        assert self.m > 0 and self.n > 0 and self.k > 0, "M/N/K must be positive"
        assert self.lanes > 0, "lanes must be positive"
        assert self.pipeline_stages > 0, "pipeline_stages must be positive"
        # 简单 sanity：累加 dtype bits 一般 >= out dtype bits（不总成立，但常见）
        if self.accumulator.dtype.bits < self.out_dtype.bits:
            # 不 raise，先警告式处理
            pass

    # =========================
    # 2.2 黑盒字段填充/枚举
    # =========================
    def is_complete(self) -> bool:
        """判断是否可以生成 complete HACG（无黑盒）。"""
        if isinstance(self.reduction_tree, BlackBoxField):
            return False
        if isinstance(self.dataflow, BlackBoxField):
            return False
        return True

    def enumerate_candidates(self) -> List["DotProductUnitSpec"]:
        """
        若包含黑盒字段，按离散 domain 枚举所有候选 DPU spec（用于搜索/反推）。
        """
        specs = [self]

        # enumerate reduction_tree
        if isinstance(self.reduction_tree, BlackBoxField) and self.reduction_tree.exploration_domain:
            new_specs = []
            for s in specs:
                for cand in self.reduction_tree.exploration_domain:
                    ss = _clone_spec(s)
                    ss.reduction_tree = cand
                    new_specs.append(ss)
            specs = new_specs

        # enumerate dataflow
        if isinstance(self.dataflow, BlackBoxField) and self.dataflow.exploration_domain:
            new_specs = []
            for s in specs:
                for cand in self.dataflow.exploration_domain:
                    ss = _clone_spec(s)
                    ss.dataflow = cand
                    new_specs.append(ss)
            specs = new_specs

        return specs


# 一个非常轻量的 clone（只用于枚举示例；你也可以改成 copy.deepcopy）
def _clone_spec(spec: DotProductUnitSpec) -> DotProductUnitSpec:
    return DotProductUnitSpec(**{k: getattr(spec, k) for k in spec.__dataclass_fields__.keys()})


# =========================
# 3) HACG 的最小图结构（先占位）
# =========================

@dataclass
class HACGNode:
    op: str                              # "mul", "add", "quant", "scale", "reduce"
    dtype_in: Optional[DTypeSpec] = None
    dtype_out: Optional[DTypeSpec] = None
    attrs: Dict[str, Any] = field(default_factory=dict)

@dataclass
class HACGEdge:
    src: int
    dst: int
    attrs: Dict[str, Any] = field(default_factory=dict)

@dataclass
class HardwareAwareComputationGraph:
    nodes: List[HACGNode] = field(default_factory=list)
    edges: List[HACGEdge] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    def add_node(self, node: HACGNode) -> int:
        self.nodes.append(node)
        return len(self.nodes) - 1

    def add_edge(self, src: int, dst: int, **attrs: Any) -> None:
        self.edges.append(HACGEdge(src=src, dst=dst, attrs=dict(attrs)))


# =========================
# 4) 从 DPU spec -> HACG（先做“结构化展开”）
# =========================

def dpu_to_hacg(spec: DotProductUnitSpec) -> HardwareAwareComputationGraph:
    """
    把 DotProductUnitSpec 展开成一个 HACG（结构 + 语义点）。
    注意：这是第一版“结构化展开”，真正的 tree 展开/映射你后续可加细节。
    """
    spec.validate()
    if not spec.is_complete():
        raise ValueError(f"DPU spec not complete: reduction_tree/dataflow still black-box. spec={spec}")

    g = HardwareAwareComputationGraph(meta={
        "vendor": spec.vendor, "arch": spec.arch, "isa_hint": spec.isa_hint,
        "m": spec.m, "n": spec.n, "k": spec.k, "lanes": spec.lanes,
        "fingerprint": spec.semantic_fingerprint(),
    })

    # 1) 输入（抽象为 “A,B,C fragments”）
    a_id = g.add_node(HACGNode(op="input", dtype_out=spec.a_dtype, attrs={"name": "A"}))
    b_id = g.add_node(HACGNode(op="input", dtype_out=spec.b_dtype, attrs={"name": "B"}))
    c_id = g.add_node(HACGNode(op="input", dtype_out=spec.c_dtype, attrs={"name": "C"}))

    # 2) mul（可插 mul_quant）
    mul_id = g.add_node(HACGNode(op="mul", dtype_in=None, dtype_out=spec.accumulator.dtype,
                                 attrs={"k": spec.k, "tile": (spec.m, spec.n)}))
    g.add_edge(a_id, mul_id, role="lhs")
    g.add_edge(b_id, mul_id, role="rhs")

    if spec.mul_quant is not None:
        q = spec.mul_quant
        qmul_id = g.add_node(HACGNode(op="quant", dtype_in=spec.accumulator.dtype, dtype_out=q.target_dtype,
                                      attrs={"rounding": q.rounding.value, "sat": q.saturation.value,
                                             "scale_ref": q.scale_ref}))
        g.add_edge(mul_id, qmul_id)
        current = qmul_id
        acc_dtype = q.target_dtype
    else:
        current = mul_id
        acc_dtype = spec.accumulator.dtype

    # 3) reduction / accumulation tree（先以一个 reduce 节点占位）
    rt: ReductionTreeSpec = spec.reduction_tree  # type: ignore
    reduce_id = g.add_node(HACGNode(op="reduce", dtype_in=acc_dtype, dtype_out=spec.accumulator.dtype,
                                    attrs={"topology": rt.topology.value,
                                           "arity": rt.arity,
                                           "depth_hint": rt.depth_hint,
                                           "deterministic": rt.deterministic,
                                           "note": rt.note}))
    g.add_edge(current, reduce_id)

    # 4) add with C (accumulate)
    add_id = g.add_node(HACGNode(op="add", dtype_in=spec.accumulator.dtype, dtype_out=spec.accumulator.dtype,
                                 attrs={"note": "accumulate with C"}))
    g.add_edge(reduce_id, add_id, role="partial_sum")
    g.add_edge(c_id, add_id, role="C")

    # 5) internal truncation points（每个都插一个 quant 节点）
    cur = add_id
    for q in spec.accumulator.internal_truncations:
        q_id = g.add_node(HACGNode(op="quant",
                                   dtype_in=spec.accumulator.dtype,
                                   dtype_out=q.target_dtype,
                                   attrs={"rounding": q.rounding.value, "sat": q.saturation.value,
                                          "scale_ref": q.scale_ref}))
        g.add_edge(cur, q_id)
        cur = q_id

    # 6) output quant
    if spec.out_quant is not None:
        q = spec.out_quant
        outq_id = g.add_node(HACGNode(op="quant",
                                      dtype_in=spec.accumulator.dtype,
                                      dtype_out=q.target_dtype,
                                      attrs={"rounding": q.rounding.value, "sat": q.saturation.value,
                                             "scale_ref": q.scale_ref}))
        g.add_edge(cur, outq_id)
        cur = outq_id

    # 7) output
    out_id = g.add_node(HACGNode(op="output", dtype_in=None, dtype_out=spec.out_dtype, attrs={"name": "D"}))
    g.add_edge(cur, out_id)

    return g


# =========================
# 5) 使用示例：先做一个“带黑盒”的 TensorCore DPU，再枚举候选
# =========================

FP16 = DTypeSpec(kind=NumericKind.FP, bits=16, name="fp16", encoding="e5m10")
FP32 = DTypeSpec(kind=NumericKind.FP, bits=32, name="fp32", encoding="e8m23")
FP8E4M3 = DTypeSpec(kind=NumericKind.FP, bits=8, name="fp8_e4m3", encoding="e4m3")
NVFP4 = DTypeSpec(kind=NumericKind.FP, bits=4, name="nvfp4_e2m1", encoding="e2m1")

# 一个黑盒 reduction tree：候选里列几个你关心的
rt_blackbox = BlackBoxField(
    name="reduction_tree",
    exploration_domain=[
        ReductionTreeSpec(topology=TreeTopology.PAIRWISE, arity=2, depth_hint=None, deterministic=True, note="candidate"),
        ReductionTreeSpec(topology=TreeTopology.CSA, arity=2, depth_hint=None, deterministic=True, note="candidate"),
        ReductionTreeSpec(topology=TreeTopology.VENDOR_OPAQUE, arity=2, deterministic=True, note="unknown exact tree"),
    ],
    note="To be inferred by microbench / reverse engineering",
)

dpu = DotProductUnitSpec(
    a_dtype=NVFP4, b_dtype=NVFP4, c_dtype=FP16, out_dtype=FP16,
    accumulator=AccumulatorSpec(dtype=FP16, internal_truncations=[]),
    m=16, n=16, k=16,
    mul_quant=None,
    add_quant=None,
    out_quant=QuantOpSpec(target_dtype=FP16, rounding=RoundingMode.RNE),
    reduction_tree=rt_blackbox,
    lanes=8,
    pipeline_stages=2,
    dataflow=BlackBoxField("dataflow", exploration_domain=["mma_fragment", "systolic", "simt_tree"]),
    vendor="nvidia",
    arch="unknown",
    isa_hint="mma/wgmma",
    note="Skeleton spec for NVFP4-like path; fill scale semantics later."
)

# 你后续会：枚举候选 -> 用 microbench 过滤 -> 选出 best -> finalize -> dpu_to_hacg()
# candidates = dpu.enumerate_candidates()
# spec_final = candidates[0] ...  (由你的实验决定)
# graph = dpu_to_hacg(spec_final)