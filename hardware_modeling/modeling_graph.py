from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple
from modeling_dpu import DotProductUnit

@dataclass
class HACGNode:
    op: str
    stage: int
    attrs: Dict[str, Any] = field(default_factory=dict)

@dataclass
class HACGEdge:
    src: int
    dst: int
    attrs: Dict[str, Any] = field(default_factory=dict)

@dataclass
class HACG:
    nodes: List[HACGNode] = field(default_factory=list)
    edges: List[HACGEdge] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    def add_node(self, op: str, stage: int, **attrs: Any) -> int:
        self.nodes.append(HACGNode(op=op, stage=stage, attrs=dict(attrs)))
        return len(self.nodes) - 1

    def add_edge(self, src: int, dst: int, **attrs: Any) -> None:
        self.edges.append(HACGEdge(src=src, dst=dst, attrs=dict(attrs)))


def build_dpu_hacg_skeleton(dpu: "DotProductUnit") -> HACG:
    g = HACG(meta={
        "name": dpu.name,
        "input_a": asdict(dpu.input_precision_a),
        "input_b": asdict(dpu.input_precision_b),
        "mul": asdict(dpu.multiplication_precision),
        "acc": asdict(dpu.accumulator_precision),
        "topology": str(dpu.accumulation_topology),
        "spatial_k_size": dpu.spatial_k_size,
        "intermediate_truncation_bits": dpu.intermediate_truncation_bits,
        "output_rounding_mode": str(dpu.output_rounding_mode),
        "output_saturation_mode": str(dpu.output_saturation_mode),
    })

    # Stage 0: inputs
    a = g.add_node("input", stage=0, name="A")
    b = g.add_node("input", stage=0, name="B")

    # Stage 0: internal cast / flush-subnormal (占位节点或 attrs)
    a0, b0 = a, b
    if dpu.internal_cast_precision is not None:
        a0 = g.add_node("cast", stage=0, to=asdict(dpu.internal_cast_precision), which="A")
        b0 = g.add_node("cast", stage=0, to=asdict(dpu.internal_cast_precision), which="B")
        g.add_edge(a, a0); g.add_edge(b, b0)

    if dpu.flush_subnormal_on_input:
        fa = g.add_node("flush_subnormal", stage=0, which="A")
        fb = g.add_node("flush_subnormal", stage=0, which="B")
        g.add_edge(a0, fa); g.add_edge(b0, fb)
        a0, b0 = fa, fb

    # Stage 1: (optional) exponent align / special handle
    # 先不强行拆太细：留一个 preprocess 占位
    pre = g.add_node("preprocess", stage=1, note="exp_align/special_handle (optional)")
    g.add_edge(a0, pre, role="A")
    g.add_edge(b0, pre, role="B")

    # Stage 2: multiply
    mul = g.add_node("mul", stage=2, out_precision=asdict(dpu.multiplication_precision))
    g.add_edge(pre, mul)

    # Stage 3: reduction + accumulate
    red = g.add_node("reduce", stage=3,
                     topology=str(dpu.accumulation_topology),
                     spatial_k_size=dpu.spatial_k_size,
                     note="opaque tree until inferred")
    g.add_edge(mul, red)

    acc = g.add_node("accumulate", stage=3, acc_precision=asdict(dpu.accumulator_precision))
    g.add_edge(red, acc)

    # Stage 4: optional intermediate truncation
    cur = acc
    if dpu.intermediate_truncation_bits is not None:
        tr = g.add_node("truncate", stage=4, mantissa_bits=dpu.intermediate_truncation_bits)
        g.add_edge(cur, tr)
        cur = tr

    # Stage 4: round + saturate
    rnd = g.add_node("round", stage=4, mode=str(dpu.output_rounding_mode))
    g.add_edge(cur, rnd)

    sat = g.add_node("saturate", stage=4, mode=str(dpu.output_saturation_mode))
    g.add_edge(rnd, sat)

    # Stage 5: output
    out = g.add_node("output", stage=5, name="D")
    g.add_edge(sat, out)

    return g