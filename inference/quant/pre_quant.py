import torch
import torch.nn as nn
from tqdm import tqdm
import gc

from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
from transformers.models.bloom.modeling_bloom import BloomForCausalLM
from transformers.models.opt.modeling_opt import OPTForCausalLM
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.models.bert.modeling_bert import BertForSequenceClassification
from transformers.models.mistral.modeling_mistral import MistralForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
from transformers.models.falcon.modeling_falcon import FalconForCausalLM

def set_op_by_name(layer, name, new_module):
    parts = name.split('.')
    mod = layer
    for p in parts[:-1]:
        mod = getattr(mod, p) if not p.isdigit() else mod[int(p)]
    setattr(mod, parts[-1], new_module)

def get_named_linears(module: nn.Module):
    # 只在“单个 block 内”找 Linear，因此不会碰到嵌入和 lm_head
    return {name: m for name, m in module.named_modules() if isinstance(m, nn.Linear)}

def get_blocks(model: nn.Module):
    if isinstance(model, LlamaForCausalLM):
        return model.model.layers
    elif isinstance(model, OPTForCausalLM):
        return model.model.decoder.layers
    elif isinstance(model, FalconForCausalLM):
        return model.transformer.h
    elif isinstance(model, GPT2LMHeadModel):
        return model.transformer.h
    elif isinstance(model, BloomForCausalLM):
        return model.transformer.h
    elif isinstance(model, BertForSequenceClassification):
        return model.bert.encoder.layer
    elif isinstance(model, MistralForCausalLM):
        return model.model.layers
    elif isinstance(model, Qwen2ForCausalLM):
        return model.model.layers
    else:
        raise NotImplementedError(type(model))

@torch.no_grad()
def replace_quant_linear(
    model: nn.Module,
    w_bit: int,
    a_bit: int,
    q_config: dict,
    use_zero_point: bool = False,
    init_only: bool = False,  # 保留形参以兼容你现在的调用
    nvfp: bool = False,
    fp8: bool = False,
    **kwargs,
):
    if nvfp:
        from .nvfp_quantizer import QuantLinear
    elif fp8:
        from .fp8_quantizer import QuantLinear
    else:
        from .quantizer import QuantLinear

    layers = get_blocks(model)
    group_size = q_config.get("q_group_size", 32)

    for i in tqdm(range(len(layers)), desc="replace quant linear...(init_only=%s)" % init_only):
        layer = layers[i]
        named_linears = get_named_linears(layer)

        targets = list(named_linears.items())

        for name, lin in targets:
            lin_gpu = lin.to("cuda")
            q_linear = QuantLinear(lin_gpu, w_bit, a_bit, group_size=group_size, use_zero_point=use_zero_point, mode=q_config["mode"])

            set_op_by_name(layer, name, q_linear)

            # 释放临时引用
            del lin_gpu, lin, q_linear
            torch.cuda.empty_cache()
    gc.collect()
