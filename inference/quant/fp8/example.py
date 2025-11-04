import torch
import scaled_fp8_ops as ops
import fp8_pseudo_quantize
import mm


FLOAT8_E4M3_MIN=-448.0
FLOAT8_E4M3_MAX=448.0
# Input tensors
a = torch.randn(128, 128, dtype=torch.bfloat16, device="cuda")

# Create tensor b with the last dimension having first 32 elements as 1 and last 32 as 0
b = torch.randn(128, 128, dtype=torch.bfloat16, device="cuda")

a_cpy=a.clone().to("cpu")
b_cpy=b.T.clone().to("cpu")

a_q=torch.empty_like(a,dtype=torch.float8_e4m3fn, device="cuda")
b_q=torch.empty_like(b,dtype=torch.float8_e4m3fn, device="cuda")

a_q=torch.empty_like(a,dtype=torch.float8_e4m3fn)
b_q=torch.empty_like(b,dtype=torch.float8_e4m3fn)

M,N=a.shape
K=b.shape[1]
group_size=32
a_s=torch.empty(M,N//group_size,dtype=torch.float32,device=a_q.device)
b_s=torch.empty(N,K//group_size,dtype=torch.float32,device=b_q.device)


ops.per_token_group_quant_fp8(
    a, a_q, a_s, group_size, 1e-8, FLOAT8_E4M3_MIN, FLOAT8_E4M3_MAX, False
)
ops.per_token_group_quant_fp8(
    b, b_q, b_s, group_size, 1e-8, FLOAT8_E4M3_MIN, FLOAT8_E4M3_MAX, False
)
# print(a_q)

output1 = mm.fp8_mm(a_q,a_s,b_q,b_s)
output2 = fp8_pseudo_quantize.fp8_pseudo_quantize_groupwise(a)@fp8_pseudo_quantize.fp8_pseudo_quantize_groupwise(b).T
#fp8 gemm is not completed
# output3=torch.zeros(M,N,dtype=torch.bfloat16,device=a_q.device)
# ops.groupwise_gemm_fp8(a_q,a_s,b_q.T.contiguous(),b_s.T.contiguous(),output3,group_size)

print(output1)
print(output2)
# print(output3)


# 太慢了
# c=torch.zeros(a.shape[0],b.shape[1],dtype=torch.float32)
# for i in range(0, a.shape[0]):
#     for j in range(0, b.shape[1]):
#         for k in range(0, a.shape[1]):
#             c[i][j]+=a_cpy[i,k]*b_cpy[k,j]
# print(c)
