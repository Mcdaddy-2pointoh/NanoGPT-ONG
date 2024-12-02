from sandbox.attention.self_attention_fns import self_attention_v1, self_attention_v2, self_attention_v3, self_attention_v4
import torch

# Random tensor
B, T, C = 16, 8, 4
x = torch.randn(size=(B,T,C))
res1 = self_attention_v1(x)
res2 = self_attention_v2(x)
res3 = self_attention_v3(x)
res4 = self_attention_v4(x)

print(res4.shape)