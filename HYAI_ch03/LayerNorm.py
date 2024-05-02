# 레이어를 정규화해 보자.

import torch
input = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
m = torch.nn.LayerNorm(input.shape[-1])
output = m(input)

print(output)
