import torch
m = torch.nn.Dropout(p=0.2)
input = torch.randn(1, 10)
output = m(input)

print(output)