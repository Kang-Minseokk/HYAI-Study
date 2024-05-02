import torch
x = torch.tensor([2, 1])
w1 = torch.tensor([[3, 2, -4], [2, -3, 1]])
b1 = 1
w2 = torch.tensor([[-1, 1], [1, 2], [3, 1]])
b2 = -1

h_preact = torch.matmul(x, w1) + b1
h = torch.nn.functional.relu(h_preact)
y = torch.matmul(h, w2) + b2

print(h_preact)
print(h)
print(y)

