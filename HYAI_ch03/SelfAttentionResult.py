# 어텐션 스코어를 만드는 코드
# 어텐션 스코어 : 주어진 쿼리(Query)와 키(Key) 벡터 간의 유사도를 나타내는 값!
import torch

x = torch.tensor([
    [1.0, 0.0, 1.0, 0.0],
    [0.0, 2.0, 0.0, 2.0],
    [1.0, 1.0, 1.0, 1.0],
])
w_query = torch.tensor([
    [1.0, 0.0, 1.0],
    [1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.0, 1.0, 1.0]
])
w_key = torch.tensor([
    [0.0, 0.0, 1.0],
    [1.0, 1.0, 0.0],
    [0.0, 1.0, 0.0],
    [1.0, 1.0, 0.0]
])
w_value = torch.tensor([
    [0.0, 2.0, 0.0],
    [0.0, 3.0, 0.0],
    [1.0, 0.0, 3.0],
    [1.0, 1.0, 0.0]
])

keys = torch.matmul(x, w_key)
querys = torch.matmul(x, w_query)
values = torch.matmul(x, w_value)
attn_scores = torch.matmul(querys, keys.T)
print(attn_scores)

# 소프트 맥스를 적용한 결과를 나타낸다.
import numpy as np
from torch.nn.functional import softmax
key_dim_sqrt = np.sqrt(keys.shape[-1])
attn_probs = softmax(attn_scores / key_dim_sqrt, dim=1)
print(attn_probs)

# 소프트맥스 확률과 밸류 벡터을 가중하는 과정을 수행한다.
weighted_values = torch.matmul(attn_probs, values)
print(attn_probs)

