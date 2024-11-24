import torch
import torch.nn as nn

# 激活函数
# 1. sigmod
m = nn.Sigmoid()
input = torch.randn(5)
output = m(input)
print(input)
print(output)

# 2. relu
m = nn.ReLU()
input = torch.randn(5)
output = m(input)
print(input)
print(output)

# 3. softmax
m = nn.Softmax(dim=1)
input = torch.randn(4, 3)
output = m(input)
print(input)
print(output)

# 4. dropout
m = nn.Dropout(p=0.2)
input = torch.randn(4, 3)
output = m(input)
print(input)
print(output)
