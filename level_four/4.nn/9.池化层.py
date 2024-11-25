import torch
import torch.nn as nn

# 长宽一致的池化，核尺寸为3x3，池化步长为2
m1 = nn.MaxPool2d(3, stride=2)

# 长宽不一致的池化
m2 = nn.MaxPool2d((3, 2), stride=(2, 1))

input = torch.randn(4, 3, 24, 24)
output1 = m1(input)
output2 = m2(input)

print("input.shape = ", input.shape)
print("output1.shape = ", output1.shape)
print("output2.shape = ", output2.shape)
