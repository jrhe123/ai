import torch
import torch.nn as nn

# 使用方形卷积核，以及相同的步长
m = nn.Conv2d(16, 33, 3, stride=2)

# 使用非方形的卷积核，以及非对称的步长和补零
m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))

# 使用非方形的卷积核，以及非对称的步长、补零和膨胀系数
m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))

input = torch.randn(20, 16, 50, 100)
output = m(input)
print(output.shape)
