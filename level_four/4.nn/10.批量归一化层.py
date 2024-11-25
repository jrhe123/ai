import torch
import torch.nn as nn

# 批量归一化层（具有可学习参数）
m_learnable = nn.BatchNorm2d(100)

# 批量归一化层（不具有可学习参数）
m_non_learnable = nn.BatchNorm2d(100, affine=False)

# 随机生成输入数据
input = torch.randn(20, 100, 35, 45)

# 应用具有可学习参数的批量归一化层
output_learnable = m_learnable(input)

# 应用不具有可学习参数的批量归一化层
output_non_learnable = m_non_learnable(input)

print("input.shape = ", input.shape)
print("output_learnable.shape = ", output_learnable.shape)
print("output_non_learnable.shape = ", output_non_learnable.shape)
