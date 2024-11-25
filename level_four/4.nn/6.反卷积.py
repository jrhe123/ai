import torch
import torch.nn as nn

# 使用长宽一致的卷积核以及相同的步长
m = nn.ConvTranspose2d(16, 33, 3, stride=2)

# 使用长宽不一致的卷积核、步长，以及补零
m = nn.ConvTranspose2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))

input = torch.randn(20, 16, 50, 100)
output = m(input)

# 可以直接指明输出的尺寸大小
input = torch.randn(1, 16, 12, 12)
downsample = nn.Conv2d(16, 16, 3, stride=2, padding=1)
upsample = nn.ConvTranspose2d(16, 16, 3, stride=2, padding=1)

h = downsample(input)
print(h.size())

output = upsample(h, output_size=input.size())
print(output.size())
