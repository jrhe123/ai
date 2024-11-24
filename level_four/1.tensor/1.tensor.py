"""
pip3 install torch torchvision torchaudio
"""

import numpy as np
import torch

# from random
tensor = torch.rand(3, 4)
print(f"shape of tensor: {tensor.shape}")
print(f"datatype of tensor: {tensor.dtype}")
print(f"device of tensor stored`: {tensor.device}")

# from 2d array
data = [
    [1, 2],
    [3, 4],
]
x_data = torch.tensor(data)
print(f"tensor: {x_data}")

# from np
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
print(f"tensor: {x_np}")

# 保持了x_np的数据属性与形状
x_ones = torch.ones_like(x_np)
print(f"tensor: {x_ones}")

# 保持了x_np的形状，重新定义数据属性
x_rand = torch.rand_like(x_np, dtype=torch.float)
print(f"tensor: {x_rand}")

# 创建一个未初始化的tensor
shape = (2, 3)
x_with_random_init = torch.rand(shape)
print(f"tensor: {x_with_random_init}")

x_with_random_init = torch.ones(shape)
print(f"tensor: {x_with_random_init}")

x_with_random_init = torch.zeros(shape)
print(f"tensor: {x_with_random_init}")

# 转移device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

sample_tensor = x_with_random_init
device_tensor = sample_tensor.to(device)
print(f"device tensor: {device_tensor}")

cpu_tensor = sample_tensor.to("cpu")
print(f"cpu tensor: {cpu_tensor}")
