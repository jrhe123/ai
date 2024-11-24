import torch

tensor = torch.randn(4, 4)

print(tensor)
print(f"first row: {tensor[0]}")
print(f"first column: {tensor[:, 0]}")
print(f"last column: {tensor[..., -1]}")

# set first column to zeros
tensor[:, 0] = 0
print(tensor)


# 乘法运算
# tensor @ tensor
tensor_1 = torch.tensor([[1, 2], [3, 4]])
result = tensor_1 @ tensor_1
print(result)

# tensor element-wise multiplication
result = tensor_1 * tensor_1
print(result)

# tensor element-wise add
result = tensor_1 + tensor_1
print(result)

# tensor aggregation
agg = tensor_1.sum()
print(agg)
agg_item = agg.item()
print(agg_item)


# 拼接两个tensor
tensor_2 = torch.tensor([[5, 6], [7, 8]])
"""
tensor([[1, 2],
        [3, 4],
        [5, 6],
        [7, 8]])
"""
result = torch.cat((tensor_1, tensor_2), dim=0)
print(result)

"""
tensor([[1, 2, 5, 6],
        [3, 4, 7, 8]])
"""
result = torch.cat((tensor_1, tensor_2), dim=1)
print(result)
