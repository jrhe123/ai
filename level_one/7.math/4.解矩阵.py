import numpy as np

# 定义矩阵
A = np.array([[1, 2], [3, 4]])
b = np.array([5, 6])

# 解线性方程组
x = np.linalg.solve(A, b)

print("解为：", x)