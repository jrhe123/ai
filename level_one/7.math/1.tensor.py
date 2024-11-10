import numpy as np

# ======================
# 示例 1: 向量的 L2 范数
# ======================
# 创建一个一维数组（向量）
vector = np.array([1, 2, 3, 4, 5])

# 计算向量的 L2 范数
# np.linalg.norm(vector) 会计算向量元素平方和的平方根
vector_norm = np.linalg.norm(vector)

# 输出向量的值和 L2 范数
print("Vector:\n", vector)
print("Vector L2 Norm:", vector_norm)
print("="*30)  # 分隔符

# ======================
# 示例 2: 矩阵的 L2 范数
# ======================
# 创建一个二维数组（矩阵）
matrix = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

# 计算矩阵的 L2 范数
# 对于矩阵，这也是所有元素平方和的平方根（Frobenius 范数）
matrix_norm = np.linalg.norm(matrix)

# 输出矩阵的值和 L2 范数
print("Matrix:\n", matrix)
print("Matrix L2 Norm:", matrix_norm)
print("="*30)  # 分隔符

# ======================
# 示例 3: 张量的 L2 范数
# ======================
# 创建一个三维数组（张量）
# 张量的形状为 (3, 2, 2)，包含三个二维数组，每个二维数组包含两个元素的数组
tensor = np.array([
    [[1, 2], [3, 4]],   # 第一个二维数组
    [[5, 6], [7, 8]],   # 第二个二维数组
    [[9, 10], [11, 12]] # 第三个二维数组
])

# 计算张量的 L2 范数
# np.linalg.norm(tensor) 会将张量中的所有元素展平成一个向量，并计算其 L2 范数
# L2 范数是所有元素的平方和的平方根
tensor_norm = np.linalg.norm(tensor)

# 输出张量的值和 L2 范数
print("Tensor:\n", tensor)
print("Tensor L2 Norm:", tensor_norm)
