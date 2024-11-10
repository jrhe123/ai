import numpy as np

# 生成两组随机数据
data1 = np.random.randn(1000)  # 正态分布随机数据1，包含1000个数据点
data2 = np.random.randn(1000)  # 正态分布随机数据2，包含1000个数据点

# 计算期望（平均值）
expectation1 = np.mean(data1)
expectation2 = np.mean(data2)

# 计算方差
variance1 = np.var(data1)
variance2 = np.var(data2)

# 计算协方差矩阵
covariance_matrix = np.cov(data1, data2)

print(f"期望值 Data1: {expectation1}, Data2: {expectation2}")
print(f"方差 Data1: {variance1}, Data2: {variance2}")
print(f"协方差矩阵:\n{covariance_matrix}")

# 从协方差矩阵中提取 data1 和 data2 的协方差
covariance = covariance_matrix[0, 1]
print(f"Data1 和 Data2 的协方差为: {covariance}")
