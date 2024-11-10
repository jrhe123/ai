import matplotlib.pyplot as plt
import numpy as np

# 创建数据
num_points = 100
x = np.random.rand(num_points)  # x坐标
y = np.random.rand(num_points)  # y坐标
colors = np.random.rand(num_points)  # 每个点的颜色
sizes = 1000 * np.random.rand(num_points)  # 每个点的大小
alphas = np.random.rand(num_points)  # 每个点的透明度

# 创建散点图
plt.scatter(x, y, c=colors, s=sizes, alpha=0.5, cmap='viridis')

# 显示颜色条
plt.colorbar()

# 显示图像
plt.show()
