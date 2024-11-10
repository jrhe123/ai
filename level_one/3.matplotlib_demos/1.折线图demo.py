import numpy as np
import matplotlib.pyplot as plt

# 创建一个x值的数组，从-2π到2π，步长为0.01
x = np.arange(-2 * np.pi, 2 * np.pi, 0.01)

# 计算每个x值对应的sin(x)值
y = np.sin(x)

# 使用matplotlib来绘制图像
plt.figure()  # 创建一个新的图像窗口
plt.plot(x, y)  # 绘制折线图
plt.title('sin(x)')  # 设置图像的标题
plt.xlabel('x')  # 设置x轴的标签
plt.ylabel('sin(x)')  # 设置y轴的标签
plt.grid(True)  # 显示网格
plt.show()  # 显示图像
