import numpy as np
import matplotlib.pyplot as plt

# 定义函数
def f(x):
    return x**2

def df(x):
    # 返回 f(x) 的导数，即 2 * x
    return 2 * x

# 生成从 -3 到 3 的等间距 x 值，共 100 个点
x = np.linspace(-3, 3, 100)

# 计算原始函数的 y 值
y = f(x)

# # 设置图形大小
plt.figure(figsize=(8, 6))

# # 绘制原始函数 f(x) 的图像
plt.plot(x, y, label="f(x) = x^2")

# # 显示绘图
# plt.legend()
# plt.grid(True)
# plt.show()


# 计算 x = 1 时的梯度和切线
x1 = 1
y1 = f(x1)
slope = df(x1)

# 定义切线方程：y = m * (x - x1) + y1
def tangent_line(x, x1, y1, slope):
    # 计算给定点处的切线方程
    return slope * (x - x1) + y1

# 在切点附近绘制切线
x_tangent = np.linspace(x1 - 1, x1 + 1, 10)  # 生成 x 的值范围，取切点前后各 1 单位，共 10 个点
y_tangent = tangent_line(x_tangent, x1, y1, slope)  # 计算切线的 y 值

# 绘制切线
plt.plot(x_tangent, y_tangent, label="Tangent at x = 1", color='red')  # 绘制切线，红色
plt.scatter([x1], [y1], color='black')  # 标记切点，用黑色

# 设置图像
plt.legend()  # 显示图例
plt.xlabel('x')  # 设置 x 轴标签
plt.ylabel('f(x)')  # 设置 y 轴标签
plt.title("Function and Tangent Line at a Point")  # 设置图的标题
plt.grid(True)  # 显示网格
plt.show()  # 显示图像