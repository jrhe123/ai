import numpy as np
import matplotlib.pyplot as plt

# 定义 Sigmoid 激活函数
def sigmoid(x):
    # 返回 x 的 sigmoid 值
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def parametric_relu(x, param=0.05):
    return np.where(x > 0, x, param * x)

def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

# 定义一个绘制激活函数的函数
def plot_activation_function(func, x, title):
    # 计算 y 值，通过将 x 传入激活函数得到
    y = func(x)
    
    # 绘制 x 和 y 的关系图
    plt.plot(x, y)
    
    # 设置图的标题
    plt.title(title)
    
    # 设置 x 轴标签
    plt.xlabel('x')
    
    # 设置 y 轴标签
    plt.ylabel('f(x)')
    
    # 显示网格
    plt.grid(True)
    
    # 显示绘制的图像
    plt.show()

# 生成从 -10 到 10，步长为 0.1 的 x 值数组
x_value = np.arange(-10, 10, 0.1)

# 调用绘图函数，绘制 Sigmoid 激活函数图像
plot_activation_function(sigmoid, x_value, title='Sigmoid')