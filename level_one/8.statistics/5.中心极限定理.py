import numpy as np
import matplotlib.pyplot as plt

# 模拟投掷一个六面骰子
def roll_dice():
    return np.random.randint(1, 7)  # 随机生成1到6之间的整数

# 进行一次实验，实验包含 num_rolls 次投掷骰子
def experiment(num_rolls):
    total = 0
    for _ in range(num_rolls):
        total += roll_dice()
    return total / num_rolls  # 返回投掷结果的平均值

# 运行多次实验，并记录平均值
def run_experiments(num_experiments, num_rolls):
    averages = []
    for _ in range(num_experiments):
        averages.append(experiment(num_rolls))
    return averages  # 返回所有实验的平均值列表

# 绘制实验平均值的直方图
def plot_histogram(averages):
    plt.hist(averages, bins=20, edgecolor='black', density=True)  # 绘制直方图
    plt.xlabel('Average of Rolls')  # x轴标签
    plt.ylabel('Frequency')  # y轴标签
    plt.title('Central Limit Theorem Demonstration')  # 标题
    plt.show()

# 参数设置
num_experiments = 10000  # 实验次数
num_rolls = 30  # 每次实验投掷次数

# 运行实验并绘制直方图
averages = run_experiments(num_experiments, num_rolls)
plot_histogram(averages)
