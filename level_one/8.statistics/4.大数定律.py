import numpy as np
import matplotlib.pyplot as plt

# 定义赌博游戏，返回硬币投掷的结果，正面为1，反面为-1
def gamble_game(trial):
    # 抛硬币游戏，正面是1，反面是-1
    results = np.random.choice(a=[1, -1], size=trial)
    return results

# 模拟赌博，计算累积盈亏和平均盈亏
def simulate_gambling(trails):
    results = gamble_game(trails)

    # 计算累计盈亏
    cumulative_sum = np.cumsum(results)

    # 计算平均盈亏
    average_win_loss = cumulative_sum / np.arange(1, trails + 1)

    return cumulative_sum, average_win_loss

# 绘制结果图表
def plot_results(trails):
    cumulative_sum, average_win_loss = simulate_gambling(trails)

    # 绘制累计盈亏图
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(cumulative_sum)
    plt.title('Cumulative Win/Loss Over Trails')
    plt.xlabel('Trails')
    plt.ylabel('Cumulative Sum')

    # 绘制平均盈亏图
    plt.subplot(1, 2, 2)
    plt.plot(average_win_loss)
    plt.axhline(y=0, color='r', linestyle='-')  # 添加 y=0 的水平线，表示盈亏平衡
    plt.title('Average Win/Loss Over Trails')
    plt.xlabel('Trails')
    plt.ylabel('Average Win/Loss')

    plt.tight_layout()
    plt.show()

# 运行绘制函数，模拟10000次实验
plot_results(10000)
