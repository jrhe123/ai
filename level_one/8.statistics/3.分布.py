import numpy as np
import matplotlib.pyplot as plt

# 设置随机数种子，以便复现实验结果
np.random.seed(0)

# 伯努利分布（假设 p 为成功概率）
p = 0.5
bernoulli_dist = np.random.binomial(n=1, p=p, size=1000)  # 生成1000个伯努利分布的随机样本

# 二项分布（参数 n 为试验次数，p 为每次成功概率）
n = 10
binomial_dist = np.random.binomial(n, p, size=1000)  # 生成1000个二项分布的随机样本

# 正态分布（参数 mu 为均值，sigma 为标准差）
mu, sigma = 0, 0.1
normal_dist = np.random.normal(mu, sigma, size=1000)  # 生成1000个正态分布的随机样本

# 指数分布（参数 lambda 为率参数，其倒数为平均间隔时间）
lambd = 1.0
exponential_dist = np.random.exponential(1 / lambd, size=1000)  # 生成1000个指数分布的随机样本

# Logistic 分布（参数 mu 为位置参数，s 为尺度参数）
mu, s = 0, 1
logistic_dist = np.random.logistic(mu, s, size=1000)  # 生成1000个Logistic分布的随机样本

# 绘制柱状图
fig, axs = plt.subplots(nrows=5, ncols=1, figsize=(5, 8))

# 伯努利分布柱状图
axs[0].hist(bernoulli_dist, bins=2)
axs[0].set_title('Bernoulli Distribution')

# 二项分布柱状图
axs[1].hist(binomial_dist, bins=range(n + 2))
axs[1].set_title('Binomial Distribution')

# 正态分布柱状图
axs[2].hist(normal_dist, bins=30)
axs[2].set_title('Normal Distribution')

# 指数分布柱状图
axs[3].hist(exponential_dist, bins=30)
axs[3].set_title('Exponential Distribution')

# Logistic 分布柱状图
axs[4].hist(logistic_dist, bins=30)
axs[4].set_title('Logistic Distribution')

# 调整布局
plt.tight_layout()
plt.show()
