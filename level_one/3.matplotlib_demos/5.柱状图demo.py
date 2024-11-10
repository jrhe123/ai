import matplotlib.pyplot as plt
import numpy as np

# 数据
labels = ['A', 'B', 'C', 'D', 'E']
values = [3, 7, 2, 5, 8]

# 设置标签的位置
x = np.arange(len(labels))

# 绘制柱状图
plt.bar(x, values, color='blue', align='center', alpha=0.7)

# 设置图表的标题和轴标签
plt.title('Simple Bar Chart')
plt.xlabel('Labels')
plt.ylabel('Values')

# 设置x轴的标签
plt.xticks(x, labels)

# 显示图像
plt.show()
