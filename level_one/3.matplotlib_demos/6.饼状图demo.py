import matplotlib.pyplot as plt

# 数据
sizes = [15, 30, 45, 10]  # 各部分的大小
labels = ['A', 'B', 'C', 'D']  # 各部分的标签
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']  # 各部分的颜色
explode = (0.1, 0, 0, 0)  # 突出显示第一个部分

# 绘制扇形图
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)

# 设置为等比例，这样扇形图就是一个圆
plt.axis('equal')

# 显示图像
plt.show()
