import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号

file_loc = 'resources/销售数据.xlsx'

data = pd.read_excel(file_loc)

print(data)

data_extract = data.groupby('商品类型')['销售金额'].sum().reset_index().sort_values('销售金额', ascending=True).reset_index(drop=True)
print(data_extract)

# 提取销售金额和大类名称
sales_amounts = data_extract['销售金额']
category_names = data_extract['商品类型']

# 计算每个类别的占比
sales_proportions = sales_amounts / sales_amounts.sum()

# 画饼状图
fig1, ax1 = plt.subplots()
ax1.pie(sales_proportions, labels=category_names, autopct='%1.1f%%', startangle=90)


plt.show()
