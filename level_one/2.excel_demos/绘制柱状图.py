import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.sans-serif']=['SimHei'] # 用黑体显示中文
matplotlib.rcParams['axes.unicode_minus']=False # 正常显示负号

file_loc = 'resources/销售数据.xlsx'

data = pd.read_excel(file_loc)

print(data)

data_extract = data.groupby('大类名称')['销售金额'].sum().reset_index().sort_values('销售金额',ascending=True).reset_index(drop=True)
print(data_extract)

# 使用大类名称作为x轴的标签
x_labels = data_extract['大类名称']
bars = plt.bar(x_labels, data_extract['销售金额'], tick_label=x_labels)
plt.xticks(rotation=45)  # 如果标签文字太长，可以旋转标签以便更好地显示

# 在每一根柱上显示对应的高度值
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom')  # ha: horizontal alignment, va: vertical alignment

plt.show()
