import os

import pandas as pd

current_folder_path = os.path.dirname(os.path.abspath(__file__))
file_name = "模仿数据读取.xlsx"
file_path = os.path.join(current_folder_path, file_name)

# 读取excel文件
df = pd.read_excel(file_path)

# 获取第一行数据
first_row = df.loc[0]
# 获取第一列数据
first_column = df.iloc[:, 0]
# 获取第一行第一列数据
df.iloc[0, 0]
# 按列排序x1
sorted_df = df.sort_values(by="x1", ascending=True)
# 填充缺失值
filled_df = df.fillna(0)
# 删除缺失值
dropped_df = df.dropna()
# 添加新的列
df["new_column"] = df["x1"].apply(lambda x: x * 2)
