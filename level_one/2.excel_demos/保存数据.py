import pandas as pd

file_loc = 'resources/销售数据.xlsx'

data = pd.read_excel(file_loc)

data_extract = data.groupby('商品类型')['销售金额'].sum()
data_extract = data_extract.reset_index()

data_extract.to_csv("处理好的表格.csv", encoding='gbk', index=False)
data_extract.to_excel("处理好的表格.xlsx", encoding='gbk', index=False)