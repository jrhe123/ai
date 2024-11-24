import os

import pandas as pd
from torch.utils.data import DataLoader, Dataset

# current_folder_path = os.path.dirname(os.path.abspath(__file__))
# file_name = "模仿数据读取.xlsx"
# file_path = os.path.join(current_folder_path, file_name)
# # 读取excel文件
# df = pd.read_excel(file_path)

# # 获取第一行数据
# first_row = df.loc[0]
# # 获取第一列数据
# first_column = df.iloc[:, 0]
# # 获取第一行第一列数据
# df.iloc[0, 0]
# # 按列排序x1
# sorted_df = df.sort_values(by="x1", ascending=True)
# # 填充缺失值
# filled_df = df.fillna(0)
# # 删除缺失值
# dropped_df = df.dropna()
# # 添加新的列
# df["new_column"] = df["x1"].apply(lambda x: x * 2)


class MyDataset(Dataset):

    def __init__(self):
        current_folder_path = os.path.dirname(os.path.abspath(__file__))
        file_name = "模仿数据读取.xlsx"
        file_path = os.path.join(current_folder_path, file_name)
        df = pd.read_excel(file_path)
        self.x1 = df["x1"]
        self.x2 = df["x2"]
        self.x3 = df["x3"]
        self.y = df["y"]

    def __len__(self):
        return len(self.x1)

    def __getitem__(self, idx):
        return (
            self.x1[idx],
            self.x2[idx],
            self.x3[idx],
            self.y[idx],
        )


if __name__ == "__main__":
    my_dataset = MyDataset()
    my_data_loader = DataLoader(
        my_dataset,
        batch_size=2,
        num_workers=2,
        shuffle=True,
    )
    for batch in my_data_loader:
        print(batch)
