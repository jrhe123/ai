from torch.utils.data import DataLoader, Dataset


class MyDataset(Dataset):
    def __init__(self):
        self.x = [i for i in range(10)]
        self.y = [i * 2 for i in range(10)]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


if __name__ == "__main__":
    my_dataset = MyDataset()
    my_data_loader = DataLoader(
        my_dataset,
        batch_size=2,  # 每次返回2条数据
        shuffle=True,  # 训练集中使用
        num_workers=2,  # 2个线程加载数据
    )
    for x_i, y_i in my_data_loader:
        print(x_i, y_i)
