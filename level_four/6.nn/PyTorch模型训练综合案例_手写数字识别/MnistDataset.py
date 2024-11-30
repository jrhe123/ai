import os
import torch
from torch.utils.data import Dataset, DataLoader
import cv2 as cv

class MNISTDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.file_list = []
        self.name_list = []
        self.id_list = []
        for root, dirs, files in os.walk(self.root_dir):
            if dirs:
                self.name_list = dirs

            for file_i in files:
                file_i_full_path = os.path.join(root,file_i)

                # mnist_images/test/0
                file_class = os.path.split(file_i_full_path)[0].split('/')[-1]

                self.id_list.append(self.name_list.index(file_class))
                self.file_list.append(file_i_full_path)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img = self.file_list[idx]
        # 0: 单通道图片信息
        img = cv.imread(img, 0)
        # 统一尺寸 28x28
        img = cv.resize(img, dsize=(28,28))
        img = torch.from_numpy(img).float()

        # 标签信息
        label = self.id_list[idx]
        label = torch.tensor(label)

        return img, label


if __name__ == '__main__':
    
    my_dataset_train = MNISTDataset(r'level_four/6.nn/PyTorch模型训练综合案例_手写数字识别/mnist_images/train')
    my_dataloader_train = DataLoader(my_dataset_train, batch_size=10, shuffle=True)
    # 尝试读取训练集数据
    print("读取训练集数据")
    # for x, y in my_dataloader_train:
    #     print(x.type(), x.shape, y)

    my_dataset_test = MNISTDataset(r'level_four/6.nn/PyTorch模型训练综合案例_手写数字识别/mnist_images/test')
    my_dataloader_test = DataLoader(my_dataset_test, batch_size=10, shuffle=False)
    # 尝试读取训练集数据
    print("读取测试集数据")
    # for x, y in my_dataloader_test:
    #     print(x.shape, y)


    # 10个labels，因为batch size为10
    """torch.Size([10, 28, 28]) tensor([9, 9, 9, 9, 9, 9, 9, 9, 9, 9])"""
    