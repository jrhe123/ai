import os

import cv2 as cv
import numpy as np
import torch
from torch.nn import Sequential
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

transform = Sequential(
    transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomHorizontalFlip(p=0.2),
)


class MyDataset(Dataset):
    def __init__(self):
        root_data = "dataset"
        self.file_name_list = []
        for root, dirs, files in os.walk(root_data):
            for file_i in files:
                file_i_full_path = os.path.join(root, file_i)
                self.file_name_list.append(file_i_full_path)

    def __len__(self):
        return len(self.file_name_list)

    def __getitem__(self, item):
        new_root = "my_new_dataset"
        file_i_loc = self.file_name_list[item]
        image_i = cv.imread(file_i_loc)

        # BGR -> RGB
        image_i = cv.cvtColor(image_i, cv.COLOR_BGR2RGB)
        image_i = cv.resize(image_i, dsize=(256, 256))
        image_i = np.transpose(image_i, (2, 0, 1))

        image_i_tensor = torch.from_numpy(image_i)
        image_i_tensor = transform(image_i_tensor)

        file_i_loc_info = file_i_loc.split("\\")
        file_i_loc_info[0] = new_root
        new_file_i_loc = os.path.join(
            file_i_loc_info[0], file_i_loc_info[1], file_i_loc_info[2]
        )
        return image_i_tensor, new_file_i_loc


if __name__ == "__main__":
    my_dataset = MyDataset()
    my_dataloader = DataLoader(my_dataset)

    for x_i, loc_i in my_dataloader:
        x_i = x_i.view(3, 256, 256)
        print(x_i.shape, loc_i)
        loc_info = loc_i[0].split("\\")
        file_dir = os.path.join(loc_info[0], loc_info[1])

        if os.path.isdir(file_dir):
            pass
        else:
            os.makedirs(file_dir)

        image = transforms.ToPILImage()(x_i)
        image.save(loc_i[0])

        # cv.imshow('img_i', image_i)
        # cv.waitKey(500)
