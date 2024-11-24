import os

import cv2 as cv
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class MyDataset(Dataset):

    def __init__(self):
        self.file_path_list = []
        current_folder_path = os.path.dirname(os.path.abspath(__file__))
        folder_path = "animal_images"
        folder_path = os.path.join(current_folder_path, folder_path)

        self.labels = []
        dir_name = []

        for root, dirs, files in os.walk(folder_path):
            files = [f for f in files if f != ".DS_Store"]
            if dirs:
                dir_name = dirs

            for file_i in files:
                file_i_full_path = os.path.join(root, file_i)
                self.file_path_list.append(file_i_full_path)

                label = root.split(os.sep)[-1]
                label_id = dir_name.index(label)
                self.labels.append(label_id)

    def __len__(self):
        return len(self.file_path_list)

    def __getitem__(self, idx):
        # 1. load image file path
        file_idx_path = self.file_path_list[idx]
        image_idx = cv.imread(file_idx_path)
        image_idx = cv.resize(image_idx, (256, 256))
        # HWC -> CHW
        image_idx = np.transpose(image_idx, (2, 1, 0))
        # convert to tensor
        image_tensor = torch.from_numpy(image_idx)

        # 2. image label
        label = self.labels[idx]

        # return
        return image_tensor, label


if __name__ == "__main__":
    my_dataset = MyDataset()
    dataloader = DataLoader(
        my_dataset,
        batch_size=4,
        num_workers=2,
        shuffle=True,
    )
    for x_i, y_i in dataloader:
        print(x_i, y_i)
