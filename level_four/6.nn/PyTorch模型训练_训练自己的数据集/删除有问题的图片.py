import cv2 as cv
import os
from tqdm import tqdm

root_path = "dataset"
file_wrong_list = []
for root, dirs, files in os.walk(root_path):
    for file_i in tqdm(files):
        file_i_full_path = os.path.join(root, file_i)
        img = cv.imread(file_i_full_path)
        if img is None:
            file_wrong_list.append(file_i_full_path)

if len(file_wrong_list) == 0:
    print("没有异常图片")

else:
    for file_i in file_wrong_list:
        print(f"正在删除{file_i}")
        os.remove(file_i)
    print("处理完毕")