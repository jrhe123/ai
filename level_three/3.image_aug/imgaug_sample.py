# coding:utf8
import imgaug as ia
import imgaug.augmenters as iaa
import numpy as np

ia.seed(1)

## 创建矩阵(16, 64, 64, 3).
images = np.array([ia.quokka(size=(64, 64)) for _ in range(16)], dtype=np.uint8)

seq = iaa.Sequential(
    [
        iaa.Fliplr(0.5),  ## 以0.5的概率进行水平翻转horizontal flips
        iaa.Crop(percent=(0, 0.1)),  ## 随机裁剪random crops
        ## 对50%的图片进行高斯模糊，标准差参数取值0～0.5.
        iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 0.5))),
        ## 对50%的通道添加高斯噪声
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
    ],
    random_order=True,
)  ## 以上所有操作，使用随机顺序

images_aug = seq(images=images)  ## 应用操作增强
grid_image = ia.draw_grid(images_aug, 4)

import imageio

imageio.imwrite("example.jpg", grid_image)
