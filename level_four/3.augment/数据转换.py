from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

plt.rcParams["savefig.bbox"] = "tight"
# orig_img = Image.open(Path('assets') / 'astronaut.jpg')
# orig_img = Image.open('image.jpeg')
orig_img = Image.open("image.png")

# if you change the seed, make sure that the randomly-applied transforms
# properly show that the image can be both transformed and *not* transformed!
torch.manual_seed(0)


def plot(
    imgs,
    title,
    with_orig=True,
    row_title=None,
    **imshow_kwargs,
):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0]) + with_orig
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    plt.title(title)

    for row_idx, row in enumerate(imgs):
        row = [orig_img] + row if with_orig else row
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if with_orig:
        axs[0, 0].set(title="Original image")
        axs[0, 0].title.set_size(8)

    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()
    plt.show()


# 边缘补充
padded_imgs = [T.Pad(padding=padding)(orig_img) for padding in (3, 10, 30, 50)]
plot(padded_imgs, "T.pad")


# 尺寸变换
resized_imgs = [T.Resize(size=size)(orig_img) for size in (30, 50, 100, orig_img.size)]
plot(resized_imgs, title="Resize")


# 中心截取
center_crops = [
    T.CenterCrop(size=size)(orig_img) for size in (30, 50, 100, orig_img.size)
]
plot(center_crops, title="CenterCrop")


# 四角及中间截取
(top_left, top_right, bottom_left, bottom_right, center) = T.FiveCrop(size=(100, 100))(
    orig_img
)
plot(
    [
        top_left,
        top_right,
        bottom_left,
        bottom_right,
        center,
    ],
    title="FiveCrop",
)


# 灰度变换
gray_img = T.Grayscale()(orig_img)
plot([gray_img], cmap="gray", title="Grayscale")


# 颜色抖动转换
jitter = T.ColorJitter(brightness=0.5, hue=0.3)
jitted_imgs = [jitter(orig_img) for _ in range(4)]
plot(jitted_imgs, title="ColorJitter")


# 高斯模糊
blurrer = T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))
blurred_imgs = [blurrer(orig_img) for _ in range(4)]
plot(blurred_imgs, title="GaussianBlur")


# 随机透视变换
perspective_transformer = T.RandomPerspective(distortion_scale=0.6, p=1.0)
perspective_imgs = [perspective_transformer(orig_img) for _ in range(4)]
plot(perspective_imgs, title="RandomPerspective")


# 随机旋转
rotater = T.RandomRotation(degrees=(0, 180))
rotated_imgs = [rotater(orig_img) for _ in range(4)]
plot(rotated_imgs, title="RandomRotation")


# 随机仿射变换
affine_transfomer = T.RandomAffine(
    degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75)
)
affine_imgs = [affine_transfomer(orig_img) for _ in range(4)]
plot(affine_imgs, title="RandomAffine")


# 弹性变换
elastic_transformer = T.ElasticTransform(alpha=250.0)
transformed_imgs = [elastic_transformer(orig_img) for _ in range(2)]
plot(transformed_imgs, title="ElasticTransform")


# 随机裁剪
cropper = T.RandomCrop(size=(128, 128))
crops = [cropper(orig_img) for _ in range(4)]
plot(crops, title="RandomCrop")


# 随机缩放裁剪
resize_cropper = T.RandomResizedCrop(size=(32, 32))
resized_crops = [resize_cropper(orig_img) for _ in range(4)]
plot(resized_crops, title="RandomResizedCrop")


# 随机颜色翻转
inverter = T.RandomInvert()
invertered_imgs = [inverter(orig_img) for _ in range(4)]
plot(invertered_imgs, title="RandomInvert")


# 随机海报化
posterizer = T.RandomPosterize(bits=2)
posterized_imgs = [posterizer(orig_img) for _ in range(4)]
plot(posterized_imgs, title="RandomPosterize")


# 随机调节锐利度
sharpness_adjuster = T.RandomAdjustSharpness(sharpness_factor=2)
sharpened_imgs = [sharpness_adjuster(orig_img) for _ in range(4)]
plot(sharpened_imgs, title="RandomAdjustSharpness")


# 随机调节对比度
autocontraster = T.RandomAutocontrast()
autocontrasted_imgs = [autocontraster(orig_img) for _ in range(4)]
plot(autocontrasted_imgs, title="RandomAutocontrast")


# 随机直方图均衡
equalizer = T.RandomEqualize()
equalized_imgs = [equalizer(orig_img) for _ in range(4)]
plot(equalized_imgs, title="RandomEqualize")

augmenter = T.RandAugment()
imgs = [augmenter(orig_img) for _ in range(4)]
plot(imgs, title="RandAugment")


# 随机垂直翻转
hflipper = T.RandomHorizontalFlip(p=0.5)
transformed_imgs = [hflipper(orig_img) for _ in range(4)]
plot(transformed_imgs, title="RandomHorizontalFlip")


# 随机水平翻转
vflipper = T.RandomVerticalFlip(p=0.5)
transformed_imgs = [vflipper(orig_img) for _ in range(4)]
plot(transformed_imgs, title="RandomVerticalFlip")


# 随机应用
applier = T.RandomApply(transforms=[T.RandomCrop(size=(64, 64))], p=0.5)
transformed_imgs = [applier(orig_img) for _ in range(4)]
plot(transformed_imgs, title="RandomApply")
