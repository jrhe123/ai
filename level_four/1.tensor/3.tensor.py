import numpy as np
import torch

# 转换成tensor
n = np.ones(5)
t = torch.from_numpy(n)
print(n)
print(t)


# 转换成numpy
n_2 = t.numpy()
print(n_2)


# 图片转tensor
from PIL import Image
from torchvision import transforms

img = Image.open(r"./level_four/1.tensor/cat.jpeg")
transform = transforms.ToTensor()
img_tensor = transform(img)

print(img_tensor)
print(type(img_tensor))

# tensor转图片
transform = transforms.ToPILImage()
transformed_image = transform(img_tensor)
save_path = r"./level_four/1.tensor/new_cat.jpeg"
transformed_image.save(save_path)
print("图片保存成功")
