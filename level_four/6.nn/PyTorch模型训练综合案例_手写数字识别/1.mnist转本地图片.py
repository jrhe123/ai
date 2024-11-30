import os
import torchvision
import torchvision.transforms as transforms

# 下载MNIST数据集
transform = transforms.Compose([transforms.ToTensor()])
mnist_trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                            download=True, transform=transform)
mnist_testset = torchvision.datasets.MNIST(root='./data', train=False,
                                           download=True, transform=transform)

# 创建一个目录来保存图像（如果它还不存在）
os.makedirs('./mnist_images/train', exist_ok=True)
os.makedirs('./mnist_images/test', exist_ok=True)

# 遍历数据集并保存图像
for idx, (image, label) in enumerate(mnist_trainset):
    # 创建类别文件夹（如果它还不存在）
    label_dir = os.path.join('./mnist_images/train', str(label))
    os.makedirs(label_dir, exist_ok=True)

    # 转换为PIL图像并保存
    pil_image = transforms.ToPILImage()(image)
    pil_image.save(os.path.join(label_dir, f'{idx}.jpg'))

# 遍历数据集并保存图像
for idx, (image, label) in enumerate(mnist_testset):
    # 创建类别文件夹（如果它还不存在）
    label_dir = os.path.join('./mnist_images/test', str(label))
    os.makedirs(label_dir, exist_ok=True)

    # 转换为PIL图像并保存
    pil_image = transforms.ToPILImage()(image)
    pil_image.save(os.path.join(label_dir, f'{idx}.jpg'))

# 打印完成消息
print("All images have been saved successfully.")
