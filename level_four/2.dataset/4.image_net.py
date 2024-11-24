import torch
import torchvision

# PIL/numpy -> tensor
transform = torchvision.transforms.ToTensor()

# 1. train dataset
train_dataset = torchvision.datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform,
)
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
)

# 2. test dataset
test_dataset = torchvision.datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform,
)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=4,
    shuffle=False,
)

for x, y in train_loader:
    print(x.shape, y)

for x, y in test_loader:
    print(x.shape, y)
