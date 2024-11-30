from torchvision.datasets import ImageFolder
import torch
from torchvision import transforms as T
from tqdm import tqdm

transform = T.Compose([
     T.RandomResizedCrop(224),
     T.ToTensor(),
])

def getStat(train_data):
    '''
    Compute mean and variance for training data
    :param train_data: 自定义类Dataset(或ImageFolder即可)
    :return: (mean, std)
    '''
    print('Compute mean and variance for training data.')
    print(len(train_data))

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=1, 
        shuffle=False, num_workers=0,
        pin_memory=True
    )
    
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for X, _ in tqdm(train_loader):
        for d in range(3):
            mean[d] += X[:, d, :, :].mean()
            std[d] += X[:, d, :, :].std()
    mean.div_(len(train_data))
    std.div_(len(train_data))

    return list(mean.numpy()), list(std.numpy())


if __name__ == '__main__':
    train_dataset = ImageFolder(
        root=r'./level_four/6.nn/PyTorch模型训练_训练自己的数据集/dataset',
        transform=transform,
    )
    mean, std = getStat(train_dataset)
    print(f"mean={mean},std={std}") 
    
    # 彩色图片：3通道
    # mean=[0.5062653, 0.46558657, 0.37899864]
    # std=[0.22566116, 0.20558165, 0.21950442]