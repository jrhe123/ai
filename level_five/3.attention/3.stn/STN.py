import torch
from torch import nn
from torch.nn import functional as F

class STN(nn.Module):
    def __init__(self, c,h,w,mode='stn'):
        assert mode in ['stn', 'cnn']

        super(STN, self).__init__()
        self.mode = mode
        self.local_net = LocalNetwork(c,h,w)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=16*8*8, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=10)
        )

    def forward(self, img):
        '''
        :param img: (b, c, h, w)
        :return: (b, c, h, w), (b,)
        '''
        batch_size,c,h,w = img.shape

        # STN 处理图片
        img = self.local_net(img)

        conv_output = self.conv(img).view(batch_size, -1)
        predict = self.fc(conv_output)
        return img, predict


class LocalNetwork(nn.Module):
    def __init__(self,c,h,w):
        super(LocalNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features=c*h*w,
                      out_features=20),
            nn.Tanh(),
            nn.Linear(in_features=20, out_features=6),
            nn.Tanh(),
        )

    def forward(self, img):
        '''
        :param img: (b, c, h, w)
        :return: (b, c, h, w)
        '''
        batch_size,c,w,h = img.shape

        theta = self.fc(img.view(batch_size, -1)).view(batch_size, 2, 3)

        ## 仿射变换采样函数
        grid = F.affine_grid(theta, torch.Size((batch_size,c,h,w)))
        img_transform = F.grid_sample(img, grid)

        return img_transform


if __name__ == '__main__':
    net = STN(3, 32, 32)
    x = torch.randn(1, 3, 32, 32)

    feature,predict = net(x)

    print(feature.shape)

