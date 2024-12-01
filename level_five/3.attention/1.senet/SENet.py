#coding:utf8
import torch
import torch.nn as nn

## SE模块搭建
class SELayer(nn.Module):
    def __init__(self, channel,reduction = 16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_features=channel, out_features=channel // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_features=channel // reduction, out_features=channel)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x
        n,c,_,_ = x.size() # [N,C,H,W]
        x = self.avg_pool(x)
        x = x.view(n,c)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        x = x.view(n,c,1,1)
        x = identity * x

        return x

## 残差模块
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, outchannels):
        super(ResidualBlock, self).__init__()

        self.channel_equal_flag = True
        if in_channels == outchannels:
            self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=outchannels, kernel_size=3, padding=1, stride=1, bias=False)
        else:
            self.conv1x1 = nn.Conv2d(in_channels=in_channels, out_channels=outchannels, kernel_size=1, stride=2, bias=False)
            self.bn1x1 = nn.BatchNorm2d(num_features=outchannels)
            self.channel_equal_flag = False

            self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=outchannels,kernel_size=3,padding=1, stride=2, bias=False)

        self.bn1 = nn.BatchNorm2d(num_features=outchannels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(in_channels=outchannels, out_channels=outchannels, kernel_size=3,padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=outchannels)

        self.selayer = SELayer(channel=outchannels)

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        if self.channel_equal_flag == True:
            pass
        else:
            identity = self.conv1x1(identity)
            identity = self.bn1x1(identity)
            identity = self.relu(identity)

        x = self.selayer(x) ## 即插即用模块

        out = identity + x
        return out

## SENet-18
class Model(nn.Module):
    def __init__(self, num_classes=1000):
        super(Model, self).__init__()
        # conv1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64,kernel_size=7,stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.relu = nn.ReLU(inplace=True)

        # conv2_x
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2,padding=1)
        self.conv2_1 = ResidualBlock(in_channels=64, outchannels=64)
        self.conv2_2 = ResidualBlock(in_channels=64, outchannels=64)

        # conv3_x
        self.conv3_1 = ResidualBlock(in_channels=64, outchannels=128)
        self.conv3_2 = ResidualBlock(in_channels=128, outchannels=128)

        # conv4_x
        self.conv4_1 = ResidualBlock(in_channels=128, outchannels=256)
        self.conv4_2 = ResidualBlock(in_channels=256, outchannels=256)

        # conv5_x
        self.conv5_1 = ResidualBlock(in_channels=256, outchannels=512)
        self.conv5_2 = ResidualBlock(in_channels=512, outchannels=512)

        # avg_pool
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1,1)) # [N, C, H, W] = [N, 512, 1, 1]

        # fc
        self.fc = nn.Linear(in_features=512, out_features=num_classes)  # [N, num_classes]

    def forward(self, x):
        # conv1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # conv2_x
        x = self.maxpool(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)

        # conv3_x
        x = self.conv3_1(x)
        x = self.conv3_2(x)

        # conv4_x
        x = self.conv4_1(x)
        x = self.conv4_2(x)

        # conv5_x
        x = self.conv5_1(x)
        x = self.conv5_2(x)

        # avgpool + fc
        x = self.avg_pool(x)

        x = torch.flatten(x, 1)

        x = self.fc(x)

        return x


if __name__ == '__main__':
    # 定义模型
    model = Model(num_classes=10)

    # 定义输入 [N, C, H, W]
    input = torch.ones([10, 3, 224, 224])
    output = model(input)
    torch.onnx.export(model,input,'senet.onnx')




