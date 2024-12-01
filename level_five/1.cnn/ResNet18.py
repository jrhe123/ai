#coding:utf8

# Copyright 2023 longpeng2008. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# If you find any problem,please contact us
#
#     longpeng2008to2012@gmail.com
#
# or create issues
# =============================================================================
import torch
import torch.nn as nn

# 残差模块
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, outchannels):
        super(ResidualBlock, self).__init__()
        self.channel_equal_flag = True
        if in_channels == outchannels:
            self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=outchannels, kernel_size=3, padding=1, stride=1)
        else:
            ## 对恒等映射分支的变换，当通道数发生变换时，分辨率变为原来的二分之一
            self.conv1x1 = nn.Conv2d(in_channels=in_channels, out_channels=outchannels, kernel_size=1, stride=2)
            self.bn1x1 = nn.BatchNorm2d(num_features=outchannels)
            self.channel_equal_flag = False

            self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=outchannels,kernel_size=3,padding=1, stride=2)

        self.bn1 = nn.BatchNorm2d(num_features=outchannels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(in_channels=outchannels, out_channels=outchannels, kernel_size=3,padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=outchannels)

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

        out = identity + x
        return out

class ResNet18(nn.Module):
    def __init__(self, num_classes=1000):
        super(ResNet18, self).__init__()
        # conv1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64,kernel_size=7,stride=2, padding=3)
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

        # softmax
        self.softmax = nn.Softmax(dim=1)

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

        # avgpool + fc + softmax
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.softmax(x)

        return x

if __name__ == '__main__':
    model = ResNet18(num_classes=1000)
    input = torch.randn([1, 3, 224, 224])
    output = model(input)

    torch.save(model, 'resnet18.pth')
    torch.onnx.export(model,input,'resnet18.onnx')




