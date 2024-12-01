# coding:utf8

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

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super(ResidualBlock, self).__init__()

        self.expansion = 4

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=stride,
                               padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels * self.expansion, kernel_size=1,
                               stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        self.stride = stride

    def forward(self, x):
        identity = x.clone()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)

        return x


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes):
        super(ResNet, self).__init__()

        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.pooling = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.res_layer(block, layers[0], out_channels=64, stride=1) #第1组残差模块，整体分辨率不下降
        self.layer2 = self.res_layer(block, layers[1], out_channels=128, stride=2)#第2组残差模块，分辨率下降1/2
        self.layer3 = self.res_layer(block, layers[2], out_channels=256, stride=2)#第2组残差模块，分辨率下降1/2
        self.layer4 = self.res_layer(block, layers[3], out_channels=512, stride=2)#第2组残差模块，分辨率下降1/2

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pooling(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)

        x = x.reshape(x.shape[0], -1)

        x = self.fc(x)

        return x

    def res_layer(self, block, num_residual_blocks, out_channels, stride):
        identity_downsample = None
        layers = []

        # 每一组残差模块的第一个残差模块需要特殊对待
        if stride != 1 or self.in_channels != out_channels * 4:
            identity_downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels * 4,
                    kernel_size=1,
                    stride=stride,
                ),
                nn.BatchNorm2d(out_channels * 4),
            )

        layers.append(
            block(self.in_channels, out_channels, identity_downsample, stride)
        )

        # 除了每一组残差模块的第一个，其他残差模块的输入通道都等于输出通道的4倍，ResNet 50,101,152
        self.in_channels = out_channels * 4 # 该值每经过一组残差模块，就会变大，64 -> 4*64=256 -> 4*128=512 -> 4*256=1024 -> 4*512=2048
        # 如resnet50的conv2_x，通道变换为256 -> 64 -> 256

        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)


def ResNet50(num_classes=1000):
    return ResNet(ResidualBlock, [3, 4, 6, 3], num_classes)


def ResNet101(num_classes=1000):
    return ResNet(ResidualBlock, [3, 4, 23, 3], num_classes)


def ResNet152(num_classes=1000):
    return ResNet(ResidualBlock, [3, 8, 36, 3], num_classes)


input = torch.randn([1, 3, 224, 224])
resnet50 = ResNet50(num_classes=1000)
output = resnet50(input)

torch.save(resnet50, 'resnet50.pth')
torch.onnx.export(resnet50, input, 'resnet50.onnx')

