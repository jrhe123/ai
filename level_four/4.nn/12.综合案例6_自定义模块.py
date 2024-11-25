from collections import OrderedDict

import torch
import torch.nn as nn
from torchsummary import summary


class MyBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv_1 = nn.Conv2d(
            in_channel,
            out_channel,
            kernel_size=3,
            padding=1,
        )
        self.bn_1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv_2 = nn.Conv2d(
            out_channel,
            out_channel,
            kernel_size=3,
            padding=1,
        )
        self.bn_2 = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        x_1 = x
        x = self.conv_1(x)
        x = self.bn_1(x)
        x = self.relu(x)
        x = self.conv_2(x)
        x = self.bn_2(x)

        result = x + x_1
        return result


class MainNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Conv2d(3, 32, 3)
        self.conv_2 = nn.Conv2d(32, 64, 3)

        self.body = nn.Sequential(
            OrderedDict(
                [
                    ("block_1", MyBlock(64, 64)),
                    ("block_2", MyBlock(64, 64)),
                    ("block_3", MyBlock(64, 64)),
                    ("block_4", MyBlock(64, 64)),
                ]
            )
        )

        self.fc = nn.Sequential(
            nn.Linear(64 * 220 * 220, 512),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)

        x = self.body(x)

        x = x.view(-1, 64 * 220 * 220)

        x = self.fc(x)

        return x


if __name__ == "__main__":
    model = MainNet()
    print(model)

    input = torch.randn(5, 3, 224, 224)
    output = model(input)
    print(output.shape)

    summary(model, (3, 224, 224))
