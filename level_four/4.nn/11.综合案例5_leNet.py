import torch
import torch.nn as nn
from torchsummary import summary


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Conv2d(3, 6, 5)
        self.pool_1 = nn.MaxPool2d(2)
        self.conv_2 = nn.Conv2d(6, 16, 5)
        self.pool_2 = nn.MaxPool2d(2)
        self.conv_3 = nn.Conv2d(16, 120, 5)
        self.fc_1 = nn.Linear(120, 84)
        self.fc_2 = nn.Linear(84, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.pool_1(x)
        x = self.conv_2(x)
        x = self.pool_2(x)
        x = self.conv_3(x)

        # 自适应
        x = x.view(-1, 120)

        x = self.fc_1(x)
        x = self.fc_2(x)
        x = self.softmax(x)
        return x


if __name__ == "__main__":
    model = LeNet()
    print(model)

    # 5张图片，3通道，32*32
    input = torch.randn(5, 3, 32, 32)
    output = model(input)
    print(output.shape)

    summary(model, (3, 32, 32))
