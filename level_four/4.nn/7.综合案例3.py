import torch
import torch.nn as nn
from torchsummary import summary


class FCN(nn.Module):
    def __init__(self):
        super().__init__()
        # 3: 输入层
        # 32: 输出层
        # 3: 卷积核大小
        self.conv_1 = nn.Conv2d(3, 32, 3)
        self.relu = nn.ReLU()
        self.conv_2 = nn.Conv2d(32, 64, 3)
        self.conv_3 = nn.Conv2d(64, 128, 3)
        self.deconv_1 = nn.ConvTranspose2d(128, 64, 3)
        self.deconv_2 = nn.ConvTranspose2d(64, 32, 3)
        self.deconv_3 = nn.ConvTranspose2d(32, 10, 3)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.relu(x)
        x = self.conv_2(x)
        x = self.relu(x)
        x = self.conv_3(x)
        x = self.relu(x)
        x = self.deconv_1(x)
        x = self.relu(x)
        x = self.deconv_2(x)
        x = self.relu(x)
        x = self.deconv_3(x)
        x = self.softmax(x)
        return x


if __name__ == "__main__":
    model = FCN()
    input = torch.randn(1, 3, 224, 224)
    output = model(input)
    print(output.shape)
    print(model)

    summary(model, (3, 224, 224))
