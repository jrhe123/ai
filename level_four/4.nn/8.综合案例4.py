import torch
import torch.nn as nn
from torchsummary import summary


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Conv2d(3, 32, kernel_size=3)
        self.relu = nn.ReLU()
        self.conv_2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc_1 = nn.Linear(64 * 220 * 220, 512)
        self.fc_2 = nn.Linear(512, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.relu(x)
        x = self.conv_2(x)
        x = self.relu(x)
        x = x.view(-1, 64 * 220 * 220)
        x = self.fc_1(x)
        x = self.relu(x)
        x = self.fc_2(x)
        x = self.softmax(x)
        return x


if __name__ == "__main__":
    model = ConvNet()
    input = torch.rand((5, 3, 224, 224))
    output = model(input)

    print(output.shape)
    print(model)

    summary(model, (3, 224, 224))
