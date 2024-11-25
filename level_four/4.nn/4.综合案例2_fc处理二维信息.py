import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256 * 256 * 3, 100)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(100, 10)
        self.fc3 = nn.Linear(10, 5)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, 256 * 256 * 3)  # 维度自适应
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x


if __name__ == "__main__":
    model = NeuralNetwork()
    input = torch.rand((10, 256, 256, 3))
    output = model(input)
    print(output.shape)

    summary(model, (256, 256, 3))
