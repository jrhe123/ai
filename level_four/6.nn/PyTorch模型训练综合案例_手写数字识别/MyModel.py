import torch
from torch import nn
from torchsummary import summary

# 定义模型
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

if __name__ == '__main__':
    model = NeuralNetwork()
    print(model)

    # 1: 表示输入图像的通道数（channels）
    # 28, 28: 表示输入图像的高度和宽度
    summary(model, (1,28,28))