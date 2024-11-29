from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28 * 3, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        result = self.linear_relu_stack(x)
        return result

if __name__ == "__main__":
    writer = SummaryWriter(log_dir="data")

    model = NeuralNetwork()

    x = torch.randn(1, 3, 28, 28)
    writer.add_graph(model, x)
    writer.close()
