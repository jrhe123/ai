import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_1 = nn.Linear(10, 5, bias=True)
        self.fc_2 = nn.Linear(5, 1, bias=True)

    def forward(self, x):
        x = self.fc_1(x)
        x = self.fc_2(x)
        return x


if __name__ == "__main__":
    model = Model()
    print(model)

    my_input = torch.rand(10)
    output = model(my_input)
    print(output)

    summary(model, (10,))

    """
      ----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
      ================================================================
                  Linear-1                    [-1, 1]              11
      ================================================================
      Total params: 11
      Trainable params: 11
      Non-trainable params: 0
      ----------------------------------------------------------------
      Input size (MB): 0.00
      Forward/backward pass size (MB): 0.00
      Params size (MB): 0.00
      Estimated Total Size (MB): 0.00
      ----------------------------------------------------------------
    """
