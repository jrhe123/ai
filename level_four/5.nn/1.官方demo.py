import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# 1. 加载数据集
train_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

# 2. 数据加载器
batch_size = 64
train_dataloader = DataLoader(
    train_data,
    batch_size=batch_size,
)
test_dataloader = DataLoader(
    test_data,
    batch_size=batch_size,
)

for x, y in test_dataloader:
    print("x shape: ", x.shape)
    print("y shape: ", y.shape)
    break

# 3. 构建模型
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device}")


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        result = self.linear_relu_stack(x)
        return result


model = NeuralNetwork().to(device)
print(model)

# 4. 损失函数
loss_fn = nn.CrossEntropyLoss()

# 5. 优化器
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


# 6. 训练
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)

    # 调到训练模式
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # 反向传播
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


# 7. 测试
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    # 调到评估模式 (不需要反向传播，不需要参数更新)
    model.eval()
    test_loss, correct = 0, 0

    # 不需要保留梯度信息
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

            # 对比真实的标签label, +1
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )


# 8. 训练和测试
# epochs = 5
# for t in range(epochs):
#     print(f"Epoch {t + 1}\n-------------------------------")
#     train(train_dataloader, model, loss_fn, optimizer)
#     test(test_dataloader, model, loss_fn)

# print("Done!")


# 9. 保存模型
# torch.save(model.state_dict(), "model.pth")
# print("Trained model saved at model.pth")


# 10. 加载模型
model = NeuralNetwork().to(device)
model.load_state_dict(torch.load("model.pth"))
classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()
with torch.no_grad():
    X, y = test_data[0][0], test_data[0][1]
    X = X.to(device)
    pred = model(X)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]

    print(f"Predicted: '{predicted}', Actual: '{actual}'")
