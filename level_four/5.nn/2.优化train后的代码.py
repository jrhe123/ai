import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from tqdm import tqdm
import matplotlib.pyplot as plt

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


"""
pred = torch.tensor([[0.1, 0.3, 0.6],
                     [0.8, 0.1, 0.1]])
print(pred.argmax(1))  # 输出: tensor([2, 0]) 表示第一个样本预测类别 2，第二个样本预测类别 0



y = torch.tensor([2, 1])  # 真实标签
pred_classes = torch.tensor([2, 0])  # 预测类别
print(pred_classes == y)  # 输出: tensor([True, False])



bool_tensor = torch.tensor([True, False])
float_tensor = bool_tensor.type(torch.float)
print(float_tensor)  # 输出: tensor([1.0, 0.0])
"""

# 6. 训练
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    # 调到训练模式
    model.train()
    loss_total = 0
    correct = 0
    for X, y in tqdm(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction and loss
        pred = model(X)

        # 计算正确预测的数量
        # 在 pred 张量中，沿着第 1 个维度（行方向）找到最大值所在的索引
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        loss = loss_fn(pred, y)
        loss_total += loss.item()

        # 反向传播
        loss.backward()
        # 优化器参数更新
        optimizer.step()
        # 优化器归零
        optimizer.zero_grad()

    loss_avg = loss_total / num_batches
    correct /= size

    return round(correct, 3), round(loss_avg, 3)


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
train_acc_list = []
train_loss_list = []
epochs = 5
for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train_acc, train_loss = train(train_dataloader, model, loss_fn, optimizer)
    train_acc_list.append(train_acc)
    train_loss_list.append(train_loss)
    test(test_dataloader, model, loss_fn)

print("Done!")

x_list = [i for i in range(len(train_acc_list))]
plt.plot(x_list, train_acc_list, label="Train")
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

plt.plot(x_list, train_loss_list, label="Train")
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()


# 9. 保存模型
# torch.save(model.state_dict(), "model.pth")
# print("Trained model saved at model.pth")


# # 10. 加载模型
# model = NeuralNetwork().to(device)
# model.load_state_dict(torch.load("model.pth"))
# classes = [
#     "T-shirt/top",
#     "Trouser",
#     "Pullover",
#     "Dress",
#     "Coat",
#     "Sandal",
#     "Shirt",
#     "Sneaker",
#     "Bag",
#     "Ankle boot",
# ]

# model.eval()
# with torch.no_grad():
#     X, y = test_data[0][0], test_data[0][1]
#     X = X.to(device)
#     pred = model(X)
#     predicted, actual = classes[pred[0].argmax(0)], classes[y]

#     print(f"Predicted: '{predicted}', Actual: '{actual}'")
