import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from torch.utils.tensorboard import SummaryWriter

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
        for X, y in tqdm(dataloader):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

            # 对比真实的标签label, +1
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    
    return round(correct, 3), round(test_loss, 3)


def writedata(txt_log_name, tensorboard_writer, epoch, train_accuracy, train_loss, test_accuracy, test_loss):
    # 保存到文档
    with open(txt_log_name, "a+") as f:
        f.write(f"Epoch:{epoch}\ttrain_accuracy:{train_accuracy}\ttrain_loss:{train_loss}\ttest_accuracy:{test_accuracy}\ttest_loss:{test_loss}\n")

    # 保存到tensorboard
    # 记录全连接层参数
    for name, param in model.named_parameters():
        if 'linear' in name:
            tensorboard_writer.add_histogram(name, param.clone().cpu().data.numpy(), global_step=epoch)

    tensorboard_writer.add_scalar('Accuracy/train', train_accuracy, epoch)
    tensorboard_writer.add_scalar('Loss/train', train_loss, epoch)
    tensorboard_writer.add_scalar('Accuracy/test', test_accuracy, epoch)
    tensorboard_writer.add_scalar('Loss/test', test_loss, epoch)

def plot_txt(log_txt_loc):
    with open(log_txt_loc, 'r') as f:
        log_data = f.read()

    # 解析日志数据
    epochs = []
    train_accuracies = []
    train_losses = []
    test_accuracies = []
    test_losses = []

    for line in log_data.strip().split('\n'):
        epoch, train_acc, train_loss, test_acc, test_loss = line.split('\t')
        epochs.append(int(epoch.split(':')[1]))
        train_accuracies.append(float(train_acc.split(':')[1]))
        train_losses.append(float(train_loss.split(':')[1]))
        test_accuracies.append(float(test_acc.split(':')[1]))
        test_losses.append(float(test_loss.split(':')[1]))

    # 创建折线图
    plt.figure(figsize=(10, 5))

    # 训练数据
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_accuracies, label='Train Accuracy')
    plt.plot(epochs, test_accuracies, label='Test Accuracy')
    plt.title('Training Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    # 设置横坐标刻度为整数
    plt.xticks(range(min(epochs), max(epochs) + 1))

    # 测试数据
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, test_losses, label='Test Loss')
    plt.title('Testing Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    # 设置横坐标刻度为整数
    plt.xticks(range(min(epochs), max(epochs) + 1))

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    log_root = "logs"
    log_txt_loc = os.path.join(log_root,"log.txt")
    if os.path.isdir(log_root):
        pass
    else:
        os.mkdir(log_root)

    # 指定TensorBoard数据的保存地址
    tensorboard_writer = SummaryWriter(log_root)

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
        print("x shape [N, C, H, W]: ", x.shape)
        print("y shape: ", y.shape)
        break

    # 3. 构建模型
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using {device}")

    model = NeuralNetwork().to(device)
    print(model)

    # 模拟输入，大小和输入相同即可
    # 模拟的输入： 1, 1, 28, 28
    init_img = torch.zeros((1, 1, 28, 28), device=device)
    tensorboard_writer.add_graph(model, init_img)

    # 4. 损失函数: 交叉熵损失 （分类）
    loss_fn = nn.CrossEntropyLoss()

    # 5. 优化器
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    # 6. 训练和测试
    best_acc = 0
    epochs = 5
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_acc, train_loss = train(train_dataloader, model, loss_fn, optimizer)
        test_acc, test_loss = test(test_dataloader, model, loss_fn)

        writedata(log_txt_loc,tensorboard_writer,t,train_acc,train_loss,test_acc,test_loss)

        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), os.path.join(log_root,"best.pth"))

        torch.save(model.state_dict(), os.path.join(log_root,"last.pth"))

    print("Done!")
    plot_txt(log_txt_loc)
    tensorboard_writer.close()

# tensorboard --logdir .