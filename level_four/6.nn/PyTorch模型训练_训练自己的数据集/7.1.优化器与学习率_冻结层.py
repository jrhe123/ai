import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm  # pip install tqdm
import matplotlib.pyplot as plt
import os
from torchsummary import summary

from torch.utils.tensorboard import SummaryWriter

import wandb
import datetime

from MyModel import SimpleCNN

from torchvision.models import resnet18, ResNet18_Weights

from MYDataset import MyDataset

# # 定义训练函数
def train(dataloader, model, loss_fn, optimizer):
    # 初始化训练数据集的大小和批次数量
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    # 设置模型为训练模式
    model.train()
    # 初始化总损失和正确预测数量
    loss_total = 0
    correct = 0
    # 遍历数据加载器中的所有数据批次
    for X, y in tqdm(dataloader):
        # 将数据和标签移动到指定设备（例如GPU）
        X, y = X.to(device), y.to(device)
        # 使用模型进行预测
        pred = model(X)
        # 计算正确预测的数量
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        # 计算预测结果和真实结果之间的损失
        loss = loss_fn(pred, y)
        # 累加总损失
        loss_total += loss.item()
        # 执行反向传播，计算梯度
        loss.backward()
        # 更新模型参数
        optimizer.step()
        # 清除梯度信息
        optimizer.zero_grad()

    # 计算平均损失和准确率
    loss_avg = loss_total / num_batches
    correct /= size
    # 返回准确率和平均损失，保留三位小数
    return round(correct, 3), round(loss_avg,3)

# 定义测试函数
def test(dataloader, model, loss_fn):
    # 初始化测试数据集的大小和批次数量
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    # 设置模型为评估模式
    model.eval()

    # 初始化测试损失和正确预测数量
    test_loss, correct = 0, 0

    # 不计算梯度，以提高计算效率并减少内存使用
    with torch.no_grad():
        # 遍历数据加载器中的所有数据批次
        for X, y in tqdm(dataloader):
            # 将数据和标签移动到指定设备（例如GPU）
            X, y = X.to(device), y.to(device)
            # 使用模型进行预测
            pred = model(X)
            # 累加预测损失
            test_loss += loss_fn(pred, y).item()
            # 累加正确预测的数量
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    # 计算平均测试损失和准确率
    test_loss /= num_batches
    correct /= size

    # 返回准确率和平均测试损失，保留三位小数
    return round(correct, 3), round(test_loss, 3)

def writedata(txt_log_name, tensorboard_writer, epoch, train_accuracy, train_loss, test_accuracy, test_loss):
    # 保存到文档
    with open(txt_log_name, "a+") as f:
        f.write(f"Epoch:{epoch}\ttrain_accuracy:{train_accuracy}\ttrain_loss:{train_loss}\ttest_accuracy:{test_accuracy}\ttest_loss:{test_loss}\n")

    # 保存到tensorboard
    # 记录全连接层参数
    for name, param in model.named_parameters():
        tensorboard_writer.add_histogram(name, param.clone().cpu().data.numpy(), global_step=epoch)

    tensorboard_writer.add_scalar('Accuracy/train', train_accuracy, epoch)
    tensorboard_writer.add_scalar('Loss/train', train_loss, epoch)
    tensorboard_writer.add_scalar('Accuracy/test', test_accuracy, epoch)
    tensorboard_writer.add_scalar('Loss/test', test_loss, epoch)

    wandb.log({"Accuracy/train": train_accuracy,
               "Loss/train": train_loss,
               "Accuracy/test": test_accuracy,
               "Loss/test": test_loss})

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
    batch_size = 64
    init_lr = 1e-3
    epochs = 20
    log_root = "logs_resnet18_pretrain_fix"
    log_txt_loc = os.path.join(log_root,"log.txt")

    # 指定TensorBoard数据的保存地址
    tensorboard_writer = SummaryWriter(log_root)

    # WandB信息保存地址
    run_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    wandb.init(
        dir=log_root,
        project='Flower Classify',
        name=f"run_resnet18_pretrain_fixparm",
        config={
            "learning_rate": init_lr,
            "batch_size": batch_size,
            "model": "resnet18",
            "dataset": "Flower10",
            "epochs": epochs,
        }
    )

    if os.path.isdir(log_root):
        pass
    else:
        os.mkdir(log_root)

    train_data = MyDataset("train.txt",train_flag=True)
    test_data = MyDataset("test.txt",train_flag=False)

    # 创建数据加载器
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    for X, y in test_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

    # 指定设备
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    print(f"Using {device} device")


    """
        使用resnet18进行预训练
    """

    # 加载具有预训练权重的ResNet18模型
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    # 保存模型权重到本地文件
    torch.save(model.state_dict(), 'resnet18_weights.pth')
    print("ResNet18 weights downloaded and saved as 'resnet18_weights.pth'")

    pretrain_model = resnet18(pretrained=False)
    print(resnet18)

    num_ftrs = pretrain_model.fc.in_features    # 获取全连接层的输入
    pretrain_model.fc = nn.Linear(num_ftrs, 10)  # 全连接层改为不同的输出

    # 预先训练好的参数， 'https://download.pytorch.org/models/resnet34-333f7ec4.pth'
    pretrained_dict = torch.load('resnet18_weights.pth')

    # # 弹出fc层的参数
    pretrained_dict.pop('fc.weight')
    pretrained_dict.pop('fc.bias')

    # # 自己的模型参数变量，在开始时里面参数处于初始状态，所以很多0和1
    model_dict = pretrain_model.state_dict()

    # # 去除一些不需要的参数
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    # # 模型参数列表进行参数更新，加载参数
    model_dict.update(pretrained_dict)

    # 改进过的预训练模型结构，加载刚刚的模型参数列表
    pretrain_model.load_state_dict(model_dict)

    '''
        冻结部分层
    '''
    # 将满足条件的参数的 requires_grad 属性设置为False
    for name, value in pretrain_model.named_parameters():
        if (name != 'fc.weight') and (name != 'fc.bias'):
            value.requires_grad = False
    #
    # filter 函数将模型中属性 requires_grad = True 的参数选出来
    params_selected = filter(lambda p: p.requires_grad, pretrain_model.parameters())    # 要更新的参数在 params_selected 当中


    model = pretrain_model.to(device)



    print(model)
    summary(model, (3,224,224))

    # 模拟输入，大小和输入相同即可
    init_img = torch.zeros((1, 3, 224, 224), device=device)
    tensorboard_writer.add_graph(model, init_img)

    # 添加wandb的模型记录
    wandb.watch(model, log='all', log_graph=True)

    # 定义损失函数
    loss_fn = nn.CrossEntropyLoss()
    # 定义优化器
    optimizer = torch.optim.SGD(params_selected, lr=init_lr)

    best_acc = 0
    # 定义循环次数，每次循环里面，先训练，再测试
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_acc, train_loss = train(train_dataloader, model, loss_fn, optimizer)
        test_acc, test_loss = test(test_dataloader, model, loss_fn)
        writedata(log_txt_loc, tensorboard_writer,t,train_acc,train_loss,test_acc,test_loss)

        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), os.path.join(log_root,"best.pth"))

        torch.save(model.state_dict(), os.path.join(log_root,"last.pth"))

    print("Done!")

    tensorboard_writer.close()
    wandb.finish()
    plot_txt(log_txt_loc)
