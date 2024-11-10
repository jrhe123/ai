import pickle
import time
import os
import numpy as np


# 模拟耗时的权重计算过程
def calculate_weights():
    print("开始计算权重...")
    time.sleep(5)  # 模拟耗时操作
    weights = np.random.rand(10, 10)  # 随机生成权重
    print("权重计算完成.")
    return weights


# 保存权重和epoch到文件
def save_weights(weights, epoch, filename='weights.pkl'):
    data = {'weights': weights, 'epoch': epoch}
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    print(f"权重和epoch已保存到{filename}.")


# 从文件加载权重和epoch
def load_weights(filename='weights.pkl'):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    print(f"权重和epoch已从{filename}加载.")
    return data['weights'], data['epoch']


# 主程序
def main():
    weights_file = 'weights.pkl'
    total_epochs = 100  # 假设我们总共需要训练100个epochs

    # 如果权重文件存在，则加载权重和epoch
    if os.path.exists(weights_file):
        weights, start_epoch = load_weights(weights_file)
    else:
        # 否则，从第一个epoch开始，并计算权重
        weights = calculate_weights()
        start_epoch = 0

    # 继续训练剩余的epochs
    for epoch in range(start_epoch, total_epochs):
        print(f"开始训练epoch {epoch}...")
        # 这里进行实际的训练代码...
        time.sleep(1)  # 模拟训练过程
        print(f"完成训练epoch {epoch}.")

        # 每个epoch结束后保存权重和epoch信息
        save_weights(weights, epoch, weights_file)

if __name__ == '__main__':
    main()
