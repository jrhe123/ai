# Copyright 2021 longpeng2008. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# If you find any problem,please contact us
# Author: longpeng
# Email: longpeng2008to2012@gmail.com

import matplotlib.pyplot as plt

# coding:utf8
import numpy as np

# 输入数据
X = np.array([[1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
# 标签
Y = np.array([[0, 1, 1, 0]])
V = (
    np.random.random((3, 4)) - 0.5
) * 2  # 第一个网络层参数矩阵，初始化输入层权值,取值范围-1到1
W = (
    np.random.random((4, 1)) - 0.5
) * 2  # 第二个网络层参数矩阵，初始化输出层权值,取值范围-1到1


def get_show():
    # 正样本
    all_positive_x = [0, 1]
    all_positive_y = [0, 1]
    # 负样本
    all_negative_x = [0, 1]
    all_negative_y = [1, 0]

    plt.figure()
    plt.plot(all_positive_x, all_positive_y, "bo")
    plt.plot(all_negative_x, all_negative_y, "yo")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


# get_show()

lr = 0.11


# 激活函数(从0～1）
def sigmoid(x):
    x = 1 / (1 + np.exp(-x))
    return x


# 激活函数的导数，f'(x)=f(x)(1-f(x)),dsigmoid(x)=sigmoid(x)*(1-sigmoid(x))
def dsigmoid(x):
    x = x * (1 - x)
    return x


# 更新权值（2个权值矩阵，V和W）
def update():
    global X, Y, W, V, lr
    L1 = sigmoid(np.dot(X, V))  # 隐藏层输出(4*3)×(3*4)=(4,4)
    L2 = sigmoid(np.dot(L1, W))  # 输出层输出(4,4)×(4*1)=(4,1)
    L2_delta = (Y.T - L2) * dsigmoid(
        L2
    )  # 输出层的误差=下一层的误差*激活函数导数*与下一层的连接权重矩阵（全为1）
    L1_delta = L2_delta.dot(W.T) * dsigmoid(
        L1
    )  # 隐藏层的误差=下一层的误差*激活函数导数*与下一层的连接权重矩阵

    W_C = lr * L1.T.dot(L2_delta)  # 输出层参数更新值=学习率*误差*上一层的激活值
    V_C = lr * X.T.dot(L1_delta)  # 隐藏层参数更新=学习率*误差*上一层的激活值
    W = W + W_C
    V = V + V_C


errors = []  # 记录误差
for i in range(100000):
    update()  # 更新权值
    if i % 1000 == 0:  # 输出误差
        L1 = sigmoid(np.dot(X, V))
        L2 = sigmoid(np.dot(L1, W))
        errors.append(np.mean(np.abs(Y.T - L2)))
        print("Error:", np.mean(np.abs(Y.T - L2)))
plt.plot(errors)
plt.ylabel("errors")
plt.show()

L1 = sigmoid(np.dot(X, V))  # 隐藏层输出(4*3)×(3*4)=(4,4)
L2 = sigmoid(np.dot(L1, W))  # 输出层输出(4,4)×(4*1)=(4,1)

print(L2)


def classify(x):
    if x > 0.5:
        return 1
    else:
        return 0


for i in map(classify, L2):  # L2一共四个数
    print(i)
