# Copyright 2021 longpeng2008. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# If you find any problem,please contact us
# Author: longpeng
# Email: longpeng2008to2012@gmail.com

#coding:utf8
import numpy as np
import matplotlib.pyplot as plt
import sys

## 导入逻辑回归模型函数
from sklearn.linear_model import LogisticRegression

## 读取数据
lines = open('level_two/1.sklearn/iris.txt').readlines()
x_features = [] ## 特征数组
y_label = [] ## 标签数组
table = {"setosa":0,"versicolor":1,"virginica":2} #创建标签转换表

for i in range(1,len(lines)):
   line = lines[i].strip().split(' ')
   feature = line[1:-1] ## 获得特征
   label = table[str(line[-1].split('\"')[1])] ## 获得标签
   x_features.append(feature)
   y_label.append(label)
   #print(str(feature)+' '+str(label))

## 调用逻辑回归模型
lr_clf = LogisticRegression()
lr_clf = lr_clf.fit(x_features, y_label) #其拟合方程为 y=w0+w1*x1+w2*x2

## 查看模型的w和b
print('the weight of Logistic Regression:',lr_clf.coef_)
print('the intercept(w0) of Logistic Regression:',lr_clf.intercept_)

