import torch
import torch.utils.data as Data
import torchvision
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import time
import os
# Configurations
from torch.nn import init

OLD_INDEX = ['Pclass','Sex','Age','UknAge','SibSp','Parch','Fare','Embarked','Survived']
OLD_INDEX_2 = ['Pclass','Sex','Age','UknAge','SibSp','Parch','Fare','Embarked']
NEW_INDEX = ['Age', 'UknAge', 'Fare',
             'Pclass_0', 'Pclass_1', 'Pclass_2',
             'Sex_0', 'Sex_1',
             'SibSp_0', 'SibSp_1', 'SibSp_2', 'SibSp_3', 'SibSp_4', 'SibSp_5', 'SibSp_8',
             'Parch_0', 'Parch_1', 'Parch_2', 'Parch_3', 'Parch_4', 'Parch_5', 'Parch_6', 'Parch_9',
             'Embarked_0', 'Embarked_1', 'Embarked_2',
             'Survived'
            ]
NEW_INDEX_2 = ['Age', 'UknAge', 'Fare',
             'Pclass_0', 'Pclass_1', 'Pclass_2',
             'Sex_0', 'Sex_1',
             'SibSp_0', 'SibSp_1', 'SibSp_2', 'SibSp_3', 'SibSp_4', 'SibSp_5', 'SibSp_8',
             'Parch_0', 'Parch_1', 'Parch_2', 'Parch_3', 'Parch_4', 'Parch_5', 'Parch_6', 'Parch_9',
             'Embarked_0', 'Embarked_1', 'Embarked_2',
            ]
MAP_Sex = {'male':0,'female':1}
MAP_Embarked = {'C':0,'Q':1,'S':2}
ONE_HOT = [[1,0],[0,1]]
FEATURES = 26
batch_size = 20
#数据预处理
def preprocess( data ):
    # Data Cleaning
    data = pd.DataFrame(data,columns=OLD_INDEX)
    data['UknAge'] = data['UknAge'].fillna(0)
    data['Survived'] = data['Survived'].fillna(0)
    #### print(data[data['Age'].isnull()])
    data.loc[data['Age'].isnull(),'UknAge'] = 1
    data['Age'] = data['Age'].fillna(0)
    #### print(data[data['Fare'].isnull()])
    data['Fare'] = data['Fare'].fillna(14.4)
    #### print(data[data['Embarked'].isnull()])
    data['Embarked'] = data['Embarked'].fillna('C')
    #### One-hot Encoding
    data['Pclass'] -= 1
    data['Sex'] = data['Sex'].map(MAP_Sex)
    data['Embarked'] = data['Embarked'].map(MAP_Embarked)
    data = pd.get_dummies(data,columns=['Pclass','Sex','SibSp','Parch','Embarked'])
    data = pd.DataFrame(data,columns=NEW_INDEX)
    data = data.fillna(0)
    #### Normalization
    for col in NEW_INDEX:
        maximum = data[col].max()
        if maximum > 0:
            data[col] /= maximum
    #### To List

    temp = np.array(data)
    print(temp.shape)
    train_data = torch.FloatTensor([temp[j][:FEATURES] for j in range(temp.shape[0] - 91)])
    train_label = torch.FloatTensor([temp[j][FEATURES] for j in range(temp.shape[0] - 91)])
    test_data = torch.FloatTensor([temp[j][:FEATURES] for j in range(800, temp.shape[0])])
    test_label = torch.FloatTensor([temp[j][FEATURES] for j in range(800, temp.shape[0])])
    # print(data,label)
    # return data

    # data = [[torch.FloatTensor(temp[j][:FEATURES]),
    #          torch.FloatTensor(int(temp[j][FEATURES]))] for j in range(temp.shape[0])]
    # print(data)
    print(test_data.shape,test_label.shape)
    return train_data,train_label,test_data,test_label
def preprocess_test( data ):
    # Data Cleaning
    data = pd.DataFrame(data,columns=OLD_INDEX_2)
    data['UknAge'] = data['UknAge'].fillna(0)
    #### print(data[data['Age'].isnull()])
    data.loc[data['Age'].isnull(),'UknAge'] = 1
    data['Age'] = data['Age'].fillna(0)
    #### print(data[data['Fare'].isnull()])
    data['Fare'] = data['Fare'].fillna(14.4)
    #### print(data[data['Embarked'].isnull()])
    data['Embarked'] = data['Embarked'].fillna('C')
    #### One-hot Encoding
    data['Pclass'] -= 1
    data['Sex'] = data['Sex'].map(MAP_Sex)
    data['Embarked'] = data['Embarked'].map(MAP_Embarked)
    data = pd.get_dummies(data,columns=['Pclass','Sex','SibSp','Parch','Embarked'])
    data = pd.DataFrame(data,columns=NEW_INDEX_2)
    data = data.fillna(0)
    #### Normalization
    for col in NEW_INDEX:
        maximum = data[col].max()
        if maximum > 0:
            data[col] /= maximum
    #### To List

    temp = np.array(data)
    print(temp.shape)
    validate_data = torch.FloatTensor([temp[j][:FEATURES - 1] for j in range(temp.shape[0])])
    print(test_data.shape,test_label.shape)
    return validate_data

origin_data = pd.read_csv("train.csv")
train_data,train_label,test_data,test_label = preprocess(origin_data)
# 将训练数据的特征和标签组合
dataset = Data.TensorDataset(train_data,train_label)
# 随机读取小批量
data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)
net = nn.Sequential(
    nn.Linear(FEATURES, 1)
    # 此处还可以传入其他层
)


init.normal_(net[0].weight, mean=0, std=0.01)
init.constant_(net[0].bias, val=0)
loss = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.03)

num_epochs = 50
for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        output = net(X)
        output = torch.sigmoid(output)
        l = loss(output, y.view(-1, 1))
        mask = output.ge(0.5).float()
        # print(mask,y)# 以0.5为阈值进行分类
        correct = 0
        for j in range(X.size(0)):
            if mask[j] == y[j]:
                correct += 1# 计算正确预测的样本个数
        acc = correct/ X.size(0)  # 计算精度
        optimizer.zero_grad() # 梯度清零，等价于net.zero_grad()
        l.backward()
        optimizer.step()
    print('epoch %d, loss: %f, acc: %f' % (epoch, l.item(), acc))

output = net(test_data)
output = torch.sigmoid(output)
l = loss(output, test_label.view(-1, 1))
mask = output.ge(0.5).float()
# print(mask,y)# 以0.5为阈值进行分类
correct = 0
for j in range(test_data.size(0)):
    if mask[j] == test_label[j]:
        correct += 1  # 计算正确预测的样本个数
acc = correct / test_data.size(0)  # 计算精度
print('loss: %f, acc: %f' % (l.item(), acc))

