import json
import logging
import os
import pickle
from os import listdir, makedirs
from os.path import join, exists, basename
import torch.nn as nn
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm
import pandas as pd
from utils.abstract import AbstractDetector
from utils.flatten import flatten_model, flatten_models
from utils.healthchecks import check_models_consistency
from utils.models import create_layer_map, load_model, \
    load_models_dirpath
from utils.padding import create_models_padding, pad_model
from utils.reduction import (
    fit_feature_reduction_algorithm,
    use_feature_reduction_algorithm,
)
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler
from archs import Net2, Net3, Net4, Net5, Net6, Net7, Net2r, Net3r, Net4r, Net5r, Net6r, Net7r, Net2s, Net3s, Net4s, Net5s, Net6s, Net7s
import torch
from torch.autograd import Variable
import pdb

# from sklearn.ensemble import RandomForestRegressor
def load_data(example_path='example_data', img_format='npy'):
    # pdb.set_trace()
    data = [os.path.join(example_path, ex) for ex in os.listdir(example_path) if ex.endswith(img_format)]

    labels = [int(ex.split('_')[1][0]) for ex in os.listdir(example_path) if ex.endswith(img_format)]

    # labels = [ex for ex in os.listdir(example_path) if ex.endswith(img_format)]

    batch = np.zeros((len(data),21),dtype = 'float32')
    # batch = np.zeros((len(data),2,768),dtype = 'float32')
    # img_org = torch.empty(len(data),256,256,3)

    for i,x in enumerate(data):
        img = np.load(x)
        # batch[i] = img.reshape(-1)
        batch[i] = torch.FloatTensor(img).reshape(-1)
        # batch[i] = torch.FloatTensor(img)
    return batch
    # return batch, torch.LongTensor(labels)

class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        self.fc1 = nn.Linear(20 , 50)  # 6*6 from image dimension

        # self.dp1  = nn.Dropout(0.15)
        self.fc2 = nn.Linear(50, 10)

        # self.bn2 = nn.BatchNorm1d(250)
        # self.dp2  = nn.Dropout(0.15)
        # self.relu = nn.ReLU(inplace=True)
        self.act = nn.ReLU()
        # self.act = nn.Softmax()
        self.fc3 = nn.Linear(10, 1)
        # self.fc3 = nn.Linear(250, 50)
        # self.fc4 = nn.Linear(10, 2)
    def forward(self, x):
        # import pdb; pdb.set_trace()
        # x = self.bn1(x)
        x = x.view(x.size(0),-1)
        x = self.fc1(x)
        # x = self.dp1(self.act(self.fc1(x)))
        # x = self.act(self.fc2(x))
        x = self.fc2(x)
        # x = self.dp2(x)
        x = self.act(x)
        x = self.fc3(x)
        # x = self.act(x)
        # x = self.fc4(x)
        return x

data_path = './counter'
# batch, label = load_data(data_path,img_format='npy')
batch= load_data(data_path,img_format='npy')
total_data = [[batch[i][:-1], batch[i][-1]] for i in range(len(batch))]
# total_data = [[batch[i][:-2], label[i]] for i in range(len(batch))]
model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr= 0.01)
# model = RandomForestRegressor(max_depth=2,random_state=0)
# model.fit(batch[:int(len(batch)*3/4)], label[:int(len(batch)*3/4)])

# pred = model.predict(batch[int(len(batch)*3/4):])
# norm_pred = np.round(pred)
# ground = label[int(len(batch)*3/4):] 

# error = sum(abs(norm_pred-ground.numpy()))
# print("error is", error, "total:",len(ground))
# label[int(len(batch)*3/4):] 
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
batch_size = 32
# pdb.set_trace()
train_loader = DataLoader(
    total_data[:int(len(batch)*4/5)],
    batch_size= batch_size, shuffle=True)

# test_loader = DataLoader(
    # total_data[int(len(batch)*3/4):],
    # batch_size= test_batch_size, shuffle=True)
test_loader = DataLoader(
    total_data[int(len(batch)*4/5):],
    batch_size= 128, shuffle=True)

def train(epoch):
    model.to(device)
    model.train()
    
    for batch_idx, [data, target] in enumerate(train_loader):
        # pdb.set_trace()
        data, target = data.to(device), target.to(device)
        data, target = Variable(data), Variable(target)

        # pdb.set_trace()
        # data.type = torch.FloatTensor
        optimizer.zero_grad()
        output = model(data)
        # pdb.set_trace()
        target = target.type(torch.FloatTensor)
        loss = criterion(output.reshape(-1), target)

        # if epoch%40==0:
        #     optimizer.param_groups[0]['lr']=optimizer.param_groups[0]['lr']*0.1

        # optimizer.zero_grad()
        loss.backward()
        # for p in list(model.parameters()):
        #     if hasattr(p,'org'):
        #         p.data.copy_(p.org)
        optimizer.step()
        # for p in list(model.parameters()):
        #     if hasattr(p,'org'):
        #         p.org.copy_(p.data.clamp_(-1,1))
    if epoch % 100 == 0:
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}. lr:{}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item(), optimizer.param_groups[0]['lr']))

        # if batch_idx % 100 == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}. lr:{}'.format(
        #         epoch, batch_idx * len(data), len(train_loader.dataset),
        #         100. * batch_idx / len(train_loader), loss.item(), optimizer.param_groups[0]['lr']))

def test(best_loss):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for [data, target] in test_loader:
            # print("###Test target is:", target)
            # import pdb; pdb.set_trace()
            # data = torch.FloatTensor(data)
            # target = torch.FloatTensor(target)
            # if args.cuda:
            #     data, target = data.cuda(), target.cuda()
            data, target = data.to(device), target.to(device)
            data, target = Variable(data), Variable(target)
            # data.type = torch.FloatTensor
            output = model(data)
            # import pdb; pdb.set_trace()
            target = target.type(torch.FloatTensor)
            test_loss += criterion(output.reshape(-1), target).item() # sum up batch loss
            
            # loss = criterion(output.reshape(-1), target)
            # print ('test_loss is', test_loss)
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            # print("pred.eq(target.data.view_as(pred)) is", pred.eq(target.data.view_as(pred)))
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    if test_loss < best_loss:
        # if is_master() and 100 - val_prec1 < best_val:
        best_loss = test_loss
        print("saving best model... Auc is ",100. * correct / len(test_loader.dataset))
        # best_val = 100 - val_prec1
        torch.save(
            {
                'model': model.state_dict(),
            },
            os.path.join('./model', 'best.pt'))
        print('New best loss: {:.3f}'.format(best_loss))
        # save latest checkpoint
    return best_loss

best_loss = 1
best_loss_new = 1
for epoch in range(1, 2000 + 1):
    train(epoch)
    best_loss = best_loss_new
    if epoch%100==0:
        best_loss_new = test(best_loss)
    if epoch%400==0:
        optimizer.param_groups[0]['lr']=optimizer.param_groups[0]['lr']*0.1