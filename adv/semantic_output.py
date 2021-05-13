import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms
from models import *
import numpy as np
import os
import time

model = semantic_ResNet18().cuda()
ckpt = torch.load('./Res18_model/net_150.pth')
batchsize=128
model.load_state_dict(ckpt)
model.eval()

y=np.load('./Cifar10_adv.npy')

index1=np.load('./True_Index.npy')

y=y[index1]

number=y.shape[0]
x=torch.from_numpy(y).cuda()
bool_i=0
with torch.no_grad():
    for batch_num in range(int(number/batchsize)+1):
        x_batch=x[batchsize*batch_num:min(batchsize*(batch_num+1),number)]
        x_adv = model(x_batch)
        if bool_i == 0:
            X_adv = x_adv.clone().cpu()
        else :
            X_adv = torch.cat((X_adv, x_adv.clone().cpu()), 0)
        bool_i +=1
print(X_adv.shape)

np.save('Semantic_Cifar10_adv.npy',X_adv)

