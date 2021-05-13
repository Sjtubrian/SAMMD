# -*- coding: utf-8 -*-

import argparse
import os
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch
from utils_HD import MatConvert, Pdist2, MMDu,SAMMD_WB

# Setup seeds
os.makedirs("images", exist_ok=True)
np.random.seed(819)
torch.manual_seed(819)
torch.cuda.manual_seed(819)
torch.backends.cudnn.deterministic = True
is_cuda = True

# parameters setting
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=1000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=100, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate for C2STs")
parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--n", type=int, default=500, help="number of samples in one set")
opt = parser.parse_args()
print(opt)
dtype = torch.float
device = torch.device("cuda:0")
cuda = True if torch.cuda.is_available() else False
N_per = 100 # permutation times
alpha = 0.05 # test threshold
N1 = opt.n # number of samples in one set
K = 20 # number of trails
N = 100 # number of test sets
N_f = 100.0 # number of test sets (float)

# Loss function
adversarial_loss = torch.nn.CrossEntropyLoss()

# Naming variables
ep_OPT = np.zeros([K])
s_OPT = np.zeros([K])
s0_OPT = np.zeros([K])
Results = np.zeros([7,K])


transform_test = transforms.Compose([transforms.ToTensor(),])
testset = datasets.CIFAR10(root='./data/cifar10', train=False, download=True, transform=transform_test)

test_loader = torch.utils.data.DataLoader(testset, batch_size=10000, shuffle=False, num_workers=0)

# Obtain CIFAR10 images
for i, (imgs, Labels) in enumerate(test_loader):
    Cifar_data_org = imgs
    label_all = Labels

Cifar_data_all= np.load('./Semantic_Cifar10_natural.npy')
index=np.load('./True_Index.npy')
Cifar_data_all=Cifar_data_all[index]
Cifar_data_org=Cifar_data_org[index]

ind_Cifar = np.random.choice(len(Cifar_data_all), len(Cifar_data_all), replace=False)
Cifar_data_all = Cifar_data_all[ind_Cifar]

data_all = Cifar_data_all[2000:]
data_all = torch.from_numpy(data_all).float()

Ind_all = np.arange(len(data_all))

Cifar_data_org = Cifar_data_org[ind_Cifar]
data_org = Cifar_data_org[2000:]

data_trans = np.load('./Semantic_Cifar10_adv.npy')
data_trans_org = np.load('./Cifar10_adv.npy')

ind_Cifar = np.random.choice(len(data_trans), len(data_trans), replace=False)
data_trans = data_trans[ind_Cifar]

data_trans_org = data_trans_org[ind_Cifar]

data_trans=data_trans[0:1000]

Ind_data = np.random.choice(len(data_trans), len(data_trans), replace=False)

data_trans = data_trans[Ind_data]
data_trans_org = data_trans_org[Ind_data]

data_trans=torch.from_numpy(data_trans).float()
data_trans_org= torch.from_numpy(data_trans_org).float()

Ind_v4_all = np.arange(len(data_trans))

# Loss function
adversarial_loss = torch.nn.CrossEntropyLoss()

# Repeat experiments K times (K = 10) and report average test power (rejection rate)
print(data_trans.shape)
print(data_all.shape)
for kk in range(K):
    torch.manual_seed(kk * 19 + N1)
    torch.cuda.manual_seed(kk * 19 + N1)
    np.random.seed(seed=1102 * (kk + 10) + N1

    if cuda:
        adversarial_loss.cuda()

    # Collect natural images
    Ind_tr = np.random.choice(len(data_all), N1, replace=False)
    Ind_te = np.delete(Ind_all, Ind_tr)
    train_data = []
    for i in Ind_tr:
       train_data.append([data_all[i], label_all[i]])

    dataloader = torch.utils.data.DataLoader(
        train_data,
        batch_size=opt.batch_size,
        shuffle=True,
    )

    # Collect adv images
    np.random.seed(seed=819 * (kk + 9) + N1)
    Ind_tr_v4 = np.random.choice(len(data_trans), N1, replace=False)
    Ind_te_v4 = np.delete(Ind_v4_all, Ind_tr_v4)
    New_CIFAR_tr = data_trans[Ind_tr_v4]
    New_CIFAR_te = data_trans[Ind_te_v4]

    # Initialize optimizers
    # Fetch training data
    s1 = data_all[Ind_tr]
    s2 = data_trans[Ind_tr_v4]
    S = torch.cat([s1.cpu(), s2.cpu()], 0).cuda()
    Sv = S.view(2 * N1, -1)

    s1 = data_org[Ind_tr]
    s2 = data_trans_org[Ind_tr_v4]
    S = torch.cat([s1.cpu(), s2.cpu()], 0).cuda()
    S_FEA = S.view(2 * N1, -1)

    # Train SAMMD

    np.random.seed(seed=1102)
    torch.manual_seed(1102)
    torch.cuda.manual_seed(1102)
    Dxy = Pdist2(Sv[:N1, :], Sv[N1:, :])
    Dxy_org = Pdist2(S_FEA[:N1, :], S_FEA[N1:, :])
    epsilonOPT = torch.log(MatConvert(np.random.rand(1) * 10 ** (-10), device, dtype))
    epsilonOPT.requires_grad = True
    sigma0 = Dxy.median()
    sigma0.requires_grad = True
    sigmaOPT = MatConvert(np.ones(1) * np.sqrt(2 * 32 * 32), device, dtype)
    sigmaOPT.requires_grad = True


    optimizer_sigma0 = torch.optim.Adam([sigma0]+[sigmaOPT]+[epsilonOPT], lr=0.0002)
    for t in range(opt.n_epochs):
        ep = torch.exp(epsilonOPT) / (1 + torch.exp(epsilonOPT))
        sigma = sigmaOPT ** 2
        TEMPa = MMDu(Sv, N1, S_FEA, sigma, sigma0, ep, is_smooth=True)
        mmd_value_tempa = -1 * (TEMPa[0] + 10 ** (-8))
        mmd_std_tempa = torch.sqrt(TEMPa[1] + 10 ** (-8))
        STAT_adaptive = torch.div(mmd_value_tempa, mmd_std_tempa)
        optimizer_sigma0.zero_grad()
        STAT_adaptive.backward(retain_graph=True)
        optimizer_sigma0.step()
        if t % 100 == 0:
            print("mmd: ", -1 * mmd_value_tempa.item(), "mmd_std: ", mmd_std_tempa.item(), "Statistic: ",
                  -1 * STAT_adaptive.item())
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # Compute test power of MMD-D and baselines

    H_adaptive = np.zeros(N)
    T_adaptive = np.zeros(N)
    M_adaptive = np.zeros(N)

    np.random.seed(1102)
    count_adp = 0

    for k in range(N):
        # Fetch test data
        np.random.seed(seed=1102 * (k + 1) + N1)
        data_all_te = data_all[Ind_te]
        N_te = len(data_trans)-N1
        Ind_N_te = np.random.choice(len(Ind_te), N_te, replace=False)#9900
        s1 = data_all_te[Ind_N_te]
        s2 = data_trans[Ind_te_v4]
        S = torch.cat([s1.cpu(), s2.cpu()], 0).cuda()
        Sv = S.view(2 * N_te, -1)

        data_all_te = data_org[Ind_te]
        s1 = data_all_te[Ind_N_te]
        s2 = data_trans_org[Ind_te_v4]
        S = torch.cat([s1.cpu(), s2.cpu()], 0).cuda()
        S_FEA = S.view(2 * N_te, -1)

        h_adaptive, threshold_adaptive, mmd_value_adaptive = SAMMD_WB(Sv, N_per, N_te, S_FEA, sigma, sigma0, ep, alpha, device, dtype)

        # Gather results

        count_adp = count_adp + h_adaptive

        print("SAMMD:", count_adp)

        H_adaptive[k] = h_adaptive
        T_adaptive[k] = threshold_adaptive
        M_adaptive[k] = mmd_value_adaptive


    # Print test power of SAMMD
    print("Reject rate_adaptive: ",H_adaptive.sum() / N_f)

    Results[3, kk] = H_adaptive.sum() / N_f

    print("Test Power of Baselines (K times): ")
    print(Results)
    print("Average Test Power of SAMMD (K times): ")
    print("SAMMD: ", (Results.sum(1) / (kk + 1))[3])
np.save('./Results_CIFAR10_' + str(N1) + 'SAMMD', Results)
