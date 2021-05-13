import numpy as np
from models import *
import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
from N_utils_HD import adaptive_MMDu

def cwloss(output, target,confidence=50, num_classes=10):
    target = target.data
    target_onehot = torch.zeros(target.size() + (num_classes,))
    target_onehot = target_onehot.cuda()
    target_onehot.scatter_(1, target.unsqueeze(1), 1.)
    target_var = Variable(target_onehot, requires_grad=False)
    real = (target_var * output).sum(1)
    other = ((1. - target_var) * output - target_var * 10000.).max(1)[0]
    loss = -torch.clamp(real - other + confidence, min=0.)  # equiv to max(..., 0.)
    loss = torch.sum(loss)
    return loss

def MMDG(model_semantic,x_adv,data):
    s1=model_semantic(data)
    s2=model_semantic(x_adv)
    N1=128

    S = torch.cat([s1.cpu(), s2.cpu()], 0).cuda()
    Sv = S.view(2 * N1, -1)

    TEMPa = adaptive_MMDu(Sv, N1, 1000)
    mmd_value_tempa = -1 * (TEMPa[0] + 10 ** (-8))
    mmd_std_tempa = torch.sqrt(TEMPa[1] + 10 ** (-8))
    STAT_adaptive = torch.div(mmd_value_tempa, mmd_std_tempa)

    return STAT_adaptive

def pgd1(model, model_semantic, data, target, epsilon, step_size, num_steps,loss_fn,category,rand_init):
    model.eval()

    if category == "trades":
        x_adv = data.detach() + 0.001 * torch.randn(data.shape).cuda().detach() if rand_init else data.detach()

    if category == "Madry":
        x_adv = data.detach() + torch.from_numpy(np.random.uniform(-epsilon, epsilon, data.shape)).float().cuda() if rand_init else data.detach()
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    for k in range(num_steps):
        x_adv.requires_grad_()
        output = model(x_adv)
        MMD_output = MMDG(model_semantic,x_adv,data)
        loss_1=nn.CrossEntropyLoss()(output, target)
        loss_2=MMD_output

        model.zero_grad()
        with torch.enable_grad():
            if loss_fn == "cent":
                loss_adv = loss_1 - loss_2

        loss_adv.backward()
        eta = step_size * x_adv.grad.sign()
        x_adv = x_adv.detach() + eta
        x_adv = torch.min(torch.max(x_adv, data - epsilon), data + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    return x_adv

def pgd2(model, model_semantic, data, target, epsilon, step_size, num_steps,loss_fn,category,rand_init):
    model.eval()

    if category == "trades":
        x_adv = data.detach() + 0.001 * torch.randn(data.shape).cuda().detach() if rand_init else data.detach()
    if category == "Madry":
        x_adv = data.detach() + torch.from_numpy(np.random.uniform(-epsilon, epsilon, data.shape)).float().cuda() if rand_init else data.detach()
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    for k in range(num_steps):
        x_adv.requires_grad_()
        MMD_output = MMDG(model_semantic,x_adv,data)
        loss_2=MMD_output

        model.zero_grad()
        with torch.enable_grad():
            if loss_fn == "cent":
                loss_adv = - loss_2

        loss_adv.backward()
        eta = step_size * x_adv.grad.sign()
        x_adv = x_adv.detach() + eta
        x_adv = torch.min(torch.max(x_adv, data - epsilon), data + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    return x_adv


def co_adaptive_generate(model, model_semantic, test_loader, perturb_steps, epsilon, step_size, loss_fn, category, rand_init):
    model.eval()
    bool_i=0
    with torch.enable_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            x_adv = pgd1(model,model_semantic,data,target,epsilon,step_size,perturb_steps,loss_fn,category,rand_init=rand_init)
            if bool_i == 0:
                X_adv = x_adv.clone().cpu()
            else :
                X_adv = torch.cat((X_adv, x_adv.clone().cpu()), 0)
            bool_i +=1
    return X_adv

def adaptive_generate(model, model_semantic, test_loader, perturb_steps, epsilon, step_size, loss_fn, category, rand_init):
    model.eval()
    bool_i=0
    with torch.enable_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            x_adv = pgd2(model,model_semantic,data,target,epsilon,step_size,perturb_steps,loss_fn,category,rand_init=rand_init)
            if bool_i == 0:
                X_adv = x_adv.clone().cpu()
            else :
                X_adv = torch.cat((X_adv, x_adv.clone().cpu()), 0)
            bool_i +=1
    return X_adv
