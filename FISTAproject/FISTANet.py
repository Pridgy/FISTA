import math
import os
import pandas as pd
import torch
from datasets import CaveDataset
from torch.utils.data import DataLoader, Dataset
import numpy as np
from argparse import ArgumentParser
import scipy.io as sio
from utils import *
import torch.optim
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import torch.nn.functional as F
import datetime
import time

try:
    # The flag below controls whether to allow TF32 on matmul. This flag defaults to True.
    torch.backends.cuda.matmul.allow_tf32 = False
    # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
    torch.backends.cudnn.allow_tf32 = False
except:
    pass


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        # init.xavier_uniform_(m.weight.data, gain=1.0)
        init.normal_(m.weight.data, mean=0.0, std=0.01)
        # init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        # init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 0.1, 0.01)
        # init.constant_(m.bias.data, 0.0)


# Define ISTA-Net Block
class BasicBlock(torch.nn.Module):
    def __init__(self):
        super(BasicBlock, self).__init__()

        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))
        self.soft_thr = nn.Parameter(torch.Tensor([0.01]))

        self.conv1_forward = nn.Conv2d(28, 56, kernel_size=3, stride=1, padding=1, bias=True).apply(weights_init_kaiming)
        self.conv2_forward = nn.Conv2d(56, 112, kernel_size=3, stride=1, padding=1, bias=True).apply(weights_init_kaiming)
        self.conv1_backward = nn.Conv2d(112, 56, kernel_size=3, stride=1, padding=1, bias=True).apply(weights_init_kaiming)
        self.conv2_backward = nn.Conv2d(56, 28, kernel_size=3, stride=1, padding=1, bias=True).apply(weights_init_kaiming)

    def forward(self, x, PhiTPhi, PhiTb):
        x = x - self.lambda_step * torch.mul(x, PhiTPhi)
        x = x + self.lambda_step * PhiTb

        # print(x.shape)      # bs*28*256*256

        x_input = x
        # F(x)
        x = self.conv1_forward(x)
        x = F.relu(x)
        x_forward = self.conv2_forward(x)

        # Soft(F(x))
        x_st = torch.mul(torch.sign(x_forward), F.relu(torch.abs(x_forward) - self.soft_thr))

        # F(X)-
        x = self.conv1_backward(x_st)
        x = F.relu(x)
        x_backward = self.conv2_backward(x)

        x_pred = x_backward.view(-1, 28, 256, 256)

        x = self.conv1_backward(x_forward)
        x = F.relu(x)
        x_est = self.conv2_backward(x)
        symloss = x_est - x_input

        return [x_pred, symloss, x_st]


# Define ISTA-Net
class ISTANet(torch.nn.Module):

    def __init__(self, LayerNo):
        super(ISTANet, self).__init__()
        onelayer = []
        self.LayerNo = LayerNo

        for i in range(LayerNo):
            onelayer.append(BasicBlock())

        self.fcs = nn.ModuleList(onelayer)

    def forward(self, x, y, Phi):
        x_old = x
        x_next = x
        t = torch.Tensor([1.]).cuda()
        PhiTPhi = torch.mul(Phi, Phi)
        # PhiTb = torch.mul(Phi, y)
        PhiTb = y
        layers_sym = []  # for computing symmetric loss
        layers_st = []

        for i in range(self.LayerNo):
            tk = (1 + math.sqrt(4 * math.pow(t, 2))) / 2.
            [x, layer_sym, layer_st] = self.fcs[i](x_next, PhiTPhi, PhiTb)
            x_next = x + ((t - 1) / tk) * (x - x_old)
            x_old = x
            t = tk
            layers_sym.append(layer_sym)
            layers_st.append(layer_st)

        x_final = x

        return [x_final, layers_sym, layers_st]


epoch_sam_num = 242
batch_size = 2
layer_num = 6
epochs = 1000
learning_rate = 0.0001
input_mask = 'Phi'
model = ISTANet(layer_num)
model = nn.DataParallel(model)
model = model.to(device)

train_path = './datasets/cave_1024_28'
mask_path = './mask'
train_set = LoadTraining(train_path)

# mask3d_batch_train, input_mask_train, PhiTPhi = init_mask(mask_path, input_mask, batch_size)
# print(mask3d_batch_train.shape, input_mask_train.shape, PhiTPhi.shape)  # torch.Size([32, 28, 256, 256]) torch.Size([32, 28, 256, 310])
mask3d = generate_masks('./mask', batch_size=batch_size)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9, last_epoch=-1)
mse = torch.nn.MSELoss().cuda()
model_dir = './model'

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
psnr_list = []
for epoch in range(epochs):
    # scheduler.step(epoch)
    print('epoch:%d' % epoch)
    psnr = 0
    batch_num = 242 // 2
    epoch_loss = 0
    psnr_total = 0
    start_time = time.time()
    for i in range(batch_num):
        gt_batch = shuffle_crop(train_data=train_set, crop_size=256, batch_size=batch_size)
        gt = Variable(gt_batch).cuda().float()
        # print(gt.shape) # 32*28*256*256

        y, x0 = gen_meas_torch(gt, mask3d)
        # print(x0.shape, y.shape)        # shape(4*28*256*256), shape(4*28*256*256)

        [model_out, loss_layers_sym, loss_layers_sparse] = model(x0, y,  mask3d)

        psnr = compare_psnr(gt.detach().cpu().numpy(), model_out.detach().cpu().numpy(), data_range=1.0)
        psnr_total = psnr_total + psnr

        loss_discrepancy = mse(model_out, gt)

        loss_constraint = torch.mean(torch.pow(loss_layers_sym[0], 2))
        for k in range(layer_num - 1):
            loss_constraint += torch.mean(torch.pow(loss_layers_sym[k + 1], 2))

        loss_sparse = torch.mean(torch.pow(loss_layers_sparse[0], 2))
        for k in range(layer_num - 1):
            loss_sparse += torch.mean(torch.pow(loss_layers_sparse[k + 1], 2))

        gamma = torch.Tensor([0.001]).to(device)
        bada = torch.Tensor([0.0001]).to(device)
        loss_all = loss_discrepancy + torch.mul(gamma, loss_constraint)
        epoch_loss += loss_all.item()

        optimizer.zero_grad()
        loss_all.backward()
        optimizer.step()
        # if (i+1) % 10 == 0:
        #     print('%2d %3d loss = %.10f time = %s' % ((epoch+1), (i+1), epoch_loss / (i+1), datetime.datetime.now()))
        if (epoch+1) % 100 == 0:
            torch.save(model.state_dict(), './model/fast_net_params_%d.pkl' % (epoch+1))
    print('%2d %3d loss = %.10f time = %s' % ((epoch + 1), (i + 1), epoch_loss / batch_num, datetime.datetime.now()))
    elapsed_time = time.time() - start_time
    # scheduler.step()
    print('epoch = %4d , loss = %.16f , Avg PSNR = %.4f ,time = %4.2f s' % (epoch + 1, epoch_loss / batch_num, psnr_total / (i + 1), elapsed_time))
    psnr_list.append(psnr_total / (i + 1))
df = pd.DataFrame(data=psnr_list)
df.to_csv('./psnr/fast_psnr_list.csv', encoding='gbk')




