import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
import os
from utils import *
import scipy.io as sio
import math


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


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

epochs = 1000
batch_size = 1
layer_num = 6
input_mask = 'Phi'
mask_path = './mask'
model = ISTANet(layer_num)
model = nn.DataParallel(model)
model = model.to(device)
model.load_state_dict(torch.load('./model/fast_net_params_1000.pkl'))

mask3d = generate_masks('./mask', batch_size)

# test_path = './datasets/KAIST_CVPR2021'
# test_data = LoadTest(test_path)
# print(test_data.shape)
# x0, y = gen_meas_torch(gt, mask3d_batch_train)
# train_data = LoadTraining('./datasets/cave_1024_28')
# index = 1
# crop_size = 256
# data = np.zeros((batch_size, crop_size, crop_size, 28), dtype=np.float32)
# for i in range(batch_size):
#     img = train_data[index]
#     h, w, _ = img.shape
#     x_index = np.random.randint(0, h - crop_size)
#     y_index = np.random.randint(0, w - crop_size)
#     data[i, :, :, :] = img[x_index:x_index + crop_size, y_index:y_index + crop_size, :]
# data = torch.from_numpy(np.transpose(data, (0, 3, 1, 2))).cuda().float()
psnr = 0
ssim = 0
with torch.no_grad():
    for i in range(1):
        # data_mat = sio.loadmat('./datasets/KAIST_CVPR2021/%d.mat' % (i + 1))
        # data = data_mat['HSI']
        data_mat = sio.loadmat('E:\datasets\TSA_simu_data\Truth\scene04.mat')
        data = data_mat['img']
        temp = np.zeros((1, 256, 256, 28), dtype=np.float32)
        img = data
        # h, w, _ = img.shape
        # x_index = np.random.randint(0, h - 256)
        # y_index = np.random.randint(0, w - 256)
        # temp[0, :, :, :] = img[x_index:x_index + 256, y_index:y_index + 256, :]
        temp[0, :, :, :] = img[:, :, :]
        temp = torch.from_numpy(np.transpose(temp, (0, 3, 1, 2))).cuda().float()
        # print(temp.shape)

        y, x0 = gen_meas_torch(temp, mask3d)
        # print(x0.shape)
        [model_out, loss_layers_sym, loss_layers_sparse] = model(x0, y, mask3d)

        print('before')
        print(torch_psnr(torch.reshape(x0, (28, 256, 256)), torch.reshape(temp, (28, 256, 256))), torch_ssim(x0, temp))

        psnr1 = torch_psnr(torch.reshape(model_out, (28, 256, 256)), torch.reshape(temp, (28, 256, 256)))
        ssim1 = torch_ssim(model_out, temp)
        print('after')
        print(psnr1, ssim1)
        psnr += psnr1
        ssim += ssim1

    print('means')
    print(psnr / 20)
    print(ssim/20)
