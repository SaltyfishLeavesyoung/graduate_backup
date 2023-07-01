import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision

import numpy as np
from maptonode import MapToNode
from semgraphconv import SemGraphConv
from self_att import Self_Attention
from resnet import r2plus1d_18

import argparse
import csv
import torch
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from torch.nn import BCELoss
import os
import yaml
import yamlloader
import random
import matplotlib.pyplot as plt
import cv2

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import utils
from util import get_mylogger
from fa68pt3D import fa68pt3D
from util import getAdjMatrix

class UNet_Pretrained_Att(nn.Module):
    def __init__(self, n_channels, n_classes, adj):
        super(UNet_Pretrained_Att, self).__init__()
        self.adj = torch.tensor(adj).float()
        self.n_channels = n_channels
        self.n_classes = n_classes
        bilinear = True

        self.vgg = r2plus1d_18(pretrained=True)
        
        self.up1 = Up_1(bilinear)
        self.double_conv1 = DoubleConv_1(512, 128)

        self.up2 = Up_1(bilinear)
        self.double_conv2 = DoubleConv_1(256, 64)

        self.up3 = Up_1(bilinear)
        self.double_conv3 = DoubleConv_1(128, 64)

        self.up4 = Up_2(bilinear)
        self.double_conv4 = DoubleConv_1(64, 64)

        self.mtn = MapToNode(512, 6)
        self.gcn = SemGraphConv(16*16*4, 512, self.adj)
        self.sa = Self_Attention(512,512,16*16)
        self.conv1 = nn.Conv3d(6, 256, kernel_size=1, padding=0)
        
        self.final = nn.Conv3d(64, self.n_classes, kernel_size=1, padding=0)

        self.fc = nn.Linear(256*256*8, 3)

    def forward(self, x):
        features = self.vgg(x)
        f = features[3].squeeze(2)

        node = self.mtn(f)
        gcn_out, adj = self.gcn(node)
        sa_attn, v = self.sa(gcn_out)
        attn = 0.9 * sa_attn + 0.1 * adj
        out = torch.bmm(attn, v).unsqueeze(2)
        out = out.unsqueeze(2)
        out = self.conv1(out)

        N, C, D, H, W = out.shape
        out = out.reshape(N, C, D, 16, 16)

        return f, node, adj, sa_attn, out


class DoubleConv_1(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv4 = nn.Conv3d(out_channels*3, out_channels, kernel_size=1)   
            
        
    def forward(self, x1, x2=None):
        if x2 is not None:
            N, C, _, _, _ = x1.shape
            _, _, D, H, W = x2.shape
            x1 = x1.reshape(N, C, D, H, W)
            x1 = torch.cat([x2, x1], dim=1)
        x = x1

        x = torch.cat([self.conv1(x), self.conv2(x), self.conv3(x)], dim=1)
        return self.conv4(x)
        

class Up_1(nn.Module):
    def __init__(self, bilinear=True):
        super().__init__()
        self.up =  nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

    def forward(self, x):
        x = self.up(x)
        return x       

class Up_2(nn.Module):
    def __init__(self, bilinear=True):
        super().__init__()
        self.up =  nn.Upsample(scale_factor=(1,2,2), mode='trilinear', align_corners=True)

    def forward(self, x):
        x = self.up(x)
        return x

"""
TEST
"""

def FFT(img):
    fft_img = np.fft.fftn(img, axes=(-2,-1))
    shape = fft_img.shape
    amp_img, pha_img = np.abs(fft_img), np.angle(fft_img)
    amp_img = amp_img.reshape(-1, shape[-2], shape[-1])
    pha_img = pha_img.reshape(-1, shape[-2], shape[-1])
    amp_mean = np.mean(amp_img, axis=0)
    pha_mean = np.mean(pha_img, axis=0)
    return amp_mean, pha_mean

def test(logger, dataloader_1, net):

    def data2variable(inputs, target, meta):
        with torch.no_grad():
            input_var = torch.autograd.Variable(inputs.cuda())
            target_var = [torch.autograd.Variable(target[i].cuda()) for i in
                          range(len(target))]
            coord_var = torch.autograd.Variable(meta['tpts_inp'].cuda())

        return input_var, target_var, coord_var

    loss_list_test = list()
    net.eval()

    ID = 0

    all_f_amp, all_f_pha, all_adj_amp, all_adj_pha, \
    all_sa_attn_amp, all_sa_attn_pha, all_out_amp, all_out_pha = \
    [], [], [], [], [], [], [], []

    for inputs, target, meta in tqdm(dataloader_1):
        input_var, target_var, coord_var = data2variable(inputs, target, meta)

        with torch.no_grad():
            f, gcn_out, adj, sa_attn, out = net.forward(input_var)
            print(f)
            print("---------")

        f_amp, f_pha = FFT(f.cpu())
        adj_amp, adj_pha = FFT(adj.cpu())
        sa_attn_amp, sa_attn_pha = FFT(sa_attn.cpu())
        out_amp, out_pha = FFT(out.cpu())
        all_f_amp.append(f_amp)
        all_f_pha.append(f_pha)
        all_adj_amp.append(adj_amp)
        all_adj_pha.append(adj_pha)
        all_sa_attn_amp.append(sa_attn_amp)
        all_sa_attn_pha.append(sa_attn_pha)
        all_out_amp.append(out_amp)
        all_out_pha.append(out_pha)
    all_fft = []
    all_fft.append(np.stack(all_f_amp, axis = 0))
    all_fft.append(np.stack(all_f_pha, axis = 0))
    all_fft.append(np.stack(all_adj_amp, axis = 0))
    all_fft.append(np.stack(all_adj_pha, axis = 0))
    all_fft.append(np.stack(all_sa_attn_amp, axis = 0))
    all_fft.append(np.stack(all_sa_attn_pha, axis = 0))
    all_fft.append(np.stack(all_out_amp, axis = 0))
    all_fft.append(np.stack(all_out_pha, axis = 0))
    '''for i in range(8):
        res = all_fft[i].reshape(all_fft[i].shape[0], -1)
        tsne = TSNE(n_components=2, init='pca')
        tsne.fit_transform(res)
        data = np.array(tsne.embedding_)
        plt.clf()
        plt.cla()
        plt.scatter(data[:,0], data[:,1])
        plt.savefig('/home/yzy/Desktop/JointLandmark/MODEL_MY3D/{}.jpg'.format(i))'''


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Joint Voxel and Coordinate Regression')

    parser.add_argument('-s', '--stacks', default=4, type=int, metavar='N',
                        help='number of hourglasses to stack')
    parser.add_argument('-b', '--blocks', default=1, type=int, metavar='N',
                        help='number of residual modules at each location in the hourglass')
    parser.add_argument('--depth_res', default=[1, 2, 4, 64], type=int, nargs="*",
                        help='Resolution of depth for the output of the corresponding hourglass')
    parser.add_argument('--resume_p2v', default='', type=str,
                        help='path to the model of voxel regression subnetwork')
    parser.add_argument('--resume_v2c', default='', type=str,
                        help='path to the model of coordinate regression subnetwork')
    parser.add_argument('--resume_p2v2c', default='/home/yzy/Desktop/JointLandmark/MODEL_MY3D/records/best_4.tar', type=str,
                        help='path to the pre-trained model')
    parser.add_argument('--gpus', default='0,1,2,3', type=str, help='set gpu IDs')
    # Training strategy
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=500, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--pretr_epochs', default=0, type=int, metavar='N',
                        help='Number of epochs for pre-training the network')
    parser.add_argument('--train-batch', default=32, type=int, metavar='N',
                        help='train batchsize')
    parser.add_argument('--test-batch', default=1, type=int, metavar='N',
                        help='test batchsize')
    parser.add_argument('--lr', '--learning-rate', default=1e-1, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-3, type=float)
    parser.add_argument('--schedule', type=int, nargs='+', default=[150, 300, 450],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='LR is multiplied by gamma on schedule.')
    parser.add_argument('--sigma', type=float, default=1,
                        help='Groundtruth Gaussian sigma.')
    parser.add_argument('--num_snapshot', default=5, type=int, metavar='N',
                        help='Frequency for saving checkpoints')
    # Miscs
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--img_size', default=256, type=int)
    parser.add_argument('--num_landmarks', default=6, type=int)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    if args.gpus == '':
        is_cuda = False
        print('Run in CPU mode.')
    else:
        is_cuda = True
    
    # Create Logger
    logger = get_mylogger()

    dataloader_1 = torch.utils.data.DataLoader(
        fa68pt3D('/home/yzy/Desktop/JointLandmark/val.txt', train=False, depth_res=args.depth_res,
            nStack=args.stacks, sigma=args.sigma),
        batch_size=args.test_batch, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # Load model
    adj = getAdjMatrix(k=4)
    net = UNet_Pretrained_Att(8, args.num_landmarks, adj)
    net = torch.nn.DataParallel(net).cuda()
    checkpoint = torch.load(args.resume_p2v2c)
    net.load_state_dict(checkpoint)

    test(logger, dataloader_1, net)
