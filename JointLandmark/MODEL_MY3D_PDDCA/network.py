import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision

import numpy as np
from maptonode import MapToNode
from semgraphconv import SemGraphConv
from self_att import Self_Attention
from resnet import r2plus1d_18, r3d_18, mc3_18

from gcn import GCNResnet

class UNet_Pretrained_Att(nn.Module):
    def __init__(self, n_channels, n_classes, adj):
        super(UNet_Pretrained_Att, self).__init__()
        self.adj = torch.tensor(adj).float()
        self.n_channels = n_channels
        self.n_classes = n_classes
        bilinear = True

        self.squeeze = nn.Conv3d(3*8, 3, kernel_size=1, padding=0)

        self.vgg = r2plus1d_18(pretrained=True)
        # self.vgg = mc3_18(pretrained=True)
        
        self.up1 = Up_1(bilinear)       
        # self.up1 = Up_2(bilinear)
        self.double_conv1 = DoubleConv_1(512, 128)

        self.up2 = Up_1(bilinear)  
        # self.up2 = Up_2(bilinear)
        self.double_conv2 = DoubleConv_1(256, 64)

        self.up3 = Up_1(bilinear)  
        # self.up3 = Up_2(bilinear)
        self.double_conv3 = DoubleConv_1(128, 64)

        self.up4 = Up_2(bilinear)
        self.double_conv4 = DoubleConv_1(64, 64)

        self.mtn = MapToNode(512, 5)
        self.gcn = SemGraphConv(16*16*4, 512, self.adj)
        # self.gcn = GCNResnet(16*16*4, 256, self.adj)
        # self.sa = Self_Attention(256,256,16*16*8)
        self.sa = Self_Attention(512,512,16*16)
        self.conv1 = nn.Conv3d(5, 256, kernel_size=1, padding=0)
        # self.conv1 = nn.Conv3d(512, 256, kernel_size=1, padding=0)
        
        self.final = nn.Conv3d(64, self.n_classes, kernel_size=1, padding=0)

        self.fc = nn.Linear(256*256*8, 3)
        # self.fc = nn.Linear(256*256*8, 2)

        #self.new_fc = nn.Linear(512, 256)

    def forward(self, x):
        x = x.view(x.shape[0], 3*8, 8, 256, 256)
        x = self.squeeze(x)

        features = self.vgg(x)
        f = features[3]
        N, C, D, H, W = f.shape
        # print(f.shape) 
        # r2plus1d: [2,64,1,128,128] - [2,512,1,16,16]
        # mc3_18: [2,512,8,16,16]
        f = f.squeeze(2)
        node = self.mtn(f)
        gcn_out, adj = self.gcn(node)
        sa_attn, v = self.sa(gcn_out)
        #print(sa_attn.shape)
        #print(adj.shape)
        attn = 0.9 * sa_attn + 0.1 * adj
        out = torch.bmm(attn, v).unsqueeze(2)
        out = out.unsqueeze(2)
        out = self.conv1(out)

        out = out.reshape(N, -1, D, H, W)

        x = self.up1(out)

        x = self.double_conv1(x, features[2])    
        x = self.up2(x)
        x = self.double_conv2(x, features[1])
        x = self.up3(x)
        x = self.double_conv3(x, features[0])
        x = self.up4(x)
        x = self.double_conv4(x)
        x = self.final(x)
        x = x.view(x.size(0), self.n_classes, -1)
        x = self.fc(x)
        x = torch.sigmoid(x)

        return x


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
            '''diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
            diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])

            # if you have padding issues, see
            # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
            # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd'''
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

if __name__ == '__main__':
    adj = torch.eye(5,5)
    test = UNet_Pretrained_Att(3, 5, adj)
    wtf = torch.randn([2, 3, 64, 256, 256], dtype=torch.float)
    x = test(wtf)
