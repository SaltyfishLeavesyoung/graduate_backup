import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision

import numpy as np
from maptonode import MapToNode
from semgraphconv import SemGraphConv
from self_att import Self_Attention
from resnet import r2plus1d_18

class UNet_Pretrained_Att(nn.Module):
    def __init__(self, n_channels, n_classes, adj):
        super(UNet_Pretrained_Att, self).__init__()
        self.adj = torch.tensor(adj).float()
        self.n_channels = n_channels
        self.n_classes = n_classes
        bilinear = True

        self.vgg =  r2plus1d_18(pretrained=True)
        
        self.up1 = Up_1(bilinear)
        self.double_conv1 = DoubleConv_1(512, 128)

        self.up2 = Up_1(bilinear)
        self.double_conv2 = DoubleConv_1(256, 64)

        self.up3 = Up_1(bilinear)
        self.double_conv3 = DoubleConv_1(128, 64)

        self.up4 = Up_2(bilinear)
        self.double_conv4 = DoubleConv_1(64, 64)

        # self.double_conv5 = DoubleConv_1(64, 64)

        self.mtn = MapToNode(512, 6)
        self.gcn = SemGraphConv(16*16*4, 512, self.adj)
        self.sa = Self_Attention(512,512,16*16)
        self.conv1 = nn.Conv3d(6, 256, kernel_size=1, padding=0)
        
        self.d3tod2 = nn.Conv2d(64*8, 64, kernel_size=1, padding=0)
        self.final = nn.Conv2d(64, self.n_classes*3, kernel_size=1, padding=0)

        self.final2 = nn.Conv2d(64, self.n_classes, kernel_size=1, padding=0)
        self.fc = nn.Linear(256*256, 8)

    def forward(self, x):  
        features = self.vgg(x)

        '''for i in range(len(features)):
            print(features[i].shape)
        torch.Size([2, 64, 8, 128, 128])
        torch.Size([2, 128, 4, 64, 64])
        torch.Size([2, 256, 2, 32, 32])
        torch.Size([2, 512, 1, 16, 16])'''
        f = features[3].squeeze(2)

        node = self.mtn(f)
        # print(node.shape) [2, 6, 1024]
        gcn_out, adj = self.gcn(node)
        # print(gcn_out.shape) [2, 6, 512]
        sa_attn, v = self.sa(gcn_out)
        attn = 0.9 * sa_attn + 0.1 * adj
        out = torch.bmm(attn, v).unsqueeze(2)
        out = out.unsqueeze(2)
        out = self.conv1(out)
        # print(out.shape) [2, 256, 1, 1, 256]
        N, C, D, H, W = out.shape
        out = out.reshape(N, C, D, 16, 16)

        x = self.up1(out)

        x = self.double_conv1(x, features[2])    

        x = self.up2(x)
        x = self.double_conv2(x, features[1])

        x = self.up3(x)
        x = self.double_conv3(x, features[0])

        x = self.up4(x)
        x = self.double_conv4(x)

        # x = self.double_conv5(x)
        N1, C1, D1, H1, W1 = x.shape
        x = x.reshape(N1, -1, H1, W1)
        x = self.d3tod2(x)
        x1 = self.final(x)

        x2 = self.final2(x)
        x2 = x2.view(N1, 6, -1)
        x2 = self.fc(x2)
        
        heatmap = torch.sigmoid(x1[:,:self.n_classes,:,:])
        regression_x = x1[:,self.n_classes:2*self.n_classes,:,:]
        regression_y = x1[:,2*self.n_classes:,:,:]
        predict_z = x2

        return heatmap, predict_z, regression_y, regression_x


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
    adj = torch.eye(6,6)
    test = UNet_Pretrained_Att(3, 6, adj)
    wtf = torch.randn([2, 3, 8, 256, 256], dtype=torch.float)
    heatmap, regression_y, regression_x= test(wtf)
    print(heatmap.shape)
    print(regression_y.shape)
    print(regression_x.shape)
