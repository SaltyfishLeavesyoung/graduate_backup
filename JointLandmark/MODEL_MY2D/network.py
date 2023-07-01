import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision

import numpy as np
from maptonode import MapToNode
from semgraphconv import SemGraphConv
from self_att import Self_Attention
from vgg import vgg19

class UNet_Pretrained_Att(nn.Module):
    def __init__(self, n_channels, n_classes, adj):
        super(UNet_Pretrained_Att, self).__init__()
        self.adj = torch.tensor(adj).float()
        self.n_channels = n_channels
        self.n_classes = n_classes
        bilinear = True

        self.vgg =  vgg19()
        
        self.up1 = Up_1(bilinear)
        self.double_conv1 = DoubleConv_1(1024, 256)

        self.up2 = Up_1(bilinear)
        self.double_conv2 = DoubleConv_1(512, 128)

        self.up3 = Up_1(bilinear)
        self.double_conv3 = DoubleConv_1(256, 64)

        self.up4 = Up_1(bilinear)
        self.double_conv4 = DoubleConv_1(128, 128)

        self.up5 = Up_1(bilinear)
        self.double_conv5 = DoubleConv_1(128, 64)

        self.mtn = MapToNode(512, 6)
        self.gcn = SemGraphConv(8*8*4, 512, self.adj)
        self.sa = Self_Attention(512,512,8*8)
        self.conv1 = nn.Conv2d(6, 512, kernel_size=1, padding=0)
        
        self.final = nn.Conv2d(64, self.n_classes, kernel_size=1, padding=0)

        self.fc = nn.Linear(256*256, 3)

    def forward(self, x):  
        _, features = self.vgg.features(x, get_features=True)

        node = self.mtn(features[4])
        gcn_out, adj = self.gcn(node)
        sa_attn, v = self.sa(gcn_out)
        attn = 0.9 * sa_attn + 0.1 * adj
        out = torch.bmm(attn, v).unsqueeze(2)
        out = self.conv1(out)

        x = self.up1(out)

        x = self.double_conv1(x, features[3])    

        x = self.up2(x)
        x = self.double_conv2(x, features[2])

        x = self.up3(x)
        x = self.double_conv3(x, features[1])

        x = self.up4(x)
        x = self.double_conv4(x, features[0])

        x = self.up5(x)
        x = self.double_conv5(x)

        x = self.final(x)
        x = x.view(x.size(0), self.n_classes, -1)
        x = self.fc(x)
        x = torch.sigmoid(x)

        return x


class DoubleConv_1(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv4 = nn.Conv2d(out_channels*3, out_channels, kernel_size=1)   
            
        
    def forward(self, x1, x2=None):
        if x2 is not None:
            '''diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
            diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])

            # if you have padding issues, see
            # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
            # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd'''
            N, C, _, _ = x1.shape
            _, _, H, W = x2.shape
            x1 = x1.reshape(N, C, H, W)
            x1 = torch.cat([x2, x1], dim=1)
        x = x1

        x = torch.cat([self.conv1(x), self.conv2(x), self.conv3(x)], dim=1)
        return self.conv4(x)
        

class Up_1(nn.Module):
    def __init__(self, bilinear=True):
        super().__init__()
        self.up =  nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.up(x)
        return x       

if __name__ == '__main__':
    adj = torch.eye(6,6)
    test = UNet_Pretrained_Att(3, 6, adj)
    wtf = torch.randn([2, 8, 256, 256], dtype=torch.float)
    from thop import profile
    macs, params = profile(test, inputs=(wtf, ))
    print('params: ', params / (1024 ** 2))
