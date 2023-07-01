import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision

import numpy as np
from maptonode import MapToNode
from semgraphconv import SemGraphConv
from self_att import Self_Attention
from attention import *
# from test_module import my_attention, CPFE

VGG = torchvision.models.vgg19

class UNet_Pretrained_Att(nn.Module):
    def __init__(self, n_channels, n_classes, adj):
        super(UNet_Pretrained_Att, self).__init__()
        self.adj = torch.tensor(adj).float()
        self.n_channels = n_channels
        self.n_classes = n_classes
        bilinear = True

        self.vgg =  VGG(pretrained=True)
        
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

        self.mtn = MapToNode(512, 19)
        self.gcn = SemGraphConv(25*20*4, 512, self.adj)
        self.sa = Self_Attention(512,512,25*20)
        self.conv1 = nn.Conv2d(19, 512, kernel_size=1, padding=0)
        '''self.conv1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)'''
        
        self.final = nn.Sequential(
            nn.Conv2d(64, self.n_classes*3, kernel_size=1, padding=0)
        )

        '''self.sa0 = SpatialAttention()
        self.sa1 = SpatialAttention()
        self.ca2 = ChannelAttention(256)
        self.ca3 = ChannelAttention(512)
        self.ca4 = ChannelAttention(512)

        self.alpha = 1
        self.beta = 1
        self.temp_alpha = []
        self.temp_beta = []

        self.ma = my_attention(self.adj, 2000, 512, 500)'''

    '''    self.gcn.register_backward_hook(self.savegrad1)
        self.sa.register_backward_hook(self.savegrad2)

    def savegrad1(self, md, grad_in, grad_out):
        self.temp_alpha.append(abs(torch.mean(grad_out[0])).cpu())

    def savegrad2(self, md, grad_in, grad_out):
        self.temp_beta.append(abs(torch.mean(grad_out[0])).cpu())

    def resetab(self):
        self.alpha = torch.mean(torch.stack(self.temp_alpha))
        self.beta = torch.mean(torch.stack(self.temp_beta))
        self.temp_alpha = []
        self.temp_beta = []
        return self.alpha, self.beta'''


    def forward(self, x):  
        _, features = self.vgg.features(x, get_features=True) 
        # print(features[4].shape) #512*25*20
        #f4 = self.ca4(features[4])
        #node = self.mtn(f4)
        """node = self.mtn(features[4])
        # print(node.shape) 19 * 2000
        gcn_out, adj = self.gcn(node)
        # print(gcn_out.shape) 19 * 512
        sa_attn, v = self.sa(gcn_out)
        # attn = torch.sigmoid(sa_attn + adj)
        #attn = (self.beta / (self.alpha + self.beta)) * sa_attn + (self.alpha / (self.alpha + self.beta)) * adj
        attn = 0.9 * sa_attn + 0.1 * adj
        out = torch.bmm(attn, v).unsqueeze(2)
        '''out = self.ma(node)'''
        out = self.conv1(out)
        # print(out.shape) 512*1*500"""

        # out = self.conv2(self.conv1(features[4]))

        x = self.up1(out)
        #f3 = self.ca3(features[3])
        #x = self.double_conv1(x, f3)
        x = self.double_conv1(x, features[3])       

        x = self.up2(x)
        #f2 = self.ca2(features[2])
        #x = self.double_conv2(x, f2)
        x = self.double_conv2(x, features[2])

        x = self.up3(x)
        #f1 = self.sa1(features[1])
        #x = self.double_conv3(x, f1)
        x = self.double_conv3(x, features[1])

        x = self.up4(x)
        #f0 = self.sa0(features[0])
        #x = self.double_conv4(x, f0)
        x = self.double_conv4(x, features[0])

        x = self.up5(x)
        x = self.double_conv5(x)
      
        x = self.final(x)
        
        heatmap = torch.sigmoid(x[:,:self.n_classes,:,:])
        regression_x = x[:,self.n_classes:2*self.n_classes,:,:]
        regression_y = x[:,2*self.n_classes:,:,:]

        return heatmap, regression_y, regression_x


class DoubleConv_1(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=5, dilation=5),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv4 = nn.Conv2d(out_channels*3, out_channels, kernel_size=1)   
            
        
    def forward(self, x1, x2=None):
        if x2 is not None:
            diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
            diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
            # if you have padding issues, see
            # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
            # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
            x1 = torch.cat([x2, x1], dim=1)
        x = x1
        # return self.double_conv(x)
        
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
    adj = torch.eye(19,19)
    test = UNet_Pretrained_Att(3, 19, adj)
    wtf = torch.randn([2, 3, 800, 640], dtype=torch.float)
    heatmap, regression_y, regression_x= test(wtf)

