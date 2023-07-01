import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision

import numpy as np

VGG = torchvision.models.vgg19

def two_d_softmax(x):
    exp_y = torch.exp(x)
    return exp_y / torch.sum(exp_y, dim=(2, 3), keepdim=True)

class global_net(nn.Module):
    def __init__(self, n_channels, n_classes, alpha = 1.0):
        super(global_net, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.alpha = alpha
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
        
        self.final = nn.Conv2d(64, self.n_classes, kernel_size=1, padding=0)

    def forward(self, x):  
        _, features = self.vgg.features(x, get_features=True)
        """
        torch.Size([2, 64, 400, 320])
        torch.Size([2, 128, 200, 160])
        torch.Size([2, 256, 100, 80])
        torch.Size([2, 512, 50, 40])
        torch.Size([2, 512, 25, 20])
        """

        x = self.up1(features[4])
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
        
        #heatmap = torch.sigmoid(x[:,:self.n_classes,:,:])
        #regression_x = x[:,self.n_classes:2*self.n_classes,:,:]
        #regression_y = x[:,2*self.n_classes:,:,:]

        heatmap = torch.sigmoid(x)
        #heatmap = F.relu(x)
        #heatmap = two_d_softmax(x)

        #return heatmap, regression_y, regression_x
        return heatmap

class local_net(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(local_net, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.predictor = nn.Sequential(
            DoubleConv_1(n_channels, 64),
            DoubleConv_1(64, 64),
            nn.Conv2d(64, self.n_classes * 2, kernel_size=1, padding=0),
            nn.BatchNorm2d(self.n_classes * 2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.predictor(x)
        regression_x = x[:,:self.n_classes,:,:]
        regression_y = x[:,self.n_classes:2*self.n_classes,:,:]
        return regression_y, regression_x

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
    test = UNet_Pretrained_Att(3, 19)
    wtf = torch.randn([2, 3, 800, 640], dtype=torch.float)
    heatmap, cov = test(wtf)
