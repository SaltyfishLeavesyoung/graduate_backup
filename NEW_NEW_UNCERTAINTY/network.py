import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision

import numpy as np

from self_att import Self_Attention, Block, PatchEmbed
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

VGG = torchvision.models.vgg19

def two_d_softmax(x):
    exp_y = torch.exp(x)
    return exp_y / torch.sum(exp_y, dim=(2, 3), keepdim=True)

class UNet_Pretrained_Att(nn.Module):
    def __init__(self, n_channels, n_classes, alpha = 1.0):
        super(UNet_Pretrained_Att, self).__init__()
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
        
        self.final = nn.Conv2d(64, self.n_classes * 3, kernel_size=1, padding=0)

        """self.pool0 = nn.AvgPool2d(2)
        self.reduce_c1 = DoubleConv_1(192, 64)
        #self.patch_embed1 = PatchEmbed(img_size=(200, 160), patch_size=1, in_chans=64, embed_dim=64)
        self.patch_embed1 = PatchEmbed(img_size=(128, 128), patch_size=1, in_chans=64, embed_dim=64)
        #self.pos_embed1 = nn.Parameter(torch.zeros(1, 200 * 160, 64))
        self.pos_embed1 = nn.Parameter(torch.zeros(1, 128 * 128, 64))
        self.attn1 = Block(64, sr_ratio = 8)
        self.pool1 = nn.AvgPool2d(2)
        self.reduce_c2 = DoubleConv_1(320, 64)
        #self.patch_embed2 = PatchEmbed(img_size=(100, 80), patch_size=1, in_chans=64, embed_dim=64)
        self.patch_embed2 = PatchEmbed(img_size=(64, 64), patch_size=1, in_chans=64, embed_dim=64)
        #self.pos_embed2 = nn.Parameter(torch.zeros(1, 100 * 80, 64))
        self.pos_embed2 = nn.Parameter(torch.zeros(1, 64 * 64, 64))
        self.attn2 = Block(64, sr_ratio = 4)
        self.pool2 = nn.AvgPool2d(2)
        self.reduce_c3 = DoubleConv_1(576, 64)
        #self.patch_embed3 = PatchEmbed(img_size=(50, 40), patch_size=1, in_chans=64, embed_dim=64)
        self.patch_embed3 = PatchEmbed(img_size=(32, 32), patch_size=1, in_chans=64, embed_dim=64)
        #self.pos_embed3 = nn.Parameter(torch.zeros(1, 50 * 40, 64))
        self.pos_embed3 = nn.Parameter(torch.zeros(1, 32 * 32, 64))
        self.attn3 = Block(64, sr_ratio = 2)
        self.pool3 = nn.AvgPool2d(2)
        self.reduce_c4 = DoubleConv_1(576, 64)
        #self.patch_embed4 = PatchEmbed(img_size=(25, 20), patch_size=1, in_chans=64, embed_dim=64)
        self.patch_embed4 = PatchEmbed(img_size=(16, 16), patch_size=1, in_chans=64, embed_dim=64)
        #self.pos_embed4 = nn.Parameter(torch.zeros(1, 501, 64))
        self.pos_embed4 = nn.Parameter(torch.zeros(1, 257, 64))
        self.attn4 = Block(64, sr_ratio = 1)"""
        
        self.ablation_patch_embed1 = PatchEmbed(img_size=(200, 160), patch_size=1, in_chans=128, embed_dim=128)
        self.ablation_pos_embed1 = nn.Parameter(torch.zeros(1, 200 * 160, 128))
        self.ablation_attn1 = Block(128, sr_ratio = 8)

        self.ablation_patch_embed2 = PatchEmbed(img_size=(100, 80), patch_size=1, in_chans=256, embed_dim=256)
        self.ablation_pos_embed2 = nn.Parameter(torch.zeros(1, 100 * 80, 256))
        self.ablation_attn2 = Block(256, sr_ratio = 4)
        
        self.ablation_patch_embed3 = PatchEmbed(img_size=(50, 40), patch_size=1, in_chans=512, embed_dim=512)
        self.ablation_pos_embed3 = nn.Parameter(torch.zeros(1, 50 * 40, 512))
        self.ablation_attn3 = Block(512, sr_ratio = 2)
        
        self.ablation_patch_embed4 = PatchEmbed(img_size=(25, 20), patch_size=1, in_chans=512, embed_dim=512)
        self.ablation_pos_embed4 = nn.Parameter(torch.zeros(1, 25 * 20, 512))
        self.ablation_attn4 = Block(512, sr_ratio = 1)     

        self.covariance = nn.Sequential(
            nn.Conv1d(64, n_classes, kernel_size=1, padding=0),
            #nn.Linear(500, 3),
            nn.Linear(256, 3),
        )
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def _get_pos_embed(self, pos_embed, patch_embed, H, W):
        #if H * W == self.patch_embed1.num_patches:
        if H * W == self.ablation_patch_embed1.num_patches:
            return pos_embed
        else:
            return F.interpolate(
                pos_embed.reshape(1, patch_embed.H, patch_embed.W, -1).permute(0, 3, 1, 2),
                size=(H, W), mode="bilinear").reshape(1, -1, H * W).permute(0, 2, 1)
        
    def forward(self, x):  
        _, features = self.vgg.features(x, get_features=True)
        """
        torch.Size([2, 64, 400, 320])
        torch.Size([2, 128, 200, 160])
        torch.Size([2, 256, 100, 80])
        torch.Size([2, 512, 50, 40])
        torch.Size([2, 512, 25, 20])
        """
        B = x.shape[0]
        
        """feat0 = self.pool0(features[0])
        cat_feat1 = torch.cat((feat0, features[1]), dim=1)
        fuse_feat1 = self.reduce_c1(cat_feat1)
        
        fuse_feat1, (H1, W1) = self.patch_embed1(fuse_feat1)
        pos_embed1 = self._get_pos_embed(self.pos_embed1, self.patch_embed1, H1, W1)
        fuse_feat1 = fuse_feat1 + pos_embed1     
        attn_feat1 = self.attn1(fuse_feat1, H1, W1)
        attn_feat1 = attn_feat1.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
        
        feat1 = self.pool1(attn_feat1)
        cat_feat2 = torch.cat((feat1, features[2]), dim=1)
        fuse_feat2 = self.reduce_c2(cat_feat2)
        
        fuse_feat2, (H2, W2) = self.patch_embed2(fuse_feat2)
        pos_embed2 = self._get_pos_embed(self.pos_embed2, self.patch_embed2, H2, W2)
        fuse_feat2 = fuse_feat2 + pos_embed2      
        attn_feat2 = self.attn2(fuse_feat2, H2, W2)
        attn_feat2 = attn_feat2.reshape(B, H2, W2, -1).permute(0, 3, 1, 2).contiguous()
        
        feat2 = self.pool2(attn_feat2)
        cat_feat3 = torch.cat((feat2, features[3]), dim=1)
        fuse_feat3 = self.reduce_c3(cat_feat3)
        
        fuse_feat3, (H3, W3) = self.patch_embed3(fuse_feat3)
        pos_embed3 = self._get_pos_embed(self.pos_embed3, self.patch_embed3, H3, W3)
        fuse_feat3 = fuse_feat3 + pos_embed3
        attn_feat3 = self.attn3(fuse_feat3, H3, W3)
        attn_feat3 = attn_feat3.reshape(B, H3, W3, -1).permute(0, 3, 1, 2).contiguous()
        
        feat3 = self.pool3(attn_feat3)
        cat_feat4 = torch.cat((feat3, features[4]), dim=1)
        fuse_feat4 = self.reduce_c4(cat_feat4)
        
        fuse_feat4, (H4, W4) = self.patch_embed4(fuse_feat4)
        pos_embed4 = self._get_pos_embed(self.pos_embed4[:, 1:], self.patch_embed4, H4, W4)
        fuse_feat4 = fuse_feat4 + pos_embed4
        attn_feat4 = self.attn4(fuse_feat4, H4, W4)
        attn_feat4 = attn_feat4.reshape(B, H4, W4, -1).permute(0, 3, 1, 2).contiguous()
        
        attn_feat4 = attn_feat4.view(attn_feat4.shape[0], attn_feat4.shape[1], -1)

        cov = self.covariance(attn_feat4)

        cholesky = Variable(torch.zeros(cov.shape).float()).cuda()

        cholesky[:, :, 1] = cov[:, :, 1]
        cholesky[:, :, 0] = myelu(cov[:,:,0], self.alpha)
        cholesky[:, :, 2] = myelu(cov[:,:,2], self.alpha)"""
        
        # ablation
        
        feat1, (H1, W1) = self.ablation_patch_embed1(features[1])
        pos_embed1 = self._get_pos_embed(self.ablation_pos_embed1, self.ablation_patch_embed1, H1, W1)
        feat1 = feat1 + pos_embed1     
        attn_feat1 = self.ablation_attn1(feat1, H1, W1)
        attn_feat1 = attn_feat1.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
        
        feat2, (H2, W2) = self.ablation_patch_embed2(features[2])
        pos_embed2 = self._get_pos_embed(self.ablation_pos_embed2, self.ablation_patch_embed2, H2, W2)
        feat2 = feat2 + pos_embed2     
        attn_feat2 = self.ablation_attn2(feat2, H2, W2)
        attn_feat2 = attn_feat2.reshape(B, H2, W2, -1).permute(0, 3, 1, 2).contiguous()
        
        feat3, (H3, W3) = self.ablation_patch_embed3(features[3])
        pos_embed3 = self._get_pos_embed(self.ablation_pos_embed3, self.ablation_patch_embed3, H3, W3)
        feat3 = feat3 + pos_embed3     
        attn_feat3 = self.ablation_attn3(feat3, H3, W3)
        attn_feat3 = attn_feat3.reshape(B, H3, W3, -1).permute(0, 3, 1, 2).contiguous()
        
        feat4, (H4, W4) = self.ablation_patch_embed4(features[4])
        pos_embed4 = self._get_pos_embed(self.ablation_pos_embed4, self.ablation_patch_embed4, H4, W4)
        feat4 = feat4 + pos_embed4     
        attn_feat4 = self.ablation_attn4(feat4, H4, W4)
        attn_feat4 = attn_feat4.reshape(B, H4, W4, -1).permute(0, 3, 1, 2).contiguous()
 
        # heatmap branch
        #x = self.up1(features[4])
        x = self.up1(attn_feat4)
        #x = self.double_conv1(x, features[3])
        x = self.double_conv1(x, attn_feat3)

        x = self.up2(x)
        #x = self.double_conv2(x, features[2])
        x = self.double_conv2(x, attn_feat2)

        x = self.up3(x)
        #x = self.double_conv3(x, features[1])
        x = self.double_conv3(x, attn_feat1)

        x = self.up4(x)
        x = self.double_conv4(x, features[0])

        x = self.up5(x)
        x = self.double_conv5(x)
      
        x = self.final(x)
        
        heatmap = torch.sigmoid(x[:,:self.n_classes,:,:])
        regression_x = x[:,self.n_classes:2*self.n_classes,:,:]
        regression_y = x[:,2*self.n_classes:,:,:]

        #heatmap = torch.sigmoid(x)
        #heatmap = F.relu(x)
        #heatmap = two_d_softmax(x)

        return heatmap, regression_y, regression_x, 0 #cholesky


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

def myelu(input, alpha):
    return F.elu(input, alpha) + alpha

if __name__ == '__main__':
    test = UNet_Pretrained_Att(3, 19)
    wtf = torch.randn([2, 3, 800, 640], dtype=torch.float)
    heatmap, cov = test(wtf)
