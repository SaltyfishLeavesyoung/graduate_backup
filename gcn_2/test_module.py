from __future__ import absolute_import, division
import numpy as np
from numpy import *
import torch
from einops import rearrange
from torch import nn
import torch.nn.functional as F
import math
from math import sqrt

class SemGraphConv(nn.Module):
    """
    Semantic graph convolution layer
    """

    def __init__(self, in_features, out_features, adj, bias=True):
        super(SemGraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Parameter(torch.zeros(size=(2, in_features, out_features), dtype=torch.float))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.adj = adj
        self.m = (self.adj > 0)
        self.e = nn.Parameter(torch.zeros(1, len(self.m.nonzero()), dtype=torch.float))
        nn.init.constant_(self.e.data, 1)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float))
            stdv = 1. / math.sqrt(self.W.size(2))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        h0 = torch.matmul(input, self.W[0])
        h1 = torch.matmul(input, self.W[1])

        adj = -9e15 * torch.ones_like(self.adj).to(input.device)
        adj[self.m] = self.e
        adj = F.softmax(adj, dim=1)

        M = torch.eye(adj.size(0), dtype=torch.float).to(input.device)
        output = torch.matmul(adj * M, h0) + torch.matmul(adj * (1 - M), h1)

        if self.bias is not None:
            return output + self.bias.view(1, 1, -1), adj
        else:
            return output, adj

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class my_attention(nn.Module):
    def __init__(self,adj,input_dim,dim_k,dim_v):
        super(my_attention, self).__init__()
        self.dim = input_dim
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.q = nn.Linear(input_dim,dim_k)
        self.k = nn.Linear(input_dim,dim_k)
        self._norm_fact = 1 / sqrt(dim_k)
        self.gcn = SemGraphConv(input_dim, dim_v, adj)

    def forward(self, x):
        
        Q = self.q(x)
        K = self.k(x)
        attn = nn.Softmax(dim=-1)(torch.bmm(Q,K.permute(0,2,1))) * self._norm_fact

        V, adj = self.gcn(x)
        return torch.bmm(attn, V).unsqueeze(2)

"""
REFERENCES:
https://github.com/sairajk/PyTorch-Pyramid-Feature-Attention-Network-for-Saliency-Detection/blob/master/src/model.py
"""
class CPFE(nn.Module):
    def __init__(self, in_channels, out_channels=16):
        super(CPFE, self).__init__()

        self.dil_rates = [3, 5, 7, 9]

        self.in_channels = in_channels

        # Define layers
        self.conv_1_1 = nn.Conv2d(in_channels=self.in_channels, out_channels=out_channels, kernel_size=1, bias=False)
        self.conv_dil_3 = nn.Conv2d(in_channels=self.in_channels, out_channels=out_channels, kernel_size=3,
                                    stride=1, dilation=self.dil_rates[0], padding=self.dil_rates[0], bias=False)
        self.conv_dil_5 = nn.Conv2d(in_channels=self.in_channels, out_channels=out_channels, kernel_size=3,
                                    stride=1, dilation=self.dil_rates[1], padding=self.dil_rates[1], bias=False)
        self.conv_dil_7 = nn.Conv2d(in_channels=self.in_channels, out_channels=out_channels, kernel_size=3,
                                    stride=1, dilation=self.dil_rates[2], padding=self.dil_rates[2], bias=False)
        self.conv_dil_9 = nn.Conv2d(in_channels=self.in_channels, out_channels=out_channels, kernel_size=3,
                                    stride=1, dilation=self.dil_rates[3], padding=self.dil_rates[3], bias=False)

        self.bn = nn.BatchNorm2d(out_channels*4)

        self.pool1 = nn.AvgPool2d(2)

        self.pool2 = nn.AvgPool2d(2)

    def forward(self, input_):
        # Extract features
        #conv_1_1_feats = self.conv_1_1(input_)
        conv_dil_3_feats = self.conv_dil_3(input_)
        conv_dil_5_feats = self.conv_dil_5(input_)
        conv_dil_7_feats = self.conv_dil_7(input_)
        conv_dil_9_feats = self.conv_dil_9(input_)

        # Aggregate features
        #concat_feats = torch.cat((conv_1_1_feats, conv_dil_3_feats, conv_dil_5_feats, conv_dil_7_feats), dim=1)
        concat_feats = torch.cat((conv_dil_3_feats, conv_dil_5_feats, conv_dil_7_feats, conv_dil_9_feats), dim=1)
        bn_feats = F.relu(self.bn(concat_feats))

        return bn_feats

