# https://github.com/Megvii-Nanjing/ML-GCN/blob/master/models.py

import torchvision.models as models
from torch.nn import Parameter
from util import *
import torch
import torch.nn as nn
import math

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCNResnet(nn.Module):
    def __init__(self, in_features, out_features, adj):
        super(GCNResnet, self).__init__()

        self.gc1 = GraphConvolution(in_features, 256)
        self.gc2 = GraphConvolution(256, out_features)
        self.relu = nn.LeakyReLU(0.2)

        self.A = Parameter(adj)

    def forward(self, inp):
        x = self.gc1(inp, self.A)
        x = self.relu(x)
        x = self.gc2(x, self.A)

        return x, self.A

    def get_config_optim(self, lr, lrp):
        return [
                {'params': self.gc1.parameters(), 'lr': lr},
                {'params': self.gc2.parameters(), 'lr': lr},
                ]


