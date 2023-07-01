from math import sqrt
import torch
import torch.nn as nn

import numpy as np
from math import sqrt

class Self_Attention(nn.Module):
    # input : batch_size * seq_len * input_dim
    # q : batch_size * input_dim * dim_k
    # k : batch_size * input_dim * dim_k
    # v : batch_size * input_dim * dim_v
    def __init__(self,input_dim,dim_k,dim_v):
        super(Self_Attention,self).__init__()
        self.dim = input_dim
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.q = nn.Linear(input_dim,dim_k)
        self.k = nn.Linear(input_dim,dim_k)
        self.v = nn.Linear(input_dim,dim_v)
        self._norm_fact = 1 / sqrt(dim_k)
    
    def forward(self,x):

        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)
         
        atten = nn.Softmax(dim=-1)(torch.bmm(Q,K.permute(0,2,1))) * self._norm_fact

        return atten, V

if __name__ == '__main__':
    x = torch.randn(2,19,800*640)
    sa = Self_Attention(800*640,512,512)
    att, V = sa(x)
    output = torch.bmm(att,V)
    print(output.size())
