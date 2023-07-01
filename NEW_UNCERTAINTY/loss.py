import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.modules.loss import _Loss

import numpy as np
from numpy import linalg as la
import scipy.sparse as sp
from scipy.stats import multivariate_normal
import math
import matplotlib.pyplot as plt

def get_sigma(conv):
    L_mat = Variable(torch.zeros((conv.shape[0], conv.shape[1], 2, 2)))
    L_mat[:, :, 0, 0] = conv[:, :, 0]                                 
    L_mat[:, :, 1, 0] = conv[:, :, 1]                                 
    L_mat[:, :, 1, 1] = conv[:, :, 2]
    Sigma = torch.matmul(L_mat, L_mat.transpose(2, 3))
    det_Sigma = Sigma[:, :, 0, 0] * Sigma[:, :, 1, 1] - Sigma[:, :, 0, 1] * Sigma[:, :, 1, 0] + 1e-5
    Sigma_inv = torch.zeros_like(Sigma)
    # Inverse of 2D matrix 
    # [a b]
    # [c d] is
    #
    # (ad-bc)^{-1} *  [d  -b]
    #                 [-c  a]
    Sigma_inv[:,:,0,0] =  Sigma[:,:,1,1]
    Sigma_inv[:,:,1,1] =  Sigma[:,:,0,0]
    Sigma_inv[:,:,1,0] = -Sigma[:,:,1,0]
    Sigma_inv[:,:,0,1] = -Sigma[:,:,0,1]
    Sigma_inverse = Sigma_inv / det_Sigma.unsqueeze(-1).unsqueeze(-1)       # batch_size x num_landmark x 2 x 2

    return Sigma, det_Sigma, Sigma_inverse

def get_sigma_double(conv, conv2):
    L_mat = Variable(torch.zeros((conv.shape[0], conv.shape[1], 2, 2)))
    L_mat[:, :, 0, 0] = conv[:, :, 0]                                 
    L_mat[:, :, 1, 0] = conv[:, :, 1]                                 
    L_mat[:, :, 1, 1] = conv[:, :, 2]
    Sigma1 = torch.matmul(L_mat, L_mat.transpose(2, 3))

    L_mat2 = Variable(torch.zeros((conv2.shape[0], conv2.shape[1], 2, 2)))
    L_mat2[:, :, 0, 0] = conv2[:, :, 0]                                 
    L_mat2[:, :, 1, 0] = conv2[:, :, 1]                                 
    L_mat2[:, :, 1, 1] = conv2[:, :, 2]
    Sigma2 = torch.matmul(L_mat2, L_mat2.transpose(2, 3))

    Sigma = 0.5 * Sigma1 + 0.5 * Sigma2

    det_Sigma = Sigma[:, :, 0, 0] * Sigma[:, :, 1, 1] - Sigma[:, :, 0, 1] * Sigma[:, :, 1, 0] + 1e-5
    Sigma_inv = torch.zeros_like(Sigma)
    # Inverse of 2D matrix 
    # [a b]
    # [c d] is
    #
    # (ad-bc)^{-1} *  [d  -b]
    #                 [-c  a]
    Sigma_inv[:,:,0,0] =  Sigma[:,:,1,1]
    Sigma_inv[:,:,1,1] =  Sigma[:,:,0,0]
    Sigma_inv[:,:,1,0] = -Sigma[:,:,1,0]
    Sigma_inv[:,:,0,1] = -Sigma[:,:,0,1]
    Sigma_inverse = Sigma_inv / det_Sigma.unsqueeze(-1).unsqueeze(-1)       # batch_size x num_landmark x 2 x 2

    return Sigma, det_Sigma, Sigma_inverse

def get_mahalanobis_distance(x, Sigma_inverse, y):
    """
        REFERENCE: https://github.com/abhi1kumar/LUVLi
        Returns x^T Sigma_inverse y
        :param x:             batch_size x num_landmark x 2
        :param Sigma_inverse: batch_size x num_landmark x 2 x 2
        :param y:             batch_size x num_landmark x 2
        
        :return: product of size batch_size x num_landmark
    """
    batch_size = Sigma_inverse.shape[0]                          # batch_size
    num_points = Sigma_inverse.shape[1]
    
    x_vec = x.unsqueeze(-1).contiguous().view(batch_size * num_points, 2, 1)             # batch_size * num_landmark x 2 x 1
    y_vec = y.unsqueeze(-1).contiguous().view(batch_size * num_points, 2, 1)             # batch_size * num_landmark x 2 x 1
    Sigma_inverse = Sigma_inverse.view(batch_size * num_points, 2, 2)       # batch_size * num_landmark x 2 x 2

    # Torch batch matrix multiplication
    # https://pytorch.org/docs/stable/torch.html#torch.bmm
    product = torch.bmm( torch.bmm(x_vec.transpose(1,2), Sigma_inverse), y_vec)    # batch_size * num_landmark x 1 x 1        
    product = product.squeeze(-1).squeeze(-1)                               # batch_size * num_landmark
    product = product.view(batch_size, num_points)                          # batch_size x num_landmark
        
    Sigma_inverse = Sigma_inverse.view(batch_size, num_points, 2, 2)        # batch_size x num_landmark x 2 x 2
    x_vec = x.squeeze(-1).contiguous().view(batch_size, num_points, 2)
    y_vec = y.squeeze(-1).contiguous().view(batch_size, num_points, 2)
    return product

def get_theta_sigma(Sigma):
    U, S, VT = la.svd(Sigma)
    theta = math.atan2(U[1][0], U[0][0])
    return theta, S

def generate_heatmap(heatmap_size, landmarks, sigmas):
    landmarks_shape = landmarks.shape
    sigmas_shape = sigmas.shape
    batch_size = landmarks_shape[0]
    num_landmarks = landmarks_shape[1]
    dim = landmarks_shape[2]

    size = [batch_size, num_landmarks] + heatmap_size
    heatmap = torch.zeros(*size)

    x, y = np.mgrid[0:heatmap_size[0]:1, 0:heatmap_size[1]:1]
    pos = np.dstack((x, y))

    lmk = landmarks.clone().detach().cpu()
    sig = sigmas.clone().detach().cpu()

    for i in range(batch_size):
        for j in range(num_landmarks):
            temp_htp = multivariate_normal([lmk[i][j][1], lmk[i][j][0]], sig[i][j]).pdf(pos)
            temp_htp = temp_htp / temp_htp[lmk[i][j][1]][lmk[i][j][0]]
            heatmap[i][j] = torch.tensor(temp_htp)

    return x, y, heatmap

def get_channel_sum(input):
    """
        Generates the sum of each channel of the input
        input  = batch_size x 68 x 64 x 64
        output = batch_size x 68
    """
    temp   = torch.sum(input, dim=3)
    output = torch.sum(temp , dim=2)
    
    return output

def generate_grid(h, w):
    """
        Generates an equally spaced grid with coordinates as integers with the
        size same as the input heatmap.

        Convention of axis:
        |----> X
        |
        |
        V Y
    """
    x = torch.linspace(0, w - 1, steps = w)
    xv = x.repeat(h, 1)

    y = torch.linspace(0, h - 1, steps = h)
    yv = y.view(-1, 1).repeat(1, w)

    return xv, yv

def normalize(htp):
    """
        Post process and then normalize to sum to 1
        Input : heatmaps     = batch_size x 68 x 64 x 64 Variable
                use_softmax  = Boolean
                tau          = scaling parameter of the softmax [float > 0]
                postprocess  = string

        Output: htp          = batch_size x 68 x 64 x 64 Variable
    """    

    batch_size = htp.shape[0]
    num_points = htp.shape[1]
    height     = htp.shape[2]
    width      = htp.shape[3]

    # Add a small EPSILON for case sum_2 entries are  all zero
    sum2 = get_channel_sum(htp) + 1e-5
    # Get the normalized heatmaps
    htp  = htp/(sum2.view(htp.size(0),htp.size(1),1,1))

    return htp

def get_spatial_mean_along_axis(xv, htp, sum_htp):
    """
        Gets spatial mean along one of the axis.
        Input : htp          = batch_size x 68 x 64 x 64
        Output: means        = batch_size x 68
    """
    batch_size = htp.shape[0]
    num_points = htp.shape[1]    
    height     = htp.shape[2]
    width      = htp.shape[3]

    # x coord * heatmap
    x_times_htp = xv.expand(batch_size,num_points,-1,-1)*htp

    # sume of x coord times heatmap
    s_x = get_channel_sum(x_times_htp)

    # x predicted pts
    # Add a small nudge when sum_htp is all zero
    x_pts = s_x/(sum_htp + 1e-5)

    return x_pts

def get_spatial_mean(heatmaps, significant_pixel_cutoff=0.05):
    """
        Gets the spatial mean of each of the heatmap from the batch of
        normalized heatmaps.
        Input : htp          = batch_size x 68 x 64 x 64
        Output: means        = batch_size x 68 x 2

        Convention:
        |----> X (0th coordinate)
        |
        |
        V Y (1st coordinate)
    """
    # htps = Variable(torch.zeros_like(heatmaps))
    # zeros = heatmaps * 2e-4
    # htps = torch.where(heatmaps > significant_pixel_cutoff, heatmaps, zeros)
    # htp = normalize(htps)

    # htp = normalize(heatmaps)
    # zeros = htp * 2e-4
    # htp[htp < significant_pixel_cutoff] = htp[htp < significant_pixel_cutoff] * 0
    # htp = torch.where(htp > significant_pixel_cutoff, htp, zeros)

    htp = normalize(heatmaps)

    batch_size = htp.shape[0]
    num_points = htp.shape[1]    
    height     = htp.shape[2]
    width      = htp.shape[3]

    # htp is the normalized heatmap
    sum_htp = get_channel_sum(htp)                                                             # batch_size x 68

    xv, yv = generate_grid(height, width)
    xv = Variable(xv)
    yv = Variable(yv)

    if htp.is_cuda:
        xv = xv.cuda()
        yv = yv.cuda()

    x_pts = get_spatial_mean_along_axis(xv, htp, sum_htp)
    y_pts = get_spatial_mean_along_axis(yv, htp, sum_htp)

    means = torch.cat((x_pts.view(batch_size,num_points,1), y_pts.view(batch_size,num_points,1)), 2)

    return means

def get_hottest_point(heatmap):
    w, h = heatmap.shape
    flattened_heatmap = torch.flatten(heatmap)
    hottest_idx = np.argmax(flattened_heatmap)
    return torch.from_numpy(np.ascontiguousarray(np.flip(np.array(np.unravel_index(hottest_idx, [w, h])))))

def getTarget(heatmap):
    b, c, _, _ = heatmap.shape
    res = torch.zeros((b, c, 2))
    for i in range(b):
        for j in range(c):
            res[i][j] = get_hottest_point(heatmap[i][j])
            print(heatmap[i][j][int(res[i][j][1])][int(res[i][j][0])])
    return res

class MahLoss(nn.Module):
    def __init__(self):
        super(MahLoss, self).__init__()

    def forward(self, pred_landmarks, ground_truth, sigma):
        sub = ground_truth - pred_landmarks
        dist = 0.5 * get_mahalanobis_distance(sub, sigma, sub)
        return torch.mean(dist, dim = 1).mean()

class SigmaReg(nn.Module):
    def __init__(self):
        super(SigmaReg, self).__init__()

    def forward(self, det_Sigma):
        return torch.mean(0.5 * torch.log(det_Sigma), dim = 1).mean()
        # return torch.mean(0.5 * det_Sigma, dim = 1).mean()

def nll_across_batch(output, target):
    nll = -target * torch.log(output.double())
    return torch.mean(torch.sum(nll, dim=(2, 3)))

if __name__ == '__main__':
    x, y, htp = generate_heatmap([256, 256], torch.tensor([[[100, 100]]]), torch.tensor([[[40, 40], [40, 40]]]))    
    fig1 = plt.figure()
    ax = fig1.add_subplot(111)
    ax.contourf(x, y, htp[0][0])
    plt.savefig("test.png")

