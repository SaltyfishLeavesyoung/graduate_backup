import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss

import numpy as np
import scipy.sparse as sp
import math

# =============================================================
# Interface for the losses
# =============================================================

'''
LOSS FOR KEY POINT DETECTION: UPSAMPLE MODULE TO PREDICT KEY POINTS
INFERENCE: https://github.com/qsyao/attack_landmark_detection/blob/master/network.py
'''
def L1Loss(pred, gt, mask=None):
    # L1 Loss for offset map
    assert(pred.shape == gt.shape)
    gap = pred - gt
    distence = gap.abs()
    if mask is not None:
        # Caculate grad of the area under mask
        distence = distence * mask
    return distence.sum() / mask.sum()

'''
LOSS FOR COORDINATE PREDICTION
INFERENCE: https://github.com/BeierZhu/Sturcture-Coherency-Face-Alignment/blob/9b4bf1285f12de1be85a969c0859e2982c96912d/losses/wing_loss.py
'''
def gen_laplacian_matrix(num_points, top_k):
    connection = np.loadtxt('/home/yzy/Desktop/faceKeyPoint/coord_net/landmark_graph.txt')
    return adj_mx_from_list(connection, num_points)


def adj_mx_from_list(connection, num_points):
    edges = connection[:, :2]
    return adj_mx_from_weighted_edges(num_points, edges)


def adj_mx_from_weighted_edges(num_pts, edges):
    edges = np.array(edges, dtype=np.int32)
    weights = np.ones(edges.shape[0])
    data, i, j = weights, edges[:, 0], edges[:, 1]
    adj_mx = sp.coo_matrix((data, (i, j)), shape=(num_pts, num_pts), dtype=np.float32).todense()

    np.fill_diagonal(adj_mx, 0)
    row_sum = adj_mx.sum(axis=1)
    row_sum[row_sum==0] = 1
    adj_mx /= row_sum
    np.fill_diagonal(adj_mx, -1)

    adj_mx = torch.tensor(adj_mx, dtype=torch.float)
    return adj_mx.unsqueeze(0)

        
class LaplacianLoss(_Loss):
    def __init__(self, loss, num_points, reduction="mean"):
        super(LaplacianLoss, self).__init__(reduction=reduction)
        self.loss = loss
        self.num_points = num_points
        self.laplacian_matrix = gen_laplacian_matrix(num_points, 3).cuda() # 106x106

    def forward(self, prediction, target):
        prediction = prediction.reshape(-1, self.num_points, 2)
        target = target.reshape(-1, self.num_points, 2)
        prediction = torch.matmul(self.laplacian_matrix, prediction)
        target = torch.matmul(self.laplacian_matrix, target)
        prediction = prediction.reshape(-1, self.num_points*2)
        target = target.reshape(-1, self.num_points*2)

        return self.loss(prediction, target)

def wing_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    width=5,
    curvature=0.5,
    reduction="mean",
):
    """
    https://arxiv.org/pdf/1711.06753.pdf
    :param prediction:
    :param target:
    :param width:
    :param curvature:
    :param reduction:
    :return:
    """
    diff_abs = (target - prediction).abs()
    loss = diff_abs.clone()

    idx_smaller = diff_abs < width
    idx_bigger = diff_abs >= width

    loss[idx_smaller] = width * torch.log(1 + diff_abs[idx_smaller] / curvature)

    C = width - width * math.log(1 + width / curvature)
    loss[idx_bigger] = loss[idx_bigger] - C

    if reduction == "sum":
        loss = loss.sum()

    if reduction == "mean":
        loss = loss.mean()

    return loss

class WingLoss(_Loss):
    def __init__(self, width=5, curvature=0.5, reduction="mean"):
        super(WingLoss, self).__init__(reduction=reduction)
        self.width = width
        self.curvature = curvature

    def forward(self, prediction, target):
        return wing_loss(
            prediction, target, self.width, self.curvature, self.reduction
        )

def wider_wing_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    thres1=2,
    thres2=20,
    curvature=2,
    reduction="mean"
):
    """
    wider_wing = 
        1. |x|                           , if |x| < thres1
        2. thres2*ln(1+|x|/curvature) + B, if |x| >= thres2
    """
    loss = (target - prediction).abs()

    idx_small = loss < thres1
    idx_big = loss >= thres1

    B = thres1 - thres2 * math.log(1 + thres1 / curvature)
    loss[idx_big] = thres2 * torch.log(1 + loss[idx_big] / curvature) + B


    if reduction == "sum":
        loss = loss.sum()
    if reduction == "mean":
        loss = loss.mean()

    return loss


class WiderWingLoss(_Loss):
    """docstring for WiderWingLoss"""
    def __init__(self, thres1=2, thres2=20, curvature=2, reduction="mean"):
        super(WiderWingLoss, self).__init__(reduction=reduction)
        self.thres1 = thres1
        self.thres2 = thres2
        self.curvature = curvature

    def forward(self, prediction, target):
        return wider_wing_loss(
            prediction, target, self.thres1, self.thres2, self.curvature, self.reduction
        )

def l2_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    reduction="mean"
):
    """Calculate the average l2 loss for multi-point samples.
    Each sample must contain `n` points, each with `d` dimensions. For example,
    in the MPII human pose estimation task n=16 (16 joint locations) and
    d=2 (locations are 2D).
    Args:
        prediction (Tensor): Predictions (B x 2N)
        target (Tensor): Ground truth target (B x 2N)
    """
    assert prediction.shape == target.shape, 'input tensors must have the same size'

    # Calculate Euclidean distances between actual and target locations
    diff = prediction - target
    dist_sq = diff.pow(2)
    dist_sq = dist_sq.mean(-1, keepdim=False)
    dist = dist_sq.sqrt() # shape (B x L)

    if reduction == "sum":
        dist = dist.sum()

    if reduction == "mean":
        dist = dist.mean()

    return dist
        
class L2Loss(_Loss):
    """docstring for L2Loss"""
    def __init__(self, reduction="mean"):
        super(L2Loss, self).__init__(reduction=reduction)
    
    def forward(self, prediction, target):
        return l2_loss(prediction, target, self.reduction)
