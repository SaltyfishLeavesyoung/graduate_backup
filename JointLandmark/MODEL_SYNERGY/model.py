import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms as T
import scipy.io as sio

import time

from backbone_nets import mobilenetv2_backbone
from backbone_nets.pointnet_backbone import MLP_for, MLP_rev

class I2P(nn.Module):
	def __init__(self, args):
		super(I2P, self).__init__()
		self.args = args
		self.backbone = mobilenetv2_backbone.MobileNetV2()

	def forward(self, input):
		"""Training time forward"""
		_3D_attr, avgpool = self.backbone(input)
		return _3D_attr, avgpool

# Main model SynergyNet definition
class SynergyNet(nn.Module):
	def __init__(self, args):
		super(SynergyNet, self).__init__()
		self.img_size = args.img_size
		self.I2P = I2P(args)
		# Forward
		self.forwardDirection = MLP_for(6)

	def forward(self, input):
		_3D_attr, avgpool = self.I2P(input)
		# print(_3D_attr.shape) [8, 18]
		vertex_lmk = _3D_attr.reshape(_3D_attr.size(0), 3, -1)
		point_residual = self.forwardDirection(vertex_lmk, avgpool)
		# print(point_residual.shape) [8, 3, 6]
		point_residual = point_residual.permute(0,2,1)
		return point_residual


if __name__ == '__main__':
	pass
