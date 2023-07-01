from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import models, transforms, utils
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import time
import os
import math
from copy import deepcopy
import pandas as pd
from MyDataLoader import Rescale, ToTensor, LandmarksDataset, MyLandmarksDataset
import MyModel
import TrainNet
import LossFunction
import MyUtils
from tqdm import tqdm

from torch.nn import MSELoss, L1Loss

plt.ion()  # interactive mode
batchSize = 1
landmarkNum = 6
image_scale = (8, 256, 256)
original_image_scale = (8, 256, 256)
cropSize = 8
use_gpu = 0
iteration = 3

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

fine_LSTM = MyModel.fine_LSTM(landmarkNum, use_gpu, iteration, cropSize).cuda(use_gpu)
corseNet = MyModel.coarseNet(landmarkNum, use_gpu, image_scale).cuda(use_gpu)

fine_LSTM.load_state_dict(torch.load("/home/yeziyang/Desktop/SA_LSTM/models/10_lr=3e-4wd=1e-4_211_8.775144876414583_l.pth"))
corseNet.load_state_dict(torch.load("/home/yeziyang/Desktop/SA_LSTM/models/10_lr=3e-4wd=1e-4_211_8.775144876414583_corse.pth"))

transform_origin = transforms.Compose([
    Rescale(image_scale),
    ToTensor()
])

val_dataset = MyLandmarksDataset('/home/yeziyang/Desktop/JointLandmark/test.txt')


val_dataloader_t = DataLoader(val_dataset, batch_size=batchSize,
                              shuffle=False, num_workers=4)


gl, gh, gw = image_scale
global_coordinate = torch.ones(gl, gh, gw, 3).float()
for i in range(gl):
    global_coordinate[i, :, :, 0] = global_coordinate[i, :, :, 0] * i
for i in range(gh):
    global_coordinate[:, i, :, 1] = global_coordinate[:, i, :, 1] * i
for i in range(gw):
    global_coordinate[:, :, i, 2] = global_coordinate[:, :, i, 2] * i
global_coordinate = global_coordinate * torch.tensor([1 / (gl - 1), 1 / (gh - 1), 1 / (gw - 1)])
global_coordinate = global_coordinate.cuda()
size_tensor = torch.tensor([1 / 255, 1 / 255, 1 / 7])


corseNet.train(False)  # Set model to evaluate mode
fine_LSTM.train(False)

losses = 0
ide = 0

ratio = torch.tensor([255.0, 255.0, 7.0]).cuda()

for data in tqdm(val_dataloader_t):

    inputs, inputs_origin, labels, image_name = data['DICOM'], data['DICOM_origin'], data[
                    'landmarks'], data['imageName']
    inputs, inputs_origin, labels = inputs.cuda(), inputs_origin.cuda(), labels.cuda()

    coarse_heatmap, coarse_features = corseNet(inputs)
    coarse_landmarks = MyUtils.get_coordinates_from_coarse_heatmaps(coarse_heatmap, global_coordinate).unsqueeze(0)
    ROIs = coarse_landmarks.cpu().detach().numpy() + np.random.uniform(0, 0.02, labels.size())

    fine_landmarks = fine_LSTM(ROIs, labels, inputs_origin, coarse_features, 'val', size_tensor, ide)

    for i in range(labels.shape[1]):
        print(torch.mul(fine_landmarks[2][i], ratio))
        print(torch.mul(labels[0][i], ratio))
        print("--------------")
        dist = (fine_landmarks[2][i][0] * 255 - labels[0][i][0] * 255) ** 2 + (fine_landmarks[2][i][1] * 255 - labels[0][i][1] * 255) ** 2 + (fine_landmarks[2][i][2] * 7 - labels[0][i][2] * 7) ** 2
        losses += np.sqrt(dist.detach().item())

    # print(fine_landmarks.shape) [3,6,3]
    # print(labels.shape) [1,6,3]

    ide += 1

avg_loss = losses / (len(val_dataloader_t) * 6)
print("MRE: {}".format( avg_loss))