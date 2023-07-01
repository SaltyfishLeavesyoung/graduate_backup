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

plt.ion()  # interactive mode
batchSize = 1
landmarkNum = 6
image_scale = (8, 256, 256)
original_image_scale = (8, 256, 256)
cropSize = 8
use_gpu = 0
iteration = 3
traincsv = 'skull_train1_m_mini.csv'
testcsv = 'skull_test1_m_mini.csv'

dataRoot = "processed_data/"
epochs = 500
saveName = "test"
testName = "190VGG19_bn_concatFPN_originOff_withIceptionkernel_newdata_32_noPretrain_try.pkl"

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

fine_LSTM = MyModel.fine_LSTM(landmarkNum, use_gpu, iteration, cropSize).cuda(use_gpu)
corseNet = MyModel.coarseNet(landmarkNum, use_gpu, image_scale).cuda(use_gpu)
#fine_LSTM.load_state_dict(torch.load("/home/yeziyang/Desktop/SA_LSTM/models/base_model/lr=3e-4wd=1e-4_126_0.879717989135107_lstm.pth"))
#corseNet.load_state_dict(torch.load("/home/yeziyang/Desktop/SA_LSTM/models/base_model/lr=3e-4wd=1e-4_126_0.879717989135107_corse.pth"))
#print(corseNet.state_dict()['encoder.Lconv1.0.weight'])
#print(corseNet.state_dict()['encoder.Lconv1.0.bias'])

print("image scale ", image_scale)

print("GPU: ", use_gpu)
print(saveName)

transform_origin = transforms.Compose([
    Rescale(image_scale),
    ToTensor()
])

train_dataset_origin = MyLandmarksDataset('/home/yeziyang/Desktop/JointLandmark/train.txt')

val_dataset = MyLandmarksDataset('/home/yeziyang/Desktop/JointLandmark/train_val.txt')

dataloaders = {}
train_dataloader = []
val_dataloader = []

train_dataloader_t = DataLoader(train_dataset_origin, batch_size=batchSize,
                                shuffle=True, num_workers=8)

for data in train_dataloader_t:
    train_dataloader.append(data)

val_dataloader_t = DataLoader(val_dataset, batch_size=batchSize,
                              shuffle=False, num_workers=0)

for data in val_dataloader_t:
    val_dataloader.append(data)

print(len(train_dataloader), len(val_dataloader))

train_dataloader_t = ''
val_dataloader_t = ''

dataloaders = {'train': train_dataloader, 'val': val_dataloader}

criterion_coarse = LossFunction.coarse_heatmap(use_gpu, batchSize, landmarkNum, image_scale)

params = list(corseNet.parameters()) + list(fine_LSTM.parameters())

optimizer_ft = optim.Adam(params, lr = 3e-4, weight_decay = 1e-4)

TrainNet.train_model(corseNet, fine_LSTM, dataloaders, criterion_coarse,
                     optimizer_ft, epochs, use_gpu, saveName, landmarkNum, image_scale)
