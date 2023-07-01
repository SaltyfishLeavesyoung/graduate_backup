from __future__ import print_function, absolute_import

import os
import numpy as np
import json
import random
import math

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from utils.osutils import *
from utils.imutils import *
from utils.transforms import *

import pandas as pd
import random
import nrrd

class fa68pt3D(data.Dataset):
    def __init__(self, pathDataset, train, num_landmark=5):

        self.list = list()
        self.size = [256, 256]
        self.num_landmark = num_landmark
        self.train = train

        for dirs in os.listdir(pathDataset):
            self.list.append(os.path.join(pathDataset, dirs))

        self.resize = transforms.Resize(self.size)
        self.as_tensor = transforms.Compose([
            self.resize,
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.transform_resize = transforms.Compose([
            self.resize,
        ])

    def resize_landmark(self, landmark, original_size):
        for i in range(len(landmark)):
            landmark[i] = int(landmark[i] * self.size[i] / original_size[i])
        return landmark

    def __getitem__(self, index):
        item = self.list[index]
        filename = os.path.join(item, 'img.nrrd')
        #readdata, header = nrrd.read(filename)
        readdata = np.load(os.path.join(item, 'img.npy'))
        imgs = np.clip(readdata, -1000, 1000)
        imgs = ((imgs + 1000) / 2000 * 255).astype('uint8')

        imgs = np.transpose(imgs, (2,0,1))
        n, h, w = imgs.shape

        min_z = 1145141919

        with open(os.path.join(item, 'xyz.csv')) as f:
            for i in range(self.num_landmark):
                min_z = min(min_z, int(f.readline().split()[2]))

        landmark_list = list()

        if n - min_z <= 64:
            imgs = imgs[n - 64:,:,:]
            img = []
            for i in range(64):
                img.append(self.as_tensor(transforms.ToPILImage()(imgs[i])))
            imgs = torch.stack(img, dim = 1)
            with open(os.path.join(item, 'xyz.csv')) as f:
                for i in range(self.num_landmark):
                    landmark = f.readline().split()
                    landmark = [int(i) for i in landmark]
                    landmark[0] = landmark[0] // 2
                    landmark[1] = landmark[1] // 2
                    landmark[2] = landmark[2] - (n - 64)
                    landmark_list.append(landmark)
        else:
            imgs = imgs[min_z:min_z + 64,:,:]
            img = []
            for i in range(64):
                img.append(self.as_tensor(transforms.ToPILImage()(imgs[i])))
            imgs = torch.stack(img, dim = 1)
            with open(os.path.join(item, 'xyz.csv')) as f:
                for i in range(self.num_landmark):
                    landmark = f.readline().split()
                    landmark = [int(i) for i in landmark]
                    landmark[0] = landmark[0] // 2
                    landmark[1] = landmark[1] // 2
                    landmark[2] = landmark[2] - min_z
                    landmark_list.append(landmark)

        pts = torch.from_numpy(np.array(landmark_list, dtype=np.float32))

        # Prepare image and groundtruth map
        inp = torch.cat([imgs, imgs, imgs], dim=0)

        meta = {'index': index}

        return inp, pts, meta

    def __len__(self):
        return len(self.list)

if __name__ == '__main__':
    for dirs in os.listdir('/home/yeziyang/Desktop/PDDCA/train'):
        filename = os.path.join('/home/yeziyang/Desktop/PDDCA/train', dirs, 'img.nrrd')
        readdata, header = nrrd.read(filename)
        print(header)
        print("------")




