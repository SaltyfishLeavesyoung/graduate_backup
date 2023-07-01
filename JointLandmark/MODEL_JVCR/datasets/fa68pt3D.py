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


class fa68pt3D(data.Dataset):
    def __init__(self, pathDataset, num_landmark=6, c2f=True, inp_res=256, out_res=64, depth_res=None, nStack=1, sigma=1, label_type='Gaussian'):

        self.list = list()
        self.size = [256, 256, 8]
        self.c2f = c2f
        self.inp_res = inp_res
        self.out_res = out_res
        self.depth_res = depth_res
        self.sigma = sigma
        self.label_type = label_type
        self.nStack = nStack
        self.num_landmark = num_landmark

        if self.c2f:
            assert len(self.depth_res) == self.nStack

        with open(pathDataset) as f:
            line = f.readline().split()[0]
            while line:
                idx = str(int(line))
                self.list.append({'ID' : idx, 'part' : 'L'})
                self.list.append({'ID' : idx, 'part' : 'R'})
                line = f.readline()

        self.resize = transforms.Resize(self.size[:2])
        self.as_tensor = transforms.Compose([
            self.resize,
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def resize_landmark(self, landmark, original_size):
        for i in range(len(landmark)):
            landmark[i] = int(landmark[i] * self.size[i] / original_size[i])
        return landmark

    def __getitem__(self, index):
        item = self.list[index]
        pth_img = np.load(os.path.join('/home/yzy/Desktop/JointLandmark/data', item['ID'], item['part']+'.npy')).astype(np.uint8)
        n, h, w = pth_img.shape

        img = []
        for i in range(n):
            img.append(self.as_tensor(transforms.ToPILImage()(pth_img[i])))
        imgs = torch.stack(img, dim = 1)
        imgs = torch.squeeze(imgs, dim=0)

        landmark_list = list()
        with open(os.path.join('/home/yzy/Desktop/JointLandmark/data', item['ID'], item['part']+'.txt')) as f:
            for i in range(self.num_landmark):
                landmark = f.readline().split()
                landmark.reverse()
                landmark = [int(i) for i in landmark]
                tmp_lmk = self.resize_landmark(landmark, [w, h, 8])
                landmark_list.append(tmp_lmk)

        pts = torch.from_numpy(np.array(landmark_list, dtype=np.float32))

        # Prepare image and groundtruth map
        inp = imgs
        tpts_inp = pts.clone()

        # Generate ground truth
        target = []
        vox_idx = range(self.nStack) if self.c2f else [-1]

        # compact volume
        for i in range(self.nStack):
            target_i = torch.zeros(self.depth_res[i], self.out_res, self.out_res)
            tpts = pts.clone()
            for j in range(tpts.size(0)):
                # if tpts[j, 2] > 0: # This is evil!!
                if tpts[j, 0] > 0:
                    target_j = torch.zeros(self.depth_res[i], self.out_res, self.out_res)

                    target_j = draw_labelvolume(target_j, tpts[j], self.sigma, type=self.label_type)
                    target_i = torch.max(target_i, target_j.float())

            target.append(target_i)
            # target = torch.cat((target, target_i))

        # Meta info
        meta = {'index': index, 'pts': pts, 'tpts': tpts, 'tpts_inp': tpts_inp}

        return inp, target, meta

    def __len__(self):
        return len(self.list)
