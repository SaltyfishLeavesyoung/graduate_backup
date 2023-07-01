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

class fa68pt3D(data.Dataset):
    def __init__(self, pathDataset, train, num_landmark=6, c2f=True, inp_res=256, out_res=64, depth_res=None, nStack=1, sigma=1, label_type='Gaussian'):

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
        self.train = train
        self.amp_label = '/home/yeziyang/Desktop/JointLandmark/label_amp.csv'

        if self.c2f:
            assert len(self.depth_res) == self.nStack

        df = pd.read_csv(self.amp_label)
        indexes = df['dir']
        labels = df['label']

        zeros = []
        ones = []

        self.abnormal = []
        self.normal = []
        for i in range(len(indexes)):
            if labels[i] == 0:
                zeros.append(str(indexes[i]))
            elif labels[i] == 1:
                ones.append(str(indexes[i]))
            else:
                pass

        with open(pathDataset) as f:
            line = f.readline().split()[0]
            while line:
                idx = str(int(line))
                if idx in zeros:
                    amp = 0
                    self.normal.append(idx)
                elif idx in ones:
                    amp = 1
                    self.abnormal.append(idx)
                else:
                    amp = 0
                self.list.append({'ID' : idx, 'part' : 'L', 'amp': amp, 'trans': False})
                self.list.append({'ID' : idx, 'part' : 'R', 'amp': amp, 'trans': False})
                '''if self.train:
                   self.list.append({'ID' : idx, 'part' : 'L', 'amp': amp, 'trans': True})
                   self.list.append({'ID' : idx, 'part' : 'R', 'amp': amp, 'trans': True})'''
                '''if amp == 1:
                    self.list.append({'ID' : idx, 'part' : 'L', 'trans': False})
                    self.list.append({'ID' : idx, 'part' : 'R', 'trans': False})
                else:
                    pass
                    self.list.append({'ID' : idx, 'part' : 'L', 'trans': True})
                    self.list.append({'ID' : idx, 'part' : 'R', 'trans': True})'''
                line = f.readline()

        self.resize = transforms.Resize(self.size[:2])
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
        pth_img = np.load(os.path.join('/home/yeziyang/Desktop/JointLandmark/data', item['ID'], item['part']+'.npy')).astype(np.uint8)
        n, h, w = pth_img.shape

        '''if item['amp'] == 0:
            sel = random.choice(self.abnormal)
            amp = np.load(os.path.join('/home/yzy/Desktop/JointLandmark/data', sel, 'amp.npy')).astype(np.uint8)
        else:
            sel = random.choice(self.normal)
            amp = np.load(os.path.join('/home/yzy/Desktop/JointLandmark/data', sel, 'amp.npy')).astype(np.uint8)'''

        img = []
        for i in range(n):
            if item['trans'] == False:
                img.append(self.as_tensor(transforms.ToPILImage()(pth_img[i])))
            else:
                sel = random.choice(self.normal)
                amp = np.load(os.path.join('/home/yeziyang/Desktop/JointLandmark/data', sel, 'amp.npy')).astype(np.uint8)
                raw_img = self.transform_resize(transforms.ToPILImage()(pth_img[i]))
                fft_img = np.fft.fft2(raw_img, axes=(-2,-1))
                pha_img = np.angle(fft_img)
                fft_transfer = amp * np.exp(1j * pha_img)
                ifft_img = np.real(np.fft.ifft2(fft_transfer, axes=(-2,-1))).astype(np.uint8)
                img.append(self.as_tensor(transforms.ToPILImage()(ifft_img)))
            '''raw_img = self.transform_resize(transforms.ToPILImage()(pth_img[i]))
            fft_img = np.fft.fft2(raw_img, axes=(-2,-1))
            pha_img = np.angle(fft_img)
            fft_transfer = np.exp(1j * pha_img)
            ifft_img = np.real(np.fft.ifft2(fft_transfer, axes=(-2,-1))).astype(np.uint8)
            img.append(self.as_tensor(transforms.ToPILImage()(ifft_img)))'''
        imgs = torch.stack(img, dim = 1)

        landmark_list = list()
        with open(os.path.join('/home/yeziyang/Desktop/JointLandmark/data', item['ID'], item['part']+'.txt')) as f:
            for i in range(self.num_landmark):
                landmark = f.readline().split()
                # landmark.reverse()
                landmark = [int(i) for i in landmark]
                tmp_lmk = self.resize_landmark(landmark, [w, h, 8])
                landmark_list.append(tmp_lmk)
                # landmark_list.append(tmp_lmk[:2])

        """for i in range(self.num_landmark):
            landmark = [0,0,0]
            tmp_lmk = self.resize_landmark(landmark, [256, 256, 8])
            landmark_list.append(tmp_lmk)"""

        pts = torch.from_numpy(np.array(landmark_list, dtype=np.float32))

        # Prepare image and groundtruth map
        inp = torch.cat([imgs, imgs, imgs], dim=0)
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
        meta = {'index': index, 'pts': pts, 'tpts': tpts, 'tpts_inp': tpts_inp, 'id': item['ID'], 'part': item['part']}

        return inp, target, meta

    def __len__(self):
        return len(self.list)

class fa68pt3D_2(data.Dataset):
    def __init__(self, pathDataset, train, num_landmark=6, c2f=True, inp_res=256, out_res=64, depth_res=None, nStack=1, sigma=1, label_type='Gaussian'):

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
        self.train = train
        self.amp_label = '/home/yeziyang/Desktop/JointLandmark/label_amp.csv'

        if self.c2f:
            assert len(self.depth_res) == self.nStack

        df = pd.read_csv(self.amp_label)
        indexes = df['dir']
        labels = df['label']

        zeros = []
        ones = []

        self.abnormal = []
        self.normal = []
        for i in range(len(indexes)):
            if labels[i] == 0:
                zeros.append(str(indexes[i]))
            elif labels[i] == 1:
                ones.append(str(indexes[i]))
            else:
                pass

        with open(pathDataset) as f:
            line = f.readline().split()[0]
            while line:
                idx = str(int(line))
                if idx in zeros:
                    amp = 0
                    self.normal.append(idx)
                elif idx in ones:
                    amp = 1
                    self.abnormal.append(idx)
                else:
                    amp = 0
                self.list.append({'ID' : idx, 'part' : 'L', 'amp': amp, 'trans': False})
                self.list.append({'ID' : idx, 'part' : 'R', 'amp': amp, 'trans': False})
                '''if self.train:
                   self.list.append({'ID' : idx, 'part' : 'L', 'amp': amp, 'trans': True})
                   self.list.append({'ID' : idx, 'part' : 'R', 'amp': amp, 'trans': True})'''
                '''if amp == 1:
                    self.list.append({'ID' : idx, 'part' : 'L', 'trans': False})
                    self.list.append({'ID' : idx, 'part' : 'R', 'trans': False})
                else:
                    pass
                    self.list.append({'ID' : idx, 'part' : 'L', 'trans': True})
                    self.list.append({'ID' : idx, 'part' : 'R', 'trans': True})'''
                line = f.readline()

        self.resize = transforms.Resize(self.size[:2])
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
        pth_img = np.load(os.path.join('/home/yeziyang/Desktop/JointLandmark/semidata', item['ID'], item['part']+'.npy')).astype(np.uint8)
        n, h, w = pth_img.shape

        '''if item['amp'] == 0:
            sel = random.choice(self.abnormal)
            amp = np.load(os.path.join('/home/yzy/Desktop/JointLandmark/data', sel, 'amp.npy')).astype(np.uint8)
        else:
            sel = random.choice(self.normal)
            amp = np.load(os.path.join('/home/yzy/Desktop/JointLandmark/data', sel, 'amp.npy')).astype(np.uint8)'''

        img = []
        for i in range(n):
            if item['trans'] == False:
                img.append(self.as_tensor(transforms.ToPILImage()(pth_img[i])))
            else:
                sel = random.choice(self.normal)
                amp = np.load(os.path.join('/home/yeziyang/Desktop/JointLandmark/data', sel, 'amp.npy')).astype(np.uint8)
                raw_img = self.transform_resize(transforms.ToPILImage()(pth_img[i]))
                fft_img = np.fft.fft2(raw_img, axes=(-2,-1))
                pha_img = np.angle(fft_img)
                fft_transfer = amp * np.exp(1j * pha_img)
                ifft_img = np.real(np.fft.ifft2(fft_transfer, axes=(-2,-1))).astype(np.uint8)
                img.append(self.as_tensor(transforms.ToPILImage()(ifft_img)))
            '''raw_img = self.transform_resize(transforms.ToPILImage()(pth_img[i]))
            fft_img = np.fft.fft2(raw_img, axes=(-2,-1))
            pha_img = np.angle(fft_img)
            fft_transfer = np.exp(1j * pha_img)
            ifft_img = np.real(np.fft.ifft2(fft_transfer, axes=(-2,-1))).astype(np.uint8)
            img.append(self.as_tensor(transforms.ToPILImage()(ifft_img)))'''
        imgs = torch.stack(img, dim = 1)

        landmark_list = list()
        with open(os.path.join('/home/yeziyang/Desktop/JointLandmark/MODEL_MY3D/pseudolabel_690', item['ID'] + item['part']+'.txt')) as f:
            for i in range(self.num_landmark):
                landmark = f.readline().split()
                landmark = [int(i) for i in landmark]
                tmp_lmk = self.resize_landmark(landmark, [256, 256, 8])
                landmark_list.append(tmp_lmk)
                # landmark_list.append(tmp_lmk[:2])

        pts = torch.from_numpy(np.array(landmark_list, dtype=np.float32))

        # Prepare image and groundtruth map
        inp = torch.cat([imgs, imgs, imgs], dim=0)
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
        meta = {'index': index, 'pts': pts, 'tpts': tpts, 'tpts_inp': tpts_inp, 'id': item['ID'], 'part': item['part']}

        return inp, target, meta

    def __len__(self):
        return len(self.list)

class fa68pt3D_3(data.Dataset):
    def __init__(self, pathDataset1, pathDataset2, train, num_landmark=6, c2f=True, inp_res=256, out_res=64, depth_res=None, nStack=1, sigma=1, label_type='Gaussian'):

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
        self.train = train

        if self.c2f:
            assert len(self.depth_res) == self.nStack

        with open(pathDataset1) as f:
            line = f.readline().split()[0]
            while line:
                idx = str(int(line))
                self.list.append({'ID' : idx, 'part' : 'L', 'amp': 0})
                self.list.append({'ID' : idx, 'part' : 'R', 'amp': 0})
                line = f.readline()

        with open(pathDataset2) as f:
            line = f.readline().split()[0]
            while line:
                idx = str(int(line))
                self.list.append({'ID' : idx, 'part' : 'L', 'amp': 1})
                self.list.append({'ID' : idx, 'part' : 'R', 'amp': 1})
                line = f.readline()

        self.resize = transforms.Resize(self.size[:2])
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
        root = ''
        root2 = ''

        if item['amp'] == 0:
            root = os.path.join('/home/yeziyang/Desktop/JointLandmark/data', item['ID'], item['part']+'.txt')
            root2 = '/home/yeziyang/Desktop/JointLandmark/data'
        else:
            root = os.path.join('/home/yeziyang/Desktop/JointLandmark/MODEL_MY3D/pseudolabel_690', item['ID']+ item['part']+'.txt')
            root2 = '/home/yeziyang/Desktop/JointLandmark/semidata'

        pth_img = np.load(os.path.join(root2, item['ID'], item['part']+'.npy')).astype(np.uint8)
        n, h, w = pth_img.shape

        img = []
        for i in range(n):
            img.append(self.as_tensor(transforms.ToPILImage()(pth_img[i])))
        imgs = torch.stack(img, dim = 1)

        landmark_list = list()
        
        with open(root) as f:
            for i in range(self.num_landmark):
                landmark = f.readline().split()
                landmark.reverse()
                landmark = [int(i) for i in landmark]
                tmp_lmk = self.resize_landmark(landmark, [w, h, 8])
                landmark_list.append(tmp_lmk)
                # landmark_list.append(tmp_lmk[:2])

        pts = torch.from_numpy(np.array(landmark_list, dtype=np.float32))

        # Prepare image and groundtruth map
        inp = torch.cat([imgs, imgs, imgs], dim=0)
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
        meta = {'index': index, 'pts': pts, 'tpts': tpts, 'tpts_inp': tpts_inp, 'id': item['ID'], 'part': item['part']}

        return inp, target, meta

    def __len__(self):
        return len(self.list)
