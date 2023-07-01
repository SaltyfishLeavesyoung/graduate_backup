import numpy as np
import os
import torch
import torch.utils.data as data
from PIL import Image
import math
import copy
import torchvision.transforms as transforms
import albumentations as A
import cv2

'''
DATALOADER OF IEEE ISBI 2015 CHALLENGE DATASET
REFERENCE: https://github.com/qsyao/attack_landmark_detection/blob/master/data_loader.py
'''

ISBI_PATH = "./dataset/"
ISBI_LANDMARK = 6
SIZE = [256, 256]
R_RATIO = 0.05
SCALE = 1

class Cephalometric(data.Dataset):

    def __init__(self, pathDataset, R_ratio=0.05, num_landmark=6, size=[256, 256]):
        
        self.num_landmark = num_landmark
        self.Radius = int(max(size) * R_ratio)
        self.size = size

        # gen mask
        mask = torch.zeros(2*self.Radius, 2*self.Radius, dtype=torch.float)
        guassian_mask = torch.zeros(2*self.Radius, 2*self.Radius, dtype=torch.float)
        for i in range(2*self.Radius):
            for j in range(2*self.Radius):
                distance = np.linalg.norm([i+1 - self.Radius, j+1 - self.Radius])
                if distance < self.Radius:
                    mask[i][j] = 1
                    # for guassian mask
                    guassian_mask[i][j] = math.exp(-0.5 * math.pow(distance, 2) /\
                        math.pow(self.Radius, 2))
        self.mask = mask
        self.guassian_mask = guassian_mask
        
        # gen offset
        self.offset_x = torch.zeros(2*self.Radius, 2*self.Radius, dtype=torch.float)
        self.offset_y = torch.zeros(2*self.Radius, 2*self.Radius, dtype=torch.float)
        for i in range(2*self.Radius):
            self.offset_x[:, i] = self.Radius - i
            self.offset_y[i, :] = self.Radius - i
        self.offset_x = self.offset_x * self.mask / self.Radius
        self.offset_y = self.offset_y * self.mask / self.Radius

        self.list = list()

        with open(pathDataset) as f:
            line = f.readline().split()[0]
            while line:
                idx = str(int(line))
                self.list.append({'ID' : idx, 'part' : 'L'})
                self.list.append({'ID' : idx, 'part' : 'R'})
                line = f.readline()
        
        self.resize = transforms.Resize(self.size)
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
        n, w, h = pth_img.shape

        landmark_list = list()
        z_list = list()
        z_list_one_hot = list()
        with open(os.path.join('/home/yzy/Desktop/JointLandmark/data', item['ID'], item['part']+'.txt')) as f:
            for i in range(self.num_landmark):
                landmark = f.readline().split()
                landmark = [int(i) for i in landmark]
                tmp_lmk = self.resize_landmark(landmark[1:], [w, h])
                landmark_list.append(tmp_lmk)
                z_list_one_hot.append([0] * 8)
                z_list_one_hot[i][landmark[0]] = 1
                z_list.append(landmark[0])
        
        z_list = torch.from_numpy(np.array(z_list, dtype=np.int64))
        z_list_one_hot = torch.from_numpy(np.array(z_list_one_hot, dtype=np.float32))
        # GT, mask, offset
        img = []
        for i in range(n):
            img.append(self.as_tensor(transforms.ToPILImage()(pth_img[i])))
        imgs = torch.stack(img, dim = 1)
        # print(imgs.shape) [1,8,256,256]
        item['image'] = imgs

        y, x = item['image'].shape[-2], item['image'].shape[-1]
        gt = torch.zeros((self.num_landmark, y, x), dtype=torch.float)
        mask = torch.zeros((self.num_landmark, y, x), dtype=torch.float)
        guassian_mask = torch.zeros((self.num_landmark, y, x), dtype=torch.float)
        offset_x = torch.zeros((self.num_landmark, y, x), dtype=torch.float)
        offset_y = torch.zeros((self.num_landmark, y, x), dtype=torch.float)

        for i, landmark in enumerate(landmark_list):

            gt[i][landmark[0]][landmark[1]] = 1

            margin_x_left = max(0, landmark[1] - self.Radius)
            margin_x_right = min(x, landmark[1] + self.Radius)
            margin_y_bottom = max(0, landmark[0] - self.Radius)
            margin_y_top = min(y, landmark[0] + self.Radius)

            mask[i][margin_y_bottom:margin_y_top, margin_x_left:margin_x_right] = \
                self.mask[0:margin_y_top-margin_y_bottom, 0:margin_x_right-margin_x_left]
            guassian_mask[i][margin_y_bottom:margin_y_top, margin_x_left:margin_x_right] = \
                self.guassian_mask[0:margin_y_top-margin_y_bottom, 0:margin_x_right-margin_x_left]
            offset_x[i][margin_y_bottom:margin_y_top, margin_x_left:margin_x_right] = \
                self.offset_x[0:margin_y_top-margin_y_bottom, 0:margin_x_right-margin_x_left]
            offset_y[i][margin_y_bottom:margin_y_top, margin_x_left:margin_x_right] = \
                self.offset_y[0:margin_y_top-margin_y_bottom, 0:margin_x_right-margin_x_left]

        
        item['image'] = torch.cat([item['image'], item['image'], item['image']], dim=0)
        # print(item['image'].shape) [3,8,256,256]

        return item['image'], mask, guassian_mask, z_list, z_list_one_hot, offset_y, offset_x, landmark_list, w, h

    def __len__(self):
        
        return len(self.list)

if __name__ == '__main__':
    test = Cephalometric('/home/yzy/Desktop/JointLandmark/train.txt')
    for i in range(1):
        img, mask, guassian_mask, offset_y, offset_x, landmark_list, w, h = test.__getitem__(i+1)
        for j in range(6):
            cv2.imwrite("mask_{}.jpg".format(j), np.float32(mask[j] * 255))
    print("pass")
