import numpy as np
import os
import torch
import torch.utils.data as data
from PIL import Image
import math
import copy
import torchvision.transforms as transforms
#import albumentations as A
import cv2

'''
DATALOADER OF IEEE ISBI 2015 CHALLENGE DATASET
REFERENCE: https://github.com/qsyao/attack_landmark_detection/blob/master/data_loader.py
'''

ISBI_PATH = "/root/hand with landmark"
ISBI_LANDMARK = 19
SIZE = [512, 512]
R_RATIO = 0.05

class Cephalometric(data.Dataset):

    def __init__(self, pathDataset, mode, num_landmark=37, size=[512, 512]):
        
        self.num_landmark = num_landmark
        self.Radius = int(max(size) * R_RATIO)
        self.size = size
        self.mode = mode

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


        self.pth_Image = pathDataset

        self.list = list()

        if mode == 'Train':
            self.pth_Image = os.path.join(self.pth_Image, 'train')
        else:
            self.pth_Image = os.path.join(self.pth_Image, 'test')
        
        self.resize = transforms.Resize(self.size)
        self.as_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        dirs = os.listdir(self.pth_Image)
        self.list = [os.path.join(self.pth_Image, dirr) for dirr in dirs]
    
    def resize_landmark(self, landmark, original_size):
        for i in range(len(landmark)):
            landmark[i] = int(landmark[i] * self.size[i] / original_size[i])
        return landmark

    def __getitem__(self, index):
        item = self.list[index]
        img = Image.open(os.path.join(item, 'img.jpg'))
        original_size = img.size
        
        landmark_list = list()
        landmarks = list()
        with open(os.path.join(item, 'landmarks.txt')) as f:
            for i in range(self.num_landmark):
                landmark1 = f.readline().split()
                landmark = [int(landmark1[i]) for i in range(len(landmark1))]
                tmp_lmk = self.resize_landmark(landmark, original_size)
                landmark_list.append(tmp_lmk)
                landmarks.append(tmp_lmk)

        img = self.resize(img.convert('RGB'))
        image = self.as_tensor(img)

        landmarks = torch.tensor(landmarks)

        # GT, mask, offset
        y, x = image.shape[-2], image.shape[-1]
        gt = torch.zeros((self.num_landmark, y, x), dtype=torch.float)
        mask = torch.zeros((self.num_landmark, y, x), dtype=torch.float)
        guassian_mask = torch.zeros((self.num_landmark, y, x), dtype=torch.float)
        offset_x = torch.zeros((self.num_landmark, y, x), dtype=torch.float)
        offset_y = torch.zeros((self.num_landmark, y, x), dtype=torch.float)

        for i, landmark in enumerate(landmark_list):
            gt[i][landmark[1]][landmark[0]] = 1

            margin_x_left = max(0, landmark[0] - self.Radius)
            margin_x_right = min(x, landmark[0] + self.Radius)
            margin_y_bottom = max(0, landmark[1] - self.Radius)
            margin_y_top = min(y, landmark[1] + self.Radius)

            mask[i][margin_y_bottom:margin_y_top, margin_x_left:margin_x_right] = \
                self.mask[0:margin_y_top-margin_y_bottom, 0:margin_x_right-margin_x_left]
            guassian_mask[i][margin_y_bottom:margin_y_top, margin_x_left:margin_x_right] = \
                self.guassian_mask[0:margin_y_top-margin_y_bottom, 0:margin_x_right-margin_x_left]
            offset_x[i][margin_y_bottom:margin_y_top, margin_x_left:margin_x_right] = \
                self.offset_x[0:margin_y_top-margin_y_bottom, 0:margin_x_right-margin_x_left]
            offset_y[i][margin_y_bottom:margin_y_top, margin_x_left:margin_x_right] = \
                self.offset_y[0:margin_y_top-margin_y_bottom, 0:margin_x_right-margin_x_left]
        
        return image, landmarks, mask, guassian_mask, offset_y, offset_x, landmark_list

    def __len__(self):
        return len(self.list)

if __name__ == '__main__':
    test = Cephalometric('/home/yeziyang/Desktop/IEEE ISBI 2015 Challenge', 'Test1')
    for i in range(10):
        img, landmark_list = test.__getitem__(i)
    print("pass")
