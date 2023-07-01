import numpy as np
import os
import torch
import torch.utils.data as data
from PIL import Image
import math
import copy
import torchvision.transforms as transforms
import albumentations as A

'''
DATALOADER OF IEEE ISBI 2015 CHALLENGE DATASET
REFERENCE: https://github.com/qsyao/attack_landmark_detection/blob/master/data_loader.py
'''

ISBI_PATH = "./dataset/"
ISBI_LANDMARK = 19
SIZE = [800, 640]
R_RATIO = 0.05
SCALE = 1

class Cephalometric(data.Dataset):

    def __init__(self, pathDataset, mode, R_ratio=0.05, num_landmark=19, size=[800, 640]):
        
        self.num_landmark = num_landmark
        self.Radius = int(max(size) * R_ratio)
        self.size = size
        self.mode = mode

        self.original_size = [2400, 1935]

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

        self.pth_Image = os.path.join(pathDataset, 'RawImage')
        self.pth_label_junior = os.path.join(pathDataset, '400_junior')
        self.pth_label_senior = os.path.join(pathDataset, '400_senior')

        self.list = list()

        if mode == 'Train':
            self.pth_Image = os.path.join(self.pth_Image, 'TrainingData')
            start = 1
            end = 150
        elif mode == 'Test1':
            self.pth_Image = os.path.join(self.pth_Image, 'Test1Data')
            start = 151
            end = 300
        else:
            self.pth_Image = os.path.join(self.pth_Image, 'Test2Data')
            start = 301
            end = 400
        
        # normalize = transforms.Normalize([0.5], [0.5])
        # transformList = []
        # transformList.append(transforms.Resize(self.size))
        # transformList.append(transforms.ToTensor())
        # transformList.append(normalize)      
        # self.transform = transforms.Compose(transformList)
        self.augment =  A.Compose([
                A.ShiftScaleRotate(shift_limit=(-0.1,0.1),scale_limit=0,rotate_limit=60,border_mode=0,p=0.4), 
            ], 
            keypoint_params=A.KeypointParams(format='xy')
        )
        
        self.resize = transforms.Resize(self.size)
        self.as_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        for i in range(start, end + 1):
            self.list.append({'ID': "{0:03d}".format(i)})
    
    def resize_landmark(self, landmark):
        for i in range(len(landmark)):
            landmark[i] = int(landmark[i] * self.size[i] / self.original_size[i])
        return landmark

    def __getitem__(self, index):


        item = self.list[index]

        # if self.transform != None:
        #     pth_img = os.path.join(self.pth_Image, item['ID']+'.bmp')
        #     item['image'] = self.transform(Image.open(pth_img).convert('RGB'))
        
        landmark_list = list()
        lmk_mat = []
        with open(os.path.join(self.pth_label_junior, item['ID']+'.txt')) as f1:
            with open(os.path.join(self.pth_label_senior, item['ID']+'.txt')) as f2:
                for i in range(self.num_landmark):
                    landmark1 = f1.readline().split()[0].split(',')
                    landmark2 = f2.readline().split()[0].split(',')
                    landmark = [int(0.5*(int(landmark1[i]) + int(landmark2[i]))) for i in range(len(landmark1))]
                    tmp_lmk = self.resize_landmark(landmark)
                    landmark_list.append(tmp_lmk)
                    lmk_mat = lmk_mat + tmp_lmk

        pth_img = os.path.join(self.pth_Image, item['ID']+'.bmp')
        img = self.resize(Image.open(pth_img).convert('RGB'))
        if self.mode == "Train":
            augments = self.augment(image = np.array(img), keypoints = landmark_list)
            item['image'] = self.as_tensor(augments['image'])
            landmark_list = augments['keypoints']
        else:
            item['image'] = self.as_tensor(img)
        
        # GT, mask, offset
        y, x = item['image'].shape[-2], item['image'].shape[-1]
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

        lmk_mat = torch.Tensor(lmk_mat) / SCALE
        
        return item['image'], mask, guassian_mask, offset_y, offset_x, landmark_list

    def __len__(self):
        
        return len(self.list)

if __name__ == '__main__':
    test = Cephalometric('dataset/', 'Test1')
    for i in range(100):
        test.__getitem__(i)
    print("pass")
