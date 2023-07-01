import torch
import numpy as np
from torch.utils.data import Dataset
import os
from scipy import ndimage

import torchvision.transforms as transforms

class MyDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, pathDataset, num_landmark=6):
        self.size = [256, 256, 8]
        self.list = list()
        self.num_landmark = num_landmark

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
        self.transform_resize = transforms.Compose([
            self.resize,
        ])

    def __len__(self):
        return len(self.list)

    def resize_landmark(self, landmark, original_size):
        for i in range(len(landmark)):
            landmark[i] = int(landmark[i] * self.size[i] / original_size[i])
        return landmark

    def __getitem__(self, index):
        item = self.list[index]
        pth_img = np.load(os.path.join('/home/yeziyang/Desktop/JointLandmark/data', item['ID'], item['part']+'.npy')).astype(np.uint8)
        n, h, w = pth_img.shape

        img = []
        for i in range(n):
            img.append(self.as_tensor(transforms.ToPILImage()(pth_img[i])))
        imgs = torch.stack(img, dim = 1)

        landmark_list = list()
        with open(os.path.join('/home/yeziyang/Desktop/JointLandmark/data', item['ID'], item['part']+'.txt')) as f:
            for i in range(self.num_landmark):
                landmark = f.readline().split()
                landmark = [int(i) for i in landmark]
                tmp_lmk = self.resize_landmark(landmark, [w, h, 8])
                landmark_list.append(tmp_lmk)

        pts = torch.from_numpy(np.array(landmark_list, dtype=np.float32))

        return imgs, pts
