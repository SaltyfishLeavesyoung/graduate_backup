import torch
from torch import nn, optim
import os
from dataset import MyDataset
from torch.utils.data import DataLoader
from network import UnetClassic3D
from tqdm import tqdm
import numpy as np
from scipy import ndimage

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

batch_size = 1

if __name__ == '__main__':

    dataset_val = MyDataset('/home/yeziyang/Desktop/JointLandmark/test.txt')
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=4)

    net = UnetClassic3D(n_classes = 6).cuda()
    net.load_state_dict(torch.load("/home/yeziyang/Desktop/JSD/models/7_0.7626711853019154.pth"))

    loss_net = nn.MSELoss()

    ratio = torch.tensor([255.0, 255.0, 7.0]).cuda()

    net.eval()
    losses = 0
    for imgs, label in tqdm(dataloader_val):
        imgs, label = imgs.cuda(), label.cuda()
        pred = net(imgs)

        pred = torch.mul(pred, ratio)

        loss_landmark = 0
        for i in range(label.shape[0]):
            for j in range(label.shape[1]):
                print(label[i][j])
                print(pred[i][j])
                print("-----------")
                dist = (label[i][j][0] - pred[i][j][0]) ** 2 + (label[i][j][1] - pred[i][j][1]) ** 2 + (label[i][j][2] - pred[i][j][2]) ** 2
                loss_landmark += np.sqrt(dist.detach().cpu())

        losses += loss_landmark

    losses = losses / (len(dataloader_val) * 6)
    print('MRE: {}'.format(losses))