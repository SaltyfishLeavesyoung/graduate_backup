import torch
from torch import nn, optim
import os
from dataset import MyDataset
from torch.utils.data import DataLoader
from network import UnetClassic3D
from tqdm import tqdm
import numpy as np
from scipy import ndimage
from torch.nn import MSELoss

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

batch_size = 2
epochs = 500
weight_decay = 1e-4
learning_rate = 3e-4
moment = 0.9

if __name__ == '__main__':

    dataset_train = MyDataset('/home/yeziyang/Desktop/JointLandmark/train.txt')
    dataset_val = MyDataset('/home/yeziyang/Desktop/JointLandmark/train_val.txt')
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=4)

    net = UnetClassic3D(n_classes = 6).cuda()

    loss_net = nn.MSELoss()

    optimizer = optim.Adam(params=net.parameters(), \
        lr=learning_rate, betas=(0.9,0.999), eps=1e-08, weight_decay=weight_decay)

    """optimizer = optim.SGD(params=net.parameters(), \
        lr=learning_rate, momentum = moment)"""

    min_loss = 1145141919

    for epoch in range(epochs):
        net.train()
        losses = 0
        for imgs, labels in tqdm(dataloader_train):
            imgs, labels = imgs.cuda(), labels.cuda()
            output = net(imgs)
            loss = (MSELoss()(output[:,:,0], labels[:,:,0] / 255.0) + MSELoss()(output[:,:,1], labels[:,:,1] / 255.0)) * 255.0 + MSELoss()(output[:,:,2], labels[:,:,2] / 7.0) * 7.0 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses += loss.detach().item()

        losses = losses / len(dataloader_train)
        print('train loss: {}'.format(losses))

        net.eval()
        losses = 0
        for imgs, labels in tqdm(dataloader_val):
            imgs, labels = imgs.cuda(), labels.cuda()
            output = net(imgs)
            loss = (MSELoss()(output[:,:,0], labels[:,:,0] / 255.0) + MSELoss()(output[:,:,1], labels[:,:,1] / 255.0)) * 255.0 + MSELoss()(output[:,:,2], labels[:,:,2] / 7.0) * 7.0 
            losses += loss.detach().item()

        losses = losses / len(dataloader_val)
        print('val loss: {}'.format(losses))
        if losses < min_loss:
            min_loss = losses
            torch.save(net.state_dict(), '/home/yeziyang/Desktop/JSD/models/{}_{}.pth'.format(epoch, losses))