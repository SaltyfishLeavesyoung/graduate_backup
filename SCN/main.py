import torch
from torch import nn, optim
import os
from dataset import MyDataset
from torch.utils.data import DataLoader
from network import network_scn, network_unet
from tqdm import tqdm
import numpy as np
from scipy import ndimage

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

batch_size = 4
epochs = 500
weight_decay = 1e-5
sigma_regularization = 1 / (256 ** 2)
learning_rate = 3e-6

def generate_heatmap_target(heatmap_size, landmarks):
    landmarks_shape = landmarks.shape
    batch_size = landmarks_shape[0]
    num_landmarks = landmarks_shape[1]
    dim = landmarks_shape[2] - 1
    assert len(heatmap_size) - 2 == dim + 1, 'Dimensions do not match.'

    sigmas = torch.ones(*heatmap_size) * 4

    squared_distances = torch.zeros(*heatmap_size)
    temp_matrix = torch.ones(*heatmap_size)

    for i in range(batch_size):
        for j in range(num_landmarks):
            temp_matrix[i][j][int(landmarks[i][j][2])][int(landmarks[i][j][0])][int(landmarks[i][j][1])] = 0
            squared_distances[i][j] = torch.from_numpy(ndimage.distance_transform_edt(temp_matrix[i][j]))
   
    heatmap = torch.exp(-squared_distances / (2 * torch.pow(sigmas, 2)))
    return heatmap

def get_landmark(image):
    img = image.detach().cpu()
    max_index = np.argmax(img)
    coord = np.array(np.unravel_index(max_index, img.shape), np.int32)
    coord = np.flip(coord, axis=0)
    return np.ascontiguousarray(coord)

if __name__ == '__main__':

    dataset_train = MyDataset('/home/yeziyang/Desktop/JointLandmark/train_65.txt')
    dataset_val = MyDataset('/home/yeziyang/Desktop/JointLandmark/train_val_196.txt')
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=4)

    net = network_scn(num_heatmaps = 6).cuda()

    loss_net = nn.MSELoss()
    loss_sigmas = nn.MSELoss()

    optimizer = optim.Adam(params=net.parameters(), \
        lr=learning_rate, betas=(0.9,0.999), eps=1e-08, weight_decay=weight_decay)

    min_loss = 1145141919

    ratio = torch.tensor([255.0, 255.0, 7.0]).cuda()

    for epoch in range(epochs):
        net.train()
        losses = 0
        for imgs, label, label_heatmap in tqdm(dataloader_train):
            imgs, label, label_heatmap = imgs.cuda(), label.cuda(), label_heatmap.cuda()
            """_, _, heatmap = net(imgs)
            # print(heatmap.shape) torch.Size([1, 6, 8, 256, 256])
            loss_heatmap = loss_net(heatmap, label_heatmap)

            loss_landmark = 0
            for i in range(label.shape[0]):
                for j in range(label.shape[1]):
                    loss_landmark += loss_sigmas(label[i][j], torch.from_numpy(get_landmark(heatmap[i][j])).cuda())
            
            loss = loss_heatmap * 10 # + sigma_regularization * loss_landmark"""
            pred, _, _ = net(imgs)
            loss = loss_sigmas(pred[:,:,0] * 255.0, label[:,:,0]) / 7.0 + loss_sigmas(pred[:,:,1] * 255.0, label[:,:,1]) / 7.0 + loss_sigmas(pred[:,:,2] * 7.0, label[:,:,2])
            loss /= 10

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses += loss.detach().item()

        losses = losses / len(dataloader_train)
        print('{} train loss: {}'.format(epoch, losses))

        net.eval()
        losses = 0
        for imgs, label, label_heatmap in tqdm(dataloader_val):
            imgs, label, label_heatmap = imgs.cuda(), label.cuda(), label_heatmap.cuda()
            """_, _, heatmap = net(imgs)
            loss_heatmap = loss_net(heatmap, label_heatmap)

            loss_landmark = 0
            for i in range(label.shape[0]):
                for j in range(label.shape[1]):
                    loss_landmark += loss_sigmas(label[i][j], torch.from_numpy(get_landmark(heatmap[i][j])).cuda())
            
            loss = loss_heatmap * 10 # + sigma_regularization * loss_landmark"""
            pred, _, _ = net(imgs)
            loss = loss_sigmas(pred[:,:,0] * 255.0, label[:,:,0]) / 7.0 + loss_sigmas(pred[:,:,1] * 255.0, label[:,:,1]) / 7.0 + loss_sigmas(pred[:,:,2] * 7.0, label[:,:,2])
            loss /= 10
    
            losses += loss.detach().item()

        losses = losses / len(dataloader_val)
        print('{} val loss: {}'.format(epoch, losses))
        if losses < min_loss:
            min_loss = losses
            torch.save(net.state_dict(), '/home/yeziyang/Desktop/SCN/models/{}_{}.pth'.format(epoch, losses))