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

batch_size = 1
sigma_regularization = 1 / (256 ** 2)

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

    dataset_val = MyDataset('/home/yeziyang/Desktop/JointLandmark/test.txt')
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=4)

    net = network_scn(num_heatmaps = 6).cuda()

    net.load_state_dict(torch.load("/home/yeziyang/Desktop/SCN/models/107_3.6798566518997657.pth"))

    loss_net = nn.MSELoss()
    loss_sigmas = nn.MSELoss()

    ratio = torch.tensor([255.0, 255.0, 7.0]).cuda()

    net.eval()
    losses = 0
    for imgs, label, label_heatmap in tqdm(dataloader_val):
        imgs, label, label_heatmap = imgs.cuda(), label.cuda(), label_heatmap.cuda()
        pred, _, _ = net(imgs)
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