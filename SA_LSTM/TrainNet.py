from __future__ import print_function, division
import torch
import numpy as np
import time
import MyUtils
from tqdm import tqdm

from torch.nn import MSELoss, L1Loss, SmoothL1Loss

def train_model(corseNet, fine_LSTM, dataloaders, criterion_coarse, optimizer,
                num_epochs, use_gpu, saveName, landmarkNum, image_scale):

    since = time.time()

    best_acc = [0, 0, 0, 0, 0, 0]

    test_epoch = 1

    gl, gh, gw = image_scale
    global_coordinate = torch.ones(gl, gh, gw, 3).float()
    for i in range(gl):
        global_coordinate[i, :, :, 0] = global_coordinate[i, :, :, 0] * i
    for i in range(gh):
        global_coordinate[:, i, :, 1] = global_coordinate[:, i, :, 1] * i
    for i in range(gw):
        global_coordinate[:, :, i, 2] = global_coordinate[:, :, i, 2] * i
    global_coordinate = global_coordinate * torch.tensor([1 / (gl - 1), 1 / (gh - 1), 1 / (gw - 1)])
    global_coordinate = global_coordinate.cuda()
    size_tensor = torch.tensor([1 / 255, 1 / 255, 1 / 7])

    min_avg_loss = 1145141919

    # ratio = torch.tensor([255.0, 255.0, 7.0]).cuda()

    torch.autograd.set_detect_anomaly(True)

    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            # print ("1")
            if phase == 'train':
                corseNet.train(True)  # Set model to training mode
                fine_LSTM.train(True)
            else:
                # ~ continue
                if epoch % test_epoch != 0:
                    continue
                corseNet.train(False)  # Set model to evaluate mode
                fine_LSTM.train(False)

            running_loss = 0.0

            # Iterate over data.
            lent = len(dataloaders[phase])

            losses = 0
            for ide in tqdm(range(lent)):
                data = dataloaders[phase][ide]

                inputs, inputs_origin, labels, image_name = data['DICOM'], data['DICOM_origin'], data[
                    'landmarks'], data['imageName']
                inputs, inputs_origin, labels = inputs.cuda(), inputs_origin.cuda(), labels.cuda()

                optimizer.zero_grad()

                # forward
                # coarse landmark detection
                coarse_heatmap, coarse_features = corseNet(inputs)
                coarse_landmarks = MyUtils.get_coordinates_from_coarse_heatmaps(coarse_heatmap, global_coordinate).unsqueeze(0)
                ROIs = coarse_landmarks.cpu().detach().numpy() + np.random.uniform(0, 0.02, labels.size())

                # fine-scale landmark detection
                # 572R error!
                fine_landmarks = fine_LSTM(ROIs, labels, inputs_origin, coarse_features, phase, size_tensor, ide)

                # calculate loss
                # loss = 0
                # loss += torch.abs(fine_landmarks - labels).sum()
                # loss += criterion_coarse(coarse_heatmap, global_coordinate, labels, size_tensor, phase) * 3
                loss = (MSELoss()(fine_landmarks[-1,:,0], labels[0,:,0]) + MSELoss()(fine_landmarks[-1,:,1], labels[0,:,1])) * 255.0 + MSELoss()(fine_landmarks[-1,:,2], labels[0,:,2]) * 7.0        
                # backward + optimize only if in training phase
                if phase == 'train':
                    with torch.autograd.detect_anomaly():
                        loss.backward()
                        optimizer.step()
                    losses += loss.detach().item()
                else:
                    losses += loss.detach().item()

            avg_loss = losses / lent
            print("{} {} loss: {}".format(epoch, phase, avg_loss))
            if phase == 'val':
                if avg_loss < min_avg_loss:
                    min_avg_loss = min(min_avg_loss, avg_loss)
                    torch.save(corseNet.state_dict(), '/home/yeziyang/Desktop/SA_LSTM/models/A_{}_{}_corse.pth'.format(epoch, avg_loss))
                    torch.save(fine_LSTM.state_dict(), '/home/yeziyang/Desktop/SA_LSTM/models/A_{}_{}_lstm.pth'.format(epoch, avg_loss))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
