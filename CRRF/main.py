import torch
import os
from tqdm import tqdm
import numpy as np
from haarfeatures import HaarFeatures3d
from scipy import ndimage
import torchvision.transforms as transforms
from sklearn.ensemble import RandomForestRegressor
import joblib
import random

def resize_landmark(landmark, original_size):
    size = [256, 256, 8]
    for i in range(len(landmark)):
        landmark[i] = int(landmark[i] * size[i] / original_size[i])
    return landmark

if __name__ == '__main__':

    # train
    train_img_list = list()
    with open('/home/yeziyang/Desktop/JointLandmark/train.txt') as f:
        line = f.readline().split()[0]
        while line:
            idx = str(int(line))
            train_img_list.append({'ID' : idx, 'part' : 'L'})
            train_img_list.append({'ID' : idx, 'part' : 'R'})
            line = f.readline()

    resize = transforms.Resize([256, 256])
    as_tensor = transforms.Compose([
            resize,
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
    ])
    transform_resize = transforms.Compose([
            resize,
    ])

    train_haars = [list(), list(), list(), list(), list(), list()]
    val_haars = list()
    train_displace = [list(), list(), list(), list(), list(), list()]
    val_displace = [list(), list(), list(), list(), list(), list()]
    val_landmark = [list(), list(), list(), list(), list(), list()]
    rfr = [RandomForestRegressor(n_estimators = 10, random_state = 0),
           RandomForestRegressor(n_estimators = 10, random_state = 0),
           RandomForestRegressor(n_estimators = 10, random_state = 0),
           RandomForestRegressor(n_estimators = 10, random_state = 0), 
           RandomForestRegressor(n_estimators = 10, random_state = 0), 
           RandomForestRegressor(n_estimators = 10, random_state = 0)]

    count = 0

    for item in tqdm(train_img_list):
        count = count + 1
        if count > 20:
            break
        pth_img = np.load(os.path.join('/home/yeziyang/Desktop/JointLandmark/data', item['ID'], item['part']+'.npy')).astype(np.uint8)
        n, h, w = pth_img.shape

        img = []
        for i in range(n):
            img.append(as_tensor(transforms.ToPILImage()(pth_img[i])))
        imgs = torch.stack(img, dim = 1)
        imgs = torch.unsqueeze(imgs, dim = 0)

        haarfeat3d = HaarFeatures3d(kernel_size=(8, 7, 7), stride=1)
        #output_haar3d = haarfeat3d(imgs)
        #print(output_haar3d.shape)
        #print(output_haar3d.shape) [1, 16, 1, 256, 256]
        #output_haar3d = output_haar3d.reshape((16 * 256 * 256,))
        #output_haar3d = output_haar3d.reshape((16 * 128 * 128,))
        #train_haars.append(output_haar3d)
        
        with open(os.path.join('/home/yeziyang/Desktop/JointLandmark/data', item['ID'], item['part']+'.txt')) as f:
            for i in range(6):
                landmark = f.readline().split()
                landmark = [int(i) for i in landmark]
                tmp_lmk = resize_landmark(landmark, [w, h, 8])
                disp = torch.zeros((3, 8, 256, 256))
                for j in range(8):
                    disp[0,j,:,:] = tmp_lmk[2] - j
                for k in range(256):
                    disp[1,:,k,:] = tmp_lmk[1] - k
                for l in range(256):
                    disp[2,:,:,l] = tmp_lmk[0] - l
                #disp = disp.reshape([3*8*256*256])
                #train_displace[i].append(disp)
                #torch.save(disp, os.path.join('/home/yeziyang/Desktop/JointLandmark/data', item['ID'], item['part'] +'_{}.pt'.format(i)))
                for j in range(20):
                    sel_x = random.randint(0, 256 - 32)
                    sel_y = random.randint(0, 256 - 32)
                    sel_img = imgs[:,:,:,sel_y:sel_y + 32, sel_x:sel_x + 32]
                    sel_haar = haarfeat3d(sel_img).reshape((16 * 32 * 32,))
                    sel_disp = disp[:,:,sel_y:sel_y + 32, sel_x:sel_x + 32]
                    sel_disp = sel_disp.reshape((3*8*32*32,))
                    train_haars[i].append(sel_haar)
                    train_displace[i].append(sel_disp)

                
        """for i in range(6):
            disp = torch.load(os.path.join('/home/yeziyang/Desktop/JointLandmark/data', item['ID'], item['part'] +'_{}.pt'.format(i)))
            train_displace[i].append(disp)"""

    for i in range(6):
        train_haars[i] = torch.stack(train_haars[i])
        torch.save(train_haars[i], "train_haars_{}_32.pt".format(i))
        train_displace[i] = torch.stack(train_displace[i])
        torch.save(train_displace[i], "train_{}_32.pt".format(i))
    print("train data done")

    idx = 0
    for regor in tqdm(rfr):
        regor.fit(train_haars[idx].detach().numpy(), train_displace[idx].detach().numpy())
        idx = idx + 1
        joblib.dump(regor, "./random_forest_{}.joblib".format(idx))

    print("train done")

    MRE = 0.0
    sums = 0

    # val
    val_img_list = list()
    with open('/home/yeziyang/Desktop/JointLandmark/train_val.txt') as f:
        line = f.readline().split()[0]
        while line:
            idx = str(int(line))
            val_img_list.append({'ID' : idx, 'part' : 'L'})
            val_img_list.append({'ID' : idx, 'part' : 'R'})
            line = f.readline()

    count = 0

    for item in tqdm(val_img_list):
        count = count + 1
        if count > 10:
            break
        pth_img = np.load(os.path.join('/home/yeziyang/Desktop/JointLandmark/data', item['ID'], item['part']+'.npy')).astype(np.uint8)
        n, h, w = pth_img.shape

        img = []
        for i in range(n):
            img.append(as_tensor(transforms.ToPILImage()(pth_img[i])))
        imgs = torch.stack(img, dim = 1)
        imgs = torch.unsqueeze(imgs, dim = 0)

        haarfeat3d = HaarFeatures3d(kernel_size=(8, 7, 7), stride=2)
        voting = torch.zeros([6, 8, 256, 256])

        for i in tqdm(range(6)):
            for m in range(8):
                for p in range(8):
                    sel_img = imgs[:,:,m * 32:(m+1) * 32,p * 32:(p+1) * 32]
                    output_haar3d = haarfeat3d(sel_img)
                    output_haar3d = output_haar3d.reshape((8 * 32 * 32,))
                    pred = rfr[i].predict(output_haar3d.detach().numpy())

                    for j in range(8):
                        for k in range(32):
                            for l in range(32):
                                pred_z = min(max(j + pred_disp[0][j][m * 32 + k][p * 32 + l], 0), 7)
                                pred_y = min(max(k + pred_disp[1][j][m * 32 + k][p * 32 + l], 0), 255)
                                pred_x = min(max(l + pred_disp[2][j][m * 32 + k][p * 32 + l], 0), 255)
                                voting[i][pred_z][pred_y][pred_x] += 1
        
        with open(os.path.join('/home/yeziyang/Desktop/JointLandmark/data', item['ID'], item['part']+'.txt')) as f:
            for i in range(6):
                landmark = f.readline().split()
                landmark = [int(i) for i in landmark]
                tmp_lmk = resize_landmark(landmark, [w, h, 8])

                max_pred = voting[i].max()
                index = np.unravel_index(voting[i].argmax(), voting[i].shape)
                MRE += np.sqrt((tmp_lmk[2] - index[0]) ** 2 + (tmp_lmk[1] - index[1]) ** 2 + (tmp_lmk[0] - index[2]) ** 2)
                sums = sums + 1

    print(MRE / sums)
