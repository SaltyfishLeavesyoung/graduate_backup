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

    idx = 0
    rfr[0] = joblib.load("./random_forest_0.joblib")

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

        haarfeat3d = HaarFeatures3d(kernel_size=(8, 7, 7), stride=1)
        voting = torch.zeros([6, 8, 256, 256])

        for i in tqdm(range(1)):
            for m in range(8):
                for p in range(8):
                    sel_img = imgs[:,:,:,m * 32:(m+1) * 32,p * 32:(p+1) * 32]
                    output_haar3d = haarfeat3d(sel_img)
                    output_haar3d = output_haar3d.reshape((1, 16 * 32 * 32))
                    pred = rfr[i].predict(output_haar3d.detach().numpy())

                    pred_disp = pred.reshape((3, 8, 32, 32))

                    for j in range(8):
                        for k in range(32):
                            for l in range(32):
                                pred_z = int(min(max(j + pred_disp[0][j][k][l], 0), 7))
                                pred_y = int(min(max(m * 32 + k + pred_disp[1][j][k][l], 0), 255))
                                pred_x = int(min(max(p * 32 + l + pred_disp[2][j][k][l], 0), 255))
                                voting[i][pred_z][pred_y][pred_x] += 1
        
        with open(os.path.join('/home/yeziyang/Desktop/JointLandmark/data', item['ID'], item['part']+'.txt')) as f:
            for i in range(1):
                landmark = f.readline().split()
                landmark = [int(i) for i in landmark]
                tmp_lmk = resize_landmark(landmark, [w, h, 8])

                max_pred = voting[i].max()
                index = np.unravel_index(voting[i].argmax(), voting[i].shape)
                MRE += np.sqrt((tmp_lmk[2] - index[0]) ** 2 + (tmp_lmk[1] - index[1]) ** 2 + (tmp_lmk[0] - index[2]) ** 2)
                sums = sums + 1

    print(MRE / sums)
