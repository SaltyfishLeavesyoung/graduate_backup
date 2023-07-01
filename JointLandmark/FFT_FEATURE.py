import os
import torch
import numpy as np
import torchvision.transforms as transforms
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd

if __name__ == "__main__":
    as_tensor = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
    tsne = TSNE(n_components=2, init='pca')
    all_amp = []
    all_dir = []
    for i in range(828):
        root = os.path.join('/home/yzy/Desktop/JointLandmark/data', str(i+1))
        if not os.path.exists(root):
            continue
        all_dir.append(i+1)
        l_img = np.load(os.path.join(root, 'L.npy')).astype(np.uint8)
        r_img = np.load(os.path.join(root, 'R.npy')).astype(np.uint8)
        imgs = np.concatenate((l_img, r_img), axis = 0)
        img = []
        for i in range(16):
            img.append(as_tensor(transforms.ToPILImage()(imgs[i])))
        img = np.concatenate(img, axis = 0)
        fft_img = np.fft.fftn(img, axes=(-2,-1))
        amp_img, pha_img = np.abs(fft_img), np.angle(fft_img)
        amp_mean = np.mean(amp_img, axis=0)
        pha_mean = np.mean(pha_img, axis=0)
        np.save(os.path.join(root, 'amp.npy'), amp_mean)
        np.save(os.path.join(root, 'pha.npy'), pha_mean)
        all_amp.append(amp_mean)
    all_amp = np.stack(all_amp, axis = 0)
    all_amp = all_amp.reshape(all_amp.shape[0], -1)
    '''label = np.zeros((all_amp.shape[0], 1))
    label[0] = 1
    label[1] = 1
    label[2] = 1
    label[3] = 2
    label[4] = 1
    label[5] = 2
    label[6] = 2
    label[7] = 2
    label[8] = 2
    label[9] = 2
    label[70] = 1
    label[71] = 1
    label[72] = 1
    label[73] = 2
    label[74] = 2
    label[80] = 1
    label[81] = 1
    label[82] = 1
    label[90] = 2
    label[91] = 2
    print(all_amp.shape)'''
    tsne.fit_transform(all_amp)
    data = np.array(tsne.embedding_)
    '''test = [0,1,2,3,4,5,6,7,8,9,70,71,72,73,74,80,81,82,90,91]
    test_X = [data[i][0] for i in test]
    test_Y = [data[i][1] for i in test]
    test_label = [label[i] for i in test]
    plt.scatter(data[:,0], data[:,1])
    plt.scatter(test_X, test_Y, c=test_label)
    plt.show()'''
    '''labels = []
    for i in range(data.shape[0]):
        if 0.5 * data[i][0] - data[i][1] - 2500 < 0:
            labels.append(0)
        else:
            labels.append(1)
    df = pd.DataFrame({'dir': all_dir,'label': labels})
    df.to_csv('label_amp.csv', index=False)'''
        








