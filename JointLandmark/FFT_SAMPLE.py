import os
import torch
import numpy as np
import torchvision.transforms as transforms
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
import cv2

if __name__ == "__main__":
    as_tensor = transforms.Compose([
            transforms.Resize((256, 256)),
        ])

    root1 = os.path.join('/home/yzy/Desktop/JointLandmark/data/1')
    root4 = os.path.join('/home/yzy/Desktop/JointLandmark/data/4')
    l_img1 = np.load(os.path.join(root1, 'L.npy')).astype(np.uint8)
    r_img1 = np.load(os.path.join(root1, 'R.npy')).astype(np.uint8)
    l_img4 = np.load(os.path.join(root4, 'L.npy')).astype(np.uint8)
    r_img4 = np.load(os.path.join(root4, 'R.npy')).astype(np.uint8)
    for i in range(8):
        l_imgx1 = as_tensor(transforms.ToPILImage()(l_img1[i]))
        l_imgx4 = as_tensor(transforms.ToPILImage()(l_img4[i]))
        r_imgx1 = as_tensor(transforms.ToPILImage()(r_img1[i]))
        r_imgx4 = as_tensor(transforms.ToPILImage()(r_img4[i]))
        fft_l_imgx1 = np.fft.fft2(l_imgx1)
        fft_l_imgx4 = np.fft.fft2(l_imgx4)
        fft_r_imgx1 = np.fft.fft2(r_imgx1)
        fft_r_imgx4 = np.fft.fft2(r_imgx4)
        amp_imglx1, pha_imglx1 = np.abs(fft_l_imgx1), np.angle(fft_l_imgx1)
        amp_imgrx1, pha_imgrx1 = np.abs(fft_r_imgx1), np.angle(fft_r_imgx1)
        amp_imglx4, pha_imglx4 = np.abs(fft_l_imgx4), np.angle(fft_l_imgx4)
        amp_imgrx4, pha_imgrx4 = np.abs(fft_r_imgx4), np.angle(fft_r_imgx4)
        fft_transferl1 = amp_imglx4 * np.exp(1j * pha_imglx1)
        fft_transferl4 = amp_imglx1 * np.exp(1j * pha_imglx4)
        fft_transferr1 = amp_imgrx4 * np.exp(1j * pha_imgrx1)
        fft_transferr4 = amp_imgrx1 * np.exp(1j * pha_imgrx4)
        ifft_imgl1 = np.real(np.fft.ifft2(fft_transferl1, axes=(-2,-1))).astype(np.uint8)
        ifft_imgl4 = np.real(np.fft.ifft2(fft_transferl4, axes=(-2,-1))).astype(np.uint8)
        ifft_imgr1 = np.real(np.fft.ifft2(fft_transferr1, axes=(-2,-1))).astype(np.uint8)
        ifft_imgr4 = np.real(np.fft.ifft2(fft_transferr4, axes=(-2,-1))).astype(np.uint8)
        cv2.imwrite('/home/yzy/Desktop/JointLandmark/sample_fft/l1_{}.png'.format(i), ifft_imgl1)
        cv2.imwrite('/home/yzy/Desktop/JointLandmark/sample_fft/l4_{}.png'.format(i), ifft_imgl4)
        cv2.imwrite('/home/yzy/Desktop/JointLandmark/sample_fft/r1_{}.png'.format(i), ifft_imgr1)
        cv2.imwrite('/home/yzy/Desktop/JointLandmark/sample_fft/r4_{}.png'.format(i), ifft_imgr4)

        '''amp_imglx1 = as_tensor2(transforms.ToPILImage()(amp_imglx1.astype(np.uint8))).numpy().astype(np.float32).squeeze()
        amp_imglx4 = as_tensor2(transforms.ToPILImage()(amp_imglx4.astype(np.uint8))).numpy().astype(np.float32).squeeze()
        amp_imgrx1 = as_tensor2(transforms.ToPILImage()(amp_imgrx1.astype(np.uint8))).numpy().astype(np.float32).squeeze()
        amp_imgrx4 = as_tensor2(transforms.ToPILImage()(amp_imgrx4.astype(np.uint8))).numpy().astype(np.float32).squeeze()
        pha_imglx1 = as_tensor2(transforms.ToPILImage()(pha_imglx1.astype(np.uint8))).numpy().astype(np.float32).squeeze()
        pha_imglx4 = as_tensor2(transforms.ToPILImage()(pha_imglx4.astype(np.uint8))).numpy().astype(np.float32).squeeze()
        pha_imgrx1 = as_tensor2(transforms.ToPILImage()(pha_imgrx1.astype(np.uint8))).numpy().astype(np.float32).squeeze()
        pha_imgrx4 = as_tensor2(transforms.ToPILImage()(pha_imgrx4.astype(np.uint8))).numpy().astype(np.float32).squeeze()'''

        print(pha_imgrx4)

        cv2.imwrite('/home/yzy/Desktop/JointLandmark/sample_fft/ampl1_{}.png'.format(i), amp_imglx1)
        cv2.imwrite('/home/yzy/Desktop/JointLandmark/sample_fft/ampl4_{}.png'.format(i), amp_imglx4)
        cv2.imwrite('/home/yzy/Desktop/JointLandmark/sample_fft/ampr1_{}.png'.format(i), amp_imgrx1)
        cv2.imwrite('/home/yzy/Desktop/JointLandmark/sample_fft/ampr4_{}.png'.format(i), amp_imgrx4)
        cv2.imwrite('/home/yzy/Desktop/JointLandmark/sample_fft/phal1_{}.png'.format(i), pha_imglx1)
        cv2.imwrite('/home/yzy/Desktop/JointLandmark/sample_fft/phal4_{}.png'.format(i), pha_imglx4)
        cv2.imwrite('/home/yzy/Desktop/JointLandmark/sample_fft/phar1_{}.png'.format(i), pha_imgrx1)
        cv2.imwrite('/home/yzy/Desktop/JointLandmark/sample_fft/phar4_{}.png'.format(i), pha_imgrx4)
        








