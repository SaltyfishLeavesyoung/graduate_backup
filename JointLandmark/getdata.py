import pydicom
import os
import pandas as pd
from tqdm import tqdm

import numpy as np
import torch
import random
from xml.dom.minidom import parse
from PIL import Image
import math

import cv2
import random

LABEL = '/home/yeziyang/Desktop/label.csv'
FRAME = '/home/yeziyang/Desktop/framecount.csv'

def extractPoint(path):
    xml = parse(path)
    points = xml.getElementsByTagName("Point")
    cordi = [list(map(float, i.getElementsByTagName("Coordinate")[0].childNodes[0].data.split(' '))) for i in points]
    return cordi

def getDCM(idx, series=None, part=None):
    imgs = []
    time = []
    sorted_imgs = []
    root = os.path.join('/home/yeziyang/Desktop/alldcm', str(idx))
    size = None
    for dcm in os.listdir(root):
        ds = pydicom.read_file(os.path.join(root, dcm), force=True)
        if "SeriesDescription" in ds:
            if ds.SeriesDescription == series or series is None:
                imgs.append(ds)
                time.append(ds.ImagePositionPatient[0])
                size = ds.pixel_array.shape
                
    sorted_time = sorted(time)
    for t in sorted_time:
        i = time.index(t)
        sorted_imgs.append(imgs[i])
    
    if part == 'left':
        res = sorted_imgs[8:]
        res.reverse()
    elif part == 'right':
        res = sorted_imgs[:8]
    else:
        res = sorted_imgs
    
    return res, size

def transformPoint(cordi, ds_list):
    ret_cord = []

    temp_pix = []
    temp_i = []
    temp_desc = []

    for c in cordi:
        min_distance = 1145141919810
        min_i = -1
    
        for i in range(len(ds_list) - 1):
            dist = (c[0] - ds_list[i].ImagePositionPatient[0]) ** 2 + \
               (c[1] - ds_list[i].ImagePositionPatient[1]) ** 2 + \
               (c[2] - ds_list[i].ImagePositionPatient[2]) ** 2
            if dist < min_distance:
                min_distance = dist
                min_i = i
            else:
                break
    
        dist_x = ((ds_list[min_i].ImageOrientationPatient[0] * c[0] + 
               ds_list[min_i].ImageOrientationPatient[1] * c[1] + 
               ds_list[min_i].ImageOrientationPatient[2] * c[2]) - 
              (ds_list[min_i].ImageOrientationPatient[0] * ds_list[min_i].ImagePositionPatient[0] +
               ds_list[min_i].ImageOrientationPatient[1] * ds_list[min_i].ImagePositionPatient[1] + 
               ds_list[min_i].ImageOrientationPatient[2] * ds_list[min_i].ImagePositionPatient[2])) / \
             math.sqrt(ds_list[min_i].ImageOrientationPatient[0] ** 2 + 
                       ds_list[min_i].ImageOrientationPatient[1] ** 2 + 
                       ds_list[min_i].ImageOrientationPatient[2] ** 2)
    
        dist_y = ((ds_list[min_i].ImageOrientationPatient[3] * c[0] + 
               ds_list[min_i].ImageOrientationPatient[4] * c[1] + 
               ds_list[min_i].ImageOrientationPatient[5] * c[2]) - 
              (ds_list[min_i].ImageOrientationPatient[3] * ds_list[min_i].ImagePositionPatient[0] +
               ds_list[min_i].ImageOrientationPatient[4] * ds_list[min_i].ImagePositionPatient[1] + 
               ds_list[min_i].ImageOrientationPatient[5] * ds_list[min_i].ImagePositionPatient[2])) / \
             math.sqrt(ds_list[min_i].ImageOrientationPatient[3] ** 2 + 
                       ds_list[min_i].ImageOrientationPatient[4] ** 2 + 
                       ds_list[min_i].ImageOrientationPatient[5] ** 2)
    
        x = int(dist_x / ds_list[min_i].PixelSpacing[0])
        y = int(dist_y / ds_list[min_i].PixelSpacing[1])
        ret_cord.append([x, y, min_i])

        count = -114514
        
        for j in range(len(temp_i)):
            if temp_i[j] == min_i:
                count = j
                break
                    
        if count != -114514:
            cv2.circle(temp_pix[count], (x, y), 3, (255, 255, 0), 1)
        else:
            temp_i.append(min_i)
            temp_pix.append(cv2.circle(cv2.cvtColor(ds_list[min_i].pixel_array, cv2.COLOR_GRAY2BGR), (x, y), 3, (255, 255, 0), 1))
            temp_desc.append(ds_list[min_i].SeriesDescription)
        
    return ret_cord, temp_pix

def getLabeledData(datalist):
    label = []
    for idx in datalist:
        if os.path.exists(os.path.join('/home/yeziyang/Desktop/Joint', str(idx), str(idx) + '-R.xml')) and \
           os.path.exists(os.path.join('/home/yeziyang/Desktop/Joint', str(idx), str(idx) + '-L.xml')):
            label.append(idx)
    return label

if __name__ == "__main__":
    labelcsv = pd.read_csv(LABEL)
    framecsv = pd.read_csv(FRAME)
    left_label = list(labelcsv['left'])
    right_label = list(labelcsv['right'])
    pdfse_count = list(framecsv['pd_fse_sag'])
    PDWTSE_count = list(framecsv['PDW_TSE'])

    idxlist = getLabeledData([i+1 for i in range(1537)])
    new_idxlist = []

    train_list = []
    val_list = []

    for idx in idxlist:
        if pdfse_count[idx - 1] == 16:
            new_idxlist.append(idx)
            series = 'pd_fse_sag'
        elif PDWTSE_count[idx - 1] == 16:
            new_idxlist.append(idx)
            series = 'PDW_TSE'
        else:
            continue
        imgs_l, original_size_l = getDCM(idx, series, 'left')
        cordi_l = extractPoint(os.path.join('/home/yeziyang/Desktop/Joint', str(idx), str(idx) + '-L.xml'))
        keypoint_list_l, sample_l = transformPoint(cordi_l, imgs_l)

        imgs_r, original_size_r = getDCM(idx, series, 'right')
        cordi_r = extractPoint(os.path.join('/home/yeziyang/Desktop/Joint', str(idx), str(idx) + '-R.xml'))
        keypoint_list_r, sample_r = transformPoint(cordi_r, imgs_r)

        path = os.path.join("/home/yeziyang/Desktop/JointLandmark/data", str(idx))
        if not os.path.exists(path):
            os.mkdir(path)

        imgl = np.array([imgs_l[i].pixel_array for i in range(8)])
        imgr = np.array([imgs_r[i].pixel_array for i in range(8)])
        np.save(os.path.join(path, "L"), imgl)
        np.save(os.path.join(path, "R"), imgr)

        for i in range(8):
            cv2.imwrite(os.path.join(path, "L" + str(i) + ".jpg"), cv2.cvtColor(imgs_l[i].pixel_array, cv2.COLOR_GRAY2BGR))
            cv2.imwrite(os.path.join(path, "R" + str(i) + ".jpg"), cv2.cvtColor(imgs_r[i].pixel_array, cv2.COLOR_GRAY2BGR))
        
        with open(os.path.join(path, 'L.txt'), 'w') as f_l:
            for i in range(6):
                f_l.write(str(keypoint_list_l[i][0]) + " " + str(keypoint_list_l[i][1]) + " " + str(keypoint_list_l[i][2]) + "\n")

        with open(os.path.join(path, 'R.txt'), 'w') as f_r:
            for i in range(6):
                f_r.write(str(keypoint_list_r[i][0]) + " " + str(keypoint_list_r[i][1]) + " " + str(keypoint_list_r[i][2]) + "\n")

        for i in range(len(sample_l)):
            cv2.imwrite(os.path.join(path, "sampleL" + str(i) + ".jpg"), sample_l[i])

        for i in range(len(sample_r)):
            cv2.imwrite(os.path.join(path, "sampleR" + str(i) + ".jpg"), sample_r[i])

        print(str(idx) + " shape: " + str(original_size_l) + ", " + str(original_size_r))

        if random.random() < 0.7:
            train_list.append(idx)
        else:
            val_list.append(idx)

    """with open('/home/yeziyang/Desktop/JointLandmark/train.txt', 'w') as f:
        for i in range(len(train_list)):
            f.write(str(train_list[i]) + "\n")

    with open('/home/yeziyang/Desktop/JointLandmark/val.txt', 'w') as f:
        for i in range(len(val_list)):
            f.write(str(val_list[i]) + "\n")"""
