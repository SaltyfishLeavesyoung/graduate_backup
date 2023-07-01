import torch
import numpy as np
import matplotlib.pyplot as plt
import csv
import pickle
import math
from util import make_dir

class Evaluater(object):
    def __init__(self, logger, size, tag='paper_figure'):
        # self.pixel_spaceing = 0.1
        self.tag = tag
        self.size = size
        make_dir(tag)
        self.tag += '/'
        self.logger = logger        
        self.RE_list = list() # MRE
        self.RE_z = list()
        self.FR_1 = 0
        self.FR_2 = 0
        self.FR_3 = 0
        self.FR_4 = 0
        self.FR_5 = 0
        self.FR_6 = 0
        self.count = 0
    
    def reset(self):
        self.RE_list.clear()
        self.FR_1 = 0
        self.FR_2 = 0
        self.FR_3 = 0
        self.FR_4 = 0
        self.FR_5 = 0
        self.FR_6 = 0
        self.count = 0

    def record(self, pred, landmark, original_size, pred_z, z_list):
        # n = batchsize = 1
        # pred : list[ c(z) ; c(y) ; c(x) ]
        # landmark: list [ (z, y, x) * c]
        scale_rate_z = 1
        scale_rate_y = original_size[0] / self.size[0]
        scale_rate_x = original_size[1] / self.size[1]

        c = pred[0].shape[0]
        self.count = self.count + c
        diff = np.zeros([c, 3], dtype=float) # y, x
        for i in range(c):
            diff[i][0] = abs(pred[0][i] - landmark[i][1]) * scale_rate_y 
            diff[i][1] = abs(pred[1][i] - landmark[i][0]) * scale_rate_x
            diff[i][2] = abs(pred_z[i] - z_list[0][i])
            error = np.sqrt(np.power(diff[i][0], 2) + np.power(diff[i][1], 2) + np.power(diff[i][2], 2))
            if error > 1:
                self.FR_1 = self.FR_1 + 1
            if error > 2:
                self.FR_2 = self.FR_2 + 1
            if error > 3:
                self.FR_3 = self.FR_3 + 1
            if error > 4:
                self.FR_4 = self.FR_4 + 1
            if error > 5:
                self.FR_5 = self.FR_5 + 1
            if error > 6:
                self.FR_6 = self.FR_6 + 1
            error_z = diff[i][2]
            self.RE_z.append(error_z)

        Radial_Error = np.sqrt(np.power(diff[:,0], 2) + np.power(diff[:,1], 2))
        self.RE_list.append(Radial_Error)
        return None

    def cal_metrics(self):
        # calculate MRE FR
        temp = np.array(self.RE_list) #shape:(total,19)
        temp_2 = np.array(self.RE_z)
        Mean_RE_channel = temp.mean(axis=0)
        self.logger.info("ALL MRE {}".format(Mean_RE_channel.mean()))
        self.logger.info("ALL MRE Z {}".format(temp_2.mean()))
        self.logger.info("FR 1 {}".format(self.FR_1 / self.count))
        self.logger.info("FR 2 {}".format(self.FR_2 / self.count))
        self.logger.info("FR 3 {}".format(self.FR_3 / self.count))
        self.logger.info("FR 4 {}".format(self.FR_4 / self.count))
        self.logger.info("FR 5 {}".format(self.FR_5 / self.count))
        self.logger.info("FR 6 {}".format(self.FR_6 / self.count))
