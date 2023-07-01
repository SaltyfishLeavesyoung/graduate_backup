import torch
import numpy as np
import matplotlib.pyplot as plt
import csv
import pickle
import math

class Evaluater(object):
    def __init__(self, logger, size):
        # self.pixel_spaceing = 0.1
        self.size = size
        self.logger = logger        
        self.RE_list = list()
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

    def record(self, pred, landmark):
        b, c, _ = pred.shape
        self.count = self.count + b * c
        diff = np.zeros([c, 3], dtype=float) # y, x
        diff = pred - landmark
        for i in range(b):
            for j in range(c):
                error = np.sqrt(np.power(diff[i][j][0], 2) + np.power(diff[i][j][1], 2) + np.power(diff[i][j][2], 2))
                if error <= 1:
                    self.FR_1 = self.FR_1 + 1
                if error <= 2:
                    self.FR_2 = self.FR_2 + 1
                if error <= 3:
                    self.FR_3 = self.FR_3 + 1
                if error <= 4:
                    self.FR_4 = self.FR_4 + 1
                if error <= 5:
                    self.FR_5 = self.FR_5 + 1
                if error <= 6:
                    self.FR_6 = self.FR_6 + 1

        Radial_Error = np.sqrt(np.power(diff[:,:,0], 2) + np.power(diff[:,:,1], 2) + np.power(diff[:,:,2], 2))
        self.RE_list.append(Radial_Error.numpy())
        return None

    def cal_metrics(self):
        # calculate MRE FR
        temp = np.array(self.RE_list)
        Mean_RE_channel = temp.mean(axis=0)
        self.logger.info("ALL MRE {}".format(Mean_RE_channel.mean()))
        self.logger.info("FR 1 {}".format(self.FR_1 / self.count))
        self.logger.info("FR 2 {}".format(self.FR_2 / self.count))
        self.logger.info("FR 3 {}".format(self.FR_3 / self.count))
        self.logger.info("FR 4 {}".format(self.FR_4 / self.count))
        self.logger.info("FR 5 {}".format(self.FR_5 / self.count))
        self.logger.info("FR 6 {}".format(self.FR_6 / self.count))
