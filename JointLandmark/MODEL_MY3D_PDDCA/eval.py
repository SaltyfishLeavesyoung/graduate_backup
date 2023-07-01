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
        self.RE_list_x = list()
        self.RE_list_y = list()
        self.RE_list_z = list()
        self.FR_1 = 0
        self.FR_2 = 0
        self.FR_3 = 0
        self.FR_4 = 0
        self.FR_5 = 0
        self.FR_6 = 0
        self.count = 0

        self.re = [0, 0, 0, 0, 0, 0]
    
    def reset(self):
        self.RE_list.clear()
        self.FR_1 = 0
        self.FR_2 = 0
        self.FR_3 = 0
        self.FR_4 = 0
        self.FR_5 = 0
        self.FR_6 = 0
        self.count = 0

        self.re = [0, 0, 0, 0, 0, 0]

    def record(self, pred, landmark):
        b, c, _ = pred.shape
        self.count = self.count + b * c
        diff = np.zeros([b, c, 3], dtype=float) # y, x
        # diff = np.zeros([c, 2], dtype=float)
        ratio = [255, 255, 63]
        for i in range(b):
            for j in range(c):
                for k in range(3):
                    diff[i][j][k] = pred[i][j][k] * ratio[k] - landmark[i][j][k]
        for i in range(b):
            for j in range(c):
                error = np.sqrt(np.power(diff[i][j][0], 2) + np.power(diff[i][j][1], 2) + np.power(diff[i][j][2], 2))
                # error = np.sqrt(np.power(diff[i][j][0], 2) + np.power(diff[i][j][1], 2))
                self.re[j] = self.re[j] + error.item()
                if error <= 1:
                    self.FR_1 = self.FR_1 + 1
                if error <= 2:
                    self.FR_2 = self.FR_2 + 1
                if error <= 2.5:
                    self.FR_3 = self.FR_3 + 1
                if error <= 3:
                    self.FR_4 = self.FR_4 + 1
                if error <= 4:
                    self.FR_5 = self.FR_5 + 1
                if error <= 8:
                    self.FR_6 = self.FR_6 + 1

        Radial_Error = np.sqrt(np.power(diff[:,:,0], 2) + np.power(diff[:,:,1], 2) + np.power(diff[:,:,2], 2))
        # Radial_Error = np.sqrt(np.power(diff[:,:,0], 2) + np.power(diff[:,:,1], 2))
        self.RE_list.append(Radial_Error)
        self.RE_list_x.append(np.sqrt(np.power(diff[:,:,0], 2)))
        self.RE_list_y.append(np.sqrt(np.power(diff[:,:,1], 2)))
        self.RE_list_z.append(np.sqrt(np.power(diff[:,:,2], 2)))
        return None

    def cal_metrics(self):
        # calculate MRE FR
        temp = np.array(self.RE_list)
        temp_x = np.array(self.RE_list_x)
        temp_y = np.array(self.RE_list_y)
        temp_z = np.array(self.RE_list_z)

        Mean_RE_channel = temp.mean(axis=0)
        Mean_RE_channel_x = temp_x.mean(axis=0)
        Mean_RE_channel_y = temp_y.mean(axis=0)
        Mean_RE_channel_z = temp_z.mean(axis=0)
        self.logger.info("ALL MRE {}".format(Mean_RE_channel.mean()))
        self.logger.info("ALL MRE X {}".format(Mean_RE_channel_x.mean()))
        self.logger.info("ALL MRE Y {}".format(Mean_RE_channel_y.mean()))
        self.logger.info("ALL MRE Z {}".format(Mean_RE_channel_z.mean()))
        self.logger.info("FR 1 {}".format(self.FR_1 / self.count))
        self.logger.info("FR 2 {}".format(self.FR_2 / self.count))
        self.logger.info("FR 3 {}".format(self.FR_3 / self.count))
        self.logger.info("FR 4 {}".format(self.FR_4 / self.count))
        self.logger.info("FR 5 {}".format(self.FR_5 / self.count))
        self.logger.info("FR 6 {}".format(self.FR_6 / self.count))

        for i in range(6):
            self.logger.info("MRES {}".format(self.re[i] / self.count * 6))
