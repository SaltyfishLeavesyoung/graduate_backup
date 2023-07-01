import torch
import numpy as np
import matplotlib.pyplot as plt
import csv
import pickle

from utils import make_dir

class Evaluater(object):
    def __init__(self, logger, size, original_size, tag='paper_figure'):
        self.pixel_spaceing = 0.1
        self.tag = tag
        make_dir(tag)
        self.tag += '/'

        self.logger = logger
        self.scale_rate_y = original_size[0] / size[0]
        self.scale_rate_x = original_size[1] / size[1]
        
        self.RE_list = list()

        self.offset_list = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
        self.diff_list = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]

        self.recall_radius = [2, 2.5, 3, 4] # 2mm etc
    
    def reset(self):
        self.RE_list.clear()

    def record(self, pred, landmark):
        # n = batchsize = 1
        # pred : list[ c(y) ; c(x) ]
        # landmark: list [ (x , y) * c]
        c = pred[0].shape[0]
        diff = np.zeros([c, 2], dtype=float) # y, x
        for i in range(c):
            diff[i][0] = abs(pred[0][i] - landmark[i][1]) * self.scale_rate_y
            diff_y = pred[0][i] - landmark[i][1]
            diff[i][1] = abs(pred[1][i] - landmark[i][0]) * self.scale_rate_x
            diff_x = pred[1][i] - landmark[i][0]
            #self.offset_list[i].append([diff_x, diff_y])
            #self.diff_list[i].append(np.sqrt(np.power(diff[i,0], 2) + np.power(diff[i,1], 2)) * self.pixel_spaceing)
        Radial_Error = np.sqrt(np.power(diff[:,0], 2) + np.power(diff[:,1], 2))
        Radial_Error *= self.pixel_spaceing
        self.RE_list.append(Radial_Error)
        # for i in range(len(Radial_Error)):
        #     if Radial_Error[i] > 10:
        #         print("Landmark {} RE {}".format(i, Radial_Error[i]))
        # if Radial_Error.max() > 10:
        #     return Radial_Error.argmax()
        return None

    def cal_metrics(self):
        # calculate MRE SDR
        temp = np.array(self.RE_list) #shape:(total,19)
        Mean_RE_channel = temp.mean(axis=0)
        self.logger.info(Mean_RE_channel)
        with open('results.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(Mean_RE_channel.tolist())
        self.logger.info("ALL MRE {}".format(Mean_RE_channel.mean()))

        #for i in range(len(self.diff_list)):
        #    temp = np.array(self.diff_list[i])
        #    self.logger.info("landmark {} MRE: {}".format(i + 1, temp.mean(axis=0)))

        for radius in self.recall_radius:
            total = temp.size
            shot = (temp < radius).sum()
            self.logger.info("ALL SDR {}mm  {}".format\
                (radius, shot * 100 / total))
