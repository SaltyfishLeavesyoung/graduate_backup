import argparse
import csv
import torch
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from torch.nn import BCELoss
import os
import yaml
import yamlloader
import random
import matplotlib.pyplot as plt
import cv2

from network import  global_net, local_net
from dataloader import Cephalometric
from mylogger import get_mylogger, set_logger_dir
from eval import Evaluater
from utils import to_Image, voting, visualize, make_dir

from spp import spp
from loss import get_spatial_mean

def L1Loss(pred, gt, mask=None):
    assert(pred.shape == gt.shape)
    gap = pred - gt
    distence = gap.abs()
    if mask is not None:
        distence = distence * mask
    return distence.sum() / mask.sum()
    # return distence.mean()

def total_loss(mask, guassian_mask, heatmap, gt_y, gt_x, pred_y, pred_x, lamda, target_list=None):
    b, k, h, w = mask.shape
    logic_loss = BCELoss()
    loss_list = list()
    for i in range(mask.shape[1]):
        channel_loss = 2 * logic_loss(heatmap[0][i], guassian_mask[0][i]) +\
            (L1Loss(pred_y[0][i], gt_y[0][i], mask[0][i]) + L1Loss(pred_x[0][i], gt_x[0][i], mask[0][i]))
        loss_list.append(channel_loss)
    total_loss = np.array(loss_list).sum()
    return total_loss

def total_loss_adaptive(mask, guassian_mask, heatmap, gt_y, gt_x, pred_y, pred_x, lamda, target_list=None):
    b, k, h, w = mask.shape
    logic_loss = BCELoss()
    loss_list = list()
    for i in range(mask.shape[1]):
        channel_loss = 2 * logic_loss(heatmap[0][i], guassian_mask[0][i]) +\
            (L1Loss(pred_y[0][i], gt_y[0][i], mask[0][i]) + L1Loss(pred_x[0][i], gt_x[0][i], mask[0][i]))
        loss_list.append(channel_loss)
    loss_list_mean = torch.tensor(loss_list).mean()
    for i in range(len(loss_list)):
        loss_list[i] *= loss_list[i] / loss_list_mean
    total_loss = np.array(loss_list).sum()
    return total_loss

class Tester(object):
    def __init__(self, logger, config, g_net=None, l_net=None, tag=None, train="", args=None):
        mode = "Test1" if train == "" else "Test2"
        dataset_1 = Cephalometric(config['dataset_pth'], mode)
        self.dataloader_1 = DataLoader(dataset_1, batch_size=1,
                                shuffle=False, num_workers=config['num_workers'])
        
        # # For anthor Testset, deprecated
        # dataset_2 = Cephalometric(config['dataset_pth'], 'Test2')
        # self.dataloader_2 = DataLoader(dataset_2, batch_size=1,
        #                         shuffle=False, num_workers=config['num_workers'])
        
        self.Radius = dataset_1.Radius
        self.config = config
        self.args = args
        
        self.g_net = g_net
        self.l_net = l_net
        
        # Creat evluater to record results
        if args is None or args.rand == "":
            self.evaluater = Evaluater(logger, dataset_1.size, \
                dataset_1.original_size)
        else:
            self.evaluater = Evaluater(logger, dataset_1.size, \
                dataset_1.original_size, args.rand)

        self.logger = logger

        self.dataset = dataset_1

        output_file_pth = os.path.join("runs", tag, tag+"_result.csv")
        output_pth = os.path.join("runs", tag)
    
        self.output_dir = make_dir(os.path.join(output_pth, 'results'))

        self.id_landmarks = [i for i in range(config['num_landmarks'])]

    def debug(self, net=None):
        # Print paper figures and debug
        if net is not None:
            self.model = net
        assert(hasattr(self, 'model'))
        ID=0
        logic_loss_list = list()
        for img, mask, guassian_mask, offset_y, offset_x, landmark_list in tqdm(self.dataloader_1):
            img, mask, offset_y, offset_x, guassian_mask = img.cuda(), mask.cuda(), \
                offset_y.cuda(), offset_x.cuda(), guassian_mask.cuda()
            with torch.no_grad():
                heatmap, regression_y, regression_x = self.model(img)
            pred_landmark = voting(heatmap, regression_y, regression_x, self.Radius)
            logic_loss = BCELoss()
            logic_loss = logic_loss(heatmap, mask)
            
            regression_x, regression_y = regression_x*mask, regression_y*mask
            for i in range(19):
                to_Image(heatmap[0][i], show=str(ID)+"/"+str(i)+"heatmap", normalize=False)
                to_Image(guassian_mask[0][i], show=str(ID)+"/"+str(i)+"mask")
                to_Image(regression_x[0][i], show=str(ID)+"/"+str(i)+"regression_x", normalize=True)
                to_Image(offset_x[0][i], show=str(ID)+"/"+str(i)+"offset_x", normalize=True)
                to_Image(regression_y[0][i], show=str(ID)+"/"+str(i)+"regression_y", normalize=True)
                to_Image(offset_y[0][i], show=str(ID)+"/"+str(i)+"offset_y", normalize=True)
            """image_gt = visualize(img, landmark_list, highlight=25)
            image_pred = visualize(img, pred_landmark, [0, 2])
            image_gt.save("./paper_figure/"+str(ID)+'/gt.png')
            image_pred.save("./paper_figure/"+str(ID)+'/pred.png')"""
            #import ipdb; ipdb.set_trace()
            ID+=1

            logic_loss_list.append(logic_loss.cpu().item())
            # import ipdb; ipdb.set_trace()
        print(sum(logic_loss_list)/self.dataset.__len__())

    def test(self, g_net=None, l_net=None):
        self.evaluater.reset()
        if g_net is not None and l_net is not None:
            self.g_net = g_net
            self.l_net = l_net
        assert(hasattr(self, 'g_net'))
        assert(hasattr(self, 'l_net'))
        ID = 0

        distance_list = dict()
        mean_list = dict()
        for i in range(19):
            distance_list[i] = list()
            mean_list[i] = list()

        self.g_net.eval()

        for img, landmark, mask, guassian_mask, offset_y_local, offset_x_local, _, _, landmark_list in tqdm(self.dataloader_1):
            img, landmark, mask, offset_y_local, offset_x_local, guassian_mask = img.cuda(), landmark.cuda(), mask.cuda(), offset_y_local.cuda(), offset_x_local.cuda(), guassian_mask.cuda()
            with torch.no_grad():
                heatmap = self.g_net(img)
                coord_from_heatmap = get_spatial_mean(heatmap)

                pred_y = list()
                pred_x = list()

                for lid in range(config['num_landmarks']):
                    local_input = torch.zeros((heatmap.shape[0], 3, 256, 256)).cuda()
                    offset_x_gt = torch.zeros((heatmap.shape[0], 1, 256, 256)).cuda()
                    offset_y_gt = torch.zeros((heatmap.shape[0], 1, 256, 256)).cuda()
                    for batch in range(heatmap.shape[0]):
                        pred_coord = coord_from_heatmap[batch][lid]
                        y_min_local = int(pred_coord[1]) - 128
                        y_max_local = int(pred_coord[1]) + 128
                        x_min_local = int(pred_coord[0]) - 128
                        x_max_local = int(pred_coord[0]) + 128
                        crop_img = img[batch,:,y_min_local:y_max_local,x_min_local:x_max_local]
                        crop_offset_x = offset_x_local[batch,lid,y_min_local:y_max_local,x_min_local:x_max_local]
                        crop_offset_y = offset_y_local[batch,lid,y_min_local:y_max_local,x_min_local:x_max_local]
                        local_input[batch,:,:crop_img.shape[1],:crop_img.shape[2]] = crop_img
                        offset_x_gt[batch,0,:crop_offset_x.shape[0],:crop_offset_x.shape[1]] = crop_offset_x
                        offset_y_gt[batch,0,:crop_offset_y.shape[0],:crop_offset_y.shape[1]] = crop_offset_y

                    self.l_net[lid].eval()
                    regression_y, regression_x = self.l_net[lid](local_input)

                    record = torch.zeros((img.shape[-2], img.shape[-1]))

                    for idy in range(256):
                        for idx in range(256):
                            py = int(pred_coord[1] + idy - 128 + regression_y[0][0][idy][idx] * img.shape[-2])
                            px = int(pred_coord[0] + idx - 128 + regression_x[0][0][idy][idx] * img.shape[-1])
                            record[py][px] += 1

                    index = torch.argmax(record)
                    pred_idy = 0
                    pred_idx = 0
                    if (index+1) % img.shape[-1] == 0:
                        pred_idy = (index+1) // img.shape[-1] - 1
                        pred_idx = img.shape[-1] - 1
                    else:
                        pred_idy = (index+1) // img.shape[-1]
                        pred_idx = ((index+1) % img.shape[-1]) - 1
                    pred_y.append([pred_idy])
                    pred_x.append([pred_idx])

                pred_y = torch.tensor(pred_y)
                pred_x = torch.tensor(pred_x)
                pred = [pred_y, pred_x]
                self.evaluater.record(pred, landmark_list)

            #pred_landmark = pred_landmark.cpu()
            #pred = [pred_landmark[0,:,1], pred_landmark[0,:,0]]
            #self.evaluater.record(pred, landmark_list)
            
            #image_gt = visualize(img, landmark_list, [i for i in range(19)]) #blue
            #image_gt_pred = visualize(torch.from_numpy(np.array(image_gt).transpose(2,0,1)).unsqueeze(0), pred_landmark)
            #image_gt_pred.save(os.path.join(self.output_dir, str(ID)+'_gt_pred.png'))
            #image_gt.save(os.path.join(self.output_dir, str(ID)+'_gt.png'))

            ID += 1
            
        self.evaluater.cal_metrics()


if __name__ == "__main__":
    # Parse command line options
    parser = argparse.ArgumentParser(description="Train a cgan Xray network")
    parser.add_argument("--tag", default='test', help="position of the output dir")
    parser.add_argument("--debug", default='', help="position of the output dir")
    parser.add_argument("--iteration", default='', help="position of the output dir")
    parser.add_argument("--config_file", default="config.yaml", help="default configs")
    parser.add_argument("--checkpoint_file", default="", help="default configs")
    parser.add_argument("--output_file", default="", help="default configs")
    parser.add_argument("--train", default="", help="default configs")
    parser.add_argument("--rand", default="", help="default configs")
    parser.add_argument("--epsilon", default="8", help="default configs")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    with open(os.path.join("runs", args.tag, args.config_file), "r") as f:
        config = yaml.load(f, Loader=yamlloader.ordereddict.CLoader)
    
    # Create Logger
    logger = get_mylogger()
        
    if args.iteration == '':
        iteration = 499
    else:
        iteration = int(args.iteration)
    
    # Load model
    g_net = global_net(3, config['num_landmarks']).cuda()
    l_net = list()
    for i in range(config['num_landmarks']):
        temp_net = local_net(3, 1)
        temp_net = torch.nn.DataParallel(temp_net)
        ckpt = torch.load(os.path.join(config['runs_dir'], \
                        "model_best_local_{}.pth".format(i)))
        temp_net.load_state_dict(ckpt)
        l_net.append(temp_net.cuda())

    logger.info("Loading checkpoints from epoch {}".format(iteration))
    #checkpoints = torch.load(os.path.join(config['runs_dir'], \
    #                    "model_epoch_{}.pth".format(iteration)))
    g_net = torch.nn.DataParallel(g_net)
    checkpoints = torch.load(os.path.join(config['runs_dir'], \
                        "model_best_global.pth"))
    g_net.load_state_dict(checkpoints)

    tester = Tester(logger, config, g_net, l_net, args.tag, args.train, args)
    if args.debug != '':
        tester.debug()
    else:
        tester.test()
