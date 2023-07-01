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

from network import UNet_Pretrained_Att
from dataloader import Cephalometric
from mylogger import get_mylogger, set_logger_dir
from eval import Evaluater
from util import to_Image, voting, visualize, make_dir, getAdjMatrix

def test(logger, dataloader_1, net, tag):
    loss_list_test = list()
    net.eval()
    output_pth = os.path.join("runs", tag)
    output_dir = make_dir(os.path.join(output_pth, 'results'))

    evaluater = Evaluater(logger, [256, 256])
    ID = 0
    for img, mask, guassian_mask, z_list, z_list_one_hot, offset_y, offset_x, landmark_list, w, h in tqdm(dataloader_1):
        img, mask, z_list, offset_y, offset_x, guassian_mask = img.cuda(), mask.cuda(), \
                z_list.cuda(), offset_y.cuda(), offset_x.cuda(), guassian_mask.cuda()
        with torch.no_grad():
            heatmap, predict_z, regression_y, regression_x = net(img)
            
        pred_landmark = voting(heatmap, regression_y, regression_x, Radius=int(256*0.05))
        pred_z = []
        for i in range(6):
            pred_z.append((predict_z[0][i].tolist()).index(max(predict_z[0][i])))
        evaluater.record(pred_landmark, landmark_list, [w, h], pred_z, z_list.cpu().detach().numpy())
        #print(pred_landmark)
        # visualize
        '''temp_img = img.squeeze(0)
        img_list = [temp_img[:, i, :, :] for i in range(img.shape[2])]
        image_gt = visualize(img_list, landmark_list, (255, 255, 0))
        temp_image_gt = [torch.from_numpy(np.array(img_gt)).permute(2,0,1) for img_gt in image_gt]
        image_pred = visualize(temp_image_gt, pred_landmark, (0, 255, 0))
        for i in range(len(image_pred)):
            image_pred[i].save(os.path.join(output_dir, str(ID) + "_" + str(i) + 'pred.png'))
        ID = ID + 1'''

    evaluater.cal_metrics()        
        

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

    with open(os.path.join("runs", args.tag, args.config_file), "r") as f:
        config = yaml.load(f, Loader=yamlloader.ordereddict.CLoader)
    
    # Create Logger
    logger = get_mylogger()
        
    if args.iteration == '':
        iteration = config['best_epoch']
    else:
        iteration = int(args.iteration)

    dataset_1 = Cephalometric(config['test_pth'])
    dataloader_1 = DataLoader(dataset_1, batch_size=1,
                                shuffle=False, num_workers=config['num_workers'])
    
    # Load model
    adj = getAdjMatrix(k=config['adj'])
    net = torch.nn.DataParallel(UNet_Pretrained_Att(3, config['num_landmarks'], adj).cuda())

    logger.info("Loading checkpoints from epoch {}".format(iteration))
    checkpoints = torch.load(os.path.join(config['runs_dir'], \
                        "model_best.pth"))
    net.load_state_dict(checkpoints)
    #net = torch.nn.DataParallel(net)

    test(logger, dataloader_1, net, args.tag)
        
