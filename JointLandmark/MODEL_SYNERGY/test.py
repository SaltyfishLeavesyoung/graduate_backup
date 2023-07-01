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

from model import SynergyNet
import utils
from utils.imutils import show_voxel, show_joints3D
from utils.misc import adjust_learning_rate
from utils.evaluation import AverageMeter, bboxNormMeanError, p2pNormMeanError, calRE
from util import get_mylogger

from fa68pt3D import fa68pt3D
from loss_definition import WingLoss
from eval import Evaluater

def test(logger, dataloader_1, net):

    def data2variable(inputs, target, meta):
        with torch.no_grad():
            input_var = torch.autograd.Variable(inputs.cuda())
            target_var = [torch.autograd.Variable(target[i].cuda()) for i in
                          range(len(target))]
            coord_var = torch.autograd.Variable(meta['tpts_inp'].cuda())

        return input_var, target_var, coord_var

    loss_list_test = list()
    net.eval()

    evaluater = Evaluater(logger, [256, 256, 8])
    ID = 0

    for inputs, target, meta in tqdm(dataloader_1):
        input_var, target_var, coord_var = data2variable(inputs, target, meta)

        with torch.no_grad():
            pred_coord = net.forward(input_var)

        pred_landmarks = (255 * pred_coord[:, 0:6, :].data).cpu()
        target_landmarks = meta['tpts_inp'][:, 0:6, :]

        evaluater.record(pred_landmarks, target_landmarks)

    evaluater.cal_metrics()        
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Joint Voxel and Coordinate Regression')

    parser.add_argument('-s', '--stacks', default=4, type=int, metavar='N',
                        help='number of hourglasses to stack')
    parser.add_argument('-b', '--blocks', default=1, type=int, metavar='N',
                        help='number of residual modules at each location in the hourglass')
    parser.add_argument('--depth_res', default=[1, 2, 4, 64], type=int, nargs="*",
                        help='Resolution of depth for the output of the corresponding hourglass')
    parser.add_argument('--resume_p2v', default='', type=str,
                        help='path to the model of voxel regression subnetwork')
    parser.add_argument('--resume_v2c', default='', type=str,
                        help='path to the model of coordinate regression subnetwork')
    parser.add_argument('--resume_p2v2c', default='/home/yzy/Desktop/JointLandmark/MODEL_SYNERGY/checkpoint/checkpoint_best_345.pth.tar', type=str,
                        help='path to the pre-trained model')
    parser.add_argument('--gpus', default='0,1,2,3', type=str, help='set gpu IDs')
    # Training strategy
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=500, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--pretr_epochs', default=0, type=int, metavar='N',
                        help='Number of epochs for pre-training the network')
    parser.add_argument('--train-batch', default=32, type=int, metavar='N',
                        help='train batchsize')
    parser.add_argument('--test-batch', default=1, type=int, metavar='N',
                        help='test batchsize')
    parser.add_argument('--lr', '--learning-rate', default=1e-1, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float)
    parser.add_argument('--schedule', type=int, nargs='+', default=[150, 300, 450],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='LR is multiplied by gamma on schedule.')
    parser.add_argument('--sigma', type=float, default=1,
                        help='Groundtruth Gaussian sigma.')
    parser.add_argument('--num_snapshot', default=5, type=int, metavar='N',
                        help='Frequency for saving checkpoints')
    # Miscs
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--img_size', default=256, type=int)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    if args.gpus == '':
        is_cuda = False
        print('Run in CPU mode.')
    else:
        is_cuda = True
    
    # Create Logger
    logger = get_mylogger()

    dataloader_1 = torch.utils.data.DataLoader(
        fa68pt3D('/home/yzy/Desktop/JointLandmark/val.txt', depth_res=args.depth_res,
            nStack=args.stacks, sigma=args.sigma),
        batch_size=args.test_batch, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # Load model
    net = SynergyNet(args)
    net = torch.nn.DataParallel(net).cuda()
    checkpoint = torch.load(args.resume_p2v2c)
    net.load_state_dict(checkpoint)

    test(logger, dataloader_1, net)
        
