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
import utils
from utils.imutils import show_voxel, show_joints3D
from utils.misc import adjust_learning_rate
from utils.evaluation import AverageMeter, bboxNormMeanError, p2pNormMeanError, calRE
from util import get_mylogger

from fa68pt3D import fa68pt3D
from loss_definition import WingLoss
from eval import Evaluater
from util import getAdjMatrix

from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import ToPILImage
to_PIL = ToPILImage()

def test(logger, dataloader_1, net):

    def data2variable(inputs, pts):
        with torch.no_grad():
            input_var = torch.autograd.Variable(inputs.cuda())
            coord_var = torch.autograd.Variable(pts.cuda())

        return input_var, coord_var

    loss_list_test = list()
    net.eval()

    evaluater = Evaluater(logger, [255, 255, 63])
    ID = 0

    for inputs, pts, meta in tqdm(dataloader_1):
        input_var, coord_var = data2variable(inputs, pts)

        with torch.no_grad():
            pred_coord = net.forward(input_var)

        pred_landmarks = (pred_coord[:, 0:5, :].data).cpu()
        target_landmarks = pts[:, 0:5, :]

        evaluater.record(pred_landmarks, target_landmarks)

        # draw
        """for i in range(8):
            temp_res = input_var[0][0][i].cpu()
            temp_img = to_PIL(torch.stack([temp_res, temp_res, temp_res], dim = 0))
            draw = ImageDraw.Draw(temp_img)

            for j in range(6):
                if int(target_landmarks[0][j][2]) == i:
                    draw.rectangle((int(target_landmarks[0][j][0])-3, int(target_landmarks[0][j][1])-3, \
                        int(target_landmarks[0][j][0])+3, int(target_landmarks[0][j][1])+3), fill='green')

            for j in range(6):
                if int(pred_landmarks[0][j][2]) == i:
                    draw.rectangle((int(pred_landmarks[0][j][0])-3, int(pred_landmarks[0][j][1])-3, \
                        int(pred_landmarks[0][j][0])+3, int(pred_landmarks[0][j][1])+3), fill='yellow')
            temp_img.save("./res/{}_{}.png".format(ID, i))"""

        # output landmark
        """idx = meta['id']
        part = meta['part']

        with open("./pseudolabel/{}{}.txt".format(idx[0], part[0]), "a") as f:
            for i in range(6):
                tl = pred_landmarks[0][i]
                f.write("{} {} {}\n".format(int(round(float(tl[0]))), int(round(float(tl[1]))), int(round(float(tl[2])))))"""

        ID = ID + 1

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
    parser.add_argument('--resume_p2v2c', default='/home/yeziyang/Desktop/JointLandmark/MODEL_MY3D_PDDCA/checkpoint/checkpoint_best_985.pth.tar')
    parser.add_argument('--gpus', default='2', type=str, help='set gpu IDs')
    # Training strategy
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=500, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--pretr_epochs', default=0, type=int, metavar='N',
                        help='Number of epochs for pre-training the network')
    parser.add_argument('--train-batch', default=24, type=int, metavar='N',
                        help='train batchsize')
    parser.add_argument('--test-batch', default=1, type=int, metavar='N',
                        help='test batchsize')
    parser.add_argument('--lr', '--learning-rate', default=1e-1, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-3, type=float)
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
    parser.add_argument('--img_size', default=512, type=int)
    parser.add_argument('--num_landmarks', default=5, type=int)
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
        fa68pt3D('/home/yeziyang/Desktop/PDDCA/val', train=False),
        batch_size=args.test_batch, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # Load model
    adj = torch.ones((5,5)).cuda()
    net = UNet_Pretrained_Att(64, args.num_landmarks, adj)
    net = torch.nn.DataParallel(net).cuda()
    checkpoint = torch.load(args.resume_p2v2c)
    net.load_state_dict(checkpoint)

    test(logger, dataloader_1, net)
        
