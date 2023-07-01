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

from fa68pt3D import fa68pt3D, fa68pt3D_2
from loss_definition import WingLoss
from eval import Evaluater
from util import getAdjMatrix

from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import ToPILImage
to_PIL = ToPILImage()

from torchcam.methods import SmoothGradCAMpp
from torchcam.utils import overlay_mask

from cam import *
import seaborn as sns

def test(logger, dataloader_1, net):

    def data2variable(inputs, target, meta):
        with torch.no_grad():
            """input_var = torch.autograd.Variable(inputs.cuda())
            target_var = [torch.autograd.Variable(target[i].cuda()) for i in
                          range(len(target))]
            coord_var = torch.autograd.Variable(meta['tpts_inp'].cuda())"""
        input_var = torch.autograd.Variable(inputs)
        target_var = [torch.autograd.Variable(target[i]) for i in
                          range(len(target))]
        coord_var = torch.autograd.Variable(meta['tpts_inp'])

        return input_var, target_var, coord_var

    loss_list_test = list()
    net.eval()

    evaluater = Evaluater(logger, [255, 255, 7])
    ID = 0

    # for name, module in net._modules['module'].named_children():
    #     print(name)

    for inputs, target, meta in tqdm(dataloader_1):
        input_var, target_var, coord_var = data2variable(inputs, target, meta)

        # cam_extractor = SmoothGradCAMpp(net, 'module.fc', input_shape = (3, 8, 256, 256))

        _, pred_coord, attn, adj = net.forward(input_var)

        print(attn)
        print(adj)

        '''attn_heatmap = sns.heatmap(attn[0].cpu().detach().numpy(), cmap='Blues', annot=True)
        attn_heatmap.get_figure().savefig(os.path.join("/home/yeziyang/Desktop/JointLandmark/MODEL_MY3D/matrix_heatmap", "{}_attn.png".format(ID)))
        plt.close()
        adj_heatmap = sns.heatmap(adj.cpu().detach().numpy(), annot=True)
        adj_heatmap.get_figure().savefig(os.path.join("/home/yeziyang/Desktop/JointLandmark/MODEL_MY3D/matrix_heatmap", "{}_adj.png".format(ID)))
        plt.close()'''

        # activation_map = cam_extractor(pred_coord.argmax().item(), pred_coord)

        pred_landmarks = (pred_coord[:, 0:6, :].data).cpu()
        target_landmarks = meta['tpts_inp'][:, 0:6, :]

        evaluater.record(pred_landmarks, target_landmarks)

        # grad_cam = GradCam(model=net, target_layer_names = ["final"], use_cuda=False)

        # mask = grad_cam(input_var, index=None)

        # print(mask.shape)

        # draw
        for i in range(8):
            temp_res = input_var[0][0][i].cpu()
            temp_img = torch.stack([temp_res, temp_res, temp_res], dim = 2)
            #temp_img = to_PIL(temp_img)

            """draw = ImageDraw.Draw(temp_img)

            for j in range(6):
                if int(target_landmarks[0][j][2] * 7) == i:
                    draw.rectangle((int(target_landmarks[0][j][0] * 255)-3, int(target_landmarks[0][j][1] * 255)-3, \
                        int(target_landmarks[0][j][0] * 255)+3, int(target_landmarks[0][j][1] * 255)+3), fill='green')

            for j in range(6):
                if int(pred_landmarks[0][j][2] * 7) == i:
                    draw.rectangle((int(pred_landmarks[0][j][0] * 255)-3, int(pred_landmarks[0][j][1] * 255)-3, \
                        int(pred_landmarks[0][j][0] * 255)+3, int(pred_landmarks[0][j][1] * 255)+3), fill='yellow')
            temp_img.save("./res2/{}_{}.png".format(ID, i))"""

            #result = overlay_mask(temp_res, to_PIL(mask[i].squeeze(0), mode='F'), alpha=0.5)
            #result = to_PIL(result)
            #result.save("./res2/cam_{}_{}.png".format(ID, i))
            #show_cam_on_image(temp_img, mask, ID, i)

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
    parser.add_argument('--resume_p2v2c', default='/home/yeziyang/Desktop/JointLandmark/MODEL_MY3D/records/best_2.tar')
    parser.add_argument('--gpus', default='1', type=str, help='set gpu IDs')
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
    parser.add_argument('--img_size', default=256, type=int)
    parser.add_argument('--num_landmarks', default=6, type=int)
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
        fa68pt3D('/home/yeziyang/Desktop/JointLandmark/test.txt', train=False, depth_res=args.depth_res,
            nStack=args.stacks, sigma=args.sigma),
        batch_size=args.test_batch, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # Load model
    adj = getAdjMatrix(k=4)
    net = UNet_Pretrained_Att(8, args.num_landmarks, adj)
    net = torch.nn.DataParallel(net)#.cuda()
    checkpoint = torch.load(args.resume_p2v2c, map_location=torch.device('cpu'))
    net.load_state_dict(checkpoint)

    test(logger, dataloader_1, net)
        
