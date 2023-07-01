from __future__ import print_function, absolute_import

import os
import sys
import argparse
from time import time
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import time
from PIL import Image
from progress.bar import Bar

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision
from torch.nn import SmoothL1Loss, MSELoss, L1Loss

from network import UNet_Pretrained_Att
import utils
from utils.imutils import show_voxel, show_joints3D
from utils.misc import adjust_learning_rate
from utils.evaluation import AverageMeter, bboxNormMeanError, p2pNormMeanError, calRE

from fa68pt3D import fa68pt3D
from loss_definition import WingLoss
from tqdm import tqdm
from util import getAdjMatrix

import torchvision.transforms as transforms

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

PART = ['L', 'R']

resize = transforms.Resize([256, 256])
as_tensor = transforms.Compose([
            resize,
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
transform_resize = transforms.Compose([
            resize,
        ])


def main(args):

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    if args.gpus == '':
        is_cuda = False
        print('Run in CPU mode.')
    else:
        is_cuda = True
        cudnn.benchmark = True

    # set training and evaluation datasets
    """train_loader = torch.utils.data.DataLoader(
        fa68pt3D('/home/yeziyang/Desktop/PDDCA/train', train=True),
        batch_size=args.train_batch, shuffle=True,
        num_workers=args.workers, pin_memory=True)


    val_loader = torch.utils.data.DataLoader(
        fa68pt3D('/home/yeziyang/Desktop/PDDCA/val', train=False),
        batch_size=args.test_batch, shuffle=False,
        num_workers=args.workers, pin_memory=True)"""

    loader_1 = torch.utils.data.DataLoader(
        fa68pt3D('/home/yeziyang/Desktop/PDDCA/3-fold/1', train=True),
        batch_size=args.train_batch, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    loader_2 = torch.utils.data.DataLoader(
        fa68pt3D('/home/yeziyang/Desktop/PDDCA/3-fold/2', train=True),
        batch_size=args.train_batch, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    loader_3 = torch.utils.data.DataLoader(
        fa68pt3D('/home/yeziyang/Desktop/PDDCA/3-fold/3', train=True),
        batch_size=args.train_batch, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    loaders = [loader_1, loader_2, loader_3]

    nParts = 6
    # create model
    adj = torch.ones((5,5)).cuda()
    model = UNet_Pretrained_Att(64, args.num_landmarks, adj)
    model = torch.nn.DataParallel(model).cuda()

    if args.pretrain:
        checkpoint = torch.load('/home/yeziyang/Desktop/JointLandmark/MODEL_MY3D_PDDCA/checkpoint/checkpoint_MRE_2630.tar')
        model.load_state_dict(checkpoint)

    # set optimizer
    print('using ADAM optimizer.')
    '''optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True)'''
    optimizer = torch.optim.Adam(params=model.parameters(), \
        lr=args.lr, betas=(0.9,0.999), eps=1e-08, weight_decay=args.weight_decay)

    # set loss criterion
    # criterion = WingLoss().cuda()
    criterion = MSELoss().cuda()
    # criterion = SmoothL1Loss().cuda()

    lr = args.lr

    best_acc = float('-inf')

    for epoch in range(args.start_epoch):
        lr_new = adjust_learning_rate(optimizer, epoch, lr, args.schedule, args.gamma)
        lr = lr_new
        print('\nEpoch: %d | LR: %.8f' % (epoch + 1, lr))

        # train for one epoch
        mode = 'pre_train' if epoch < args.pretr_epochs else 'train'
        print(mode+'ing...')
        run(best_acc, model, train_loader, mode, criterion, optimizer)

        # evaluation
        mode = 'evaluate'
        log_info, nme_results, current_acc = run(best_acc, model, val_loader, mode, criterion, optimizer)

        is_best = current_acc > best_acc
        if is_best:
            print('new record:{}'.format(current_acc))
            best_acc = current_acc
            torch.save(model.state_dict(), os.path.join(args.checkpoint, 'checkpoint_best_{}_{}.pth.tar'.format(epoch, current_acc)))
        """if (epoch + 1) % args.num_snapshot == 0:
            torch.save(model.state_dict(), os.path.join(args.checkpoint, 'checkpoint_{}_{}.pth.tar'.format(epoch, current_acc)))"""

    for epoch in range(args.start_epoch, args.epochs):
    # for epoch in range(args.epochs):
        lr_new = adjust_learning_rate(optimizer, epoch, lr, args.schedule, args.gamma)
        lr = lr_new
        print('\nEpoch: %d | LR: %.8f' % (epoch + 1, lr))

        # train for one epoch
        mode = 'pre_train' if epoch < args.pretr_epochs else 'train'
        print(mode+'ing...')
        #run(best_acc, model, train_loader, mode, criterion, optimizer)
        #run(best_acc, model, train_loader_2, mode, criterion, optimizer)
        #run(best_acc, model, train_loader_3, mode, criterion, optimizer)
        run(best_acc, model, loaders[epoch % 3], mode, criterion, optimizer)
        run(best_acc, model, loaders[(epoch + 1) % 3], mode, criterion, optimizer)

        # evaluation
        mode = 'evaluate'
        # _, nme_results, current_acc = run(best_acc, model, val_loader, mode, criterion, optimizer)
        _, nme_results, current_acc = run(best_acc, model, loaders[(epoch + 2) % 3], mode, criterion, optimizer)

        is_best = current_acc > best_acc
        if is_best:
            print('new record:{}'.format(current_acc))
            best_acc = current_acc
            torch.save(model.state_dict(), os.path.join(args.checkpoint, 'checkpoint_best_{}.pth.tar'.format(epoch)))
        """if (epoch + 1) % args.num_snapshot == 0:
            torch.save(model.state_dict(), os.path.join(args.checkpoint, 'checkpoint_{}.pth.tar'.format(epoch)))"""

def run(best_acc, model, data_loader, mode, criterion, optimizer):
    # self.epoch += 1
    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses_coord = AverageMeter()
    errs = AverageMeter()

    RATIO = torch.Tensor([255.0, 255.0, 63.0]).cuda()

    log_key = ['losses_coord', 'errs']
    log_info = dict.fromkeys(log_key)

    # normalized mean error results
    dataset_len = data_loader.dataset.__len__()
    nme_results = torch.Tensor(dataset_len, 1)

    def data2variable(inputs, pts):
        if mode in ['pre_train', 'train']:
            input_var = torch.autograd.Variable(inputs.cuda())
            coord_var = torch.autograd.Variable(pts.cuda())
        else:
            with torch.no_grad():
                input_var = torch.autograd.Variable(inputs.cuda())
                coord_var = torch.autograd.Variable(pts.cuda())

        return input_var, coord_var

    # switch mode
    if mode in ['pre_train', 'train']:
        model.train()
    else:
        model.eval()

    data_num = 0
    data_length = len(data_loader.dataset)

    # measure time
    end = time.time()

    bar = Bar('Processing', max=len(data_loader))
    for i, (inputs, pts, meta) in enumerate(data_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input_var, coord_var = data2variable(inputs, pts)

        # run forward
        pred_coord = model.forward(input_var)
        #print(pred_coord[0])

        if mode == 'train':
            loss_coord = criterion(pred_coord[:,:,0], coord_var[:,:,0] / 255.0) * 255.0 + criterion(pred_coord[:,:,1], coord_var[:,:,1] / 255.0) * 255.0 + criterion(pred_coord[:,:,2], coord_var[:,:,2] / 63.0) * 63.0
            #loss_coord = criterion(pred_coord, coord_var / 511.0)
            # loss_coord *= 10
            optimizer.zero_grad()
            loss_coord.backward()
            optimizer.step()

        else:
            loss_coord = criterion(pred_coord[:,:,0], coord_var[:,:,0] / 255.0) * 255.0 + criterion(pred_coord[:,:,1], coord_var[:,:,1] / 255.0) * 255.0 + criterion(pred_coord[:,:,2], coord_var[:,:,2] / 63.0) * 63.0
            #loss_coord = criterion(pred_coord, coord_var / 511.0)
            # loss_coord *= 10
        pred_landmarks = (torch.mul(pred_coord[:, 0:5, :], RATIO).data).cpu()
        target_landmarks = pts[:, 0:5, :]
        box_nme = bboxNormMeanError(pred_landmarks, target_landmarks)
        box_nme = np.array(box_nme)

        for n in range(len(meta['index'])):
            nme_results[meta['index'][n]] = box_nme[n]

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        data_num += len(input_var)

        # measure nme and record loss
        losses_coord.update(loss_coord.item(), inputs.size(0))
        errs.update(np.mean(box_nme), inputs.size(0))

        log_info['losses_coord'] = losses_coord.avg
        log_info['errs'] = errs.avg

        # plot progress
        bar.suffix = '({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | LOSS {loss} | NME: {nme: .4f}'.format(
            batch=data_num,
            size=data_length,
            data=data_time.val,
            bt=batch_time.val,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss='coord: {:.4f}'.format(loss_coord.item()),
            nme=errs.avg
        )
        bar.next()

    bar.finish()

    if mode == 'evaluate':
        current_acc = -errs.avg
        print('Performance(NME) current: {}, best:{}'.format(-current_acc, -best_acc))
    else:
        current_acc = 0

    log_info['errs'] = -errs.avg

    return log_info, nme_results, current_acc


if __name__ == '__main__':
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
    parser.add_argument('--resume_p2v2c', default='/home/yzy/Desktop/JointLandmark/MODEL_MY3D/checkpoint/checkpoint_best_435.pth.tar', type=str,
                        help='path to the pre-trained model')
    parser.add_argument('--gpus', default='1,2,3,4', type=str, help='set gpu IDs')
    # Training strategy
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=1000, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--pretr_epochs', default=0, type=int, metavar='N',
                        help='Number of epochs for pre-training the network')
    parser.add_argument('--train-batch', default=4, type=int, metavar='N',
                        help='train batchsize')
    parser.add_argument('--test-batch', default=4, type=int, metavar='N',
                        help='test batchsize')
    parser.add_argument('--lr', '--learning-rate', default=3e-5, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float)
    parser.add_argument('--schedule', type=int, nargs='+', default=[200,400,600,800],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--gamma', type=float, default=1 / 3.0,
                        help='LR is multiplied by gamma on schedule.')
    parser.add_argument('--sigma', type=float, default=1,
                        help='Groundtruth Gaussian sigma.')
    parser.add_argument('--num_snapshot', default=10, type=int, metavar='N',
                        help='Frequency for saving checkpoints')
    # Miscs
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--img_size', default=512, type=int)
    parser.add_argument('--num_landmarks', default=5, type=int)
    parser.add_argument('--pretrain', default=True, type=bool)

    main(parser.parse_args())
