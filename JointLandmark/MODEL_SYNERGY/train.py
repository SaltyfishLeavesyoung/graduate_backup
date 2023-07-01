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

from model import SynergyNet
import utils
from utils.imutils import show_voxel, show_joints3D
from utils.misc import adjust_learning_rate
from utils.evaluation import AverageMeter, bboxNormMeanError, p2pNormMeanError, calRE

from fa68pt3D import fa68pt3D
from loss_definition import WingLoss
from tqdm import tqdm


def main(args):

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    if args.gpus == '':
        is_cuda = False
        print('Run in CPU mode.')
    else:
        is_cuda = True
        cudnn.benchmark = True

    # set training and evaluation datasets
    train_loader = torch.utils.data.DataLoader(
        fa68pt3D('/home/yzy/Desktop/JointLandmark/train.txt', depth_res=args.depth_res,
            nStack=args.stacks, sigma=args.sigma),
        batch_size=args.train_batch, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        fa68pt3D('/home/yzy/Desktop/JointLandmark/val.txt', depth_res=args.depth_res,
            nStack=args.stacks, sigma=args.sigma),
        batch_size=args.test_batch, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    nParts = 6
    # create model
    print("==> creating model: stacks={}, blocks={}, z-res={}".format(args.stacks, args.blocks, args.depth_res))
    model = SynergyNet(args)
    model = torch.nn.DataParallel(model).cuda()

    # set optimizer
    print('using ADAM optimizer.')
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True)

    # set loss criterion
    criterion = WingLoss().cuda()

    lr = args.lr

    best_acc = float('-inf')


    for epoch in range(args.start_epoch, args.epochs):
        lr_new = adjust_learning_rate(optimizer, epoch, lr, args.schedule, args.gamma)
        lr = lr_new
        print('\nEpoch: %d | LR: %.8f' % (epoch + 1, lr))

        # train for one epoch
        mode = 'pre_train' if epoch < args.pretr_epochs else 'train'
        print(mode+'ing...')
        run(best_acc, model, train_loader, mode, criterion, optimizer)\
        # evaluation
        mode = 'evaluate'
        _, nme_results, current_acc = run(best_acc, model, val_loader, mode, criterion, optimizer)

        is_best = current_acc > best_acc
        if is_best:
            print('new record:{}'.format(current_acc))
            best_acc = current_acc
            torch.save(model.state_dict(), os.path.join(args.checkpoint, 'checkpoint_best_{}.pth.tar'.format(epoch)))
        torch.save(model.state_dict(), os.path.join(args.checkpoint, 'checkpoint_{}.pth.tar'.format(epoch)))

def run(best_acc, model, data_loader, mode, criterion, optimizer):
    # self.epoch += 1
    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses_coord = AverageMeter()
    errs = AverageMeter()

    log_key = ['losses_coord', 'errs']
    log_info = dict.fromkeys(log_key)

    # normalized mean error results
    dataset_len = data_loader.dataset.__len__()
    nme_results = torch.Tensor(dataset_len, 1)

    def data2variable(inputs, target, meta):
        if mode in ['pre_train', 'train']:
            input_var = torch.autograd.Variable(inputs.cuda())
            target_var = [torch.autograd.Variable(target[i].cuda()) for i in range(len(target))]
            coord_var = torch.autograd.Variable(meta['tpts_inp'].cuda())
        else:
            with torch.no_grad():
                input_var = torch.autograd.Variable(inputs.cuda())
                target_var = [torch.autograd.Variable(target[i].cuda()) for i in
                          range(len(target))]
                coord_var = torch.autograd.Variable(meta['tpts_inp'].cuda())

        return input_var, target_var, coord_var

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
    for i, (inputs, target, meta) in enumerate(data_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input_var, target_var, coord_var = data2variable(inputs, target, meta)

        # run forward
        pred_coord = model.forward(input_var)
        #print(pred_coord[0])


        if mode == 'train':
            loss_coord = criterion(pred_coord, coord_var / 255)

            optimizer.zero_grad()
            loss_coord.backward()
            optimizer.step()

        else:
            loss_coord = criterion(pred_coord, coord_var / 255)

        pred_landmarks = (255 * pred_coord[:, 0:6, :].data).cpu()
        target_landmarks = meta['tpts_inp'][:, 0:6, :]
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
    parser.add_argument('--test-batch', default=16, type=int, metavar='N',
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

    main(parser.parse_args())
