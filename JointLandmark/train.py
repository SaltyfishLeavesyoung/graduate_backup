import argparse
import os
from pathlib import Path
import yaml
import yamlloader
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch import optim
from torch.nn import functional as F
from torch.nn import BCELoss, SmoothL1Loss, L1Loss, CrossEntropyLoss

from network import  UNet_Pretrained_Att
from dataloader import Cephalometric
from mylogger import get_mylogger, set_logger_dir

from torch.utils.tensorboard import SummaryWriter 
from datetime import datetime, timedelta
from util import getAdjMatrix

import numpy as np


TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now() + timedelta(hours=8))
writer = SummaryWriter('./logs/gcn_k3_aug/'+ TIMESTAMP)


def L1Loss_offset(pred, gt, mask=None):
    # L1 Loss for offset map
    assert(pred.shape == gt.shape)
    gap = pred - gt
    distence = gap.abs()
    if mask is not None:
        # Caculate grad of the area under mask
        distence = distence * mask
    return distence.sum() / mask.sum()
    # return distence.mean()

if __name__ == "__main__":
    # Parse command line options
    parser = argparse.ArgumentParser(description="Train Unet landmark detection network")
    parser.add_argument("--tag", default='', help="name of the run")
    parser.add_argument("--config_file", default="config.yaml", help="default configs")
    args = parser.parse_args()
 
    # Load yaml config file
    with open(args.config_file) as f:
        config = yaml.load(f, Loader=yamlloader.ordereddict.CLoader)
    
    # Create Logger
    logger = get_mylogger()
    logger.info(config)

    # Create runs dir
    tag = str(datetime.now()).replace(' ', '-') if args.tag == '' else args.tag
    runs_dir = "./runs/" + tag
    runs_path = Path(runs_dir)
    config['runs_dir'] = runs_dir
    if not runs_path.exists():
        runs_path.mkdir()
    set_logger_dir(logger, runs_dir)

    dataset = Cephalometric(config['train_pth'])
    dataloader = DataLoader(dataset, batch_size=config['batch_size'],
                                shuffle=True, num_workers=config['num_workers'])

    dataset_1 = Cephalometric(config['test_pth'])
    dataloader_1 = DataLoader(dataset_1, batch_size=1,
                                shuffle=False, num_workers=config['num_workers'])
    
    adj = getAdjMatrix(k=config['adj'])
    net = UNet_Pretrained_Att(3, config['num_landmarks'], adj)

    net = torch.nn.DataParallel(net)
    net = net.cuda()
    logger.info(net)

    optimizer = optim.Adam(params=net.parameters(), \
        lr=config['learning_rate'], betas=(0.9,0.999), eps=1e-08, weight_decay=1e-4)
    
    scheduler = StepLR(optimizer, config['decay_step'], gamma=config['decay_gamma'])

    # loss
    loss_logic_fn = BCELoss()
    loss_regression_fn = L1Loss_offset
    loss_z_fn = SmoothL1Loss()

    min_loss = 1145141919

    for epoch in range(config['num_epochs']):
        loss_list_train = list()
        net.train()
        i = 0
        for img, mask, guassian_mask, z_list, z_list_one_hot, offset_y, offset_x, _, _, _ in tqdm(dataloader):
            img, mask, z_list, z_list_one_hot, offset_y, offset_x, guassian_mask = img.cuda(), mask.cuda(), \
                z_list.cuda(), z_list_one_hot.cuda(), offset_y.cuda(), offset_x.cuda(), guassian_mask.cuda()

            heatmap, predict_z, regression_y, regression_x = net(img)
            logic_loss = loss_logic_fn(heatmap, guassian_mask)
            loss_z = loss_z_fn(predict_z, z_list_one_hot)
            regression_loss_y = loss_regression_fn(regression_y, offset_y, mask)
            regression_loss_x = loss_regression_fn(regression_x, offset_x, mask)
            loss = regression_loss_x + regression_loss_y + loss_z * 0.1 + logic_loss * config['lambda']
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_list_train.append(loss.cpu().item())         
        logger.info("Epoch {} train_loss {} ".format(epoch, sum(loss_list_train) / dataset.__len__()))
        writer.add_scalar('train/train_loss', sum(loss_list_train) / dataset.__len__(), epoch)

        loss_list_test = list()
        net.eval()
        for img, mask, guassian_mask, z_list, z_list_one_hot, offset_y, offset_x, _, _, _ in tqdm(dataloader_1):
            img, mask, z_list, z_list_one_hot, offset_y, offset_x, guassian_mask = img.cuda(), mask.cuda(), \
                z_list.cuda(), z_list_one_hot.cuda(), offset_y.cuda(), offset_x.cuda(), guassian_mask.cuda()
            with torch.no_grad():
                heatmap, predict_z, regression_y, regression_x = net(img)
                
                logic_loss = loss_logic_fn(heatmap, guassian_mask)
                loss_z = loss_z_fn(predict_z, z_list_one_hot)
                regression_loss_y = loss_regression_fn(regression_y, offset_y, mask)
                regression_loss_x = loss_regression_fn(regression_x, offset_x, mask)
                
                loss =  regression_loss_x + regression_loss_y + loss_z + logic_loss * config['lambda']
                    
            loss_list_test.append(loss.cpu().item())

        logger.info("Epoch {} test_loss {}".format(epoch, sum(loss_list_test) / dataset_1.__len__()))
        writer.add_scalar('test/test_loss', sum(loss_list_test) / dataset_1.__len__(), epoch)

        #neta, netb = net.module.resetab()
        #logger.info("alpha: {} beta: {}".format(neta, netb))

        # save model
        if (epoch + 1) % config['save_seq'] == 0:
            logger.info(runs_dir + "/model_epoch_{}.pth".format(epoch))
            torch.save(net.state_dict(), runs_dir + "/model_epoch_{}.pth".format(epoch))
    
            config['last_epoch'] = epoch

        if sum(loss_list_test) / dataset_1.__len__() < min_loss:
            min_loss = sum(loss_list_test) / dataset_1.__len__()
            logger.info(runs_dir + "/model_epoch_{}.pth".format(epoch))
            torch.save(net.state_dict(), runs_dir + "/model_best.pth")
            config['best_epoch'] = epoch

        # dump yaml
        with open(runs_dir + "/config.yaml", "w") as f:
            yaml.dump(config, f)
