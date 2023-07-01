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
from torch.nn import BCELoss, MSELoss
from torch.nn import L1Loss

from network import  global_net, local_net
from dataloader import Cephalometric
from mylogger import get_mylogger, set_logger_dir
from test import Tester

from torch.utils.tensorboard import SummaryWriter 
from datetime import datetime, timedelta
from spp import spp
from loss import get_spatial_mean

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
    else:
        return distence.mean()

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6"

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

    dataset = Cephalometric(config['dataset_pth'], 'Train')
    dataloader = DataLoader(dataset, batch_size=config['batch_size'],
                                shuffle=True, num_workers=config['num_workers'])
    
    g_net = global_net(3, config['num_landmarks'])

    g_net = torch.nn.DataParallel(g_net)
    g_net = g_net.cuda()
    #logger.info(net)

    global_optimizer = optim.Adam(params=g_net.parameters(), \
        lr=config['learning_rate'], betas=(0.9,0.999), eps=1e-08, weight_decay=1e-4)
    
    global_scheduler = StepLR(global_optimizer, config['decay_step'], gamma=config['decay_gamma'])

    # loss
    loss_logic_fn = BCELoss()
    loss_regression_fn = L1Loss_offset

    # Tester
    tester = Tester(logger, config, tag=args.tag)

    min_loss_global = 1145141919

    for epoch in range(config['num_epochs']):
        loss_list_train = list()
        g_net.train()
        for img, landmark, mask, guassian_mask, offset_y_local, offset_x_local, _, _, _ in tqdm(dataloader):
            img, landmark, mask, offset_y_local, offset_x_local, guassian_mask = img.cuda(), landmark.cuda(), mask.cuda(), offset_y_local.cuda(), offset_x_local.cuda(), guassian_mask.cuda()

            heatmap = g_net(img)

            logic_loss = loss_logic_fn(heatmap, guassian_mask)

            coord_from_heatmap = get_spatial_mean(heatmap)
            cov_loss = MSELoss()(coord_from_heatmap, landmark.float())
            loss_global = logic_loss * config['lambda'] + cov_loss

            global_optimizer.zero_grad()
            loss_global.backward()
            global_optimizer.step()

        g_net.eval()
        loss_list_test_global = list()
        for img, landmark, mask, guassian_mask, offset_y_local, offset_x_local, _, _, _ in tqdm(tester.dataloader_1):
            img, landmark, mask, offset_y_local, offset_x_local, guassian_mask = img.cuda(), landmark.cuda(), mask.cuda(), offset_y_local.cuda(), offset_x_local.cuda(), guassian_mask.cuda()
            with torch.no_grad():
                heatmap = g_net(img)
                
                logic_loss = loss_logic_fn(heatmap, guassian_mask)

                coord_from_heatmap = get_spatial_mean(heatmap)
                cov_loss = MSELoss()(coord_from_heatmap, landmark.float())
                loss_global = logic_loss * config['lambda'] + cov_loss

                loss_list_test_global.append(loss_global.cpu().item())

        # save model
        '''if (epoch + 1) % config['save_seq'] == 0:
            logger.info(runs_dir + "/model_epoch_{}.pth".format(epoch))
            torch.save(net.state_dict(), runs_dir + "/model_epoch_{}.pth".format(epoch))
    
            config['last_epoch'] = epoch'''

        if sum(loss_list_test_global) / tester.dataset.__len__() < min_loss_global:
            min_loss_global = sum(loss_list_test_global) / tester.dataset.__len__()
            logger.info(runs_dir + "/model_global_epoch_{}.pth".format(epoch))
            torch.save(g_net.state_dict(), runs_dir + "/model_best_global.pth")
            config['best_epoch_global'] = epoch

        # dump yaml
        with open(runs_dir + "/config.yaml", "w") as f:
            yaml.dump(config, f)

    # # Test
    # tester.test(net)
