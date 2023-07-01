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

from network import  UNet_Pretrained_Att
# from dataloader_cepha import Cephalometric
from dataloader_hand import Cephalometric
from mylogger import get_mylogger, set_logger_dir
from test import Tester
from loss import *
from torch.nn import MSELoss, BCELoss

from torch.utils.tensorboard import SummaryWriter 
from datetime import datetime, timedelta

from eval import Evaluater

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
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4"

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
    
    net = UNet_Pretrained_Att(3, config['num_landmarks'])

    net = torch.nn.DataParallel(net)
    net = net.cuda()
    logger.info(net)

    if config['pretrain']:
        net.load_state_dict(torch.load(config['pretrain_path']))

    optimizer = optim.Adam(params=net.parameters(), \
        lr=config['learning_rate'], betas=(0.9,0.999), eps=1e-08, weight_decay=1e-4)
    
    scheduler = StepLR(optimizer, config['decay_step'], gamma=config['decay_gamma'])

    # loss
    loss_heatmap = MSELoss()
    loss_cov = MahLoss()
    reg = SigmaReg()
    loss_regression_fn = L1Loss_offset
    loss_logic_fn = BCELoss()

    # Tester
    tester = Tester(logger, config, tag=args.tag)

    min_loss = 1145141919

    for epoch in range(config['num_epochs']):
        loss_list_train = 0
        net.train()
        i = 0
        for img, landmark, mask, guassian_mask, offset_y, offset_x, _ in tqdm(dataloader):
            img, landmark, mask,  guassian_mask, offset_y, offset_x = img.cuda(), landmark.cuda(), mask.cuda(), guassian_mask.cuda(), offset_y.cuda(), offset_x.cuda()

            heatmap, regression_y, regression_x, cov = net(img)
            #heatmap, cov = net(img)
            sigma, det, sigma_inv = get_sigma(cov)
            sigma, det, sigma_inv = sigma.cuda(), det.cuda(), sigma_inv.cuda()
            coord_from_heatmap = get_spatial_mean(heatmap)
            #cov_loss = loss_cov(coord_from_heatmap, landmark, sigma_inv)
            logic_loss = loss_logic_fn(heatmap, guassian_mask)
            regression_loss_y = loss_regression_fn(regression_y, offset_y, mask)
            regression_loss_x = loss_regression_fn(regression_x, offset_x, mask)
            cov_loss_2 = MSELoss()(coord_from_heatmap, landmark.float())
            
            sigma_reg = reg(det)
            #loss_htm = nll_across_batch(heatmap, channels)
            #loss = cov_loss_2 + 0.1 * sigma_reg + (regression_loss_x + regression_loss_y) + logic_loss * 3
            loss = cov_loss_2 + (regression_loss_x + regression_loss_y) + logic_loss * 3
            #loss = cov_loss + 0.1 * sigma_reg
            #loss = loss_htm
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_list_train += loss.detach().cpu().item()         
        logger.info("Epoch {} train_loss {} ".format(epoch, loss_list_train / dataset.__len__()))
        writer.add_scalar('train/train_loss', loss_list_train / dataset.__len__(), epoch)

        loss_list_test = 0
        net.eval()
        tester.evaluater.reset()

        for img, landmark, mask, guassian_mask, offset_y, offset_x, landmark_list in tqdm(tester.dataloader_1):
            with torch.no_grad():
                img, landmark, mask,  guassian_mask, offset_y, offset_x = img.cuda(), landmark.cuda(), mask.cuda(), guassian_mask.cuda(), offset_y.cuda(), offset_x.cuda()

                heatmap, regression_y, regression_x, cov = net(img)
                #heatmap, cov = net(img)
                sigma, det, sigma_inv = get_sigma(cov)
                sigma, det, sigma_inv = sigma.cuda(), det.cuda(), sigma_inv.cuda()
                #print(sigma)
                #print(det)
                coord_from_heatmap = get_spatial_mean(heatmap)
                #cov_loss = loss_cov(coord_from_heatmap, landmark, sigma_inv)
                logic_loss = loss_logic_fn(heatmap, guassian_mask)
                regression_loss_y = loss_regression_fn(regression_y, offset_y, mask)
                regression_loss_x = loss_regression_fn(regression_x, offset_x, mask)
                cov_loss_2 = MSELoss()(coord_from_heatmap, landmark.float())
                #sigma_reg = reg(det)
                #loss_htm = nll_across_batch(heatmap, channels)
                #loss = cov_loss_2 + 0.1 * sigma_reg + (regression_loss_x + regression_loss_y) + logic_loss * 3
                loss = cov_loss_2 + (regression_loss_x + regression_loss_y) + logic_loss * 3
                #loss = cov_loss + loss_htm + 0.1 * sigma_reg
                #loss = loss_htm
                    
                loss_list_test += loss.detach().cpu().item()

                coord_from_heatmap = coord_from_heatmap.cpu()
                pred = [coord_from_heatmap[0,:,1], coord_from_heatmap[0,:,0]]
                tester.evaluater.record(pred, landmark_list)

        tester.evaluater.cal_metrics()

        logger.info("Epoch {} test_loss {}".format(epoch, loss_list_test / tester.dataset.__len__()))
        writer.add_scalar('test/test_loss', loss_list_test / tester.dataset.__len__(), epoch)

        #neta, netb = net.module.resetab()
        #logger.info("alpha: {} beta: {}".format(neta, netb))

        # save model
        if (epoch + 1) % config['save_seq'] == 0:
            logger.info(runs_dir + "/model_epoch_{}.pth".format(epoch))
            torch.save(net.state_dict(), runs_dir + "/model_epoch_{}.pth".format(epoch))
    
            config['last_epoch'] = epoch

        if loss_list_test / tester.dataset.__len__() < min_loss:
            min_loss = loss_list_test / tester.dataset.__len__()
            logger.info(runs_dir + "/model_epoch_{}.pth".format(epoch))
            torch.save(net.state_dict(), runs_dir + "/model_best.pth")
            config['best_epoch'] = epoch

        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()

        # dump yaml
        with open(runs_dir + "/config.yaml", "w") as f:
            yaml.dump(config, f)

    # # Test
    # tester.test(net)
