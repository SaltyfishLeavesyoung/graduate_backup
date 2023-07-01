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
from dataloader_cepha import Cephalometric
from mylogger import get_mylogger, set_logger_dir
from eval import Evaluater
from utils import to_Image, voting, visualize, make_dir

from loss import *

from matplotlib.patches import Ellipse
from torchvision.transforms import ToPILImage
from PIL import Image, ImageDraw

from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

to_PIL = ToPILImage()

def multivariate_gaussian(pos, mu, Sigma):
    """Return the multivariate Gaussian distribution on array pos."""

    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2*np.pi)**n * Sigma_det)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)

    return np.exp(-fac / 2) / N

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
    def __init__(self, logger, config, net=None, tag=None, train="", args=None):
        mode = "Test1" if train == "" else "Test2"
        dataset_1 = Cephalometric(config['dataset_pth'], mode)
        self.dataloader_1 = DataLoader(dataset_1, batch_size=1,
                                shuffle=False, num_workers=config['num_workers'])
        
        # # For anthor Testset, deprecated
        # dataset_2 = Cephalometric(config['dataset_pth'], 'Test2')
        # self.dataloader_2 = DataLoader(dataset_2, batch_size=1,
        #                         shuffle=False, num_workers=config['num_workers'])
        
        self.Radius = 40
        self.config = config
        self.args = args
        
        self.model = net 
        
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
    
        self.output_dir = make_dir(os.path.join(output_pth, 'results', "uncertain_1"))

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
            image_gt = visualize(img, landmark_list, highlight=25)
            image_pred = visualize(img, pred_landmark, [0, 2])
            image_gt.save("./paper_figure/"+str(ID)+'/gt.png')
            image_pred.save("./paper_figure/"+str(ID)+'/pred.png')
            #import ipdb; ipdb.set_trace()
            ID+=1

            logic_loss_list.append(logic_loss.cpu().item())
            # import ipdb; ipdb.set_trace()
        print(sum(logic_loss_list)/self.dataset.__len__())

    def test(self, net=None):
        self.evaluater.reset()
        if net is not None:
            self.model = net
        assert(hasattr(self, 'model'))
        ID = 0

        distance_list = dict()
        mean_list = dict()
        for i in range(19):
            distance_list[i] = list()
            mean_list[i] = list()

        for img, landmark, mask, guassian_mask, offset_y, offset_x, landmark_list in tqdm(self.dataloader_1):
            img, landmark, mask,  guassian_mask, offset_y, offset_x = img.cuda(), landmark.cuda(), mask.cuda(), guassian_mask.cuda(), offset_y.cuda(), offset_x.cuda()
            with torch.no_grad():
                heatmap, regression_y, regression_x, cov = self.model(img)
            
            pred_landmark = get_spatial_mean(heatmap)
            #pred_landmark = getTarget(heatmap.detach().cpu())
            #pred_landmark = voting(heatmap, regression_y, regression_x, self.Radius)
            sigma, det, sigma_inv = get_sigma(cov)
            pred_landmark = pred_landmark.cpu()
            pred = [pred_landmark[0,:,1], pred_landmark[0,:,0]]
            self.evaluater.record(pred, landmark_list)
            #self.evaluater.record(pred_landmark, landmark_list)
            
            #image_gt = visualize(img, landmark_list, [i for i in range(19)]) #blue

            if len(img.shape) == 3:
                img = img.unsqueeze(0)
            Radius = 3
            img = torch.true_divide((img - img.min()), (img.max() - img.min()))
            img = img.cpu()
            cyan = (0, 255, 255)
            yellow = (255, 255, 0)

            image = to_PIL(img[0]).convert("RGBA")
            _, hh, ww = img[0].shape
            draw = ImageDraw.Draw(image)

            for i in range(19):
                landmark = pred_landmark[0][i]
                draw.rectangle((landmark[0]-Radius, landmark[1]-Radius, \
                    landmark[0]+Radius, landmark[1]+Radius), fill=cyan)

                ell = Image.new(mode='RGBA', size=(100, 100))
                draw_ell = ImageDraw.Draw(ell)

                sig = sigma[0][i]

                vals, vecs = np.linalg.eigh(sig)
                order = vals.argsort()[::-1]
                vals  = vals[order]
                vecs  = vecs[:, order]

                theta = np.degrees(np.arctan(vecs[1, 0] / vecs[0, 0]))

                h, w  = 2 * 8 * np.sqrt(vals)

                draw_ell.ellipse((50 - w // 2, 50 - h // 2, 50 + w // 2, 50 + h // 2), outline ="yellow", width=3)
                #draw_ell.ellipse((30, 30, 70, 70), outline ="yellow", width=3)
                ell = ell.rotate(theta)
                image.paste(ell, (int(landmark[0] - 50), int(landmark[1] - 50)), ell)

                if i == 9:
                    N = 200
                    X = np.linspace(landmark[0]-7, landmark[0]+7, N)
                    Y = np.linspace(landmark[1]-7, landmark[1]+7, N)
                    X, Y = np.meshgrid(X, Y)

                    mu = np.array(landmark)
                    Sigma1 = np.array([[ np.sqrt(40.) , 0.], [0.,  np.sqrt(40.)]])
                    Sigma2 = np.array(sig)

                    pos = np.empty(X.shape + (2,))
                    pos[:, :, 0] = X
                    pos[:, :, 1] = Y

                    Z1 = multivariate_gaussian(pos, mu, Sigma1)
                    Z2 = multivariate_gaussian(pos, mu, Sigma2)

                    fig = plt.figure()
                    ax1 = fig.add_subplot(1,1,1,projection='3d')

                    ax1.invert_yaxis()

                    ax1.plot_surface(X, Y, Z1, rstride=3, cstride=3, linewidth=1, antialiased=True,
                cmap=cm.viridis)
                    ax1.view_init(55,-70)
                    ax1.set_xticks([])
                    ax1.set_yticks([])
                    ax1.set_zticks([])
                    ax1.set_xlabel('X')
                    ax1.set_ylabel('Y')
                    plt.savefig(os.path.join(self.output_dir, str(ID)+'gt_distribution.png'))

                    plt.close()
                    fig = plt.figure()

                    ax2 = fig.add_subplot(1,1,1,projection='3d')

                    ax2.invert_yaxis()

                    ax2.plot_surface(X, Y, Z2, rstride=3, cstride=3, linewidth=1, antialiased=True,
                cmap=cm.viridis)
                    ax2.view_init(55,-70)
                    ax2.set_xticks([])
                    ax2.set_yticks([])
                    ax2.set_zticks([])
                    ax2.set_xlabel('X')
                    ax2.set_ylabel('Y')
                    plt.savefig(os.path.join(self.output_dir, str(ID)+'pred_distribution.png'))  
                    plt.close()       
            
            
            image.save(os.path.join(self.output_dir, str(ID)+'_gt_uncertain.png'))
            
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

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    args = parser.parse_args()

    with open(os.path.join("runs", args.tag, args.config_file), "r") as f:
        config = yaml.load(f, Loader=yamlloader.ordereddict.CLoader)
    
    # Create Logger
    logger = get_mylogger()
        
    if args.iteration == '':
        iteration = config['last_epoch']
    else:
        iteration = int(args.iteration)
    
    # Load model
    net = torch.nn.DataParallel(UNet_Pretrained_Att(3, config['num_landmarks']).cuda())

    logger.info("Loading checkpoints from epoch {}".format(iteration))
    #checkpoints = torch.load(os.path.join(config['runs_dir'], \
    #                    "model_epoch_{}.pth".format(iteration)))
    #checkpoints = torch.load(os.path.join(config['runs_dir'], \
    #                    "model_best.pth"))
    checkpoints = torch.load('/home/yeziyang/Desktop/UNCERTAINTY/runs/2022-11-26-16:44:06.455524/model_epoch_299.pth')
    net.load_state_dict(checkpoints)
    #net = torch.nn.DataParallel(net)

    tester = Tester(logger, config, net, args.tag, args.train, args)
    if args.debug != '':
        tester.debug()
    else:
        tester.test()
