import errno
import logging
import os
import os.path
import shutil
import sys
from logging import INFO, DEBUG, WARNING, CRITICAL, ERROR
from datetime import datetime
from six.moves import input
from termcolor import colored
from PIL import Image
from torchvision.transforms import ToPILImage
from PIL import Image, ImageDraw, ImageFont
from multiprocessing import Process, Queue
from pathlib import Path
import torch
import numpy as np
import copy
'''
LOGGER
REFERENCE: https://github.com/qsyao/attack_landmark_detection/blob/master/mylogger.py
'''

INFO = INFO
DEBUG = DEBUG
WARNING = WARNING
CRITICAL = CRITICAL
ERROR = ERROR

def get_mylogger(level=logging.INFO, flag="MyLogger", log_dir=None, action='k', file_name='log.log'):
    logger = logging.getLogger(flag)
    logger.propagate = False
    logger.setLevel(level)
    handler = logging.StreamHandler()
    handler.setFormatter(_MyFormatter(datefmt='%m%d %H:%M:%S'))
    logger.addHandler(handler)
    if log_dir is not None:
        set_logger_dir(logger, log_dir, action, file_name)
    return logger

class _MyFormatter(logging.Formatter):
    def format(self, record):
        date = colored('[%(asctime)s @%(filename)s:%(lineno)d]', 'green')
        msg = '%(message)s'
        if record.levelno == logging.WARNING:
            fmt = date + ' ' + colored('WRN', 'yellow', attrs=['blink']) + ' ' + msg
        elif record.levelno == logging.ERROR :
            fmt = date + ' ' + colored('ERR', 'red', attrs=['blink']) + ' ' + msg
        elif record.levelno == logging.DEBUG:
            fmt = date + ' ' + colored('DBG', 'magenta', attrs=['bold']) + ' ' + msg
        elif record.levelno == logging.INFO:
            fmt = date + ' ' + colored('INFO', 'cyan', attrs=['bold']) + ' ' + msg
        elif record.levelno == logging.CRITICAL:
            fmt = date + ' ' + colored('CRITICAL', 'red', attrs=['blink', 'underline']) + ' ' + msg
        else:
            fmt = date + ' ' + msg
        if hasattr(self, '_style'):
            # Python3 compatibility
            self._style._fmt = fmt
        self._fmt = fmt
        return super(_MyFormatter, self).format(record)

def _get_time_str():
    return datetime.now().strftime('%m%d-%H%M%S')

def _set_file(logger, path):
    if os.path.isfile(path):
        backup_name = path + '.' + _get_time_str()
        shutil.move(path, backup_name)
        logger.info("Existing log file '{}' backuped to '{}'".format(path, backup_name))  # noqa: F821
    hdl = logging.FileHandler(
        filename=path, encoding='utf-8', mode='w')
    hdl.setFormatter(_MyFormatter(datefmt='%m%d %H:%M:%S'))

    _FILE_HANDLER = hdl
    logger.addHandler(hdl)
    logger.info("Argv: " + ' '.join(sys.argv))


def set_logger_dir(logger, dirname='log', action='k', file_name='log.log'):
    """
    Set the directory for global logging.
    Args:
        dirname(str): log directory
        action(str): an action of ["k","d","q"] to be performed
            when the directory exists. Will ask user by default.
                "d": delete the directory. Note that the deletion may fail when
                the directory is used by tensorboard.
                "k": keep the directory. This is useful when you resume from a
                previous training and want the directory to look as if the
                training was not interrupted.
                Note that this option does not load old models or any other
                old states for you. It simply does nothing.
                "b" : copy the old dir
                "n" : New an new dir by time
    """
    def dir_nonempty(dirname):
        # If directory exists and nonempty (ignore hidden files), prompt for action
        return os.path.isdir(dirname) and len([x for x in os.listdir(dirname) if x[0] != '.'])

    if dir_nonempty(dirname):
        if action == 'b':
            backup_name = dirname + _get_time_str()
            shutil.move(dirname, backup_name)
            logger.info("Directory '{}' backuped to '{}'".format(dirname, backup_name))  # noqa: F821
        elif action == 'd':
            shutil.rmtree(dirname, ignore_errors=True)
            if dir_nonempty(dirname):
                shutil.rmtree(dirname, ignore_errors=False)
        elif action == 'n':
            dirname = dirname + _get_time_str()
            logger.info("Use a new log directory {}".format(dirname))  # noqa: F821
        elif action == 'k':
            pass
        else:
            raise OSError("Directory {} exits!".format(dirname))
    mkdir_p(dirname)
    _set_file(logger, os.path.join(dirname, file_name))

def mkdir_p(dirname):
    """ Like "mkdir -p", make a dir recursively, but do nothing if the dir exists
    Args:
        dirname(str):
    """
    assert dirname is not None
    if dirname == '' or os.path.isdir(dirname):
        return
    try:
        os.makedirs(dirname)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise e

'''
OTHER FUNC
REFERENCE: https://github.com/qsyao/attack_landmark_detection/blob/master/utils.py
'''
to_PIL = ToPILImage()

def pred2gt(pred):
    if len(pred) != 2: 
        return pred
    # Convert predicts to GT format
    # pred :  list[ c(y) ; c(x) ]
    out = list()
    for i in range(pred[0].shape[-1]):
        out.append([int(pred[1][i]), int(pred[0][i])])
    return out

def distance(pred, landmark, k):
    diff = np.zeros([2], dtype=float) # y, x
    diff[0] = abs(pred[0] - landmark[k][1]) * 3.0
    diff[1] = abs(pred[1] - landmark[k][0]) * 3.0
    Radial_Error = np.sqrt(np.power(diff[0], 2) + np.power(diff[1], 2))
    Radial_Error *= 0.1
    # if Radial_Error > 40:
    #     return Radial_Error
    return 0

def to_Image(tensor, show=None, normalize=False):
    if normalize:
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    tensor = tensor.cpu()
    image = to_PIL(tensor)
    if show:
        image.save(show + ".png")
    return image

def voting_channel(k, heatmap, regression_y, regression_x,\
     Radius, spots_y, spots_x, queue, num_candi):
    n, c, h, w = heatmap.shape

    score_map = np.zeros([h, w], dtype=int)
    for i in range(num_candi):
        vote_x = regression_x[0, k, int(spots_y[0, k, i]), int(spots_x[0, k, i])]
        vote_y = regression_y[0, k, int(spots_y[0, k, i]), int(spots_x[0, k, i])]
        vote_x = spots_x[0, k, i] + int(vote_x * Radius)
        vote_y = spots_y[0, k, i] + int(vote_y * Radius)
        if vote_x < 0 or vote_x >= w or vote_y < 0 or vote_y >= h:
            # Outbounds
            continue
        score_map[int(vote_y), int(vote_x)] += 1
    score_map = score_map.reshape(-1)
    candidataces = score_map.argsort()[-10:]
    candidataces_x = candidataces % w
    candidataces_y = candidataces / w
    # import ipdb; ipdb.set_trace()
    # Print Big mistakes
    # gg = distance([candidataces_y[-1], candidataces_x[-1]], gt, k)
    # if gg:
    #     print("Landmark {} RE {}".format(k, gg))
    #     print(candidataces_y.astype(int))
    #     print(candidataces_x.astype(int))
    #     print(gt[k][1], gt[k][0])
    queue.put([k, score_map.argmax()])

def voting(heatmap, regression_y, regression_x, Radius):
    # n = batchsize = 1
    heatmap = heatmap.cpu()
    regression_x, regression_y = regression_x.cpu(), regression_y.cpu()
    n, c, h, w = heatmap.shape
    assert(n == 1)

    num_candi = int(3.14 * Radius * Radius)

    # Collect top num_candi points
    score_map = torch.zeros(n, c, h, w, dtype=torch.int16)
    spots_heat, spots = heatmap.view(n, c, -1).topk(dim=-1, \
        k=num_candi)
    spots_y = spots / w
    spots_x = spots % w

    # for mutiprocessing debug
    # voting_channel(0, heatmap,\
    #         regression_y, regression_x, Radius, spots_y, spots_x, None, None, None)
            
    # MutiProcessing
    # Each process votes for one landmark
    process_list = list()
    queue = Queue()
    for k in range(c):
        process = Process(target=voting_channel, args=(k, heatmap,\
            regression_y, regression_x, Radius, spots_y, spots_x, queue, num_candi))
        process_list.append(process)
        process.start()
    for process in process_list:
        process.join()
    
    landmark = np.zeros([c], dtype=int)
    for i in range(c):
        out = queue.get()
        landmark[out[0]] = out[1]

        # This is for guassian mask
        # landmark[i] = heatmap[0][i].view(-1).max(0)[1]
    landmark_y = landmark / w
    landmark_x = landmark % w
    return [landmark_y.astype(int), landmark_x]

def visualize(img, landmarks, red_marks=None, blue_marks=None, ratio=0.01, highlight=55):
    # img : tensor [1, 3, h, w]
    if len(img.shape) == 3:
        img = img.unsqueeze(0)
    h, w = img.shape[-2], img.shape[-1]
    Radius_Base = int(min(h, w) * ratio)
    img = (img - img.min()) / (img.max() - img.min())
    img = img.cpu()
    num_landmarks = len(pred2gt(landmarks))
    # Draw Landmark
    # Green [0, 1, 0] Red [1, 0, 0]
    Channel_R = {'Red': 1, 'Green': 0, 'Blue': 0}
    Channel_G = {'Red': 0, 'Green': 1, 'Blue': 0}
    Channel_B = {'Red': 0, 'Green': 0, 'Blue': 1}
    red = (255, 0, 0)
    green = (0, 255, 0)
    blue = (0, 0, 255)
    yellow = (255, 255, 0)

    if blue_marks is not None:
        landmarks = landmarks + blue_marks
    landmarks = pred2gt(landmarks)

    # for i, landmark in enumerate(landmarks):
    #     if red is not None and i in red: color = 'Red'
    #     elif i >= num_landmarks: color = 'Blue'
    #     else: color = 'Green'
    #     img[0][0][landmark[1]-Radius:landmark[1]+Radius,\
    #         landmark[0]-Radius:landmark[0]+Radius] = Channel_R[color]
    #     img[0][1][landmark[1]-Radius:landmark[1]+Radius, \
    #         landmark[0]-Radius:landmark[0]+Radius] = Channel_G[color]
    #     img[0][2][landmark[1]-Radius:landmark[1]+Radius, \
    #         landmark[0]-Radius:landmark[0]+Radius] = Channel_B[color]
    image = to_PIL(img[0])
    draw = ImageDraw.Draw(image)
    for i, landmark in enumerate(landmarks):
        id = i
        if red_marks is not None and i in red_marks: 
            color = red
            Radius = Radius_Base * 2
        elif i >= num_landmarks: 
            color = green
            id = landmark[2]
            Radius = Radius_Base * 2
        else: 
            color = green
            Radius = Radius_Base * 1.5
        if i % 19 != highlight:
            draw.rectangle((landmark[0]-Radius, landmark[1]-Radius, \
                landmark[0]+Radius, landmark[1]+Radius), fill=color)
        else:
            draw.polygon((landmark[0], landmark[1]-Radius*1.5, landmark[0]-Radius*1.5, landmark[1]+Radius*1.5,\
                landmark[0]+Radius*1.5, landmark[1]+Radius*1.5), fill=color)
        font = ImageFont.truetype('./ARIAL.TTF' , size=18)
        draw.text((landmark[0]-Radius, landmark[1]-Radius), str(id%19 + 1), \
            fill=red, font=font)
    
    # # Highlight
    # landmark = landmarks[highlight]
    # draw.rectangle((landmark[0]-Radius, landmark[1]-Radius, \
    #     landmark[0]+Radius, landmark[1]+Radius), fill=red)
    
    return image

def make_dir(pth):
    dir_pth = Path(pth)
    if not dir_pth.exists():
        dir_pth.mkdir()
    return pth

def getAdjMatrix(k=5, path = '/home/yzy/Desktop/IEEE ISBI 2015 Challenge'):
    
    landmark_list = list()
    for j in range(400):
        temp_list = list()
        with open(os.path.join(path, '400_junior', str(j+1).zfill(3) + '.txt')) as f1:
            with open(os.path.join(path, '400_senior', str(j+1).zfill(3) + '.txt')) as f2:
                for i in range(19):
                    landmark1 = f1.readline().split()[0].split(',')
                    landmark2 = f2.readline().split()[0].split(',')
                    landmark = [int(0.5*(int(landmark1[i]) + int(landmark2[i]))) for i in range(len(landmark1))]
                    temp_list.append(landmark)
        landmark_list.append(temp_list)

    landmark_list = np.array(landmark_list)
    # landmark_list.shape: [400, 19, 2]
    lx = landmark_list[:, :, 0]
    ly = landmark_list[:, :, 1]
    cx = np.corrcoef(lx.T)
    cy = np.corrcoef(ly.T)
    c = 0.5 * (np.abs(cx) + np.abs(cy))
    adj = np.zeros((19, 19))
    idx = np.argpartition(c, [i for i in range(18)])[:, -(k+1):]
    for i in range(19):
        adj[i][idx[i]] = 1
    return adj

if __name__ == "__main__":
    img = Image.open('./dataset/RawImage/TrainingData/001.bmp').convert('RGB')
    keypoint = list()
    with open(os.path.join('./dataset/AnnotationsByMD/400_junior', '001.txt')) as f1:
        with open(os.path.join('./dataset/AnnotationsByMD/400_senior', '001.txt')) as f2:
            for i in range(19):
                landmark1 = f1.readline().split()[0].split(',')
                landmark2 = f2.readline().split()[0].split(',')
                landmark = [int(0.5*(int(landmark1[i]) + int(landmark2[i]))) for i in range(len(landmark1))]
                keypoint.append(landmark)
    Radius = 10
    for i in range(10):
        temp_img = copy.deepcopy(img)
        draw = ImageDraw.Draw(temp_img)
        adj = getAdjMatrix(k=i+1)
        for j in range(19):
            draw.rectangle((keypoint[j][0]-Radius, keypoint[j][1]-Radius, \
                keypoint[j][0]+Radius, keypoint[j][1]+Radius), fill=(255, 0, 0))
        for j in range(19):
            for k in range(19):
                if j != k and adj[j][k] == 1:
                    draw.line((keypoint[j][0], keypoint[j][1], keypoint[k][0], keypoint[k][1]), 'cyan', 10)
        temp_img.save(os.path.join('./visual_adj', str(i+1)+'_adj.png'))
        print(adj)
        break










