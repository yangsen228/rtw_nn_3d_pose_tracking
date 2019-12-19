import os
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable

from model.model import LinearModel
from data.kinect_data import Kinect

import sys
sys.path.append('..')
from utils.helper import *

N_JOINT = 15
N_INPUT = 2 * N_JOINT * 3
N_HIDDEN = 2048
N_OUTPUT = N_JOINT * 3

EPOCH = 300
BATCH_SIZE = 256
LR = 0.001

TRAIN_SET = 'dl_enhanced_063_068_train'
TEST_SET = 'dl_enhanced_063_068_train'
DATA_PATH = 'data'

TEST_IMG = '063_068_train'
INPUT_DIR = '../rtw_tracking/data/processed/Kinect'
OUTPUT_DIR = 'png'

def main():
    # create model
    print(">>> creating model")
    model = LinearModel(N_INPUT, N_HIDDEN, N_OUTPUT, is_train_good=True)
    model = model.cuda()
    if isinstance(model, nn.Linear):
        nn.init.kaiming_normal(model.weight)
    print(model)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=8, verbose=True, min_lr=0.00001)
    loss_func = nn.MSELoss(size_average=True).cuda()

    # load dataset
    print(">>> loading data")
    train_loader = DataLoader(
            dataset=Kinect(TRAIN_SET, TEST_SET, DATA_PATH, is_train=True),
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=4,
            drop_last = True)
    test_loader = DataLoader(
            dataset=Kinect(TRAIN_SET, TEST_SET, DATA_PATH, is_train=False),
            batch_size=8000,
            shuffle=False,
            num_workers=4)
    print(">>> data loaded")

    # start training
    cudnn.benchmark = True
    for epoch in range(EPOCH):
        print("===========================")
        print(">>> epoch: {} | lr: {:.4f}".format(epoch+1, LR))

        train(train_loader, model, optimizer, loss_func)
        start = time.time()
        loss = test(test_loader, model, loss_func, epoch)
        end = time.time()
        print(">>> runtime: %f" % (end-start))
        scheduler.step(loss,epoch)
    print(">>> training finished")

    # save net parameters
    torch.save(model.state_dict(), 'model/model_parameters_%s.pkl' % TRAIN_SET)

def train(train_loader, model, optimizer, loss_func):
    for step, (x, y) in enumerate(train_loader):
        b_x = Variable(x).cuda()
        b_y = Variable(y).cuda()

        output = model(b_x)
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #print(">>> train loss: {:.4f}".format(loss.data))

def test(test_loader, model, loss_func, epoch):
    for step, (x, y) in enumerate(test_loader):
        b_x = Variable(x).cuda()
        b_y = Variable(y).cuda()

        output = model(b_x)
        loss_1 = loss_func(b_x[:,:(N_INPUT>>1)], b_y)
        loss_2 = loss_func(output, b_y)
        print(">>> rtw loss: {:.7f}".format(loss_1.cpu().data))  # loss of the results before neural network
        print(">>> new loss: {:.7f}".format(loss_2.cpu().data))  # loss of the results after neural network

        # visualization
        if epoch+1 >= EPOCH:
            p1 = output.cpu().detach().numpy()
            p2 = b_y.cpu().detach().numpy()
            print(p1.shape)
            #visualization3D(p1,p2)
            p1 = p1.reshape((len(p1), N_JOINT, 3))
            print(p1.shape)
            #for i in range(len(p1)):
            #    p1[i] = world2pixel(p1[i], 0) 
            #visualization2D(p1)

    return loss_2

def visualization2D(p1):
    imgs = np.load(os.path.join(INPUT_DIR, 'kinect_depth_images_%s.npy' % TEST_IMG))
    png_folder = 'nn_%s_[%s]/' % (TRAIN_SET, TEST_IMG)
    if not os.path.exists(os.path.join(OUTPUT_DIR, png_folder)):
        os.makedirs(os.path.join(OUTPUT_DIR, png_folder))
    for test_idx in range(len(imgs)):
        png_path = os.path.join(OUTPUT_DIR, png_folder, str(test_idx) + '.png')
        drawTest(imgs[test_idx], p1[test_idx], png_path)

def visualization3D(p1,p2):
    fig = plt.figure()
    ax = Axes3D(fig)
    plt.ion()
    for i in range(len(p1)):
        plt.cla()
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(0.5, 4.5)
        ax.set_zlim(-1.5, 1.5)
        ax.text(-1.4, 4, 1.4, '%d'%(i+1), color='red')

        # draw estimated joint points and skeleton (blue dots and lines)
        xdata1 = p1[i][0::3]
        ydata1 = -p1[i][1::3]
        zdata1 = p1[i][2::3]
        ax.scatter3D(xdata1, zdata1, ydata1, color='blue')
        xlimb1 = [xdata1[k] for k in [1,0,5,6,7]]
        ylimb1 = [ydata1[k] for k in [1,0,5,6,7]]
        zlimb1 = [zdata1[k] for k in [1,0,5,6,7]]
        ax.plot(xlimb1,zlimb1,ylimb1,color='blue')
        xlimb2 = [xdata1[k] for k in [0,2,3,4]]
        ylimb2 = [ydata1[k] for k in [0,2,3,4]]
        zlimb2 = [zdata1[k] for k in [0,2,3,4]]
        ax.plot(xlimb2,zlimb2,ylimb2,color='blue')
        xlimb3 = [xdata1[k] for k in [0,14,12,8,9]]
        ylimb3 = [ydata1[k] for k in [0,14,12,8,9]]
        zlimb3 = [zdata1[k] for k in [0,14,12,8,9]]
        ax.plot(xlimb3,zlimb3,ylimb3,color='blue')
        xlimb4 = [xdata1[k] for k in [14,13,10,11]]
        ylimb4 = [ydata1[k] for k in [14,13,10,11]]
        zlimb4 = [zdata1[k] for k in [14,13,10,11]]
        ax.plot(xlimb4,zlimb4,ylimb4,color='blue')

        # draw joint posints truth value (red dots)
        xdata2 = p2[i][0::3]
        ydata2 = -p2[i][1::3]
        zdata2 = p2[i][2::3]
        ax.scatter3D(xdata2, zdata2, ydata2, color='red')

        plt.pause(0.03)
    plt.ioff()
    plt.show()

                
if __name__ == '__main__':
    main()
