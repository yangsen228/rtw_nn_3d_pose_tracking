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

from model.model import Linear
from data.kinect_data import Kinect

N_INPUT = 15 * 3
N_HIDDEN = 2048
N_OUTPUT = 15 * 3

BATCH_SIZE = 2700

TRAIN_SET = 'dl_051_059_train'
TEST_SET = 'dl_051_059_train'
DATA_PATH = 'data'

def main():
    # create model
    print(">>> creating model")
    model = Linear(N_INPUT, N_HIDDEN, N_OUTPUT, is_train_good=False)
    model = model.cuda()
    model.load_state_dict(torch.load('model/model_parameters_%s.pkl' % TRAIN_SET))
    print(model)

    loss_func = nn.MSELoss(size_average=True).cuda()

    # load dataset
    print(">>> loading data")
    test_loader = DataLoader(
            dataset=Kinect(TRAIN_SET, TEST_SET, DATA_PATH, is_train=False),
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=4)
    print(">>> data loaded")

    # start testing
    cudnn.benchmark = True
    start = time.time()
    output, b_y = test(test_loader, model, loss_func)
    end = time.time()
    print(">>> runtime: %f" % (end-start))

    # visualization
    visualization(output, b_y)

def test(test_loader, model, loss_func):
    for step, (x, y) in enumerate(test_loader):
        b_x = Variable(x).cuda()
        b_y = Variable(y).cuda()

        output = model(b_x)
        loss_1 = loss_func(b_x, b_y)
        loss_2 = loss_func(output, b_y)
        print(">>> rtw loss: {:.4f}".format(loss_1.cpu().data))  # loss of the results before neural network
        print(">>> new loss: {:.4f}".format(loss_2.cpu().data))  # loss of the results after neural network

        return output, b_y

def visualization(output, b_y):
    p1 = output.cpu().detach().numpy()
    p2 = b_y.cpu().detach().numpy()
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
        xdata1 = -p1[i][0::3]
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
        xdata2 = -p2[i][0::3]
        ydata2 = -p2[i][1::3]
        zdata2 = p2[i][2::3]
        ax.scatter3D(xdata2, zdata2, ydata2, color='red')

        plt.pause(0.03)
    plt.ioff()
    plt.show()

                
if __name__ == '__main__':
    main()
