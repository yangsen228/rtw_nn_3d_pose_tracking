import os
import sys
import argparse

import numpy as np
import joblib
from sklearn.tree import DecisionTreeRegressor
from sklearn.cluster import MiniBatchKMeans
from multiprocessing import Process, Queue
from multiprocessing.pool import ThreadPool

from utils.helper import *
from time import time
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable

from nn.model.model import Linear
from nn.data.kinect_data import Kinect

###############################################################################
# Parser arguments
###############################################################################

parser = argparse.ArgumentParser(description='Random Tree Walks algorithm.')

# Loading options for the model and data
# parser.add_argument('--load-params', action='store_true',
#                     help='Load the parameters')
parser.add_argument('--load-model', action='store_true',
                    help='Load a pretrained model')
parser.add_argument('--load-test', action='store_true',
                    help='Run trained model on test set')

# Location of data directories
parser.add_argument('--input-dir', type=str, default='rtw_tracking/data/processed',
                    help='Directory of the processed input')
parser.add_argument('--dataset', type=str, default='Kinect', # NTU-RGBD, CAD-60
                    help='Name of the dataset to load')

# Location of output saved data directories
parser.add_argument('--model-dir', type=str, default='rtw_tracking/output/random-tree-walks/Kinect/models',
                    help='Directory of the saved model')
parser.add_argument('--preds-dir', type=str, default='rtw_tracking/output/random-tree-walks/Kinect/preds',
                    help='Directory to save predictions')
parser.add_argument('--png-dir', type=str, default='rtw_tracking/output/random-tree-walks/Kinect/png',
                    help='Directory to save prediction images')

# Training options
parser.add_argument('--seed', type=int, default=1111,
                    help='Random seed')
parser.add_argument('--shuffle', type=int, default=1,
                    help='Shuffle the data')
parser.add_argument('--multithread', action='store_true',
                    help='Train each joint on a separate threads')
# parser.add_argument('--num-threads', type=int, default=3,
#                     help='Number of threads to use to concurrently process joints.')

# Evaluation hyperparameters
parser.add_argument('--num-steps', type=int, default=16,
                    help='Number of steps during evaluation')
parser.add_argument('--step-size', type=int, default=5,
                    help='Step size (in cm) during evaluation')

# Output options
parser.add_argument('--make-png', action='store_true',
                    help='Draw predictions on top of inputs')

args = parser.parse_args()

###############################################################################
# Training hyperparameters
###############################################################################

# Train-test set 
SMALL_DATA_SIZE = 5000
TEST_SET = '035'
TRAIN_SET = 'dl_030_train'

# Dimension of each feature vector
NUM_FEATS = 500
MAX_FEAT_OFFSET = 200

# Number of clusters for K-Means regression
K = 20

# Mininum samples in leaf node
MIN_SAMPLES_LEAF = 400

# Dimension of neural network
N_INPUT = 15 * 3 
N_HIDDEN = 1024
N_OUTPUT = 15 * 3 

###############################################################################
# Dataset Constants
###############################################################################

# Depth image dimension
#H, W = 240, 320
H, W = 424, 512

# See https://help.autodesk.com/view/MOBPRO/2018/ENU/?guid=__cpp_ref__nui_image_camera_8h_source_html
C = 3.8605e-3 # NUI_CAMERA_DEPTH_NOMINAL_INVERSE_FOCAL_LENGTH_IN_PIXELS

###############################################################################
# Skeleton Constants
###############################################################################

# Number of joints in a skeleton
NUM_JOINTS = 15

# List of joint names
JOINT_NAMES = ['NECK (0)', 'HEAD (1)', \
                'LEFT SHOULDER (2)', 'LEFT ELBOW (3)', 'LEFT HAND (4)', \
                'RIGHT SHOULDER (5)', 'RIGHT ELBOW (6)', 'RIGHT HAND (7)', \
                'LEFT KNEE (8)', 'LEFT FOOT (9)', \
                'RIGHT KNEE (10)', 'RIGHT FOOT (11)', \
                'LEFT HIP (12)', \
                'RIGHT HIP (13)', \
                'TORSO (14)']

# Map from joint names to index
JOINT_IDX = {
    'NECK': 0,
    'HEAD': 1,
    'LEFT SHOULDER': 2,
    'LEFT ELBOW': 3,
    'LEFT HAND': 4,
    'RIGHT SHOULDER': 5,
    'RIGHT ELBOW': 6,
    'RIGHT HAND': 7,
    'LEFT KNEE': 8,
    'LEFT FOOT': 9,
    'RIGHT KNEE': 10,
    'RIGHT FOOT': 11,
    'LEFT HIP': 12,
    'RIGHT HIP': 13,
    'TORSO': 14,
}

###############################################################################
# Load dataset
###############################################################################

def load_dataset(processed_dir, is_mask=False, small_data=False):
    """Loads the depth images and joints from the processed dataset.

    Note that each joint is a coordinate of the form (im_x, im_y, depth_z).
    Each depth image is an H x W image containing depth_z values.

    depth_z values are in meters.

    @return:
        depth_images : depth images (N x H x W)
        joints : joint positions (N x NUM_JOINTS x 3)
    """
    logger.debug('Loading data from directory %s', processed_dir)

    # Load input and labels from numpy files
    depth_images = np.load(os.path.join(processed_dir, 'kinect_depth_images_%s_test.npy' % TEST_SET)) # N x H x W depth images
    joints = np.load(os.path.join(processed_dir, 'kinect_joints_%s_test.npy' % TEST_SET)) # N x NUM_JOINTS x 3 joint locations

    assert depth_images.shape[1] == H and depth_images.shape[2] == W, "Invalid dimensions for depth image"

    # Load and apply mask to the depth images
    if is_mask:
        depth_mask = np.load(os.path.join(processed_dir, 'depth_mask.npy')) # N x H x W depth mask
        depth_images = depth_images * depth_mask

    # Run experiments on random subset of data
    if small_data:
        random_idx = np.random.choice(depth_images.shape[0], SMALL_DATA_SIZE, replace=False)
        depth_images, joints = depth_images[random_idx], joints[random_idx]

    logger.debug('Data loaded: # data: %d', depth_images.shape[0])
    return depth_images, joints

###############################################################################
# Calculate features
###############################################################################

def compute_theta(num_feats=NUM_FEATS, max_feat_offset=MAX_FEAT_OFFSET):
    """Computes the theta for each skeleton.

    @params:
        max_feat_offset : the maximum offset for features (before divided by d)
        num_feats : the number of features of each offset point
    """
    logger.debug('Computing theta...')
    np.random.seed(0)
    # Compute the theta = (-max_feat_offset, max_feat_offset) for 4 coordinates (x1, x2, y1, y2)
    theta = np.random.randint(-max_feat_offset, max_feat_offset + 1, (4, num_feats)) # (4, num_feats)

    return theta

def get_features(img, q, z, theta):
    """Gets the feature vector for a single example.

    @params:
        img : depth image = (H x W)
        q : joint xyz position with some random offset vector
        z : z-value of body center
        theta : (-max_feat_offset, max_feat_offset) = (4, num_feats)
    """
    # Retrieve the (y, x) of the joint offset coordinates
    coor = q[:2][::-1] # coor: flip x, y -> y, x
    coor[0] = np.clip(coor[0], 0, H-1) # limits y between 0 and H
    coor[1] = np.clip(coor[1], 0, W-1) # limits x between 0 and W
    coor = np.rint(coor).astype(int) # rounds to nearest integer

    # Find z-value of joint offset by indexing into depth imag
    LARGE_NUM = 4500
    img[img == 0] = LARGE_NUM # no division by zero
    dq = z if (img[tuple(coor)] == LARGE_NUM) else img[tuple(coor)] / 1000.0 # initialize to LARGE_NUM

    # Normalize x theta by z-value
    x1 = np.clip(coor[1] + theta[0] / dq, 0, W-1).astype(int)
    x2 = np.clip(coor[1] + theta[2] / dq, 0, W-1).astype(int)

    # Normalize y theta by z-value
    y1 = np.clip(coor[0] + theta[1] / dq, 0, H-1).astype(int)
    y2 = np.clip(coor[0] + theta[3] / dq, 0, H-1).astype(int)

    # Get the feature vector as difference of depth-values
    feature = img[y1, x1] - img[y2, x2]
    return feature

###############################################################################
# Evaluate model
###############################################################################

def test_rtw(regressor, L, theta, qm0, img, body_center, num_steps=args.num_steps, step_size=args.step_size):
    """Test the model on a single example.
    """
    qm = np.zeros((num_steps + 1, 3))
    qm[0] = qm0
    joint_pred = np.zeros(3)

    for i in range(num_steps):
        body_center_z = body_center[2]
        f = get_features(img, qm[i], body_center_z, theta).reshape(1, -1) # flatten feature vector
        leaf_id = regressor.apply(f)[0]
        
        idx = np.random.choice(L[leaf_id][0].shape[0], p=L[leaf_id][0]) # L[leaf_id][0] = weights
        u = L[leaf_id][1][idx] # L[leaf_id][1] = centers

        qm[i+1] = qm[i] + u * step_size
        qm[i+1][0] = np.clip(qm[i+1][0], 0, W-1) # limit x between 0 and W
        qm[i+1][1] = np.clip(qm[i+1][1], 0, H-1) # limit y between 0 and H
        qm[i+1][2] = img[int(qm[i+1][1]), int(qm[i+1][0])] # index (y, x) into image for z position
        joint_pred += qm[i+1]

    joint_pred = joint_pred / num_steps
    return joint_pred

def test_nn(y_pred, y_nn, model, test_idx):
    y_pred[test_idx,:,2] = y_pred[test_idx,:,2] / 1000
    y_pred[test_idx] = pixel2world(y_pred[test_idx], C)

    x_nn = torch.from_numpy(y_pred[test_idx].flatten()).float()
    y_nn[test_idx] = (model(Variable(x_nn).cuda())).cpu().detach().numpy()

###############################################################################
# Visualize predictions
###############################################################################

def visualization(y_nn, y_test):
    fig = plt.figure()
    ax = Axes3D(fig)
    plt.ion()
    for i in range(len(y_nn)):
        plt.cla()
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(0.5, 4.5)
        ax.set_zlim(-1.5, 1.5)
        ax.text(-1.4, 4, 1.4, '%d'%(i+1), color='red')

        # draw estimated joint points and skeleton (blue dots and lines)
        xdata1 = (y_nn[i])[0::3]
        ydata1 = (y_nn[i])[1::3]
        zdata1 = (y_nn[i])[2::3]
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
        xdata2 = (y_test[i].flatten())[0::3]
        ydata2 = (y_test[i].flatten())[1::3]
        zdata2 = (y_test[i].flatten())[2::3]
        ax.scatter3D(xdata2, zdata2, ydata2, color='red')

        plt.pause(0.03)
    plt.ioff()
    plt.show()

###############################################################################
# Main
###############################################################################

def main():
    # Load dataset
    processed_dir = os.path.join(args.input_dir, args.dataset) # directory of saved numpy files
    X_test, y_test = load_dataset(processed_dir)

    num_test = X_test.shape[0]
        
    # Load rtw model
    logger.debug('\n------- Load rtw models ------- ')

    folder = '%s_%d_%d/' % (TRAIN_SET, K, MIN_SAMPLES_LEAF)
    regressors, Ls = {}, {}
    for joint_id in range(NUM_JOINTS):
        regressor_path = os.path.join(args.model_dir, folder, 'regressor' + str(joint_id) + '.pkl')
        L_path = os.path.join(args.model_dir, folder, 'L' + str(joint_id) + '.pkl')
        regressors[joint_id] = joblib.load(regressor_path)
        Ls[joint_id] = joblib.load(L_path)

    # Load nn model
    logger.debug('\n------- Load nn models ------- ')

    model = Linear(N_INPUT, N_HIDDEN, N_OUTPUT, is_train_good=False)
    model = model.cuda()
    model.load_state_dict(torch.load('nn/model/model_parameters.pkl'))
    loss_func = nn.MSELoss(reduction='mean').cuda()

    # Evaluate model
    logger.debug('\n------- Testing starts ------- ')
    logger.debug('\n------- Number of test ------- %d', num_test)

    cudnn.benchmark = True
    y_pred = np.zeros((num_test, NUM_JOINTS, 3))   # predicted results from rtw
    y_nn = np.zeros((num_test, NUM_JOINTS*3))      # predicted results from nn
    
    theta = compute_theta()

    t1 = time()
    previous_test_idx = -1
    for test_idx in range(num_test):
        if test_idx % 100 == 0:
            logger.debug('Processing image %d / %d', test_idx, num_test)
        for joint_id in range(NUM_JOINTS): 
            qm0 = y_test[test_idx][joint_id] if previous_test_idx == -1 else y_pred[previous_test_idx][joint_id]
            y_pred[test_idx][joint_id] = test_rtw(regressors[joint_id], Ls[joint_id], theta, qm0, X_test[test_idx], y_test[test_idx][JOINT_IDX['TORSO']])

        test_nn(y_pred, y_nn, model, test_idx)
        y_pred[test_idx] = y_nn[test_idx].reshape((NUM_JOINTS, 3))
        y_pred[test_idx] = world2pixel(y_pred[test_idx], C)
        y_pred[test_idx,:,2] = y_pred[test_idx,:,2] * 1000

        previous_test_idx += 1
    t2 = time()
    logger.debug('runtime: %f', (t2-t1)/num_test)

    # Calculate the loss
    for i in range(len(y_test)):
        y_test[i] = pixel2world(y_test[i], C)

    loss = loss_func(Variable(torch.from_numpy(y_nn).float()).cuda(), Variable(torch.from_numpy(y_test.reshape((num_test, NUM_JOINTS*3))).float()).cuda())
    logger.debug('new loss: {:.4f}'.format(loss.cpu().data))

    # Visualization
    visualization(y_nn, y_test)

if __name__=='__main__':
    main()
