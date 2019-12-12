import os
import sys
import argparse

import numpy as np
import joblib
from sklearn.tree import DecisionTreeRegressor
from sklearn.cluster import MiniBatchKMeans
from sklearn import preprocessing
from multiprocessing import Process, Queue, Array
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
parser.add_argument('--num-steps', type=int, default=8,
                    help='Number of steps during evaluation')
parser.add_argument('--step-size', type=int, default=2,
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
TEST_SET = '070'
TRAIN_SET = 'dl_063_068_train'

# Dimension of each feature vector
NUM_FEATS = 500
MAX_FEAT_OFFSET = 150

# Number of clusters for K-Means regression
K = 20

# Mininum samples in leaf node
MIN_SAMPLES_LEAF = 400

# Dimension of neural network
N_INPUT = 15 * 3 
N_HIDDEN = 1024
N_OUTPUT = 15 * 3 

OUTPUT_DIR = 'png'
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

# Set the kinematic tree (starting from torso body center)
kinem_order =  [14,  0, 13, 12, 1, 2, 5, 3, 6, 4, 7,  8, 10, 9, 11]
kinem_parent = [-1, 14, 14, 14, 0, 0, 0, 2, 5, 3, 6, 12, 13, 8, 10]

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

def test(y_rtw, y_nn, X_test, y_test, regressors, Ls, theta, model, num_test):
    # Start rtw in multiple processes
    processes = []
    for idx,j in enumerate(kinem_order):
        p = Process(target = test_rtw, args=(y_rtw, y_nn, regressors[j], Ls[j], theta, j, num_test, X_test, y_test))
        processes.append(p)
        p.start()

    # Start nn in main process
    test_nn(y_rtw, y_nn, model, num_test)

    for p in processes:
        p.join()

def test_rtw(y_rtw, y_nn, regressor, L, theta, j, num_test, X_test, y_test, num_steps=args.num_steps, step_size=args.step_size):
    for i in range(num_test):
        img = X_test[i]
        # Initialization
        if i == 0:
            y_rtw[(i*NUM_JOINTS*3 + j*3):(i*NUM_JOINTS*3 + (j+1)*3)] = (y_test[i][j]).tolist()
            #print('init %i' % j)
        else:
            while(1):
                # Start rtw when the starting position qm0 is updated 
                qm0 = y_nn[((i-1)*NUM_JOINTS*3 + j*3):((i-1)*NUM_JOINTS*3 + (j+1)*3)]     # set starting position
                if(qm0 != [0,0,0]):  # qm0 is updated
                    #t1 = time()
                    #print(('%i_start' % i),t1)
                    qm = np.zeros((num_steps + 1, 3))
                    qm[0] = np.array(qm0)
                    joint_pred = np.zeros(3)
                    for m in range(num_steps):
                        body_center_z = (y_test[i][JOINT_IDX['TORSO']])[2]
                        f = get_features(img, qm[m], body_center_z, theta).reshape(1, -1) # flatten feature vector
                        leaf_id = regressor.apply(f)[0]

                        idx = np.random.choice(L[leaf_id][0].shape[0], p=L[leaf_id][0])   # L[leaf_id][0] = weights
                        u = L[leaf_id][1][idx]                                            # L[leaf_id][1] = centers

                        qm[m+1] = qm[m] + u * step_size
                        qm[m+1][0] = np.clip(qm[m+1][0], 0, W-1)                          # limit x between 0 and W
                        qm[m+1][1] = np.clip(qm[m+1][1], 0, H-1)                          # limit y between 0 and H
                        qm[m+1][2] = img[int(qm[m+1][1]), int(qm[m+1][0])] / 1000.0       # index (y, x) into image for z position
                        joint_pred += qm[m+1]
                    joint_pred = joint_pred / num_steps
                    y_rtw[(i*NUM_JOINTS*3 + j*3):(i*NUM_JOINTS*3 + (j+1)*3)] = joint_pred.tolist()
                    #print('rtw_%i' % i)
                    #t2 = time()
                    #print(('%i_end' % i),t2)
                    #print(t2-t1)
                    break
                else:  # qm0 is not updated: keep waiting
                    #print('rtw_waiting_%i' % i)
                    continue

def test_nn(y_rtw, y_nn, model, num_test):
    for i in range(num_test):
        if i % 100 == 0:
            logger.debug('Testing %ith image', i)
        while(1):
            # Judge whether all the rtw processes finished
            judge = [y_rtw[(i*NUM_JOINTS*3 + j*3):(i*NUM_JOINTS*3 + (j+1)*3)] == [0,0,0] for j in range(NUM_JOINTS)]
            if True in judge:    # unfinished rtw process exists: keep waiting
                #print('nn_waiting_%i' % i)
                continue
            else:                # all the rtw processes finished
                #t1 = time()
                tmp = np.array(y_rtw[(i*NUM_JOINTS*3):((i+1)*NUM_JOINTS*3)])
                tmp[0::3] = tmp[0::3] / W           # normalization for x
                tmp[1::3] = tmp[1::3] / H           # normalization for y
                tmp[2::3] = tmp[2::3] / 4.5         # normalization for z

                x_tmp = torch.from_numpy(tmp).float()
                y_tmp = (model(Variable(x_tmp).cuda())).cpu().detach().numpy()    # run nn

                y_tmp[0::3] = y_tmp[0::3] * W       # denormalization for x
                y_tmp[1::3] = y_tmp[1::3] * H       # denormalization for y
                y_tmp[2::3] = y_tmp[2::3] * 4.5     # denormalization for z

                y_nn[(i*NUM_JOINTS*3):((i+1)*NUM_JOINTS*3)] = y_tmp.tolist()      # update y_nn
                #print('nn_%i' % i)
                #t2 = time()
                #print(t2-t1)
                break

###############################################################################
# Visualize predictions
###############################################################################

def visualization2D(y_nn, imgs):
    png_folder = 'rtw_nn_%s_[%s]/' % (TRAIN_SET, TEST_SET)
    if not os.path.exists(os.path.join(OUTPUT_DIR, png_folder)):
        os.makedirs(os.path.join(OUTPUT_DIR, png_folder))
    for test_idx in range(len(imgs)):
        png_path = os.path.join(OUTPUT_DIR, png_folder, str(test_idx) + '.png')
        drawTest(imgs[test_idx], y_nn[test_idx], png_path)

###############################################################################
# Run evaluation metrics
###############################################################################

def get_distances(y_test, y_pred):
    """Compute the raw world distances between the prediction and actual joint
    locations.
    """
    assert y_test.shape == y_pred.shape, "Mismatch of y_test and y_pred"

    distances = np.zeros((y_test.shape[:2]))
    for i in range(y_test.shape[0]):
        p1 = pixel2world(y_test[i], C)
        p2 = pixel2world(y_pred[i], C)
        distances[i] = np.sqrt(np.sum((p1-p2)**2, axis=1))
    return distances

###############################################################################
# Main
###############################################################################

def main():
    # Load dataset
    processed_dir = os.path.join(args.input_dir, args.dataset) # directory of saved numpy files
    X_test, y_test = load_dataset(processed_dir)
    X_test_draw = X_test.copy()

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

    model = Linear(N_INPUT, N_HIDDEN, N_OUTPUT, is_train_good=True)
    model = model.cuda()
    model.load_state_dict(torch.load('nn/model/model_parameters_%s.pkl' % TRAIN_SET))
    loss_func = nn.MSELoss(size_average=True).cuda()

    # Evaluate model
    logger.debug('\n------- Testing starts ------- ')
    logger.debug('\n------- Number of test ------- %d', num_test)

    cudnn.benchmark = True
    y_rtw = Array('f', [0 for i in range(num_test*NUM_JOINTS*3)])    # predicted results from rtw
    y_nn = Array('f', [0 for i in range(num_test*NUM_JOINTS*3)])      # predicted results from nn
    y_pred = np.zeros((num_test, NUM_JOINTS, 3))
    
    theta = compute_theta()

    t1 = time()

    test(y_rtw, y_nn, X_test, y_test, regressors, Ls, theta, model, num_test)

    t2 = time()
    logger.debug('runtime: %f', (t2-t1)/num_test)

    y_pred = np.array(y_nn).reshape((num_test, NUM_JOINTS, 3))

    y_mid = np.array(y_rtw).reshape((num_test, NUM_JOINTS, 3))
    joblib.dump(y_mid, os.path.join(args.preds_dir, 'y_pred_%s_%s.pkl' % (TRAIN_SET, TEST_SET+'t')))
    joblib.dump(y_test, os.path.join(args.preds_dir, 'y_test_%s_%s.pkl' % (TRAIN_SET, TEST_SET+'t')))

    #loss = loss_func(Variable(torch.from_numpy(y_nn).float()).cuda(), Variable(torch.from_numpy(y_test.reshape((num_test, NUM_JOINTS*3))).float()).cuda())
    #logger.debug('new loss: {:.4f}'.format(loss.cpu().data))

    # Visualization
    visualization2D(y_pred, X_test_draw)

    # Save failed cases
    #joblib.dump(np.concatenate((y_mid[0:80], y_mid[276:299]), axis=0), os.path.join(args.preds_dir, 'y_pred_%s_%s.pkl' % (TRAIN_SET, TEST_SET+'f'))) 
    #joblib.dump(np.concatenate((y_mid[0:80], y_mid[276:299]), axis=0), os.path.join(args.preds_dir, 'y_test_%s_%s.pkl' % (TRAIN_SET, TEST_SET+'f')))

    #y_pred[:,:,2] = y_test[:,:,2]
    distances = get_distances(y_test, y_pred) * 100.0 # convert from m to cm
    #mean_dist = np.mean(distances, axis=1)
    #mAD = []
    #for i in range(distances.shape[0]):
    #    if i > 1:
    #        mAD.append(np.mean(mean_dist[0:i]))
    #plt.plot(mAD)
    #plt.title('Mean Average Distance')
    #plt.xlabel('num of frames')
    #plt.ylabel('mAD(cm)')
    #plt.ylim(0,30)
    #plt.savefig('mAD.png')
    #plt.show()

    mAP_10 = 0
    mAP_5 = 0
    mAP_2 = 0
    mAP_joint = []
    for i in range(NUM_JOINTS):
        mAP_10 += np.sum(distances[:, i] < 10) / float(distances.shape[0])
        mAP_joint.append(np.sum(distances[:, i] < 10) / float(distances.shape[0]))
        mAP_5 += np.sum(distances[:, i] < 5) / float(distances.shape[0])
        mAP_2 += np.sum(distances[:, i] < 2) / float(distances.shape[0])
    print(mAP_joint)
    logger.debug('mAP (10cm): %f', mAP_10 / NUM_JOINTS)
    logger.debug('mAP (5cm): %f', mAP_5 / NUM_JOINTS)
    logger.debug('mAP (2cm): %f', mAP_2 / NUM_JOINTS)

if __name__=='__main__':
    main()
