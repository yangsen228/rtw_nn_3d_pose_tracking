import sys
import argparse

import numpy as np
import pickle

import joblib
from sklearn.tree import DecisionTreeRegressor
from sklearn.cluster import MiniBatchKMeans
from multiprocessing import Process, Queue
from multiprocessing.pool import ThreadPool

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from time import time

from helper import *

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
parser.add_argument('--input-dir', type=str, default='../../data/processed',
                    help='Directory of the processed input')
parser.add_argument('--dataset', type=str, default='Kinect', # NTU-RGBD, CAD-60
                    help='Name of the dataset to load')

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

# Set location of output saved files
args.model_dir = '../../output/random-tree-walks/' + args.dataset + '/models'
args.preds_dir = '../../output/random-tree-walks/' + args.dataset + '/preds'
args.png_dir = '../../output/random-tree-walks/' + args.dataset + '/png'

###############################################################################
# Training hyperparameters
###############################################################################

# Train-test ratio
TRAIN_RATIO = 0
SMALL_DATA_SIZE = 5000
TEST_SET = '072'
TRAIN_SET = 'dl_enhanced_063_068_train'

# Dimension of each feature vector
NUM_FEATS = 500
MAX_FEAT_OFFSET = 150

# Number of clusters for K-Means regression
K = 20

# Mininum samples in leaf node
MIN_SAMPLES_LEAF = 400

###############################################################################
# Dataset Constants
###############################################################################

# Depth image dimension
H, W = 424, 512

# See https://help.autodesk.com/view/MOBPRO/2018/ENU/?guid=__cpp_ref__nui_image_camera_8h_source_html
C = 3.8605e-3 # NUI_CAMERA_DEPTH_NOMINAL_INVERSE_FOCAL_LENGTH_IN_PIXELS

###############################################################################
# RTW Constants
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
# Load dataset splits
###############################################################################

def load_dataset(processed_dir, is_mask=False, small_data=False):
    """Loads the depth images and joints from the processed dataset.

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
# Train model
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

def multitesting(y_test, X_test, previous_test_idx, joint_id, num_test, regressors, Ls, theta, q1, q2):
    r1 = np.zeros((num_test, args.num_steps+1, 3))
    r2 = np.zeros((num_test, 3))
    for test_idx in range(num_test):
        if previous_test_idx == -1:
            qm0 = y_test[test_idx][joint_id]
        elif (previous_test_idx+1) % 5 == 0:
            qm0 = y_test[previous_test_idx][joint_id]
        else:
            qm0 = r2[previous_test_idx]
            
        #qm0 = y_test[test_idx][joint_id] if (previous_test_idx == -1 or (previous_test_idx+1) % 5 == 0) else r2[previous_test_idx]
        #qm0 = y_test[test_idx][joint_id] if previous_test_idx == -1 else y_test[previous_test_idx][joint_id]
        r1[test_idx], r2[test_idx] = test_model(regressors[joint_id], Ls[joint_id], theta, qm0, X_test[test_idx], y_test[test_idx][JOINT_IDX['TORSO']], joint_id, test_idx)
        previous_test_idx += 1
    q1.put({joint_id:r1})
    q2.put({joint_id:r2})

def test_model(regressor, L, theta, qm0, img, body_center, joint_id, test_idx, num_steps=args.num_steps, step_size=args.step_size):
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
        tmp = img[int(qm[i+1][1]), int(qm[i+1][0])]
        if tmp < 4000:
            qm[i+1][2] = tmp / 1000.0
        joint_pred += qm[i+1]

    #print(U)
    joint_pred += qm0
    joint_pred = joint_pred / (num_steps + 1)
    return qm, joint_pred

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
    # Load dataset splits
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

    # Evaluate model
    logger.debug('\n------- Testing starts ------- ')
    logger.debug('\n------- Number of test ------- %d', num_test)

    theta = compute_theta()
    qms = np.zeros((num_test, NUM_JOINTS, args.num_steps+1, 3))
    y_pred = np.zeros((num_test, NUM_JOINTS, 3))

    t1 = time()
    processes = []
    q1 = Queue()
    q2 = Queue()
    for kinem_idx, joint_id in enumerate(kinem_order):
        logger.debug('Testing %s model', JOINT_NAMES[joint_id])
        previous_test_idx = -1
        p = Process(target = multitesting, args=(y_test,X_test,previous_test_idx,joint_id,num_test,regressors,Ls,theta,q1,q2))
        processes.append(p)
        p.start()
    qms_tmp = [q1.get() for p in processes]
    y_pred_tmp = [q2.get() for p in processes]
    for i in qms_tmp:
        qms[:,list(i.keys())[0],:,:] = list(i.values())[0]
    for i in y_pred_tmp:
        y_pred[:,list(i.keys())[0],:] = list(i.values())[0]
    for p in processes:
        p.join()
    t2 = time()
    logger.debug('average running time = %f', (t2-t1)/num_test)

    #y_pred[:, :, 2] = y_test[:, :, 2]
    joblib.dump(y_pred, os.path.join(args.preds_dir, 'y_pred_%s_%s.pkl' % (TRAIN_SET, TEST_SET)))
    joblib.dump(y_test, os.path.join(args.preds_dir, 'y_test_%s_%s.pkl' % (TRAIN_SET, TEST_SET)))
    ###############################################################################
    # Visualize predictions
    ###############################################################################
    
    # if args.make_png:
    logger.debug('\n------- Saving prediction visualizations -------')
    
    png_folder = 'g_%s_%d_%d_[%s]/' % (TRAIN_SET, K, MIN_SAMPLES_LEAF, TEST_SET)
    if not os.path.exists(os.path.join(args.png_dir, png_folder)):
        os.makedirs(os.path.join(args.png_dir, png_folder))
    for test_idx in range(num_test):
        png_path = os.path.join(args.png_dir, png_folder, str(test_idx) + '.png')
        drawPred(X_test_draw[test_idx], y_pred[test_idx], qms[test_idx], y_test[test_idx][JOINT_IDX['TORSO']], png_path, NUM_JOINTS, JOINT_NAMES)

    # Run evaluation metrics
    logger.debug('\n------- Computing evaluation metrics -------')

    #y_pred[:, :, 2] = y_test[:, :, 2]
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

    #failures = []
    # Find failures
    #for i in range(num_test):
    #    for j in range(NUM_JOINTS):
    #        if distances[i,j] > 10:
    #            failures.append(i+1)
    #            break
    #print(failures)
    #failures = [str(x) for x in failures]
    #failcases_path = '../../data/datasets/failed_cases.txt'
    #with open(failcases_path, 'a') as f:
    #    f.write('\n'+','.join(failures))
                
if __name__ == '__main__':
    main()
