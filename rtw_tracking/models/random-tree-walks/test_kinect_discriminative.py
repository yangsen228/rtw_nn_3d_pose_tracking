import sys
import argparse

import numpy as np
import pickle

from sklearn.externals import joblib
from sklearn.tree import DecisionTreeRegressor
from sklearn.cluster import MiniBatchKMeans
from multiprocessing import Process, Queue
from multiprocessing.pool import ThreadPool

from helper import *
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from time import time

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

# Location of output saved data directories
# parser.add_argument('--model-dir', type=str, default='../../output/random-tree-walks/models',
#                     help='Directory of the saved model')
# parser.add_argument('--preds-dir', type=str, default='../../output/random-tree-walks/preds',
#                     help='Directory to save predictions')
# parser.add_argument('--png-dir', type=str, default='../../output/random-tree-walks/png',
#                     help='Directory to save prediction images')

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
parser.add_argument('--num-steps', type=int, default=64,
                    help='Number of steps during evaluation')
parser.add_argument('--step-size', type=int, default=10,
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
TEST_SET = '035'
TRAIN_SET = '030_train'

# Dimension of each feature vector
NUM_FEATS = 1000
MAX_FEAT_OFFSET = 300

# Number of samples for each joint for each example
NUM_SAMPLES = [1500]

# Set maximum XYZ offset from each joint
MAX_XY_OFFSET = [65] # image xy coordinates (pixels)
MAX_Z_OFFSET = 0.6 # z-depth coordinates (meters)

# Number of clusters for K-Means regression
K = [20]

# Mininum samples in leaf node
MIN_SAMPLES_LEAF = [400]

# Nunber of steps and step size
#NUM_STEPS = [16, 32]
#STEP_SIZE = [2]

###############################################################################
# Dataset Constants
###############################################################################

# Depth image dimension
#H, W = 240, 320
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

def split_dataset(X, y, train_ratio, k):
    """Splits the dataset according to the train-test ratio.

    @params:
        X : depth images (N x H x W)
        y : joint positions (N x NUM_JOINTS x 3)
        train_ratio : ratio of training to test
    """
    test_ratio = 1.0 - train_ratio
    num_test = int(X.shape[0] * test_ratio)
    X_train, y_train = np.concatenate([X[:k*num_test],X[(k+1)*num_test:]]), np.concatenate([y[:k*num_test],y[(k+1)*num_test:]])
    X_test, y_test = X[k*num_test:(k+1)*num_test], y[k*num_test:(k+1)*num_test]

    logger.debug('Data split: # training data: %d, # test data: %d', X_train.shape[0], X_test.shape[0])
    return X_train, y_train, X_test, y_test

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

def get_random_offset(max_offset_xy=MAX_XY_OFFSET, max_offset_z=MAX_Z_OFFSET):
    """Gets xyz vector with uniformly random xy and z offsets.
    """
    offset_xy = np.random.randint(-max_offset_xy, max_offset_xy + 1, 2)
    offset_z = np.random.uniform(-max_offset_z, max_offset_z, 1)
    offset = np.concatenate((offset_xy, offset_z)) # xyz offset
    return offset

def get_training_samples(joint_id, X, y, theta, num_feats=NUM_FEATS, num_samples=NUM_SAMPLES):
    """Generates training samples for each joint.

    Each sample is (i, q, u, f) where:
         i is the index of the depth image,
         q is the random offset point from the joint,
         u is the unit direction vector toward the joint location,
         f is the feature array

    @params:
        X : depth images (N x H x W)
        y : joint position = (N x NUM_JOINTS x 3) = (im_x, im_y, depth_z)
        joint_id : current joint id
        num_samples : number of samples of each joint
        max_offset_xy : maximum offset for samples in (x, y) axes
        max_offset_z : maximum offset for samples in z axis

    @return:
        S_f : samples feature array (N x num_samples x num_feats)
        S_u : samples unit direction vectors (N x num_samples x 3)
    """
    num_train, _, _ = X.shape

    S_f = np.zeros((num_train, num_samples, num_feats), dtype=np.float64)
    S_u = np.zeros((num_train, num_samples, 3), dtype=np.float64)

    for train_idx in range(num_train):
        if train_idx % 300 == 0:
            logger.debug('Joint %s: Processing image %d / %d', JOINT_NAMES[joint_id], train_idx, num_train)

        # Create samples for each training example
        for sample_idx in range(num_samples):
            depth_im = X[train_idx]
            offset = get_random_offset()
            unit_offset = 0 if np.linalg.norm(offset) == 0 else (-offset / np.linalg.norm(offset))
            body_center_z = y[train_idx][JOINT_IDX['TORSO']][2] # body center (torso) index, 2 = z_index

            S_f[train_idx, sample_idx] = get_features(depth_im, y[train_idx][joint_id] + offset, body_center_z, theta)
            S_u[train_idx, sample_idx] = unit_offset

    return S_f, S_u

def stochastic(regressor, features, unit_directions, k_value):
    """Applies stochastic relaxation when choosing the unit direction. Training
    samples at the leaf nodes are further clustered using K-means.
    """
    L = {}
    print(regressor.tree_.max_depth)
    indices = regressor.apply(features) # leaf id of each sample
    leaf_ids = np.unique(indices) # array of unique leaf ids
    #print(leaf_ids)
    #print(indices.shape)
    #print(unit_directions.shape)

    logger.debug('Running stochastic (minibatch) K-Means...')
    for leaf_id in leaf_ids:
        kmeans = MiniBatchKMeans(n_clusters=k_value, batch_size=1000)
        labels = kmeans.fit_predict(unit_directions[indices == leaf_id])
        #print("number of directions %d" % unit_directions[indices == leaf_id].shape[0])
        #print(labels)
        weights = np.bincount(labels).astype(float) / labels.shape[0]
        #print(weights)

        # Normalize the centers
        centers = kmeans.cluster_centers_
        #print("centers: ", centers)
        centers /= np.linalg.norm(centers, axis=1)[:, np.newaxis]
        #print("normalized centers: ", centers)
        # checkUnitVectors(centers)

        L[leaf_id] = (weights, centers)
    return L

def train(joint_id, model_dir, samples_leaf, k_value, num_samples, xy_offset):
    """Trains a regressor tree on the unit directions towards the joint.

    @params:
        joint_id : current joint id
        X : samples feature array (N x num_samples x num_feats)
        y : samples unit direction vectors (N x num_samples x 3)
        min_samples_split : minimum number of samples required to split an internal node
        load_models : load trained models from disk (if exist)
    """
    logger.debug('Start training %s model...', JOINT_NAMES[joint_id])

    folder = '%s_%d_%d_%d_%d/' % (TRAIN_SET, k_value, samples_leaf, num_samples, xy_offset)
    regressor_path = os.path.join(model_dir, folder, 'regressor' + str(joint_id) + '.pkl')
    L_path = os.path.join(model_dir, folder, 'L' + str(joint_id) + '.pkl')
    #regressor_path = os.path.join(model_dir, '011/regressor' + str(joint_id) + '.pkl')
    #L_path = os.path.join(model_dir, '011/L' + str(joint_id) + '.pkl')
    #regressor_path = os.path.join(model_dir, 'regressor' + str(joint_id) + '.pkl')
    #L_path = os.path.join(model_dir, 'L' + str(joint_id) + '.pkl')

    #X_reshape = X.reshape(X.shape[0] * X.shape[1], X.shape[2]) # (N x num_samples, num_feats)
    #y_reshape = y.reshape(y.shape[0] * y.shape[1], y.shape[2]) # (N x num_samples, 3)

    # Count the number of valid (non-zero) samples
    #valid_rows = np.logical_not(np.all(X_reshape == 0, axis=1)) # inverse of invalid samples
    #logger.debug('Model %s - Valid samples: %d / %d', JOINT_NAMES[joint_id], X_reshape[valid_rows].shape[0], X_reshape.shape[0])

    # Fit decision tree to samples
    #regressor = DecisionTreeRegressor(min_samples_leaf=samples_leaf)
    #regressor.fit(X_reshape[valid_rows], y_reshape[valid_rows])
    #L = stochastic(regressor, X_reshape, y_reshape, k_value)
    
    # Print statistics on leafs
    #leaf_ids = regressor.apply(X_reshape)
    #bin = np.bincount(leaf_ids)
    #unique_ids = np.unique(leaf_ids)
    #biggest = np.argmax(bin)
    #smallest = np.argmin(bin[bin != 0])

    #logger.debug('Model %s - # Leaves: %d', JOINT_NAMES[joint_id], unique_ids.shape[0])
    #logger.debug('Model %s - Smallest Leaf ID: %d, # Samples: %d/%d', JOINT_NAMES[joint_id], smallest, bin[bin != 0][smallest], np.sum(bin))
    #logger.debug('Model %s - Biggest Leaf ID: %d, # Samples: %d/%d', JOINT_NAMES[joint_id], biggest, bin[biggest], np.sum(bin))
    #logger.debug('Model %s - Average Leaf Size: %d', JOINT_NAMES[joint_id], np.sum(bin) / unique_ids.shape[0])

    # Save models to disk
    #pickle.dump(regressor, open(regressor_path, 'wb'))
    #pickle.dump(L, open(L_path, 'wb'))
    #joblib.dump(regressor, regressor_path)
    #joblib.dump(L, L_path)
    regressor = joblib.load(regressor_path)
    L = joblib.load(L_path)
    
    return regressor, L

def train_series(joint_id, X, y, theta, model_dir, samples_leaf, k_value, num_samples, xy_offset):
    """Train each joint sequentially.
    """
    #S_f, S_u = get_training_samples(joint_id, X, y, theta)
    #regressor, L = train(joint_id, S_f, S_u, model_dir, samples_leaf, k_value)
    regressor, L = train(joint_id, model_dir, samples_leaf, k_value, num_samples, xy_offset)
    return regressor, L

###############################################################################
# Evaluate model
###############################################################################

def test_model(regressor, L, theta, qm0, img, body_center, k_value, joint_id, test_idx, num_steps=args.num_steps, step_size=args.step_size):
    """Test the model on a single example.
    """
    qm = np.zeros((num_steps + 1, 3))
    qm[0] = qm0
    joint_pred = np.zeros(3)

    #vector_folder = 'g_%s_[%s]_vectors/' % (TRAIN_SET, TEST_SET)
    #if not os.path.exists(os.path.join(args.png_dir, vector_folder)):
    #    os.makedirs(os.path.join(args.png_dir, vector_folder))
    for i in range(num_steps):
        body_center_z = body_center[2]
        f = get_features(img, qm[i], body_center_z, theta).reshape(1, -1) # flatten feature vector
        leaf_id = regressor.apply(f)[0]
        #if(k_value != L[leaf_id][0].shape[0]):
            #print('leaf_id = %d' % leaf_id)
            #print(L[leaf_id][0].shape)
            #print("qm[%d] = "%i, qm[i])
            #print(L[leaf_id][0])
            #print(L[leaf_id][0].sum())
        
        idx = np.random.choice(L[leaf_id][0].shape[0], p=L[leaf_id][0]) # L[leaf_id][0] = weights
        #idx = np.where(L[leaf_id][0]==np.max(L[leaf_id][0]))[0][0]
        #print(idx)
        u = L[leaf_id][1][idx] # L[leaf_id][1] = centers
        #if i==100 and test_idx == 1 and joint_id == 3:
            #print(L[leaf_id][1])
            #print(L[leaf_id][0])
            #fig = plt.figure()
            #ax = fig.gca(projection='3d')
            #for j in range(L[leaf_id][0].shape[0]):
                #print(L[leaf_id][0][j])
                #x = np.array([0, L[leaf_id][1][j][0]*L[leaf_id][0][j]])
                #y = np.array([0, L[leaf_id][1][j][1]*L[leaf_id][0][j]])
                #z = np.array([0, L[leaf_id][1][j][2]*L[leaf_id][0][j]])
                #ax.plot(x,y,z,color='brown')
            #x = np.array([0, u[0]*L[leaf_id][0][idx]])
            #y = np.array([0, u[1]*L[leaf_id][0][idx]])
            #z = np.array([0, u[2]*L[leaf_id][0][idx]])
            #ax.plot(x,y,z,color='b')
            #plt.show()
            #plt.savefig(os.path.join(args.png_dir, vector_folder, str(i) + '.png'), format='png')

        qm[i+1] = qm[i] + u * step_size
        qm[i+1][0] = np.clip(qm[i+1][0], 0, W-1) # limit x between 0 and W
        qm[i+1][1] = np.clip(qm[i+1][1], 0, H-1) # limit y between 0 and H
        qm[i+1][2] = img[int(qm[i+1][1]), int(qm[i+1][0])] # index (y, x) into image for z position
        joint_pred += qm[i+1]

    joint_pred = joint_pred / num_steps
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






results = []
processed_dir = os.path.join(args.input_dir, args.dataset) # directory of saved numpy files
depth_images, joints = load_dataset(processed_dir)
for k_idx in range(len(K)):
    for leaf_idx in range(len(MIN_SAMPLES_LEAF)):
        set_10 = []
        set_5 = []
        set_2 = []
        for samples_idx in range(len(NUM_SAMPLES)):
            for xy_offset_idx in range(len(MAX_XY_OFFSET)):
                for round_idx in range(1):
                    # Load dataset splits
                    X_train, y_train, X_test, y_test = split_dataset(depth_images, joints, TRAIN_RATIO, round_idx)
                    X_test_draw = X_test.copy()
        
                    num_train = X_train.shape[0]
                    num_test = X_test.shape[0]
        
                    # Train model
                    logger.debug('\n------- Training models ------- %d-%d-%d-%d-%d', k_idx, leaf_idx, samples_idx, xy_offset_idx, round_idx)
                    logger.debug('\n------- Training models ------- %d-%d-%d-%d-%d', K[k_idx], MIN_SAMPLES_LEAF[leaf_idx], NUM_SAMPLES[samples_idx], MAX_XY_OFFSET[xy_offset_idx], round_idx)
        
                    theta = compute_theta()
        
                    regressors, Ls = {}, {}
        
                    if not args.multithread:
                        for joint_id in range(NUM_JOINTS):
                            regressors[joint_id], Ls[joint_id] = train_series(joint_id, X_train, y_train, theta, args.model_dir, MIN_SAMPLES_LEAF[leaf_idx], K[k_idx], NUM_SAMPLES[samples_idx], MAX_XY_OFFSET[xy_offset_idx])
        
                    # Evaluate model
                    logger.debug('\n------- Testing models ------- %d-%d-%d-%d-%d', k_idx, leaf_idx, samples_idx, xy_offset_idx, round_idx)
                    logger.debug('\n------- Number of test ------- %d', num_test)
        
                    qms = np.zeros((num_test, NUM_JOINTS, args.num_steps+1, 3))
                    y_pred = np.zeros((num_test, NUM_JOINTS, 3))
                    
                    t1 = time()
                    for kinem_idx, joint_id in enumerate(kinem_order):
                        logger.debug('Testing %s model', JOINT_NAMES[joint_id])
                        parent_joint_id = kinem_parent[kinem_idx]
                        for test_idx in range(num_test):
                            #if test_idx % 100 == 0:
                            #    logger.debug('(%d)Joint %s: Processing image %d / %d', kinem_idx, JOINT_NAMES[joint_id], test_idx, num_test)
                            qm0 = y_test[test_idx][JOINT_IDX['TORSO']] if parent_joint_id == -1 else y_pred[test_idx][parent_joint_id]
                            qms[test_idx][joint_id], y_pred[test_idx][joint_id] = test_model(regressors[joint_id], Ls[joint_id], theta, qm0, X_test[test_idx], y_test[test_idx][JOINT_IDX['TORSO']], K[k_idx], joint_id, test_idx)

                    t2 = time()
                    logger.debug('average running time = %f', (t2-t1)/num_test)
        
                    y_pred[:, :, 2] = y_test[:, :, 2]
                    ###############################################################################
                    # Visualize predictions
                    ###############################################################################
                    
                    # if args.make_png:
                    logger.debug('\n------- Saving prediction visualizations -------')
                    
                    png_folder = 'd_%s_%d_%d_%d_%d_[%s]/' % (TRAIN_SET, K[k_idx], MIN_SAMPLES_LEAF[leaf_idx], NUM_SAMPLES[samples_idx], MAX_XY_OFFSET[xy_offset_idx], TEST_SET)
                    if not os.path.exists(os.path.join(args.png_dir, png_folder)):
                        os.makedirs(os.path.join(args.png_dir, png_folder))
                    for test_idx in range(num_test):
                        png_path = os.path.join(args.png_dir, png_folder, str(test_idx) + '.png')
                        drawPred(X_test_draw[test_idx], y_pred[test_idx], qms[test_idx], y_test[test_idx][JOINT_IDX['TORSO']], png_path, NUM_JOINTS, JOINT_NAMES)
        
                    # Run evaluation metrics
                    logger.debug('\n------- Computing evaluation metrics -------')
        
                    distances = get_distances(y_test, y_pred) * 100.0 # convert from m to cm
                    #print(distances.shape)
                    #for i in range(distances.shape[1]):
                    #    plt.plot(distances[:,i])
                    #plt.show()
        
                    distances_pixel = np.zeros((y_test.shape[:2]))
                    for i in range(y_test.shape[0]):
                        p1 = y_test[i]
                        p2 = y_pred[i]
                        distances_pixel[i] = np.sqrt(np.sum((p1-p2)**2, axis=1))
        
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
                    print(np.mean(mAP_joint))
        
                    logger.debug('mAP (10cm): %f', mAP_10 / NUM_JOINTS)
                    logger.debug('mAP (5cm): %f', mAP_5 / NUM_JOINTS)
                    logger.debug('mAP (2cm): %f', mAP_2 / NUM_JOINTS)
                    set_10.append(mAP_10/NUM_JOINTS)
                    set_5.append(mAP_5/NUM_JOINTS)
                    set_2.append(mAP_2/NUM_JOINTS)
                
                set_10 = np.array(set_10)
                set_5 = np.array(set_5)
                set_2 = np.array(set_2)
                logger.debug('5 fold average mAP (10cm): %f', np.mean(set_10))
                logger.debug('5 fold average mAP (5cm): %f', np.mean(set_5))
                logger.debug('5 fold average mAP (2cm): %f', np.mean(set_2))
                
                results.append([K[k_idx],MIN_SAMPLES_LEAF[leaf_idx],round_idx,np.mean(set_10),np.mean(set_5),np.mean(set_2)])

#results = np.array(results)
#print(results)
#np.savetxt("g_results_012_test.txt", results, delimiter=',')

