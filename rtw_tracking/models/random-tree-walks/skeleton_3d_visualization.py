import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.externals import joblib
from helper import *

H, W, D = 424, 512, 4500
C = 3.8605e-3

TEST_SET = '035'
pred_path = '../../output/random-tree-walks/Kinect/preds/y_pred_%s.pkl' % TEST_SET
test_path = '../../output/random-tree-walks/Kinect/preds/y_test_%s.pkl' % TEST_SET

# load data
print("===============================")
print(">>> loading data")
y_pred = joblib.load(pred_path)
y_test = joblib.load(test_path)
print(">>> data loaded")

print("\n>>> load pred data from %s" % pred_path)
print(">>> load test data from %s" % test_path)

num_test = len(y_pred)
num_joint = len(y_pred[0])

print(y_pred.shape)
print(">>> num_test: %d" % num_test)
print(">>> num_joint: %d" % num_joint)
'''
# coordinates transformation (from depth image to point cloud)
for i in range(num_test):
   y_pred[i] = pixel2world(y_pred[i], C)
   y_test[i] = pixel2world(y_test[i], C)

joblib.dump(y_pred, pred_path)
joblib.dump(y_test, test_path)
'''
# calculate MSE
print("\n>>> MSE: %f" % (sum([(x-y)**2 for x,y in zip(y_pred.flatten(),y_test.flatten())])/(y_pred.shape[0]*y_pred.shape[1]*y_pred.shape[2])))

# data visulization
fig = plt.figure()
ax = Axes3D(fig)
plt.ion()
for i in range(num_test):
    plt.cla()
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(0.5, 4.5)
    ax.set_zlim(-1.5, 1.5)
    ax.text(-1.4, 4, 1.4, '%d'%(i+1), color='red')

    # draw predicted joints in blue
    p1 = y_pred[i]
    xdata1 = p1[:,0]
    ydata1 = p1[:,1]
    zdata1 = p1[:,2]
    ax.scatter3D(xdata1, zdata1, ydata1, color='blue')

    # draw joint index
    for j in range(num_joint):
        ax.text(xdata1[j], zdata1[j], ydata1[j], '%d'%j, color='blue')

    # draw predicted limbs in blue
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

    # draw label joints in red 
    p2 = y_test[i]
    xdata2 = p2[:,0]
    ydata2 = p2[:,1]
    zdata2 = p2[:,2]
    ax.scatter3D(xdata2, zdata2, ydata2, color='red')

    plt.pause(0.03)

plt.ioff()
plt.show()

