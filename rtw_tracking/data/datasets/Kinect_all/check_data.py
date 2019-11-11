'''
Check whether the labelled joint positions in the dataset are reasonable (1000 < z < 4000)
If not, print the unreasonable frame index and joint index
'''

import os
import cv2
import numpy as np

NUM_IMG = 0
NUM_JOINT = 15
DATA = '035'
content = []

with open(os.path.join(DATA + '.txt')) as f:
    content = [line.rstrip() for line in f.readlines()]
    NUM_IMG = len(content)
    print('number of images: %d' % NUM_IMG)

valid_joint = [2,3,4,5,7,8,9,11,13,15,17,19,12,16,1]
failure = {}
failure_list = []
for i in range(NUM_IMG):
    path = os.path.join(DATA, '%d.xml' % (i+1))
    fs = cv2.FileStorage(path, cv2.FileStorage_READ)
    img = fs.getNode('bufferMat').mat()
    fs.release()
    for ind, j in enumerate(valid_joint):
        x = content[i].split(',')[4*j+2]
        y = content[i].split(',')[4*j+3]
        z = content[i].split(',')[4*j+4]
        #if abs((float(z)*1000) - img[int(y),int(x)]) > 0.1:
        if (float(z)*1000) > 4000 or (float(z)*1000) < 1000:
            failure[i+1] = (j,)
            failure_list.append(i+1)
            print('\nimg: z = %f' % img[int(y),int(x)])
            print('label: z = %f' % (float(z)*1000))
            continue
        
print(failure)
print(failure_list)
