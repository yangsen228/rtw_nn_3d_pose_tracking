import os
import cv2
import numpy as np

NUM_IMG = 0
NUM_JOINT = 15
DATA = '070'
content = []

counter = 0

with open(os.path.join(DATA + '.txt')) as f:
    content = [line.rstrip() for line in f.readlines()]
    NUM_IMG = len(content)
    print('number of images: %d' % NUM_IMG)

with open(os.path.join(DATA + '_corrected.txt'),'w') as f:
    for i in range(NUM_IMG):
        if i % 100 == 0:
            print('processing %d images' % i)

        in_path = os.path.join(DATA+'_color', '%d.png' % (i+1))
        img = cv2.imread(in_path)

        xml_path = os.path.join(DATA, '%d.xml' % (i+1))
        fs = cv2.FileStorage(xml_path, cv2.FileStorage_READ)
        xml = fs.getNode('bufferMat').mat()
        fs.release()

        tmp = content[i].split(',')
        out_path = os.path.join(DATA+'_corrected', '%d.png' % (i+1))
        for j in range(NUM_JOINT):
            x = tmp[4*j+2]
            y = tmp[4*j+3]
            z = tmp[4*j+4]
            xml_z = xml[int(y),int(x)] / 1000
            if float(z) != xml_z:
                tmp[4*j+4] = str(xml_z)
                counter += 1
            cv2.circle(img, (int(x),int(y)), 5, (0,0,255), -1)
            cv2.imwrite(out_path, img)
        res = ",".join(tmp)
        f.write('\n'+res)

print(counter)
