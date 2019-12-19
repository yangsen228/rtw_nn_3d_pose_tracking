import cv2
import os

H, W = 424, 512
TEST_SET = '070'
img_path = '../png/rtw_nn_dl_enhanced_063_068_train_[%s]/' % TEST_SET
out_path = '../png/enhanced_%s.avi' % TEST_SET
#img_path = '../rtw_tracking/output/random-tree-walks/Kinect/png/g_dl_063_068_train_20_400_[%s]/' % TEST_SET
#out_path = '../rtw_tracking/output/random-tree-walks/Kinect/png/%s.avi' % TEST_SET
fps = 30
size = (W, H)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
videoWriter = cv2.VideoWriter(out_path, fourcc, fps, size)

for i in range(1000):
    frame = cv2.imread(img_path+str(i+1)+'.png')
    videoWriter.write(frame)
videoWriter.release()
