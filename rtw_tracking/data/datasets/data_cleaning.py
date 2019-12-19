import numpy as np
import os
import shutil # delete folders and files
import imageio
import cv2

NUM_IMG = 1000
NUM_JOINT = 15
DATA = ['072']
#DATA = ['063','064','065','066','067','068']
#DATA = ['060','061','062','069']
#JOINT_INDEX = [2,3,4,5,7,8,9,11,13,15,17,19,12,16,1]
FAIL_DATA = ['072','071','069','062','061','060']
FAILURE = {}

SOURCE_DIR = 'Kinect_all/'
TARGET_DIR = 'Kinect/'
TARGET_FOLDER = '228'

# find all the failures in each data file
def check_failure(source_dir, source_folder, failure):
    # get label contents
    content = []
    with open(os.path.join(source_dir, (source_folder + '.txt'))) as f:
        content = [line.rstrip() for line in f.readlines()]
        print('[%s] Read label contents: %d images' % (source_folder, len(content)))
    
    failure_details = {}
    failure_list = []
    for i in range(len(content)):
        path = os.path.join(source_dir, source_folder, '%d.xml' % (i+1))
        fs = cv2.FileStorage(path, cv2.FileStorage_READ)
        img = fs.getNode('bufferMat').mat()
        fs.release()
        #for ind, j in enumerate(JOINT_INDEX):
        for j in range(NUM_JOINT):
            x = content[i].split(',')[4*j+2]
            y = content[i].split(',')[4*j+3]
            z = content[i].split(',')[4*j+4]
            #if abs((float(z)*1000) - img[int(y),int(x)]) > 0.1:
            if (float(z)*1000) > 4000 or (float(z)*1000) < 1000:
                failure_details[i+1] = (content[i].split(',')[4*j+1],float(z)*1000,img[int(y),int(x)])
                failure_list.append(i+1)
                continue
    print('[%s] Failure list: ' % source_folder, end='')
    print(failure_details)
    failure[source_folder] = failure_list

# delete all the original data
def delete_all(target_dir):
    print('\n================ empty the folder ===============')
    for f in os.listdir(target_dir):                   # traverse all the floder&file names
        file_path = os.path.join(target_dir, f)
        if os.path.isfile(file_path):
            os.remove(file_path)                       # delete file
            print('%s has been deleted' % f)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path, True)             # delete folder and subfiles
            print('%s folder has been deleted' % f)

# copy images
def copy_files(source_dir, source_folder, target_dir, target_folder, failure_list, last_index, flag):
    source_path = os.path.join(source_dir, source_folder)
    target_path = os.path.join(target_dir, target_folder)
    if not os.path.exists(target_path):
        os.makedirs(target_path)
        print('\n%s folder has been created' % os.path.basename(target_path))

    index = 0
    for root, dirs, files in os.walk(source_path, topdown=False):
        for f in files:
            pre_index = int(os.path.splitext(f)[0])
            if (not pre_index in failure_list) and flag:     # if it's training set, flag = True, not bad cases will be copied
                file_source_path = os.path.join(source_path, '%d.xml' % pre_index)
                file_target_path = os.path.join(target_path, '%d.xml' % (pre_index + last_index))
                shutil.copyfile(file_source_path, file_target_path)
            if (pre_index in failure_list) and (not flag):   # if it's test set, flag = False, failed cases will be copied
                file_source_path = os.path.join(source_path, '%d.xml' % pre_index)
                file_target_path = os.path.join(target_path, '%d.xml' % (pre_index + last_index))
                shutil.copyfile(file_source_path, file_target_path)
            index = pre_index + last_index
    last_index = index
    print('\nall valid files have been copied to %s' % target_path)

# copy joint labels 
def copy_lines(source_dir, source_folder, target_dir, target_folder, failure_list, last_index, flag):
    source_path = os.path.join(source_dir, '%s.txt' % source_folder)
    target_path = os.path.join(target_dir, '%s.txt' % target_folder)
    with open(target_path, 'a') as f_write:
        with open(source_path, 'r') as f_read:
            for line in f_read.readlines():
                if (not int(line.split(',')[0]) in failure_list) and flag:     # if it's training set, flag = True, not bad cases will be copied
                    f_write.write(line)
                if (int(line.split(',')[0]) in failure_list) and (not flag):   # if it's test set, flag = False, failed cases will be copied
                    f_write.write(line)
    print('all valid labels have been copied to %s' % target_path)

# sample the useful data
def sampling(target_dir, target_folder, sample_stride):
    for f in os.listdir(target_dir):                  
        file_path = os.path.join(target_dir, f)
        if os.path.isfile(file_path):
            content = []
            with open(file_path, 'r') as f_read:
                content = [line for line in f_read.readlines()]
                content = content[::sample_stride]
            os.remove(file_path)
            print('length of .txt = %d' % len(content))
            with open(file_path, 'w') as f_write:
                for idx in range(len(content)):
                    tmp = content[idx].split(',')
                    tmp[0] = '%d' % (idx+1)
                    tmp = ','.join(map(lambda x:str(x), tmp))
                    f_write.write(tmp)
        if os.path.isdir(file_path):
            for root, dirs, files in os.walk(file_path, topdown=False):
                files.sort(key=lambda x:int(x[:-4]))
                for idx in range(len(files)):
                    if idx%sample_stride!=0:
                        os.remove(os.path.join(file_path, files[idx]))
            for root, dirs, files in os.walk(file_path, topdown=False):
                files.sort(key=lambda x:int(x[:-4]))
                for idx in range(len(files)):
                    os.rename(os.path.join(file_path, files[idx]), os.path.join(file_path, '%d.xml' % (idx+1)))

def main():
    last_index_ = 0
    delete_all(TARGET_DIR)
    for i in range(len(DATA)):
        print('\n================ processing %s ================' % DATA[i])
        check_failure(SOURCE_DIR, DATA[i], FAILURE)
        copy_files(SOURCE_DIR, DATA[i], TARGET_DIR, TARGET_FOLDER, FAILURE[DATA[i]], i*NUM_IMG, True) 
        copy_lines(SOURCE_DIR, DATA[i], TARGET_DIR, TARGET_FOLDER, FAILURE[DATA[i]], i*NUM_IMG, True) 
        last_index_ += NUM_IMG
'''
    # add failed cases on test set
    print(last_index_)
    failcases_path = 'failed_cases.txt'
    with open(failcases_path, 'r') as f:
        failcases = [line.rstrip() for line in f.readlines()]       # load fail-case index in each test set
    for i in range(len(FAIL_DATA)):
        print('\n================ processing %s ================' % FAIL_DATA[i])
        f_input = [int(x) for x in failcases[i].split(',')]
        copy_files(SOURCE_DIR, FAIL_DATA[i], TARGET_DIR, TARGET_FOLDER, f_input, last_index_ + i*NUM_IMG, False) 
        copy_lines(SOURCE_DIR, FAIL_DATA[i], TARGET_DIR, TARGET_FOLDER, f_input, last_index_ + i*NUM_IMG, False) 
'''

if __name__=='__main__':
    main()
