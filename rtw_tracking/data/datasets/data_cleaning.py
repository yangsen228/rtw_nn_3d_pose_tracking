import numpy as np
import os
import shutil # delete folders and files
import imageio

source_dir_ = 'Kinect_all/'
target_dir_ = 'Kinect/'
source_folder_1_ = '030'
source_folder_2_ = '030'
target_folder_ = '228'

sample_stride_ = 2
failure_list_ = [733, 808, 810, 815, 829, 856, 1236]
#failure_list_ = [115,137,207,208,209,210,299,303,304,305,308,309,310,323,325,326,327,335,336,351,352,353,357,358,359,383,385,407,414,415,419,420,421,422,423,424,425,426,427,428,445,449,450,451,452,453,489,670,677,679,680,730,731,746,747,752,753,754,755,756,758,773,791,803,804,847,848,849,865,866,875,878,879,880,881,882,883,884,885,886,888,926,959,964]

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

# create new folder
def create_new(target_dir, target_folder):
    new_folder = os.path.join(target_dir, target_folder)
    if not os.path.exists(new_folder):
        os.makedirs(new_folder)
        print('\n%s folder has been created' % os.path.basename(new_folder))

# copy images
def copy_files(source_dir, source_folder, target_dir, target_folder, failure_list):
    print('\n================ copy the images ================')
    source_path = os.path.join(source_dir, source_folder)
    target_path = os.path.join(target_dir, target_folder)
    if not os.path.exists(target_path):
        os.makedirs(target_path)
        print('%s folder has been created' % os.path.basename(target_path))

    for root, dirs, files in os.walk(source_path, topdown=False):
        for f in files:
            index = os.path.splitext(f)[0]
            if not int(index) in failure_list:
                file_source_path = os.path.join(source_path, '%d.xml' % int(index))
                file_target_path = os.path.join(target_path, '%d.xml' % int(index))
                shutil.copyfile(file_source_path, file_target_path)
    print('all valid files have been copied to %s' % target_path)

# copy joint labels 
def copy_lines(source_dir, source_folder, target_dir, target_folder, failure_list):
    print('\n================ copy the labels ================')
    source_path = os.path.join(source_dir, '%s.txt' % source_folder)
    target_path = os.path.join(target_dir, '%s.txt' % target_folder)
    with open(target_path, 'w') as f_write:
        print('%s.txt has been created' % target_folder)
        with open(source_path, 'r') as f_read:
            for line in f_read.readlines():
                if not int(line.split(',')[0]) in failure_list:
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
    delete_all(target_dir_)
    copy_files(source_dir_, source_folder_1_, target_dir_, target_folder_, failure_list_) 
    copy_lines(source_dir_, source_folder_1_, target_dir_, target_folder_, failure_list_) 
    #sampling(target_dir_, target_folder_, sample_stride_) 

if __name__=='__main__':
    main()
