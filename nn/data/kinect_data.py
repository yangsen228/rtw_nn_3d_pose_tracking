import os
import torch
from torch.utils.data import Dataset
from sklearn.externals import joblib

class Kinect(Dataset):
    '''
    x_train.shape -> (n x 15 x 3)
    y_train.shape -> (n x 15 x 3)
    x_test.shape -> (m x 15 x 3)
    y_test.shape -> (m x 15 x 3)

    inputs.shape -> (45 x 1)
    outputs.shape -> (45 x 1)
    '''
    def __init__(self, train_set, test_set, data_path, is_train=True):
        self.train_set = train_set
        self.test_set = test_set
        self.data_path = data_path
        self.is_train = is_train
        self.x_train, self.y_train, self.x_test, self.y_test = [], [], [], []
        
        # loading data
        if self.is_train:
            self.x_train = joblib.load(os.path.join(data_path, 'y_pred_%s.pkl'%self.train_set))
            self.y_train = joblib.load(os.path.join(data_path, 'y_test_%s.pkl'%self.train_set))
        else:
            self.x_test = joblib.load(os.path.join(data_path, 'y_pred_%s.pkl'%self.test_set))
            self.y_test = joblib.load(os.path.join(data_path, 'y_test_%s.pkl'%self.test_set))


    def __getitem__(self, index):
        if self.is_train:
            inputs = torch.from_numpy(self.x_train[index].flatten()).float()
            outputs = torch.from_numpy(self.y_train[index].flatten()).float()
        else:
            inputs = torch.from_numpy(self.x_test[index].flatten()).float()
            outputs = torch.from_numpy(self.y_test[index].flatten()).float()

        return inputs, outputs

    def __len__(self):
        if self.is_train:
            return len(self.x_train)
        else:
            return len(self.x_test)
