import torch.nn as nn

class Linear(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, is_train_good=False):
        super(Linear, self).__init__()
        self.fc1 = nn.Linear(n_input, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, n_hidden)
        self.fc4 = nn.Linear(n_hidden, n_hidden)
        self.fc5 = nn.Linear(n_hidden, n_hidden)
        self.fc6 = nn.Linear(n_hidden, n_output)
        self.relu = nn.ReLU(inplace=True)
        self.is_train_good = is_train_good

        if self.is_train_good == True:
            self.dropout = nn.Dropout(p = 0.5)

    def forward(self, x):
        if self.is_train_good == True:  # if the results on training set is good
            y = self.fc1(x)
            y = self.relu(y)
            y = self.fc2(y)
            y = self.relu(y)
            y = self.dropout(y)
            y = self.fc3(y)
            y = self.relu(y)
            y = self.dropout(y)
            y = self.fc4(y)
            y = self.relu(y)
            y = self.dropout(y)
            y = self.fc5(y)
            y = self.relu(y)
            y = self.dropout(y)
            y = self.fc6(y)
        else:
            y = self.fc1(x)
            y = self.relu(y)
            y = self.fc2(y)
            y = self.relu(y)
            y = self.fc3(y)
            y = self.relu(y)
            y = self.fc4(y)
            y = self.relu(y)
            y = self.fc5(y)
            y = self.relu(y)
            y = self.fc6(y)
        return y
        
