import torch.nn as nn

class Linear(nn.Module):
    def __init__(self, n_hidden, is_train_good=False):
        super(Linear, self).__init__()
        self.fc1 = nn.Linear(n_hidden, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.bn1 = nn.BatchNorm1d(n_hidden)
        self.bn2 = nn.BatchNorm1d(n_hidden)
        self.relu = nn.ReLU(inplace=True)
        self.is_train_good = is_train_good

        if self.is_train_good == True:
            self.dropout = nn.Dropout(p = 0.5)

    def forward(self, x):
        if self.is_train_good == True:  # if the results on training set is good
            y = self.fc1(x)
            y = self.bn1(y)
            y = self.relu(y)
            y = self.dropout(y)

            y = self.fc2(y)
            y = self.bn2(y)
            y = self.relu(y)
            y = self.dropout(y)

            out = x + y
        else:
            y = self.fc1(x)
            y = self.bn1(y)
            y = self.relu(y)

            y = self.fc2(y)
            y = self.bn2(y)
            y = self.relu(y)

            out = x + y
        return out

class LinearModel(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, is_train_good=False):
        super(LinearModel, self).__init__()
        self.num_stage = 2
        self.is_train_good = is_train_good
        self.fc1 = nn.Linear(n_input, n_hidden)
        self.bn1 = nn.BatchNorm1d(n_hidden)

        self.linear_stages = []
        for l in range(self.num_stage):
            self.linear_stages.append(Linear(n_hidden, self.is_train_good))
        self.linear_stages = nn.ModuleList(self.linear_stages)

        self.fc2 = nn.Linear(n_hidden, n_output)
        self.relu = nn.ReLU(inplace=True)
        if self.is_train_good == True:
            self.dropout = nn.Dropout(p = 0.5)

    def forward(self, x):
        if self.is_train_good == True:  # if the results on training set is good
            y = self.fc1(x)
            y = self.bn1(y)
            y = self.relu(y)
            y = self.dropout(y)

            for i in range(self.num_stage):
                y = self.linear_stages[i](y)

            y = self.fc2(y) 
        else:
            y = self.fc1(x)
            y = self.bn1(y)
            y = self.relu(y)

            for i in range(self.num_stage):
                y = self.linear_stages[i](y)

            y = self.fc2(y) 
        return y
        
