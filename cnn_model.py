import torch.nn as nn
import torch.nn.functional as F
import logging
FORMAT = '%(filename)s line:%(lineno)d\t%(message)s'
logging.basicConfig(level=logging.INFO,format=FORMAT)
print = logging.info

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 30 * 30, 15)
        self.fc2 = nn.Linear(15, 3)

    def forward(self, x):
        # print(x.shape)
        x = self.pool(F.relu(self.conv1(x)))
        # print(x.shape)
        x = self.pool(F.relu(self.conv2(x)))
        # print(x.shape)
        x = x.view(-1, 16 * 30 * 30)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x