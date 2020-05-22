import torch.nn as nn
import torch.nn.functional as F
import logging

FORMAT = '%(filename)s line:%(lineno)d\t%(message)s'
logging.basicConfig(level=logging.INFO,format=FORMAT)
print = logging.info

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.fc1 = nn.Linear(64 * 29 * 29, 128)
        self.fc2 = nn.Linear(128, 3)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        # x = self.pool(F.relu(self.conv2(x)))
        x = F.dropout(self.pool(F.relu(self.conv2(x))),p=0.25,training=self.training)
        x = F.dropout(self.pool(F.relu(self.conv3(x))),p=0.25,training=self.training)

        # print(x.shape)    
        x = x.view(-1, 64 * 29 * 29)
    
        x = F.dropout(F.relu(self.fc1(x)),p=0.25,training=self.training)
        x = self.fc2(x)
        return x
