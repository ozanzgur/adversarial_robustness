import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import weight_norm

class Net(nn.Module):
    def __init__(self, **kwargs):
        n_kernels = 80
        
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, n_kernels, kernel_size=5, bias = True)
        self.bn1 = nn.BatchNorm2d(n_kernels)
        self.relu1 = nn.ReLU()
        
        part_size = n_kernels
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(part_size, part_size*2, kernel_size=3, bias = True)
        self.bn2 = nn.BatchNorm2d(part_size*2)
        self.relu2 = nn.ReLU()
        
        part_size = n_kernels * 2
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(part_size, part_size*2, kernel_size=3, bias = True)
        self.bn3 = nn.BatchNorm2d(part_size*2)
        self.relu3 = nn.ReLU()
        
        self.pool3 = nn.MaxPool2d(2)
        self.flatten1 = nn.Flatten(1)
        self.fc1 = nn.Linear(1280, 75)
        self.relu_out = nn.ReLU()
        self.fc2 = nn.Linear(75, 10)
        self.softmax = nn.LogSoftmax()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        
        x = self.pool3(x)
        fc1_in = self.flatten1(x)
        fc1_out = self.fc1(fc1_in)
        relu_out = self.relu_out(fc1_out)
        fc2_out = self.fc2(relu_out)
        logsoftmax_output = self.softmax(fc2_out)
            
        return logsoftmax_output