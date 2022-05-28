import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import weight_norm
import math

class NoiseLayer(nn.Module):
    def __init__(self, std = 0.1, **kwargs):
        super(NoiseLayer, self).__init__()
        self.std = std
        
    def forward(self, x):
        return x + torch.randn(x.shape[-1]).cuda() * self.std

class Net(nn.Module):
    def __init__(self, **kwargs):
        super(Net, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 20, kernel_size=3, bias = True)
        self.bn1 = nn.BatchNorm2d(20)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        
        self.conv2 = nn.Conv2d(20, 5, kernel_size=1, bias = True)
        self.bn2 = nn.BatchNorm2d(5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        
        self.flatten1 = nn.Flatten(1)
        self.fc1 = nn.Linear(245, 40)
        #self.noise1 = NoiseLayer(std=1)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(40, 10)
        self.noise2 = NoiseLayer(std=0.02)
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
        
        x = self.flatten1(x)
        x = self.fc1(x)
        #x = self.noise1(x)
        x = self.relu4(x)
        x = self.fc2(x)
        x = self.noise2(x)
        x = self.softmax(x)
            
        return x

