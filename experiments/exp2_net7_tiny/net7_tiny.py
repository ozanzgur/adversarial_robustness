from xml.dom import xmlbuilder
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import weight_norm
import numpy as np

def downscale(x, size = 3):
    return torch.AvgPool2d()

class Net(nn.Module):
    def __init__(self, **kwargs):
        n_kernels = 20
        
        super(Net, self).__init__()
        
        self.downscale = nn.AvgPool2d(3, stride=3)
        self.conv1 = nn.Conv2d(1, n_kernels, kernel_size=3, bias = True)
        self.bn1 = nn.BatchNorm2d(n_kernels)
        self.activation = nn.ReLU()
        self.feature_gate = nn.Parameter(torch.ones(1, n_kernels, 7, 7), requires_grad=False)
        self.pool = nn.MaxPool2d(2)
        
        self.flatten1 = nn.Flatten(1)
        self.fc2 = nn.Linear(180, 10)
        self.softmax = nn.LogSoftmax()

    def forward(self, x):
        x = self.downscale(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.feature_gate * x
        x = self.pool(x)
        x = self.flatten1(x)
        x = self.fc2(x)
        x = self.softmax(x)
            
        return x