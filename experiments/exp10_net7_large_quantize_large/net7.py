from xml.dom import xmlbuilder
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import weight_norm
import numpy as np

def sigmoid_quantize(x, thresholds):
    x_out = torch.zeros(x.shape).cuda()
    for i_ch in range(x.shape[1]):
        output = torch.zeros([x.shape[0], thresholds.shape[0], x.shape[-2], x.shape[-1]]).cuda()
        for i, th in enumerate(thresholds):
            output[:, i] = sigmoid(x[:, i_ch, :, :], mean=th)
            
        output = output.mean(axis=1)
        x_out[:, i_ch] = output
    
    return x_out

def sigmoid_quantize_stack(x, thresholds):
    output = torch.zeros(size=[x.shape[0], x.shape[1] * len(thresholds), x.shape[-2], x.shape[-1]]).cuda()
    for ch in range(3):
        for i, th in enumerate(thresholds):
            output[:, ch * i] = sigmoid(x[:, ch, :, :], mean=th)
            
    return output
        
def sigmoid(x, multiplier = 10000, mean = 0):
    return 1 / (1 + torch.exp(-(x - mean) * multiplier))

def downscale(x, size = 3):
    return torch.AvgPool2d()

class Net(nn.Module):
    def __init__(self, **kwargs):
        n_kernels = 20
        
        super(Net, self).__init__()
        self.thresholds = torch.tensor(np.linspace(0.07, 0.93, 15), requires_grad=False).cuda()
        
        self.downscale = nn.AvgPool2d(3, stride=3)
        self.conv1 = nn.Conv2d(3, n_kernels, kernel_size=5, bias = True)
        self.bn1 = nn.BatchNorm2d(n_kernels)
        self.activation1 = nn.ReLU()
        
        self.pool2 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(n_kernels, n_kernels*2, kernel_size=3, bias = True)
        self.bn2 = nn.BatchNorm2d(n_kernels*2)
        self.activation2 = nn.ReLU()
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten1 = nn.Flatten(1)
        self.fc2 = nn.Linear(40, 10)
        self.softmax = nn.LogSoftmax()

    def forward(self, x):
        """output = torch.zeros(size=[x.shape[0], 15, x.shape[-2], x.shape[-1]]).cuda()
        for ch in range(3):
            for i, th in enumerate(self.thresholds):
                output[:, ch * i] = sigmoid(x[:, ch, :, :], mean=th)"""
        
        x = self.downscale(x)
        x = sigmoid_quantize(x, self.thresholds)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation1(x)
        
        x = self.pool2(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation2(x)
        
        x = self.avgpool(x)
        x = self.flatten1(x)
        x = self.fc2(x)
        x = self.softmax(x)
            
        return x