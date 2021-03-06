import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import weight_norm

def sigmoid_quantize_sparse(x, thresholds):
    output = torch.zeros(size=[x.shape[0], 30, x.shape[-2], x.shape[-1]]).cuda()
    for ch in range(3):
        for i, th in enumerate(thresholds):
            output[:, ch * i] = sigmoid(x[:, ch, :, :], mean=th)
            
    return output
        
def sigmoid(x, multiplier = 1000, mean = 0):
    return 1 / (1 + torch.exp(-(x - mean) * multiplier))

class Net(nn.Module):
    def __init__(self, **kwargs):
        n_kernels = 20
        super(Net, self).__init__()
        
        self.thresholds = torch.tensor([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], requires_grad=False).cuda()
        
        self.conv1 = nn.Conv2d(30, n_kernels, kernel_size=5, bias = True)
        self.bn1 = nn.BatchNorm2d(n_kernels)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(n_kernels, 20, kernel_size=3, bias = True)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.flatten1 = nn.Flatten(1)
        self.fc1 = nn.Linear(500, 50)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(50, 10)
        self.softmax = nn.LogSoftmax()

    def forward(self, x):
        x = sigmoid_quantize_sparse(x)
        conv1_out = self.conv1(x)
        conv1_out = self.bn1(conv1_out)
        relu1_out = self.relu1(conv1_out)
        relu1_out_pool = self.pool1(relu1_out)
        conv2_out = self.conv2(relu1_out_pool)
        relu2_out = self.relu2(conv2_out)
        relu2_out_pool = self.pool2(relu2_out)
        fc1_in = self.flatten1(relu2_out_pool)
        fc1_out = self.fc1(fc1_in)
        relu3_out = self.relu3(fc1_out)
        fc2_out = self.fc2(relu3_out)
        logsoftmax_output = self.softmax(fc2_out)
            
        return logsoftmax_output