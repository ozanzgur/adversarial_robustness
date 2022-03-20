from xml.dom import xmlbuilder
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import weight_norm

"""
def se_block(in_block, ch, ratio=16):
    x = GlobalAveragePooling2D()(in_block)
    x = Dense(ch//ratio, activation='relu')(x)
    x = Dense(ch, activation='sigmoid')(x)
    return multiply()([in_block, x])"""


class SEBlock(nn.Module):
    def __init__(self, n_channels, ratio=16, **kwargs):
        super(SEBlock, self).__init__()
        self.n_channels = n_channels
        self.hidden_size = n_channels//ratio
        self.sigmoid_output = None
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten1 = nn.Flatten(1)
        self.fc1 = nn.Linear(self.n_channels, self.hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_size, self.n_channels)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, in_block):
        #print(f"in_block.shape={in_block.shape}")
        x = self.pool(in_block)
        #print(f"pool.shape={x.shape}")
        x = self.flatten1(x)
        #print(f"flatten1.shape={x.shape}")
        x = self.fc1(x)
        #print(f"fc1.shape={x.shape}")
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        y = x.clone()
        y[y < 0.3] = 0
        self.sigmoid_output = y.clone()
        # print(f"sigmoid.shape={x.shape}")
        # print(f"in_block.shape={in_block.shape}")
        return y.unsqueeze(-1).unsqueeze(-1) * in_block
    

class Net(nn.Module):
    def __init__(self, **kwargs):
        n_kernels = 20
        
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, n_kernels, kernel_size=5, bias = True)
        self.bn1 = nn.BatchNorm2d(n_kernels)
        self.se1 = SEBlock(n_channels=n_kernels, ratio=5)
        self.relu1 = nn.ReLU()
        
        part_size = n_kernels
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(part_size, part_size*2, kernel_size=3, bias = True)
        self.bn2 = nn.BatchNorm2d(part_size*2)
        self.se2 = SEBlock(n_channels=part_size*2, ratio=5)
        self.relu2 = nn.ReLU()
        
        part_size = n_kernels * 2
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(part_size, part_size*2, kernel_size=3, bias = True)
        self.bn3 = nn.BatchNorm2d(part_size*2)
        self.se3 = SEBlock(n_channels=part_size*2, ratio=5)
        self.relu3 = nn.ReLU()
        
        self.pool3 = nn.MaxPool2d(2)
        self.flatten1 = nn.Flatten(1)
        """self.fc1 = nn.Linear(320, 75)
        self.relu_out = nn.ReLU()"""
        self.fc2 = nn.Linear(320, 10)
        self.softmax = nn.LogSoftmax()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.se1(x)
        x = self.relu1(x)
        
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.se2(x)
        x = self.relu2(x)
        
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.se3(x)
        x = self.relu3(x)
        
        x = self.pool3(x)
        fc1_in = self.flatten1(x)
        """fc1_out = self.fc1(fc1_in)
        relu_out = self.relu_out(fc1_out)"""
        fc2_out = self.fc2(fc1_in)
        logsoftmax_output = self.softmax(fc2_out)
            
        return logsoftmax_output