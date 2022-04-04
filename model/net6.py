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
    def __init__(self, n_channels, h, w, ratio=16, **kwargs):
        super(SEBlock, self).__init__()
        self.fixed_weights = nn.parameter.Parameter(torch.ones(1, n_channels, h, w), requires_grad=False)
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
        if self.training or self.fixed_weights is None:
            x = self.pool(in_block)
            x = self.flatten1(x)
            x = self.fc1(x)
            x = self.relu1(x)
            x = self.fc2(x)
            x = self.sigmoid(x)
            
            y = x.clone()
            y = y.mean(axis=0)
            y = y.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            
            if self.training:
                #with torch.no_grad():
                if self.fixed_weights is None:
                    self.fixed_weights.data = y.clone()
                    #self.fixed_weights = self.fixed_weights.detach()
                else:
                    self.fixed_weights.data =  self.fixed_weights.data * 0.9 + y * 0.1 #
                    #self.fixed_weights = self.fixed_weights.detach()
            
            final_y = self.fixed_weights.data#.clone()
            self.sigmoid_output = final_y
            #final_y[final_y < 0.3] = 0
            return final_y * in_block
        else:
            final_y = self.fixed_weights.data.clone()
            #final_y[final_y < 0.3] = 0
            return final_y * in_block
        
    
class SEBlock2(nn.Module):
    def __init__(self, n_channels, h, w, ratio=16, **kwargs):
        super(SEBlock2, self).__init__()
        self.fixed_weights = nn.parameter.Parameter(torch.ones(1, n_channels, h, w), requires_grad=False)
        self.n_channels = n_channels
        self.hidden_size = n_channels//ratio
        self.sigmoid_output1 = None
        self.sigmoid_output2 = None
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten1 = nn.Flatten(1)
        self.fc1 = nn.Linear(self.n_channels, self.hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_size, self.n_channels)
        self.sigmoid = nn.Sigmoid()
        
        self.hidden_size1 = self.n_channels//ratio
        self.conv1 = nn.Conv2d(self.n_channels, self.hidden_size1, kernel_size=1, bias = True, padding=0)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(self.hidden_size1, self.n_channels, kernel_size=1, bias = True)
        self.sigmoid1 = nn.Sigmoid()
        #self.register_buffer("fixed_weights", torch.zeros((1, 1, n_channels, h, w)))
        
    def forward(self, in_block):
        if self.training:
            x = self.pool(in_block)
            x = self.flatten1(x)
            x = self.fc1(x)
            x = self.relu1(x)
            x = self.fc2(x)
            x = self.sigmoid(x)
            y = x.clone()
            y = y.mean(axis=0)
            y = y.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            #y[y<0.03] = 0
            in_block1 = y * in_block
            self.sigmoid_output1 = y
            
            z = self.conv1(in_block1)
            z = self.relu2(z)
            z = self.conv2(z)
            #z = z.mean(axis=0).unsqueeze(0)
            z = self.sigmoid1(z)
            t = z.clone()
            t = t.mean(axis=0).unsqueeze(0)
            #t[t < 0.03] = 0
            self.sigmoid_output2 = t
            
            #if self.training:
            if self.fixed_weights is None:
                self.fixed_weights.data = t * y
                #self.fixed_weights = self.fixed_weights.detach()
            else:
                self.fixed_weights.data =  self.fixed_weights.data * 0.9 + (t * y) * 0.1 #
                #self.fixed_weights = self.fixed_weights.detach()
            
            #print(self.fixed_weights.data.mean())
            final_y = self.fixed_weights.data#.clone()
            #self.sigmoid_output = final_y
            return final_y * in_block
        else:
            final_y = self.fixed_weights.data.clone()
            self.sigmoid_output = final_y
            #final_y[final_y < 0.3] = 0
            return final_y * in_block
    
class SEBlock3(nn.Module):
    def __init__(self, n_channels, ratio=16, **kwargs):
        super(SEBlock2, self).__init__()
        self.n_channels = n_channels
        self.hidden_size = n_channels//ratio
        self.sigmoid_output1 = None
        self.sigmoid_output2 = None
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten1 = nn.Flatten(1)
        self.fc1 = nn.Linear(self.n_channels, self.hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_size, self.n_channels)
        self.sigmoid = nn.Sigmoid()
        
        self.hidden_size1 = self.n_channels//ratio
        self.conv1 = nn.Conv2d(self.n_channels, self.hidden_size1, kernel_size=1, bias = True)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(self.hidden_size1, self.n_channels, kernel_size=1, bias = True)
        self.sigmoid1 = nn.Sigmoid()
        
        
    def forward(self, in_block):
        x = self.pool(in_block)
        x = self.flatten1(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        y = x.clone()
        y[y < 0.3] = 0
        self.sigmoid_output1 = y.clone()
        in_block1 = y.unsqueeze(-1).unsqueeze(-1) * in_block
        
        z = in_block1.mean(axis=1, keep_dims=True)
        z = self.conv1(in_block1)
        z = self.relu1(z)
        z = self.conv2(z)
        z = self.sigmoid1(z)
        t = z.clone()
        t[t < 0.3] = 0
        t[y < 0.3] = 0
        self.sigmoid_output2 = t.clone()
        self.sigmoid_output = self.sigmoid_output2
        
        return t * in_block1
    

class Net(nn.Module):
    def __init__(self, **kwargs):
        n_kernels = 20
        
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, n_kernels, kernel_size=5, bias = True)
        self.bn1 = nn.BatchNorm2d(n_kernels)
        self.se1 = SEBlock2(n_channels=n_kernels, ratio=5, h=28, w=28)
        self.relu1 = nn.ReLU()
        
        part_size = n_kernels
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(part_size, part_size*2, kernel_size=3, bias = True)
        self.bn2 = nn.BatchNorm2d(part_size*2)
        self.se2 = SEBlock2(n_channels=part_size*2, ratio=5, h=12, w=12)
        self.relu2 = nn.ReLU()
        
        part_size = n_kernels * 2
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(part_size, part_size*2, kernel_size=3, bias = True)
        self.bn3 = nn.BatchNorm2d(part_size*2)
        self.se3 = SEBlock(n_channels=part_size*2, ratio=5, h=4, w=4)
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