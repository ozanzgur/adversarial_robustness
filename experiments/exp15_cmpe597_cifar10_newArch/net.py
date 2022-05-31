import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import weight_norm
import math

class RouterNet(nn.Module):
    def __init__(self, **kwargs):
        super(RouterNet, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 20, kernel_size=3, bias = True)
        self.bn1 = nn.BatchNorm2d(20)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        
        self.conv2 = nn.Conv2d(20, 5, kernel_size=1, bias = True)
        self.bn2 = nn.BatchNorm2d(5)
        self.relu2 = nn.ReLU()
        
        self.flatten1 = nn.Flatten(1)
        self.fc1 = nn.Linear(1125, 64)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        x = self.flatten1(x)
        x = self.fc1(x)
        x = F.normalize(x)
            
        return x

class NetLvl2(nn.Module):
    def __init__(self, **kwargs):
        super(NetLvl2, self).__init__()
        self.w = nn.Parameter(torch.FloatTensor(1, 64))
        nn.init.xavier_normal_(self.w)

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
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(40, 10)
        self.softmax = nn.LogSoftmax()

    def get_latent_out(self, x):
        return F.linear(F.normalize(self.w), x).squeeze()

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
        x = self.relu4(x)
        x = self.fc2(x)
        x = self.softmax(x)
            
        return x

class Net(nn.Module):
    def __init__(self, **kwargs):
        super(Net, self).__init__()
        self.lvl1net = RouterNet()
        self.lvl2nets = nn.ModuleDict({f"lvl2net_{i}": NetLvl2() for i in range(5)})
        self.router_softmax = nn.Softmax()
        self.softmax = nn.LogSoftmax()

        self.report_i = 0
        self.report_freq = 50

    def forward(self, x):
        y_latent = self.lvl1net(x)
        lvl2_similarities = torch.stack([n.get_latent_out(y_latent) for k, n in self.lvl2nets.items()], dim=1)
        lvl2_similarities = self.router_softmax(lvl2_similarities)
        #print(lvl2_similarities[0])
        #print(lvl2_similarities.shape)
        #lvl2_similarities = lvl2_similarities / lvl2_similarities.sum(dim=1, keepdims=True)

        #if self.training:
        lvl2_outputs = torch.zeros((x.shape[0], len(self.lvl2nets), 10), dtype=torch.float32).cuda()
        for i, (k, n) in enumerate(self.lvl2nets.items()):
            lvl2_outputs[:, i, :] = n(x)

        lvl2_outputs = lvl2_outputs * lvl2_similarities.unsqueeze(-1)
        lvl2_outputs = torch.sum(lvl2_outputs, dim=1)
        lvl2_outputs = self.softmax(lvl2_outputs)
        return lvl2_outputs

        """else:
            self.report_i += 1
            max_lvl2_i = torch.argmax(lvl2_similarities, dim=1)[0]

            if self.report_i == self.report_freq:
                self.report_i = 0
            return list(self.lvl2nets.values())[max_lvl2_i](x)"""
            
