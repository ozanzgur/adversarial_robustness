import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import weight_norm
import math

def normalize_2d(w):
    n_kernel, n_ch, height, width = w.shape
    w = (F.normalize(w.view((n_kernel, n_ch, -1)), dim=1)).view(n_kernel, n_ch, height, width)
    return w

class ConvNorm(nn.Module):
    def __init__(self, in_channels, n_kernels):
        super(ConvNorm, self).__init__()
        self.w = nn.Parameter(torch.FloatTensor(n_kernels, in_channels, 1, 1))
        nn.init.xavier_normal_(self.w)
        
    def forward(self, x):
        return F.conv2d(normalize_2d(x), normalize_2d(self.w), padding=0) * 10

class ArcFaceModule(nn.Module):
    def __init__(self, in_features, out_features, s = 10):
        super(ArcFaceModule, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.center = nn.Parameter(torch.FloatTensor(1, in_features) * 0.1)
        self.r = nn.Parameter(torch.ones(1, dtype=torch.float32) * 10)
        nn.init.xavier_normal_(self.weight)
    
    def forward(self, x):
        cos_th = F.normalize(F.linear(F.normalize(x - self.center), F.normalize(self.weight)))
        #cos_th = cos_th.clamp(-1, 1)
        return cos_th * self.r
    
class PlaneProjection(nn.Module):
    def __init__(self, in_features, out_features):
        super(PlaneProjection, self).__init__()
        self.normal = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_normal_(self.normal)
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_normal_(self.weight)
        
    def forward(self, x):
        #print(x.shape)
        normal_normalized = F.normalize(self.normal, dim=0)
        cos_th = F.linear(F.normalize(x), normal_normalized).unsqueeze(1)
        # print(cos_th.shape)
        # print(normal_normalized.transpose(0, 1).unsqueeze(0).shape)
        x = x.unsqueeze(2) - cos_th * normal_normalized.transpose(0, 1).unsqueeze(0)
        x = x + self.normal.transpose(0, 1).unsqueeze(0)
        
        # x: batch, in_features, out_features
        x = x * self.weight.transpose(0, 1).unsqueeze(0)
        x = x.sum(axis=1)
        
        return x
    
class PlaneProjection2(nn.Module):
    def __init__(self, in_features, out_features):
        super(PlaneProjection2, self).__init__()
        self.normal = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_normal_(self.normal)
        
        self.plane = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_normal_(self.plane)
        
    def project_w_onto_normal(self, x):
        cos_th = (F.normalize(x, dim=1) * F.normalize(self.normal, dim=1)).sum(axis=1, keepdim=True)
        x = x - self.normal * cos_th
        x = F.normalize(x, dim=1)
        return x
        
    def forward(self, x):
        normal_normalized = F.normalize(self.normal, dim=0)
        cos_th = F.linear(x, normal_normalized).unsqueeze(1)
        x = x.unsqueeze(2) - cos_th * normal_normalized.transpose(0, 1).unsqueeze(0)
        
        plane_proj_to_normal = self.project_w_onto_normal(self.plane)
        # print(x.shape)
        # print(plane_proj_to_normal.shape)
        x = F.normalize(x, dim=1)
        cos_th_plane = (x * plane_proj_to_normal.transpose(0, 1).unsqueeze(0)).sum(axis=1)
        # print(cos_th_plane.shape)
        return cos_th_plane
        
class Net(nn.Module):
    def __init__(self, **kwargs):
        super(Net, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5, bias = True)
        self.bn1 = nn.BatchNorm2d(20)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        
        #self.conv2 = ConvNorm(20, 5)
        self.conv2 = nn.Conv2d(20, 5, kernel_size=1, bias = True)
        self.bn2 = nn.BatchNorm2d(5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        
        self.flatten1 = nn.Flatten(1)
        self.fc1 = PlaneProjection2(in_features=180, out_features=40)
        self.relu4 = nn.ReLU()
        self.fc2 = ArcFaceModule(in_features=40, out_features=10)
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
        x = self.relu4(x)
        x = self.fc2(x)
        x = self.softmax(x)
            
        return x

