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

class TinyNet(nn.Module):
    def __init__(self, n_classes = 10, softmax = True, **kwargs):
        n_kernels = 20
        self.softmax = softmax
        
        super(TinyNet, self).__init__()
        
        self.downscale = nn.AvgPool2d(3, stride=3)
        self.conv1 = nn.Conv2d(1, n_kernels, kernel_size=3, bias = True)
        self.bn1 = nn.BatchNorm2d(n_kernels)
        self.activation = nn.ReLU()
        self.feature_gate = nn.Parameter(torch.ones(1, n_kernels, 7, 7), requires_grad=False)
        self.pool = nn.MaxPool2d(2)
        
        self.flatten1 = nn.Flatten(1)
        self.fc2 = nn.Linear(180, n_classes)
        if self.softmax:
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
        
        if self.softmax:
            x = self.softmax(x)
            
        return x

class TwoStageNet(nn.Module):
    def __init__(self, n_classes = 10, **kwargs):
        super(TwoStageNet, self).__init__()
        self.stage_2_enabled = True
        self.model_template = "{}_{}"
        
        self.stage1 = TinyNet(n_classes=10, softmax=True)
        self.softmax = nn.LogSoftmax()
        
        # Create stage 2 nets
        self.stage2_nets = {}
        for i1 in range(n_classes):
            for i2 in range(i1+1, n_classes):
                self.stage2_nets[self.model_template.format(i1, i2)] = TinyNet(n_classes=2, softmax=False).cuda()
                
        print(f"# stage 2 models: {len(self.stage2_nets)}")
                

    def forward(self, x):
        batch_size = x.shape[0]
        stage1_output = self.stage1(x)
        
        if not self.stage_2_enabled:
            return stage1_output
        
        stage2_idx = stage1_output.argsort(axis=1)[:, -2:].sort(axis=1).values
        output = torch.zeros((batch_size, 10)).cuda()
        
        for i_example in range(batch_size):
            stage2_name = self.model_template.format(stage2_idx[i_example, 0], stage2_idx[i_example, 1])
            stage2_output = self.stage2_nets[stage2_name](x[i_example].unsqueeze(0))
            
            # Map 2 class output to 10 class output
            output[i_example, stage2_idx[0]] = stage2_output[0, 0]
            output[i_example, stage2_idx[1]] = stage2_output[0, 1]
        
        output = self.softmax(output)
        return [stage1_output, output]