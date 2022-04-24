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
    def __init__(self, n_classes = 10, softmax = True, do_downscale = True, **kwargs):
        n_kernels = 20
        self.softmax = softmax
        self.do_downscale = do_downscale
        
        super(TinyNet, self).__init__()
        
        if do_downscale:
            self.downscale = nn.AvgPool2d(3, stride=3)
        self.conv1 = nn.Conv2d(1, n_kernels, kernel_size=3, bias = True)
        self.bn1 = nn.BatchNorm2d(n_kernels)
        self.activation = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        
        self.flatten1 = nn.Flatten(1)
        self.fc2 = nn.Linear(180, n_classes)
        if self.softmax:
            self.softmax = nn.LogSoftmax()

    def forward(self, x):
        if self.do_downscale:
            x = self.downscale(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.pool(x)
        x = self.flatten1(x)
        x = self.fc2(x)
        
        if self.softmax:
            x = self.softmax(x)
            
        return x
    
class TinyNet2(nn.Module):
    def __init__(self, n_classes = 10, softmax = True, do_downscale = True, **kwargs):
        n_kernels = 15
        self.softmax = softmax
        self.do_downscale = do_downscale
        
        super(TinyNet2, self).__init__()
        
        if do_downscale:
            self.downscale = nn.AvgPool2d(3, stride=3)
        self.conv1 = nn.Conv2d(1, n_kernels, kernel_size=5, bias = True)
        self.bn1 = nn.BatchNorm2d(n_kernels)
        self.activation = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        
        self.flatten1 = nn.Flatten(1)
        self.fc2 = nn.Linear(60, n_classes)
        if self.softmax:
            self.softmax = nn.LogSoftmax()

    def forward(self, x):
        if self.do_downscale:
            x = self.downscale(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.pool(x)
        x = self.flatten1(x)
        x = self.fc2(x)
        
        if self.softmax:
            x = self.softmax(x)
            
        return x
    
class Stage2Net(nn.Module):
    def __init__(self, n_classes = 10, softmax = True, do_downscale = True, **kwargs):
        n_kernels = 20
        self.softmax = softmax
        
        super(Stage2Net, self).__init__()
        self.conv1 = nn.Conv2d(1, n_kernels, kernel_size=5, bias = True)
        self.bn1 = nn.BatchNorm2d(n_kernels)
        self.activation = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        
        self.conv2 = nn.Conv2d(n_kernels, n_kernels * 2, kernel_size=5, bias = True)
        self.bn2 = nn.BatchNorm2d(n_kernels * 2)
        self.activation2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        
        self.flatten1 = nn.Flatten(1)
        self.fc1 = nn.Linear(16*40, 50)
        self.fc2 = nn.Linear(50, n_classes)
        if self.softmax:
            self.softmax = nn.LogSoftmax()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation2(x)
        x = self.pool2(x)
        
        x = self.flatten1(x)
        x = self.fc1(x)
        x = self.fc2(x)
        
        if self.softmax:
            x = self.softmax(x)
            
        return x

class TwoStageNet(nn.Module):
    def __init__(self, n_classes = 10, **kwargs):
        super(TwoStageNet, self).__init__()
        self.stage_2_enabled = True
        self.output_stage1 = True
        self.model_template = "{}_{}"
        
        self.stage1 = TinyNet(n_classes=10, softmax=False)
        self.softmax1 = nn.LogSoftmax()
        self.softmax2 = nn.LogSoftmax()
        
        # Create stage 2 nets
        self.stage2_nets = nn.ModuleDict()
        for i1 in range(n_classes):
            for i2 in range(i1+1, n_classes):
                model_name = self.model_template.format(i1, i2)
                self.stage2_nets[model_name] = TinyNet2(n_classes=2, softmax=False).cuda()
                #self.stage2_nets[model_name].train()
                
        print(f"# stage 2 models: {len(self.stage2_nets)}")
                

    def forward(self, x):
        batch_size = x.shape[0]
        stage1_output = self.stage1(x)
        stage1_output_softmax = self.softmax1(stage1_output)
        
        if not self.stage_2_enabled:
            return stage1_output_softmax
        
        stage2_idx = stage1_output_softmax.argsort(axis=1)[:, -2:].sort(axis=1).values
        stage2_closeness = stage1_output_softmax.sort(axis=1).values[:, -3:-1]
        stage2_closeness = stage2_closeness[:, 1] - stage2_closeness[:, 0]
        #print(stage2_idx[0])
        output = torch.ones((batch_size, 2 if self.training else 10), requires_grad=True).cuda() * -100
        stage2_names = []
        
        for i_example in range(batch_size):
            used_stage1 = False
            stage2_name = self.model_template.format(stage2_idx[i_example, 0], stage2_idx[i_example, 1])
            stage2_names.append(stage2_name)
            
            if not self.training:
                if stage1_output_softmax[i_example].max() > 0.3:
                    output[i_example] = 0 if self.training else stage1_output[i_example]
                    used_stage1 = True
                
            """if stage2_closeness[i_example] < 0.15 and not used_stage1:
                output[i_example] = 0 if self.training else stage1_output[i_example]
                used_stage1 = True"""
                
            if not used_stage1:
                
                stage2_output = self.stage2_nets[stage2_name](x[i_example].unsqueeze(0))
                
                if self.training:
                    output[i_example] = stage2_output[0]
                    
                else:
                    output[i_example, stage2_idx[i_example, 0]] = stage2_output[0, 0]
                    output[i_example, stage2_idx[i_example, 1]] = stage2_output[0, 1]
            
            """if stage2_idx[i_example, 0] == 1 and stage2_idx[i_example, 1] == 7:
                print("Out:")
                print(stage2_output)
                print(output[i_example])"""
        
        """print("OUT:")
        print(stage2_idx[0])
        print(stage2_output[0])
        print(stage1_output[0])
        print(output[0])"""
        output = self.softmax2(output)
        
        if self.output_stage1:
            return [stage1_output, output, stage2_names]
        else:
            return output