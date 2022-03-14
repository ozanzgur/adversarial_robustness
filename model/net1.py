import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import weight_norm

# activation_stats1 = []
# conv1_out = None
# conv2_out = None
# fc1_out = None
# fc2_out = None
# example = None

def batchnorm_inverse(output, layer):
    """Calculate batch norm input from output and layer parameters
    
    """
    beta = layer.bias.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    gamma = layer.weight.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    mean = layer.running_mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    var = (torch.sqrt(layer.running_var + layer.eps)).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    return (output - beta) / gamma * var + mean

class Net(nn.Module):
    def __init__(self, n_kernels=10, n_active_layers='all'):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, n_kernels, kernel_size=5, bias = True)
        self.bn1 = nn.BatchNorm2d(n_kernels)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(n_kernels, 20, kernel_size=5, bias = True)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(320, 50)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(50, 10)
        self.softmax = nn.LogSoftmax()

        self.nn_layers = [self.conv1, self.conv2, self.fc1, self.fc2]
        self.n_active_layers = n_active_layers
        self.n_layers = len(self.nn_layers)
        
        if isinstance(self.n_active_layers, str):
            if not self.n_active_layers == 'all':
                raise ValueError("n_active_layers can be an int or 'all'")

    def forward(self, x):
        if self.n_active_layers < 1:
            raise ValueError("n_active_layers cannot be less than 1")
        
        conv1_out = self.conv1(x)
        #conv1_out = self.bn1(conv1_out)
        relu1_out = self.relu1(conv1_out)
        
        if self.n_active_layers == 1:
            return x, relu1_out # conv1_out #

        relu1_out_pool = self.pool1(relu1_out)
        if self.n_active_layers == 'all' or self.n_active_layers > 1:
            conv2_out = self.conv2(relu1_out_pool)
            relu2_out = self.relu2(conv2_out)
            if self.n_active_layers == 2:
                return relu1_out_pool, relu2_out # conv2_out #

        relu2_out_pool = self.pool2(relu2_out)
        if self.n_active_layers == 'all' or self.n_active_layers > 2:
            fc1_in = torch.flatten(relu2_out_pool, 1)
            fc1_out = self.fc1(fc1_in)
            relu3_out = self.relu3(fc1_out)
            
            if self.n_active_layers == 3:
                return fc1_in, relu3_out # fc1_out #

        if self.n_active_layers == 'all' or self.n_active_layers > 3:
            fc2_out = self.fc2(relu3_out)
            logsoftmax_output = self.softmax(fc2_out)
            
            return relu3_out, logsoftmax_output
    
    def layer_reconstruction_loss(self, i_layer, layer_output, layer_input, is_conv = False, class_idx = None, bn_layer = None):
        layer = self.nn_layers[i_layer]
        #print(layer_output.size())
        bias_shape = [1] * len(layer_output.size())
        bias_shape [1] = -1
        #print(f'f i_layer: {i_layer} ==============================================')
        #print(f'f bias_shape: {bias_shape}')
        reflection = None
        #print(f'this_layer_input: {layer_input.size()}')
        #print(f'this_layer_output: {layer_output.size()}')
        if is_conv:
            reflection = layer_output.clone()
            if not bn_layer is None:
                reflection = batchnorm_inverse(reflection, bn_layer)
            
            assert class_idx is None, "class_idx must be None if is_conv=True"

            reflection = F.conv_transpose2d(reflection, self.nn_layers[i_layer].weight)# * self.deconv1_n_steps
        else:
            output_unsqueeze = (layer_output - self.nn_layers[i_layer].bias.reshape(bias_shape)).unsqueeze(2)
            W = self.nn_layers[i_layer].weight.unsqueeze(0)
            #print(f'W: {W.size()}')
            #print(f'this_layer_output_unsqueeze: {output_unsqueeze.size()}')

            reflection = (output_unsqueeze * W)
            if not class_idx is None:
                #print(f'ref: {reflection.size()}')
                reflection = reflection.index_select(1, class_idx)
            else:
                reflection = torch.sum(reflection, axis = 1)
        
        """if i_layer > 0: # ???
            reflection[reflection < 0] = 0.0"""
        
        """print(f'is_conv: {is_conv}')
        print(f'reflection size: {reflection.size()}')
        print(f'this_layer_input: {layer_input.size()}')"""
        
        reconstruction_error = torch.square(reflection - layer_input)# * torch.abs(layer_input)
        loss = torch.sqrt(torch.sum(reconstruction_error))# MSE loss math.sqrt()
        return loss


    def total_reconstruction_loss(self, inputs, outputs):
        layer0_loss = 0
        layer1_loss = 0
        layer2_loss = 0
        layer3_loss = 0
        layer2_class_loss = 0
        class_loss = 0

        if self.n_active_layers == 1:
            i_layer = 0
            layer0_loss = self.layer_reconstruction_loss(i_layer, outputs, inputs, is_conv = True) # , bn_layer = self.bn1
            #layer0_losses.append(layer0_loss)

        if self.n_active_layers == 2:
            i_layer = 1
            layer1_loss = self.layer_reconstruction_loss(i_layer, outputs, inputs, is_conv = True)
            #layer1_losses.append(layer1_loss)

        if self.n_active_layers == 3:
            i_layer = 2
            layer2_loss = self.layer_reconstruction_loss(i_layer, outputs, inputs)

            #layer2_class_loss = - torch.mean(torch.max(torch.exp(outputs[i_layer]), dim = 1)[0]) * 1.0
            #layer2_class_losses.append(layer2_class_loss)
            #layer2_losses.append(layer2_loss)

        if self.n_active_layers == 4:
            i_layer = 3
            #max_vals, max_idx = torch.max(torch.exp(outputs), dim = 1)

            layer3_loss = self.layer_reconstruction_loss(i_layer, outputs, inputs)
            #layer3_losses.append(layer3_loss)
            
            #print(max_vals.size())
            #class_loss = - torch.mean(max_vals) * 0.01

        #class_losses.append(class_loss)

        total_loss = layer0_loss + layer1_loss + layer2_loss + layer3_loss# + class_loss# + layer2_class_loss
        return total_loss