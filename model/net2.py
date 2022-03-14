import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import weight_norm

class Net(nn.Module):
    def __init__(self, n_kernels=10, n_active_layers='all'):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, n_kernels, kernel_size=5, bias = True)
        self.conv2 = nn.Conv2d(n_kernels, n_kernels*2, kernel_size=5, bias = True)
        self.conv3 = nn.Conv2d(n_kernels*2, n_kernels*4, kernel_size=3, bias = True)
        self.fc1 = nn.Linear(160, 50)
        self.fc2 = nn.Linear(50, 10)

        self.nn_layers = [self.conv1, self.conv2, self.conv3, self.fc1, self.fc2]
        self.n_active_layers = n_active_layers
        self.n_layers = len(self.nn_layers)
        
        if isinstance(self.n_active_layers, str):
            if not self.n_active_layers == 'all':
                raise ValueError("n_active_layers can be an int or 'all'")

    def forward(self, x):
        # global activation_stats1
        # global conv1_out
        # global conv2_out
        # global fc1_out
        
        inputs = [x]
        outputs = []

        relu1_out = F.relu(self.conv1(x))
        outputs.append(relu1_out)

        """activation_stats1.append(relu1_out.detach().cpu().numpy())
        conv1_out = activation_stats1[-1]"""

        if self.n_active_layers == 'all' or self.n_active_layers > 1:
            #print(relu1_out.shape)
            relu1_out_pool = F.max_pool2d(relu1_out, 2)
            relu2_out = F.relu(self.conv2(relu1_out_pool))

            inputs.append(relu1_out_pool)
            outputs.append(relu2_out)
            #conv2_out = relu2_out.detach().cpu().numpy()
            
        if self.n_active_layers == 'all' or self.n_active_layers > 2:
            #print(relu1_out.shape)
            relu2_out_pool = F.max_pool2d(relu2_out, 2)
            relu3_out = F.relu(self.conv3(relu2_out_pool))

            inputs.append(relu2_out_pool)
            outputs.append(relu3_out)
            #conv2_out = relu2_out.detach().cpu().numpy()

        if self.n_active_layers == 'all' or self.n_active_layers > 3:
            relu3_out_pool = F.max_pool2d(relu3_out, 2)
            fc1_in = torch.flatten(relu3_out_pool, 1)
            #print(fc1_in.shape)
            fc1_out = F.relu(self.fc1(fc1_in))
            inputs.append(fc1_in)
            outputs.append(fc1_out)

        if self.n_active_layers == 'all' or self.n_active_layers > 4:
            fc2_out = self.fc2(fc1_out)
            logsoftmax_output = F.log_softmax(fc2_out)
            inputs.append(fc1_out)
            outputs.append(fc2_out)
            outputs.append(logsoftmax_output)

        return inputs, outputs
    
    def layer_reconstruction_loss(self, i_layer, layer_output, layer_input, is_conv = False, class_idx = None):
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
            assert class_idx is None, "class_idx must be None if is_conv=True"

            reflection = F.conv_transpose2d(layer_output, self.nn_layers[i_layer].weight)# * self.deconv1_n_steps
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

        if i_layer > 0:
            reflection[reflection < 0] = 0.0
        
        reconstruction_error = torch.square(reflection - layer_input)# * torch.abs(layer_input)
        loss = torch.mean(reconstruction_error)# * layer_input.size()[1]# MSE loss math.sqrt()
        return loss


    def total_reconstruction_loss(self, inputs, outputs):
        # global layer0_losses
        # global layer1_losses
        # global layer2_losses
        # global layer3_losses

        layer0_loss = 0
        layer1_loss = 0
        layer2_loss = 0
        layer3_loss = 0
        layer4_loss = 0
        layer2_class_loss = 0
        class_loss = 0

        if self.n_active_layers == 1:
            i_layer = 0
            layer0_loss = self.layer_reconstruction_loss(i_layer, outputs[i_layer], inputs[i_layer], is_conv = True)
            #layer0_losses.append(layer0_loss)

        if self.n_active_layers == 2:
            i_layer = 1
            layer1_loss = self.layer_reconstruction_loss(i_layer, outputs[i_layer], inputs[i_layer], is_conv = True)
            #layer1_losses.append(layer1_loss)
            
        if self.n_active_layers == 3:
            i_layer = 2
            layer2_loss = self.layer_reconstruction_loss(i_layer, outputs[i_layer], inputs[i_layer], is_conv = True)
            #layer1_losses.append(layer1_loss)

        if self.n_active_layers == 4:
            i_layer = 3
            layer3_loss = self.layer_reconstruction_loss(i_layer, outputs[i_layer], inputs[i_layer])

            #layer2_class_loss = - torch.mean(torch.max(torch.exp(outputs[i_layer]), dim = 1)[0]) * 1.0
            #layer2_class_losses.append(layer2_class_loss)
            #layer2_losses.append(layer2_loss)

        if self.n_active_layers == 5:
            i_layer = 4
            max_vals, max_idx = torch.max(torch.exp(outputs[-1]), dim = 1)

            layer4_loss = self.layer_reconstruction_loss(i_layer, outputs[i_layer], inputs[i_layer])
            #layer3_losses.append(layer3_loss)
            
            #print(max_vals.size())
            #class_loss = - torch.mean(max_vals) * 0.01

        #class_losses.append(class_loss)

        total_loss = layer0_loss * 0.01 + layer1_loss * 10 + layer2_loss * 100 + layer3_loss + layer4_loss# + class_loss# + layer2_class_loss
        return total_loss