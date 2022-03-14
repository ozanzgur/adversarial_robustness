import torch
import torch.nn.functional as F

def batchnorm_inverse(output, layer):
    """Calculate batch norm input from output and layer parameters
    
    """
    beta = layer.bias.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    gamma = layer.weight.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    mean = layer.running_mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    var = (torch.sqrt(layer.running_var + layer.eps)).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    return (output - beta) / gamma * var + mean

def conv_inverse(output, layer):
    bias_clone = layer.bias.clone().unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    return F.conv_transpose2d((output - bias_clone) / layer.weight.square().sum(),
                              weight = layer.weight,
                              stride = layer.stride, 
                              padding = layer.padding, 
                              output_padding = layer.output_padding, 
                              groups = layer.groups, 
                              dilation = layer.dilation)
    
def conv_inverse_channelwise(output, layer, i_channel):
    bias_clone = layer.bias.clone()[i_channel].unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    w = layer.weight[i_channel].unsqueeze(0)
    return F.conv_transpose2d((output - bias_clone) / w.square().sum(),
                              weight = w,
                              stride = layer.stride, 
                              padding = layer.padding, 
                              output_padding = layer.output_padding, 
                              groups = layer.groups, 
                              dilation = layer.dilation)
        
def fc_inverse(output, layer):
    bias_shape = [1] * len(output.size())
    bias_shape[1] = -1
    output_unsqueeze = (output - layer.bias.reshape(bias_shape)).unsqueeze(2)
    W = layer.weight.unsqueeze(0)
    reconstruction = (output_unsqueeze * W)
    reconstruction = torch.sum(reconstruction, axis = 1)
    return reconstruction