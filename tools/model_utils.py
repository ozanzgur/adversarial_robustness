import torch
import torch.optim as optim
import torch.nn.functional as F

def nll(y_pred, y_label):
    """Torch F.nll_loss function
    """
    return F.nll_loss(y_pred, y_label)

def disable_training_all_layers(model : torch.nn.Module):
    for i_layer, l in enumerate(model.nn_layers):
        print(f'Layer {i_layer} trainig: OFF')
        l.bias.requires_grad = False
        l.weight.requires_grad = False
        
    model.bn1.weight.requires_grad = False
    model.bn1.bias.requires_grad = False

def enable_training_all_layers(model : torch.nn.Module):
    for i_layer, l in enumerate(model.nn_layers):
        print(f'Layer {i_layer} trainig: ON')
        l.bias.requires_grad = True
        l.weight.requires_grad = True
        
    model.bn1.weight.requires_grad = True
    model.bn1.bias.requires_grad = True
                
def enable_layer_training(model : torch.nn.Module, i_layer : int):
    if i_layer < 0:
        raise ValueError("i_layer must be greater than 0")
    
    print(f'Layer {i_layer} trainig: ON')
    model.nn_layers[i_layer].weight.requires_grad = True
    model.nn_layers[i_layer].bias.requires_grad = True
    
    if i_layer == 0:
        model.bn1.weight.requires_grad = True
        model.bn1.bias.requires_grad = True
    
def disable_layer_training(model : torch.nn.Module, i_layer : int):
    if i_layer < 0:
        raise ValueError("i_layer must be greater than 0")
    
    print(f'Layer {i_layer} trainig: OFF')
    model.nn_layers[i_layer].weight.requires_grad = False
    model.nn_layers[i_layer].bias.requires_grad = False
    
    if i_layer == 0:
        model.bn1.weight.requires_grad = False
        model.bn1.bias.requires_grad = False
    
def get_trainable_layers(model):
    return filter(lambda p: p.requires_grad, model.parameters())