import torch.nn as nn
import torch
from types import MethodType

FORWARD_ON = 'forward_on'
TYPE_CONV = 'conv'
TYPE_BN = 'bn'
TYPE_RELU = 'relu'
TYPE_LINEAR = 'linear'
TYPE_POOL = 'pool'
TYPE_OTHER = 'other'
LAYER_TYPES = {
    nn.Conv2d : TYPE_CONV,
    nn.BatchNorm2d : TYPE_BN,
    nn.ReLU : TYPE_RELU,
    nn.Linear : TYPE_LINEAR,
    nn.MaxPool2d : TYPE_POOL
}

SAVED_INPUT_NAME = 'saved_input'
SAVED_OUTPUT_NAME = 'saved_output'

class PartManager:
    def __init__(self, model):
        self.train_part_i = 0
        model_layers = get_model_layers(model)
        self.parts = create_parts(model_layers)
        print(f"Created {len(self.parts)} parts.")
        """self.disable_all()
        self.enable_part(0)
        self.enable_part_training(0)"""
    
    def enable_all(self):
        print(f"Enable all parts")
        for p in self.parts:
            p.enable_all()

    def enable_all_training(self):
        for i in range(len(self.parts)):
            self.enable_part_training(i)
            
    def disable_all_training(self):
        for i in range(len(self.parts)):
            self.disable_part_training(i)
            
    def disable_all(self):
        print(f"Disable all parts")
        for p in self.parts:
            p.disable_all()
            
        for i in range(len(self.parts)):
            self.disable_part_training(i)
            
    def enable_part(self, i):
        print(f"Enable part {i}")
        self.parts[i].enable_all()
        
    def change_part_training(self, part_i, value):
        for i, l in enumerate(self.parts[part_i].layers):
            if self.parts[part_i].layer_types[i] in [TYPE_BN, TYPE_CONV, TYPE_LINEAR]:
                if hasattr(l, 'weight'):
                    if hasattr(l.weight, 'requires_grad'):
                        l.weight.requires_grad = value
                if hasattr(l, 'bias'):
                    if hasattr(l.bias, 'requires_grad'):
                        l.bias.requires_grad = value
        
    def enable_part_training(self, part_i):
        print(f'Part {part_i} trainig: ON')
        self.change_part_training(part_i, True)
        
    def disable_part_training(self, part_i):
        print(f'Part {part_i} trainig: OFF')
        self.change_part_training(part_i, False)
        
    def enable_to_loss_layer(self, i):
        print(f"Enable part {i} up to loss layer")
        self.parts[i].enable_to_loss_layer()
            
    def disable_part(self, i):
        print(f"Disable part {i}")
        self.parts[i].disable_all()
        
    def part_step(self):
        if self.train_part_i == len(self.parts) - 1:
            print("Already made a step for each part, will not step.")
            return
        
        self.train_part_i += 1
        self.disable_part_training(self.train_part_i - 1)
        self.enable_part(self.train_part_i - 1)
        
        self.enable_to_loss_layer(self.train_part_i)
        self.enable_part_training(self.train_part_i)
        
        
class NNPart:
    def __init__(self, index):
        self.index = index
        self.layers = []
        self.layer_types = []
        self.loss_start = None
        self.loss_end = None
        self.loss_start_i = None
        self.loss_end_i = None
        
        self.has_conv = False
        self.conv_layer = None
        self.has_relu = False
        self.relu_layer = None
        self.has_bn = False
        self.bn_layer = None
        self.has_linear = False
        self.linear_layer = None
        self.has_pool = False
        self.pool_layer = None
        
    def add_layer(self, layer, layer_type):
        self.layers.append(layer)
        self.layer_types.append(layer_type)
        
        if layer_type == TYPE_RELU:
            self.has_relu = True
            self.relu_layer = layer
        elif layer_type == TYPE_BN:
            self.has_bn = True
            self.bn_layer = layer
        elif layer_type == TYPE_POOL:
            self.has_pool = True
            self.pool_layer = layer
        elif layer_type == TYPE_LINEAR:
            self.has_linear = True
            self.linear_layer = layer
        elif layer_type == TYPE_CONV:
            self.conv_layer = layer
            self.has_conv = True
            
        # Modify forward method of model
        layer_add_forward_toggle(layer)
        if self.loss_start is None:
            if layer_type in [TYPE_CONV, TYPE_LINEAR]:
                self.loss_start = layer
                self.loss_start_i = len(self.layers) - 1
                
    def end_part(self):
        # End layer is bn or relu
        for i in reversed(range(len(self.layers))):
            if self.layer_types[i] in [TYPE_BN, TYPE_RELU, TYPE_LINEAR]:
                self.loss_end = self.layers[i]
                self.loss_end_i = i
                break
    
    def enable_all(self):
        for l in self.layers:
            setattr(l, FORWARD_ON, True)
    
    def enable_to_loss_layer(self):
        for i in range(self.loss_end_i + 1):
            setattr(self.layers[i], FORWARD_ON, True)
            
    def disable_all(self):
        for l in self.layers:
            setattr(l, FORWARD_ON, False)
            
    def get_conv_layer(self):
        if self.conv_layer is None:
            raise AttributeError(f"conv layer does not exist for part {self.index}.")
        return self.conv_layer
    
    def get_relu_layer(self):
        if self.relu_layer is None:
            raise AttributeError(f"relu layer does not exist for part {self.index}.")
        return self.relu_layer
    
    def get_bn_layer(self):
        if self.bn_layer is None:
            raise AttributeError(f"bn layer does not exist for part {self.index}.")
        return self.bn_layer
    
    def get_linear_layer(self):
        if self.linear_layer is None:
            raise AttributeError(f"linear layer does not exist for part {self.index}.")
        return self.linear_layer
    
    def get_loss_end_layer(self):
        return self.layers[self.loss_end_i]
    
    def get_loss_start_layer(self):
        return self.layers[self.loss_start_i]
            
    def __repr__(self):
        layers_str = '\n' + '\n'.join([str(i) + '- ' + str(l) for i, l in enumerate(self.layers)]) + '\n====================================='
        return f"loss_start_i={self.loss_start_i}\nloss_end_i={self.loss_end_i}\nlayers={layers_str}"
        
def create_parts(layers):
    in_part = False
    parts = []
    layer_types = [get_layer_type(l) for l in layers]
    
    for i in range(len(layers)):
        layer = layers[i]
        layer_type = layer_types[i]
        if not in_part:
            in_part = True
            parts.append(NNPart(len(parts)))
            parts[-1].add_layer(layer, layer_type)
        else:
            parts[-1].add_layer(layer, layer_type)
            if (parts[-1].has_conv or parts[-1].has_linear) and is_end_layer(layer_types, i):
                # End part
                in_part = False
                parts[-1].end_part()
    
    # Last layer did not have a conv end        
    if in_part:
        parts[-1].end_part()
                
    return parts


def get_layer_type(layer : torch.nn.Module) -> str:
    for t, t_str in LAYER_TYPES.items():
        if isinstance(layer, t):
            return t_str
    return TYPE_OTHER

def is_layer_start(layer_types, i_layer):
    # Each part starts with a conv
    return layer_types[i_layer] in [TYPE_CONV, TYPE_LINEAR]

def is_end_layer(layer_types, i):    
    # A part ends at a layer if it is relu
    if layer_types[i] == TYPE_RELU:
        return True
    
    if len(layer_types) == i + 1:
        #raise AttributeError("Reached the end of lists layer without and end to part.")
        return False
    
    # A part ends at a layer if next layer is conv
    if layer_types[i + 1] in [TYPE_CONV, TYPE_LINEAR]:
        return True
        
    return False

def get_model_layers(model: torch.nn.Module):
    # get children form model!
    children = list(model.children())
    flatt_children = []
    if children == []:
        # if model has no children; model is last child! :O
        return model
    else:
       # look for children from children... to the last child!
       for child in children:
            try:
                flatt_children.extend(get_model_layers(child))
            except TypeError:
                flatt_children.append(get_model_layers(child))
    return flatt_children

"""def create_forward_toggle(self, func):
    def forward_toggle(*args, **kwargs):
        if self.__dict__.get(FORWARD_ON):
            print(args)
            return func(*args, **kwargs)
        else:
            # Identity function if layer was turned off
            return args[0]
        
    return forward_toggle"""

def forward_toggle(nn, *args, **kwargs):
    if nn.__dict__.get(FORWARD_ON):
        # Keep input
        setattr(nn, SAVED_INPUT_NAME, args[0].clone())
        output = nn.forward_original(*args, **kwargs)
        
        # Keep output
        setattr(nn, SAVED_OUTPUT_NAME, output.clone())
        return output
    else:
        # Identity function if layer was turned off
        return args[0]

def layer_add_forward_toggle(layer):
    """Modifies forward method of layer so that is can be toggled
    on and off. When on, layer operates as usual. When off, layer
    is an identity layer, it outputs input.

    Args:
        layer (torch.nn.Module): torch layer
    """
    if hasattr(layer, FORWARD_ON):
        print("Layer already has a FORWARD_ON flag, will not modify forward method.")
        return
    
    # All parts are on by default
    #print(f"layer id: {id(layer)}")
    setattr(layer, FORWARD_ON, True)
    #setattr(layer, 'create_forward_toggle', create_forward_toggle)
    #layer.forward = layer.create_forward_toggle(layer.forward)
    layer.forward_original = layer.forward
    layer.forward = MethodType(forward_toggle, layer)
    #layer.forward = create_forward_toggle(layer, layer.forward_original)
    #layer._forward_impl = MethodType(_forward_impl_toggle, layer)