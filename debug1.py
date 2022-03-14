# %%
import torchvision.models as models
import torch.nn as nn
import torch
from model import model_part
import numpy as np

# https://web.stanford.edu/~nanbhas/blog/forward-hooks-pytorch/#extracting-activations-from-a-layer

# %%
model = models.resnet18()
model.eval()
model_layers = model_part.get_model_layers(model)

# %%
l = model_layers[0]

# %%
"""class SumClass:
    def __init__(self, w = 3):
        self.w = w
    def forward(self, input):
        return self.w * input
    
mysum = SumClass()

mysum.forward = create_forward_toggle(mysum, mysum.forward)

mysum.forward_on = True
mysum.forward(1)"""

# %%
parts = model_part.create_parts(model_layers)

# %%
l = model_layers[0]

# %%
parts[0]

# %%
x = np.ones((1, 3, 32, 32))
x = torch.tensor(x).float()

# %%
y = model(x)

# %%



