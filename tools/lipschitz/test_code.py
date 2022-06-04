

import torch
from torchvision import transforms, datasets
from lipschitz_approximations import lipschitz_opt_lb
from .lipschitz_utils import *

def random_dataset(input_size, scale=1, fn=lambda x: x):
    tensor_dataset = scale * torch.randn(1,1,input_size,input_size)
    return torch.utils.data.TensorDataset(tensor_dataset, fn(tensor_dataset))

model = torch.load('C:\\Users\\osero\\Desktop\\Phyton Projects\\adversarial_robustness-main\\tools\\lipschitz\\trained_model.pt')
dataset = random_dataset(28, scale=2)

lipschitz_opt_lb(model)

# compute_lipschitz_approximations(model, random_dataset(28, scale=2))
asdasda = 5
