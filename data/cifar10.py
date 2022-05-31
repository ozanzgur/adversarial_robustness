import torch
import torchvision
import numpy as np
import torchvision.transforms as tt

DATASET_SIZE = 50000

stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
train_tfms = tt.Compose([tt.RandomCrop(32, padding=4, padding_mode='reflect'), 
                         tt.RandomHorizontalFlip(), 
                         tt.ToTensor(), 
                         tt.Normalize(*stats,inplace=True)])
valid_tfms = tt.Compose([tt.ToTensor(), tt.Normalize(*stats)])

"""
torchvision.transforms.Compose([
    torchvision.transforms.Resize((size, size)),
    torchvision.transforms.ToTensor()]

[
torchvision.transforms.Resize((size, size)),
torchvision.transforms.ToTensor()
]

"""

def torch_seed(seed=42):
    torch.manual_seed(seed)

def get_train_loader(batch_size=10, loader_sizes = None, size=32, **kwargs):
    dataset = torchvision.datasets.CIFAR10('/files/', train=True, download=True,
                                transform=train_tfms
                                )
    
    print(len(dataset))
    
    if not loader_sizes is None:
        n_select = np.sum(loader_sizes)
        if n_select < DATASET_SIZE:
            loader_sizes.append(DATASET_SIZE - n_select)
            
        datas = torch.utils.data.random_split(dataset, loader_sizes, generator=torch.Generator().manual_seed(42))
        loaders = [torch.utils.data.DataLoader(d, batch_size=batch_size, shuffle=True) for d in datas]
        
        if n_select < DATASET_SIZE:
            return loaders[:-1]
        else:
            return loaders
    else:
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

def get_test_loader(batch_size=1, size=32, **kwargs):
    return torch.utils.data.DataLoader(
    torchvision.datasets.CIFAR10('/files/', train=False, download=True,
                                transform=valid_tfms
                                ),
    batch_size=batch_size, shuffle=True)
    
def get_train_dataset(size=32, **kwargs):
    train_dataset = torchvision.datasets.CIFAR10('/files/', train=False, download=True,
                                transform=train_tfms
                                )
    return train_dataset

def get_test_dataset(size=32, **kwargs):
    return torchvision.datasets.CIFAR10('/files/', train=False, download=True,
                                transform=valid_tfms
                                )