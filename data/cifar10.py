import torch
import torchvision
import numpy as np

DATASET_SIZE = 50000

def torch_seed(seed=42):
    torch.manual_seed(seed)

def get_train_loader(batch_size=10, loader_sizes = None, **kwargs):
    dataset = torchvision.datasets.CIFAR10('/files/', train=True, download=True,
                                transform=torchvision.transforms.Compose([
                                torchvision.transforms.Resize((32, 32)),
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
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

def get_test_loader(batch_size=10, **kwargs):
    return torch.utils.data.DataLoader(
    torchvision.datasets.CIFAR10('/files/', train=False, download=True,
                                transform=torchvision.transforms.Compose([
                                torchvision.transforms.Resize((32, 32)),
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
                                ),
    batch_size=batch_size, shuffle=True)
    
def get_train_dataset(**kwargs):
    train_dataset = torchvision.datasets.CIFAR10('/files/', train=False, download=True,
                                transform=torchvision.transforms.Compose([
                                torchvision.transforms.Resize((32, 32)),
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
                                )
    return train_dataset

def get_test_dataset(**kwargs):
    return torchvision.datasets.CIFAR10('/files/', train=False, download=True,
                                transform=torchvision.transforms.Compose([
                                torchvision.transforms.Resize((32, 32)),
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
                                )