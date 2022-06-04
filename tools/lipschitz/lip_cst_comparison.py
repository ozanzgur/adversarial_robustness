# encoding: utf-8

import os
import argparse

import numpy as np

import scipy.stats as st

import torch
import torch.optim as optim
from torch.optim import lr_scheduler

from torch.autograd import Variable

from torchvision import datasets, transforms, models

from .lipschitz_utils import *

from .lipschitz_approximations import *

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm

# Global parameters
use_cuda = torch.cuda.is_available()

#parser = argparse.ArgumentParser()
#parser.add_argument('--dataset', required=True, help='MNIST | CIFAR')
#parser.add_argument('--root', required=True, help='path to dataset')
#parser.add_argument('--batchSize', type=int, default=128, help='size of input batch')

#opt = parser.parse_args()
#print(opt)

def create_dataset(data_type):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    # MNIST
    if data_type is 'MNIST':
        return datasets.MNIST(root='data/', download=True, train=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    normalize]))

    # CIFAR
    if data_type is 'CIFAR':
        return datasets.CIFAR10(root='data/', download=True, train=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    normalize]))

    # RANDOM
    if data_type is 'RANDOM':
        input_size = 10
        return random_dataset([input_size], 1000)


def compute_lipschitz_approximations(model, data):
    # print(model)
    # Don't compute gradient for the projector: speedup computations
    for p in model.parameters():
        p.requires_grad = False

    # Compute input sizes for all modules of the model
    input_size = get_input_size(data)
    try:
        print(f"input_size: {input_size}, model.shape{model.shape}")
    except:  
        print(f"input_size: {input_size}")
    compute_module_input_sizes(model, input_size)

    # Lipschitz lower bound through optimization of the gradient norm
    print('Computing lip_opt...')
    lip_opt = lipschitz_annealing(model, n_iter=1000, temp=1, batch_size=1)
    #lip_opt = 0

    # Lipschitz lower bound on dataset
    print('Computing lip_data...')
    lip_data = lipschitz_data_lb(model, data, max_iter=1000)

    # Lipschitz upper bound using the product of spectral norm
    print('Computing lip_spec...')
    lip_spec = lipschitz_spectral_ub(model).data[0]

    # Lipschitz upper bound using the product of Frobenius norm
    print('Computing lip_frob...')
    lip_frob = lipschitz_frobenius_ub(model).data[0]

    print('Computing lip_secorder greedy...')
    try:
        lip_secorder_greedy = lipschitz_second_order_ub(model, algo='greedy')
    except:
        lip_secorder_greedy = 9999999


    print('Computing lip_secorder BFGS...')
    try:
        lip_secorder_bfgs = lipschitz_second_order_ub(model, algo='bfgs')
    except:
        lip_secorder_bfgs = 9999999

    print('Lipschitz approximations:\nLB-dataset:\t{:.0f}\n'
          'LB-optim:\t{:.0f}\nUB-frobenius:\t{:.0f}\nUB-spectral:\t{:.0f}\n'
          'UB-secorder greedy:\t{:.0f}\nUB-secorder bfgs:\t{:.0f}'.format(lip_data, lip_opt, lip_frob, lip_spec,
                 lip_secorder_greedy, lip_secorder_bfgs))


def plot_model(model, window_size=1, num_points=100):
    fig = plt.figure()
    ax = Axes3D(fig)
    points = window_size * Variable(torch.randn(num_points, 2))
    fun_val = model(points).data

    ax.scatter(points[:, 0], points[:, 1], fun_val)
    plt.show()


def plot_data(x, y):
    fig = plt.figure(figsize=(9, 9))
    ax = Axes3D(fig)

    # z = y[:]
    # x, y = np.meshgrid(x[:, 0], x[:, 1])

    # ax.plot_surface(x, y, z, cmap=cm.coolwarm)
    # ax.plot_trisurf(x[:, 0], x[:, 1], y, cmap=cm.coolwarm)
    ax.plot_trisurf(x[:, 0], x[:, 1], y, cmap=cm.RdYlBu_r)
    plt.show()


def plot_model_data(model, n_points, data=None, data_y=None, coef=1):
    x_train = coef * torch.randn(n_points, 2)
    y = model(Variable(x_train, volatile=True))
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x_train[:, 0], x_train[:, 1], y.data, c='b')
    if (data is not None) and (data_y is not None):
        ax.scatter(data[:, 0], data[:, 1], data_y, c='g')
    plt.show()


def plot_models(model1, model2, n_points):
    x_train = 200 * torch.randn(n_points, 2)
    y1 = model1(Variable(x_train, volatile=True))
    y2 = model2(Variable(x_train, volatile=True))

    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x_train[:, 0], x_train[:, 1], y1.data, c='b')

    ax.scatter(x_train[:, 0], x_train[:, 1], y2.data, c='g')
    plt.show()
