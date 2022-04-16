import torch
from autoattack import AutoAttack
import numpy as np

def do_attack(model, device, loader, fast = True, first_n = None):
    n_examples = 0

    adversary = None
    
    if fast:
        adversary = AutoAttack(model, norm='Linf', eps=8/255, version='custom', attacks_to_run=['apgd-ce', 'apgd-dlr'])
    else:
        adversary = AutoAttack(model, norm='Linf', eps=8/255)
        
    x_all = []
    y_all = []
    for x, y in loader:
        x_all.append(x)
        y_all.append(y)
        n_examples += x.shape[0]
        
    x_all = torch.concat(x_all, dim=0).to(device)
    y_all = torch.concat(y_all, dim=0).to(device)
    
    if not first_n is None:
        x_all = x_all[:first_n]
        y_all = y_all[:first_n]
        
    return adversary.run_standard_evaluation(x_all, y_all, bs=250)