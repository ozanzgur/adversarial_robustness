from tools import utils, config, parts
from tools import trainer_multistage
from tools import trainer
import argparse
from contextlib import redirect_stdout
from tools.logging import log_print
from os import path
import numpy as np

import torch
import torchattacks
from tqdm import tqdm
import yaml

def get_accuracy(model, device, test_loader, attack_on = True):
    correct = 0
    n_examples = 0

    atk = torchattacks.PGD(model, eps=8/255, alpha=2/255, steps=4)
    # adv_images = None
    # y = None
    # adv_pred = None

    for x, y in test_loader:
        x.to(device)
        y.to(device)
        n_examples += x.shape[0]
        adv_images = atk(x, y) if attack_on else x.to(device)
        adv_pred = model(adv_images).data.max(1, keepdim=True)[1].cpu()
        correct += adv_pred.eq(y.data.view_as(adv_pred)).sum()
        
    acc = correct / n_examples
    return acc

from autoattack import AutoAttack

def get_acc_autoattack(model, device, loader, fast = True, eps = None):
    if eps is None:
        eps = 8/255
        
    correct = 0
    n_examples = 0

    adversary = None
    
    if fast:
        adversary = AutoAttack(model, norm='Linf', eps=eps, version='custom', attacks_to_run=['apgd-ce', 'apgd-dlr'])
    else:
        adversary = AutoAttack(model, norm='Linf', eps=eps)
        
    x_all = []
    y_all = []
    for x, y in loader:
        x_all.append(x)
        y_all.append(y)
        n_examples += x.shape[0]
        
    x_all = torch.concat(x_all, dim=0).to(device)
    y_all = torch.concat(y_all, dim=0).to(device)
    
    SAMPLE_SIZE = 1000
    np.random.seed(42)
    sample_idx = np.random.choice(range(SAMPLE_SIZE), SAMPLE_SIZE, replace=False)
    x_all = x_all[sample_idx]
    y_all = y_all[sample_idx]
        
    _ = adversary.run_standard_evaluation(x_all, y_all, bs=250)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("c")
    args = parser.parse_args()
    
    print(f"Loading config from {args.c}")
    
    # Load config
    cfg = config.from_yaml(args.c)            
    with log_print(open(path.join(cfg.trainer_sup.checkpoint_path, "log.txt"), 'w')):
        print("Experiment Config:\n")
        print(yaml.safe_dump(cfg))
        print("########################################")
        
        metrics = []
        metrics_adv = []
        
        dataset_sup = utils.load_dataset_module(**cfg.data_supervised)
        dataset_unsup = utils.load_dataset_module(**cfg.data_unsupervised)
        
        # Supervised dataset
        train_loader_sup, val_loader_sup = dataset_sup.get_train_loader(**cfg.data_supervised)
        test_loader_sup = dataset_sup.get_test_loader(batch_size=25)
        
        # Unsupervised dataset
        train_loader, val_loader = dataset_unsup.get_train_loader(**cfg.data_unsupervised)
        for i_run in range(cfg.n_repeat):
            
            # Load model
            model = utils.load_model(**cfg.model)
            
            if hasattr(cfg, 'trainer_unsup') and cfg.trainer_unsup.enabled:
                print(f"UNSUPERVISED TRAINING {i_run + 1}/{cfg.n_repeat}")
                 # Load dataset
                dataset_unsup.torch_seed()
                part_manager = parts.PartManager(model)
                part_manager.disable_all()
                part_manager.enable_part(0)
                part_manager.enable_part_training(0)
                # Turn off training for all layers
                trn = trainer.ModelTrainer(model=model, cfg=cfg.trainer_unsup, part_manager=part_manager)
                
                # Train part by part
                for i_layer in range(len(part_manager.parts) if cfg.trainer_unsup.train_n_layers == 'all' else cfg.trainer_unsup.train_n_layers):
                    trn.train(train_loader=train_loader, val_loader=val_loader)
                    part_manager.part_step()
            
            if cfg.trainer_sup.enabled:
                print(f"SUPERVISED TRAINING {i_run + 1}/{cfg.n_repeat}")
                part_manager = parts.PartManager(model)
                part_manager.enable_all()
                part_manager.disable_all_training()
                n_parts = len(part_manager.parts)
                for i in range(0 if cfg.trainer_sup.finetune_n_layers == 'all' else n_parts - cfg.trainer_sup.finetune_n_layers, n_parts): # 
                    part_manager.enable_part_training(i)
                
                part_manager.train_part_i = cfg.trainer_sup.train_part_i
                trn = trainer.ModelTrainer(model=model, cfg=cfg.trainer_sup_single_stage, part_manager=part_manager)
                model.stage_2_enabled = False
                trn.train(train_loader=train_loader_sup, val_loader=val_loader_sup)
                
                model.stage1.eval()
                trn = trainer_multistage.ModelTrainer(model=model, cfg=cfg.trainer_sup, part_manager=part_manager)
                model.stage_2_enabled = True
                model.output_stage1 = True
                
                trn.train(train_loader=train_loader_sup, val_loader=val_loader_sup)
                
                model.output_stage1 = False
                model.eval()
                
                is_fast = (not "adv_is_fast" in cfg) or cfg.adv_is_fast
                print(f"Autoattack fast version: {is_fast}")
                get_acc_autoattack(model, trn.device, test_loader_sup, fast=is_fast)
            
        # print(f'Metrics:\n{metrics}, Mean: {np.mean(metrics)}')
        # print(f'Metrics adv:\n{metrics_adv}, Mean: {np.mean(metrics_adv)}')
        
        
        
        