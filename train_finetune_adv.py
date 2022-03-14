from tools import utils, config, trainer, parts
import argparse
from contextlib import redirect_stdout
from tools.logging import log_print
from os import path
import numpy as np

import torchattacks
from tqdm import tqdm

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("c")
    args = parser.parse_args()
    
    print(f"Loading config from {args.c}")
    
    # Load config
    cfg = config.from_yaml(args.c)            
    with log_print(open(path.join(cfg.trainer_sup.checkpoint_path, "log.txt"), 'w')):
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
            print(f"UNSUPERVISED TRAINING {i_run + 1}/{cfg.n_repeat}")
            
            # Load model
            model = utils.load_model(**cfg.model)
            
            if cfg.trainer_unsup.enabled:
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
                trn = trainer.ModelTrainer(model=model, cfg=cfg.trainer_sup, part_manager=part_manager)
                trn.train(train_loader=train_loader_sup, val_loader=val_loader_sup)
                
                """part_manager.enable_all_training()
                trn.train(train_loader=train_loader, val_loader=val_loader)"""
            
                # Test
                
                accuracy = trn.test_accuracy(test_loader_sup)
                print(f"Accuracy: {accuracy}")
                metrics.append(accuracy)
                
                metrics_adv.append(get_accuracy(model, trn.device, test_loader_sup, attack_on = True))
            
        print(f'Metrics:\n{metrics}, Mean: {np.mean(metrics)}')
        print(f'Metrics adv:\n{metrics_adv}, Mean: {np.mean(metrics_adv)}')
        
        
        
        