from tools import utils, config, trainer, parts
import argparse
from contextlib import redirect_stdout
from tools.logging import log_print
from os import path
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("c")
    args = parser.parse_args()
    
    print(f"Loading config from {args.c}")
    
    # Load config
    cfg = config.from_yaml(args.c)            
    with log_print(open(path.join(cfg.trainer_sup.checkpoint_path, "log.txt"), 'w')):
        metrics = []
        
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
                
                trn = trainer.ModelTrainer(model=model, cfg=cfg.trainer_sup, part_manager=part_manager)
                trn.train(train_loader=train_loader_sup, val_loader=val_loader_sup)
                
                """part_manager.enable_all_training()
                trn.train(train_loader=train_loader, val_loader=val_loader)"""
            
                # Test
                
                accuracy = trn.test_accuracy(test_loader_sup)
                print(f"Accuracy: {accuracy}")
                metrics.append(accuracy)
            
        print(f'Metrics:\n{metrics}, Mean: {np.mean(metrics)}')
        
        
        
        