from tools import utils, config, trainer, parts
import argparse
from contextlib import redirect_stdout
from tools.logging import log_print
from os import path
import numpy as np
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("c")
    args = parser.parse_args()
    
    print(f"Loading config from {args.c}")
    
    # Load config
    cfg = config.from_yaml(args.c)            
    with log_print(open(path.join(cfg.trainer_sup.checkpoint_path, "log.txt"), 'w')):
        dataset_sup = utils.load_dataset_module(**cfg.data_supervised)
        
        i_part = cfg.trainer_sup.train_part_i
        
        # Supervised dataset
        train_loader_sup, val_loader_sup = dataset_sup.get_train_loader(**cfg.data_supervised)
            
        # Load model
        model = utils.load_model(**cfg.model)
        model.load_state_dict(torch.load(cfg.model.weights_path))
        part_manager = parts.PartManager(model)
        for i in range(i_part):
            part_manager.enable_part(i)
        part_manager.disable_all_training()
        part_manager.train_part_i = i_part
        
        trn = trainer.ModelTrainer(model=model, cfg=cfg.trainer_sup, part_manager=part_manager)
        trn.set_aux_layer(part_manager.parts[i_part], size_multiplier=1)
        trn.train(train_loader=train_loader_sup, val_loader=val_loader_sup)
        
        
        
        