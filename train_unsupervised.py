from tools import utils, config, trainer, parts
import argparse
from contextlib import redirect_stdout
from tools.logging import log_print
from os import path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("c")
    args = parser.parse_args()
    
    print(f"Loading config from {args.c}")
    
    # Load config
    cfg = config.from_yaml(args.c)            
    with log_print(open(path.join(cfg.trainer.checkpoint_path, "log.txt"), 'w')):
        print("UNSUPERVISED TRAINING")
        
        # Load dataset
        dataset = utils.load_dataset_module(**cfg.data)
        dataset.torch_seed()
        train_loader, val_loader = dataset.get_train_loader(**cfg.data)
        # Load model
        model = utils.load_model(**cfg.model)
        
        part_manager = parts.PartManager(model)
        part_manager.disable_all()
        part_manager.enable_part(0)
        part_manager.enable_part_training(0)
        # Turn off training for all layers
        trainer = trainer.ModelTrainer(model=model, cfg=cfg.trainer, part_manager=part_manager)
        
        # Train part by part
        for i_layer in range(len(part_manager.parts) - 2):
            trainer.train(train_loader=train_loader, val_loader=val_loader)
            part_manager.part_step()

        print(f"Done")