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
        # Load dataset
        dataset = utils.load_dataset_module(**cfg.data)
        dataset.torch_seed()
        train_loader, val_loader = dataset.get_train_loader(**cfg.data)
        # Load model
        model = utils.load_model(**cfg.model)

        # Create parts
        part_manager = parts.PartManager(model)
        trainer = trainer.ModelTrainer(model=model, cfg=cfg.trainer, part_manager=part_manager)
        trainer.train(train_loader=train_loader, val_loader=val_loader)
        
        # Test
        test_loader = dataset.get_test_loader(batch_size=25)
        accuracy = trainer.test_accuracy(test_loader)
        print(f"Accuracy: {accuracy}")