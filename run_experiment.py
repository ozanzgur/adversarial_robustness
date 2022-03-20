from tools import config
import argparse
from subprocess import call
import yaml

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("c", nargs='+', default=[])
    args = parser.parse_args()
    
    print(f"Running {len(args.c)} experiments:")
    for config_path in args.c:
        print(config_path)
    for config_path in args.c:
        # Load config
        cfg = config.from_yaml(config_path)
        script_path = cfg.script_path
        
        print(f"Running experiment: {config_path}")
        call(["python", script_path, config_path])