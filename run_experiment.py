from tools import config
import argparse
from subprocess import call

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("c")
    args = parser.parse_args()
    
    # Load config
    cfg = config.from_yaml(args.c)
    script_path = cfg.script_path
    
    print(f"Running experiment: {args.c}")
    call(["python", script_path, args.c])