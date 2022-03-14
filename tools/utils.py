import importlib.util
from os.path import basename
import os
import shutil

def load_module(path):
    spec = importlib.util.spec_from_file_location(basename(path), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def load_model(path: str, module_name : str, **kwargs):
    module = load_module(path)
    return getattr(module, module_name)(**kwargs)

def load_dataset_module(path: str, **kwargs):
    print(f"Loading dataset: {path}")
    spec = importlib.util.spec_from_file_location(basename(path), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def remove(path):
    """ param <path> could either be relative or absolute. """
    if os.path.isfile(path) or os.path.islink(path):
        os.remove(path)  # remove the file
    elif os.path.isdir(path):
        shutil.rmtree(path)  # remove dir and all contains
    else:
        raise ValueError("file {} is not a file or dir.".format(path))