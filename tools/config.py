import munch

def from_yaml(path : str):
    with open(path) as f:
        return munch.Munch.fromYAML(f)

class Config:
    def __init__(self, cfg : dict):
        self.__dict__.update(cfg)
        
    @classmethod
    def from_yaml(cls, path : str):
        with open(path) as f:
            #return cls(yaml.load(f, Loader=yaml.FullLoader))
            munch.Munch.fromYAML(f)