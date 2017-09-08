from lattice import LatticeWorld
from maze import MazeWorld

def load(config):
    cls_name = config.world.name
    try:
        cls = globals()[cls_name]
        return cls(config)
    except KeyError:
        raise Exception("No such world: {}".format(cls_name))
