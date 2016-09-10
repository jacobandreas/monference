from reinforcement import ReinforcementTrainer
from imitation import ImitationTrainer

def load(config):
    cls_name = config.trainer.name
    try:
        cls = globals()[cls_name]
        return cls(config)
    except KeyError:
        raise Exception("No such trainer: {}".format(cls_name))
