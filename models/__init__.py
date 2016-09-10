from reflex import ReflexModel
from recurrent import RecurrentModel
from planner import PlannerModel

def load(config):
    cls_name = config.model.name
    try:
        cls = globals()[cls_name]
        return cls(config)
    except KeyError:
        raise Exception("No such model: {}".format(cls_name))
