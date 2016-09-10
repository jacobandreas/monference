#!/usr/bin/env python2

from misc.util import Struct
import models
import trainers
import worlds

import logging
import numpy as np
import random
import tensorflow as tf
import yaml

def main():
    config = configure()
    world = worlds.load(config)
    model = models.load(config)
    trainer = trainers.load(config)
    trainer.train(model, world)

def configure():
    np.random.seed(0)
    random.seed(0)
    tf.set_random_seed(0)
    with open("config.yaml") as config_f:
        config = Struct(**yaml.load(config_f))
    log_name = "logs/%s-%d_%s-%d_%s.log" % (
            config.world.name, 
            config.world.size,
            config.model.name, 
            config.model.depth,
            config.trainer.name)
    logging.basicConfig(filename=log_name, level=logging.DEBUG,
            format='%(asctime)s %(levelname)-8s %(message)s')
    logging.info("BEGIN")
    logging.info(str(config))
    return config

if __name__ == "__main__":
    main()
