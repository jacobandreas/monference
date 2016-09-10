from misc.experience import Transition

import logging
import numpy as np
import tensorflow as tf

N_ITERS = 1000000
MAX_TIMESTEPS = 20
N_UPDATE = 1000

class ImitationTrainer(object):
    def __init__(self, config):
        pass

    def do_rollout(self, model, world):
        total_reward = 0.
        scenario = world.sample_scenario()
        state_before = scenario.init()
        model.init(state_before)
        for t in range(MAX_TIMESTEPS):
            action = model.act(state_before, randomize=False)
            reward, state_after, terminate = state_before.step(action)
            total_reward += reward
            state_before = state_after
            if terminate:
                break

        actions = scenario.get_demonstration()
        state_before = scenario.init()
        transitions = []
        for action in actions:
            reward, state_after, terminate = state_before.step(action)
            transitions.append(Transition(state_before, action, state_after,
                    reward))
            state_before = state_after
            #total_reward += reward
        return transitions, total_reward

    def train(self, model, world):
        model.prepare(world)
        total_err = 0.
        total_reward = 0.
        for i_iter in range(N_ITERS):
            demonstration, reward = self.do_rollout(model, world)
            model.demonstrate(demonstration)
            total_reward += reward
            total_err += model.train_im()

            if (i_iter + 1) % N_UPDATE == 0:
                #logging.info("sample transitions: " + \
                #        str([t.a for t in transitions]))
                logging.info("[err] %8.3f" % (total_err / N_UPDATE))
                logging.info("[rew] %8.3f" % (total_reward / N_UPDATE))
                logging.info("")

                total_err = 0.
                total_reward = 0.
