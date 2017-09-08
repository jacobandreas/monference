from misc.experience import Transition

import logging
import numpy as np
import tensorflow as tf

N_ITERS = 1000000
MAX_TIMESTEPS = 20
N_UPDATE = 1000
#N_UPDATE = 1

class ImitationTrainer(object):
    def __init__(self, config):
        pass

    def do_rollout(self, model, world, vis=False):
        total_reward = 0.
        scenario = world.sample_scenario()
        state_before = scenario.init()
        model.init(state_before)
        actions = []
        for t in range(MAX_TIMESTEPS):
            action = model.act(state_before, randomize=False)
            actions.append(action)
            reward, state_after, terminate = state_before.step(action)
            total_reward += reward
            state_before = state_after
            if terminate:
                break

        if vis:
            logging.info((scenario.start, scenario.goal))
            logging.info(scenario.get_demonstration())
            logging.info([tuple(a.round().astype(int)) for a in actions])

        actions = scenario.get_demonstration()
        state_before = scenario.init()
        transitions = []
        true_reward = 0
        for action in actions:
            reward, state_after, terminate = state_before.step(action, verify=True)
            true_reward += reward
            transitions.append(Transition(state_before, action, state_after,
                    reward))
            state_before = state_after
            if terminate:
                break

        return transitions, total_reward

    def train(self, model, world):
        model.prepare(world)
        total_err = 0.
        total_reward = 0.
        for i_iter in range(N_ITERS):
            vis = (i_iter + 1) % N_UPDATE == 0

            demonstration, reward = self.do_rollout(model, world, vis)
            model.demonstrate(demonstration)
            total_reward += reward
            e = model.train_im()
            total_err += e

            if vis:
                #logging.info("sample transitions: " + \
                #        str([t.a for t in transitions]))
                logging.info("[err] %8.3f" % (total_err / N_UPDATE))
                logging.info("[rew] %8.3f" % (total_reward / N_UPDATE))
                logging.info("")

                total_err = 0.
                total_reward = 0.
