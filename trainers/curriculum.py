from misc import util
from misc.experience import Transition
from worlds.cookbook import Cookbook

import numpy as np
import yaml

N_ITERS = 1000000
N_UPDATE = 5000
MAX_TIMESTEPS = 40

class CurriculumTrainer(object):
    def __init__(self, config):
        self.cookbook = Cookbook(config.recipes)
        self.action_index = util.Index()
        with open(config.trainer.hints) as hints_f:
            self.hints = yaml.load(hints_f)

    def do_rollout(self, model, world):
        transitions = []
        goal = np.random.choice(self.hints.keys())
        goal_action, goal_arg = util.parse_fexp(goal)
        steps = [util.parse_fexp(s) for s in self.hints[goal]]
        steps = [(self.action_index.index(a), self.cookbook.index[b])
                for a, b in steps]

        scenario = world.sample_scenario_with_goal(goal_arg)
        state_before = scenario.init()

        model.init(state_before, steps)

        total_reward = 0
        hit = 0
        for t in range(MAX_TIMESTEPS):
            model_state_before = model.get_state()
            action, terminate = model.act(state_before)
            model_state_after = model.get_state()
            if terminate:
                win = state_before.inventory[self.cookbook.index[goal_arg]] > 0
                reward = 1 if win else 0
                if win:
                    hit += 1
                state_after = scenario.terminal
            elif action >= world.n_actions:
                state_after = state_before
                partial = state_before.inventory[self.cookbook.index["wood"]] > 0
                reward = 0
                if partial:
                    hit += 1
            else:
                reward, state_after = state_before.step(action)

            reward = max(min(reward, 1), -1)
            transitions.append(Transition(state_before, model_state_before,
                action, state_after, model_state_after, reward))
            total_reward += reward
            if terminate:
                break

            state_before = state_after

        #if hit == 2: print "both", total_reward

        return transitions, total_reward, steps

    def train(self, model, world):
        model.prepare(world)
        total_err = 0.
        total_reward = 0.
        for i_iter in range(N_ITERS):
            transitions, reward, steps = self.do_rollout(model, world)
            model.experience(transitions)
            total_reward += reward
            total_err += model.train()

            if (i_iter + 1) % N_UPDATE == 0:
                print steps
                print reward
                print [t.a for t in transitions]
                print "%5.3f %5.3f" % \
                        (total_err / N_UPDATE, total_reward / N_UPDATE)
                print

                if reward == 0:
                    transitions[0].s1.features()
                    print transitions[0].s1.pos
                    print transitions[0].s1.gf
                    print transitions[0].s1.gfb
                    print [t.a for t in transitions]
                    print "===\n"

                total_err = 0.
                total_reward = 0.
                model.roll()
