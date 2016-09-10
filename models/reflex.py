import net

import numpy as np
import tensorflow as tf

N_SEQ_BATCH = 100
N_BATCH = 256

N_HIDDEN = 256

DISCOUNT = 0.9
MAX_EXPERIENCES = 50000

class ReflexModel(object):
    def __init__(self, config):
        self.experiences = []
        self.demonstrations = []
        self.world = None
        self.config = config.model

    def prepare(self, world):
        assert self.world is None
        self.world = world
        self.n_actions = world.n_actions

        def predictor(scope):
            with tf.variable_scope(scope):
                t_feats = tf.placeholder(tf.float32, shape=[None, world.n_features])
                layers = [N_HIDDEN] * self.config.depth + [self.n_actions]
                t_scores, v = net.mlp(t_feats, layers)
            return t_feats, t_scores, v

        opt = tf.train.AdamOptimizer()

        t_feats, t_scores, v = predictor("now")
        t_feats_n, t_scores_n, v_n = predictor("next")

        # rl
        t_rewards = tf.placeholder(tf.float32, shape=[None])
        t_action_mask = tf.placeholder(tf.float32, shape=[None, self.n_actions])
        t_score_chosen = tf.reduce_sum(t_scores * t_action_mask, reduction_indices=(1,))
        t_score_n_best = tf.reduce_max(t_scores_n, reduction_indices=(1,))
        t_td = t_rewards + DISCOUNT * t_score_n_best - t_score_chosen
        t_err_rl = tf.reduce_mean(tf.minimum(tf.square(t_td), 1))
        t_train_rl_op = opt.minimize(t_err_rl, var_list=v)
        t_assign_rl_ops = [w_n.assign(w) for (w, w_n) in zip(v, v_n)]

        # im
        t_actions = tf.placeholder(tf.int32, shape=[None])
        t_err_im = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(t_scores,
                t_actions))
        t_train_im_op = opt.minimize(t_err_im, var_list=v)

        self.session = tf.Session()
        self.session.run(tf.initialize_all_variables())

        self.t_feats = t_feats
        self.t_feats_n = t_feats_n
        self.t_action_mask = t_action_mask
        self.t_actions = t_actions
        self.t_rewards = t_rewards
        self.t_scores = t_scores
        self.t_err_rl = t_err_rl
        self.t_train_rl_op = t_train_rl_op
        self.t_assign_rl_ops = t_assign_rl_ops
        self.t_err_im = t_err_im
        self.t_train_im_op = t_train_im_op

        self.step_count = 0

    def init(self, state):
        pass

    def experience(self, transitions):
        #self.experiences.append(transitions)
        self.experiences += transitions
        self.experiences = self.experiences[-MAX_EXPERIENCES:]

    def demonstrate(self, demonstration):
        self.demonstrations += demonstration
        self.demonstrations = self.demonstrations[-MAX_EXPERIENCES:]

    def act(self, state, randomize=True):
        eps = max(1. - self.step_count / 100000., 0.1)
        if randomize and np.random.random() < eps:
            action = np.random.randint(self.n_actions)
        else:
            preds = self.session.run([self.t_scores], 
                    feed_dict={
                        self.t_feats: [state.features()],
                    })[0][0, :]
            action = np.argmax(preds)

        return action

    def train_rl(self):
        if len(self.experiences) < N_BATCH:
            return 0
        batch_indices = [np.random.randint(len(self.experiences))
                for _ in range(N_BATCH)]
        batch_exp = [self.experiences[i] for i in batch_indices]
        s1, a, s2, r = zip(*batch_exp)
        feats1 = [s.features() for s in s1]
        feats2 = [s.features() for s in s2]
        a_mask = np.zeros((len(batch_exp), self.n_actions))
        for i, act in enumerate(a):
            a_mask[i, act] = 1

        feed_dict = {
            self.t_feats: feats1,
            self.t_action_mask: a_mask,
            self.t_feats_n: feats2,
            self.t_rewards: r
        }

        _, err = self.session.run([self.t_train_rl_op, self.t_err_rl], feed_dict=feed_dict)

        self.step_count += 1

        return err

    def train_im(self):
        if len(self.demonstrations) < N_BATCH:
            return 0
        batch_indices = [np.random.randint(len(self.demonstrations))
                for _ in range(N_BATCH)]
        batch_exp = [self.demonstrations[i] for i in batch_indices]
        s1, a, _, _ = zip(*batch_exp)
        feats = [s.features() for s in s1]

        feed_dict = {
            self.t_feats: feats,
            self.t_actions: a
        }

        _, err = self.session.run([self.t_train_im_op, self.t_err_im], feed_dict=feed_dict)

        return err

    def roll(self):
        self.session.run(self.t_assign_rl_ops)
