import net

import numpy as np
import tensorflow as tf

N_BATCH = 100

N_HIDDEN = 256

DISCOUNT = 0.9
MAX_EXPERIENCES = 50000
MAX_REPLAY_LEN = 10

class RecurrentModel(object):
    def __init__(self, config):
        self.experiences = []
        self.demonstrations = []

    def prepare(self, world):
        self.world = world
        self.initializer = tf.nn.rnn_cell.LSTMStateTuple(
                np.zeros((1, N_HIDDEN), dtype=np.float32),
                np.zeros((1, N_HIDDEN), dtype=np.float32))
        self.batch_initializer = tf.nn.rnn_cell.LSTMStateTuple(
                np.zeros((N_BATCH, N_HIDDEN), dtype=np.float32),
                np.zeros((N_BATCH, N_HIDDEN), dtype=np.float32))

        def predictor(scope):
            with tf.variable_scope(scope) as vs:
                t_ep_feats = tf.placeholder(tf.float32,
                            shape=(N_BATCH, MAX_REPLAY_LEN, world.n_features))
                t_ep_l = tf.placeholder(tf.int32, shape=(N_BATCH,))
                t_feats = tf.placeholder(tf.float32,
                            shape=(1, world.n_features))
                t_rnn_state = (
                        tf.placeholder(tf.float32, shape=(1, N_HIDDEN)),
                        tf.placeholder(tf.float32, shape=(1, N_HIDDEN)))
                t_ep_rnn_state = tf.nn.rnn_cell.LSTMStateTuple(
                        tf.placeholder(tf.float32, shape=(N_BATCH, N_HIDDEN)),
                        tf.placeholder(tf.float32, shape=(N_BATCH, N_HIDDEN)))

                cell = tf.nn.rnn_cell.LSTMCell(N_HIDDEN, state_is_tuple=True)
                proj_cell = tf.nn.rnn_cell.OutputProjectionWrapper(cell, world.n_actions)

                # TODO input projection
                t_ep_scores, _ = tf.nn.dynamic_rnn(proj_cell, t_ep_feats, 
                        sequence_length=t_ep_l, initial_state=t_ep_rnn_state, 
                        dtype=tf.float32, scope=vs)
                vs.reuse_variables()
                t_scores, t_next_rnn_state = \
                        proj_cell(t_feats, t_rnn_state)

                v = tf.get_collection(tf.GraphKeys.VARIABLES, scope=vs.name)

            return t_ep_feats, t_ep_rnn_state, t_ep_l, t_ep_scores, \
                t_feats, t_rnn_state, t_scores, t_next_rnn_state, v

        opt = tf.train.AdamOptimizer()

        t_ep_feats, t_ep_rnn_state, t_ep_l, t_ep_scores, \
                t_feats, t_rnn_state, t_scores, t_next_rnn_state, v = predictor("now")

        t_ep_feats_n, t_ep_rnn_state_n, t_ep_l_n, t_ep_scores_n, \
                _, _, _, _, v_n = predictor("next")

        t_loss_mask = tf.placeholder(tf.float32, shape=(N_BATCH, MAX_REPLAY_LEN))

        # rl
        t_rewards = tf.placeholder(tf.float32, shape=(N_BATCH, MAX_REPLAY_LEN))
        t_action_masks = tf.placeholder(tf.float32, shape=(N_BATCH, MAX_REPLAY_LEN, world.n_actions))
        t_scores_chosen = tf.reduce_sum(t_ep_scores * t_action_masks, reduction_indices=(2,))
        t_scores_n_best = tf.reduce_max(t_ep_scores_n, reduction_indices=(2,))
        t_td = t_rewards + DISCOUNT * t_scores_n_best - t_scores_chosen
        t_err_rl = tf.reduce_mean(tf.minimum(t_loss_mask * tf.square(t_td), 1))
        t_train_rl_op = opt.minimize(t_err_rl, var_list=v)
        t_assign_rl_ops = [w_n.assign(w) for (w, w_n) in zip(v, v_n)]

        # im
        t_actions = tf.placeholder(tf.int32, shape=[None, MAX_REPLAY_LEN])
        t_err_im = tf.reduce_mean(t_loss_mask *
                tf.nn.sparse_softmax_cross_entropy_with_logits(t_ep_scores,
                    t_actions))
        t_train_im_op = opt.minimize(t_err_im, var_list=v)

        self.session = tf.Session()
        self.session.run(tf.initialize_all_variables())

        self.t_loss_mask = t_loss_mask
        self.t_ep_feats = t_ep_feats
        self.t_ep_l = t_ep_l
        self.t_feats = t_feats
        self.t_ep_rnn_state = t_ep_rnn_state
        self.t_rnn_state = t_rnn_state
        self.t_scores = t_scores
        self.t_next_rnn_state = t_next_rnn_state
        self.t_ep_feats_n = t_ep_feats_n
        self.t_ep_rnn_state_n = t_ep_rnn_state_n
        self.t_ep_l_n = t_ep_l_n

        self.t_action_masks = t_action_masks
        self.t_rewards = t_rewards
        self.t_err_rl = t_err_rl
        self.t_train_rl_op = t_train_rl_op
        self.t_assign_rl_ops = t_assign_rl_ops

        self.t_actions = t_actions
        self.t_err_im = t_err_im
        self.t_train_im_op = t_train_im_op

        self.step_count = 0

    def init(self, state):
        self.rnn_state = self.initializer

    def experience(self, episode):
        self.experiences.append(episode)
        self.experiences = self.experiences[-MAX_EXPERIENCES:]

    def demonstrate(self, episode):
        self.demonstrations.append(episode)
        self.demonstrations = self.demonstrations[-MAX_EXPERIENCES:]

    def act(self, state, randomize=True):
        eps = max(1. - self.step_count / 100000., 0.1)
        if randomize and np.random.random() < eps:
            return np.random.randint(self.world.n_actions)
        preds, self.rnn_state = self.session.run(
                [self.t_scores, self.t_next_rnn_state],
                feed_dict={
                    self.t_feats: [state.features()],
                    self.t_rnn_state: self.rnn_state
                })
        preds = preds[0, :]
        return np.argmax(preds)

    def train_rl(self):
        if len(self.experiences) < N_BATCH:
            return 0
        batch_indices = [np.random.randint(len(self.experiences))
                for _ in range(N_BATCH)]
        batch_episodes = [self.experiences[i] for i in batch_indices]
        batch_offsets = [np.random.randint(len(e)) for e in batch_episodes]
        #batch_offsets = [0 for e in batch_episodes]
        sliced_episodes = [e[o:o+MAX_REPLAY_LEN]
                for e, o in zip(batch_episodes, batch_offsets)]
        s1 = np.zeros((N_BATCH, MAX_REPLAY_LEN, self.world.n_features))
        a = np.zeros((N_BATCH, MAX_REPLAY_LEN, self.world.n_actions))
        s2 = np.zeros((N_BATCH, MAX_REPLAY_LEN, self.world.n_features))
        r = np.zeros((N_BATCH, MAX_REPLAY_LEN))
        l = np.zeros(N_BATCH)
        loss_mask = np.zeros((N_BATCH, MAX_REPLAY_LEN))

        for i_episode, episode in enumerate(sliced_episodes):
            s1[i_episode, :len(episode), :] = [t.s1.features() for t in episode]
            actions = [t.a for t in episode]
            a[i_episode, :len(episode), actions] = 1
            s2[i_episode, :len(episode), :] = [t.s2.features() for t in episode]
            r[i_episode, :len(episode)] = [t.r for t in episode]
            l[i_episode] = len(episode)
            loss_mask[i_episode, :len(episode)] = 1

        feed_dict = {
            self.t_ep_feats: s1,
            self.t_ep_rnn_state: self.batch_initializer,
            self.t_action_masks: a,
            self.t_ep_feats_n: s2,
            self.t_ep_rnn_state_n: self.batch_initializer,
            self.t_rewards: r,
            self.t_ep_l: l,
            self.t_ep_l_n: l,
            self.t_loss_mask: loss_mask
        }

        _, err = self.session.run([self.t_train_rl_op, self.t_err_rl], 
                feed_dict=feed_dict)

        self.step_count += 1

        return err

    def roll(self):
        self.session.run(self.t_assign_rl_ops)

    def train_im(self):
        if len(self.demonstrations) < N_BATCH:
            return 0
        batch_indices = [np.random.randint(len(self.demonstrations))
                for _ in range(N_BATCH)]
        batch_episodes = [self.demonstrations[i] for i in batch_indices]
        batch_offsets = [np.random.randint(len(e)) for e in batch_episodes]
        #batch_offsets = [0 for e in batch_episodes]
        sliced_episodes = [e[o:o+MAX_REPLAY_LEN]
                for e, o in zip(batch_episodes, batch_offsets)]
        s = np.zeros((N_BATCH, MAX_REPLAY_LEN, self.world.n_features))
        a = np.zeros((N_BATCH, MAX_REPLAY_LEN))
        l = np.zeros(N_BATCH)
        loss_mask = np.zeros((N_BATCH, MAX_REPLAY_LEN))

        for i_episode, episode in enumerate(sliced_episodes):
            s[i_episode, :len(episode), :] = [t.s1.features() for t in episode]
            actions = [t.a for t in episode]
            a[i_episode, :len(episode)] = actions
            l[i_episode] = len(episode)
            loss_mask[i_episode, :len(episode)] = 1

        feed_dict = {
            self.t_ep_feats: s,
            self.t_ep_rnn_state: self.batch_initializer,
            self.t_actions: a,
            self.t_ep_l: l,
            self.t_loss_mask: loss_mask
        }

        _, err = self.session.run([self.t_train_im_op, self.t_err_im], feed_dict=feed_dict)

        return err

