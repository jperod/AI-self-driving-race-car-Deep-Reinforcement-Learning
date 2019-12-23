from __future__ import generator_stop
from exp_replay import ExperienceReplay
import numpy as np
import tensorflow.contrib.slim as slim
import tensorflow as tf
import re
from processimage import processimage


class DQN:

    def __init__(self,
            env,
            batchsize=64,
            pic_size=(96, 96),
            num_frame_stack=3,
            gamma=0.95,
            frame_skip=3,
            train_freq=3,
            initial_epsilon=1,
            min_epsilon=0.05,
            render=False,
            epsilon_decay_steps=int(100000),
            min_experience_size=int(1000),
            experience_capacity=int(100000),
            target_network_update_freq=1000,
            regularization = 1e-6,
            optimizer_params = None,
            action_map=None
    ):
        self.exp_history = ExperienceReplay(
            num_frame_stack,
            capacity=experience_capacity,
            pic_size=pic_size
        )

        # in playing mode we don't store the experience to agent history
        # but this cache is still needed to get the current frame stack
        self.playing_cache = ExperienceReplay(
            num_frame_stack,
            capacity=num_frame_stack * 5 + 10,
            pic_size=pic_size
        )

        if action_map is not None:
            self.dim_actions = len(action_map)
        else:
            self.dim_actions = env.action_space.n

        self.target_network_update_freq = target_network_update_freq
        self.action_map = action_map
        self.env = env
        self.batchsize = batchsize
        self.num_frame_stack = num_frame_stack
        self.gamma = gamma
        self.frame_skip = frame_skip
        self.train_freq = train_freq
        self.initial_epsilon = initial_epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay_steps = epsilon_decay_steps
        self.render = render
        self.min_experience_size = min_experience_size
        self.pic_size = pic_size
        self.regularization = regularization
        # These default magic values always work with Adam
        self.global_step = tf.Variable(0, trainable=False)
        self.increment_global_step_op = tf.assign(self.global_step, self.global_step+1)
        self.decayed_lr = tf.train.exponential_decay(0.001, self.global_step, 200000, 0.7, staircase=False)
        lr = self.decayed_lr
        # lr = 0.001
        self.optimizer_params = optimizer_params or dict(learning_rate=lr, epsilon=1e-7)

        self.do_training = True
        self.playing_epsilon = 0.0
        self.session = None

        self.state_size = (self.num_frame_stack,) + self.pic_size
        self.global_counter = 0
        self.episode_counter = 0

    def build_graph(self):
        input_dim_general = (None, self.pic_size[0], self.pic_size[1], self.num_frame_stack)   # (None, 4, 96, 96) changed to (None, 96, 96, 4)
        input_dim_with_batch = (self.batchsize, self.pic_size[0], self.pic_size[1], self.num_frame_stack) #Input dimensions: (64, 4, 96, 96) changed to (64, 96, 96, 4)

        self.input_prev_state = tf.compat.v1.placeholder(tf.float32, input_dim_general, "prev_state")
        self.input_next_state = tf.compat.v1.placeholder(tf.float32, input_dim_with_batch, "next_state")
        self.input_reward = tf.compat.v1.placeholder(tf.float32, self.batchsize, "reward")
        self.input_actions = tf.compat.v1.placeholder(tf.int32, self.batchsize, "actions")
        self.input_done_mask = tf.compat.v1.placeholder(tf.int32, self.batchsize, "done_mask")

        # The target Q-values come from the fixed network
        with tf.compat.v1.variable_scope("fixed"): #64 96 96 3
            # Create target network which is gonna be fixed and updated every C parameters
            qsa_targets = self.create_network(self.input_next_state, trainable=False)

        with tf.compat.v1.variable_scope("train"): # ? 96 96 3
            # Create Prediction/Estimate network which will be trained/updated every 3 frames
            # Create Prediction/Estimate network which will be trained/updated every 3 frames
            qsa_estimates = self.create_network(self.input_prev_state, trainable=True)

        self.best_action = tf.argmax(qsa_estimates, axis=1)

        not_done = tf.cast(tf.logical_not(tf.cast(self.input_done_mask, "bool")), "float32")
        # select the chosen action from each row
        # in numpy this is qsa_estimates[range(batchsize), self.input_actions]
        action_slice = tf.stack([tf.range(0, self.batchsize), self.input_actions], axis=1)
        #
        q_estimates_for_input_action = tf.gather_nd(qsa_estimates, action_slice)

        #Taken from paper : Loss = [(r + gamma*max Qtarget)-(Q estimate)^2]
        q_target = tf.reduce_max(qsa_targets, -1) * self.gamma * not_done + self.input_reward
        training_loss = tf.nn.l2_loss(q_target - q_estimates_for_input_action) / self.batchsize

        # reg_loss = tf.add_n(tf.losses.get_regularization_losses())
        reg_loss = [0]

        #Adam optimizer
        optimizer = tf.train.AdamOptimizer(**(self.optimizer_params))
        #Adadelta optimizer:
        # optimizer = tf.train.RMSPropOptimizer(**(self.optimizer_params))

        self.train_op = optimizer.minimize(reg_loss + training_loss)

        train_params = self.get_variables("train")
        fixed_params = self.get_variables("fixed")


        assert (len(train_params) == len(fixed_params))
        self.copy_network_ops = [tf.assign(fixed_v, train_v) for train_v, fixed_v in zip(train_params, fixed_params)]

    def get_variables(self, scope):
        vars = [t for t in tf.compat.v1.global_variables()
            if "%s/" % scope in t.name and "Adam" not in t.name]
        return sorted(vars, key=lambda v: v.name)

    def create_network(self, input, trainable):
        if trainable:
            # wr = None
            wr = tf.compat.v1.keras.regularizers.l2(l=self.regularization)
        else:
            wr = None

        net = tf.layers.conv2d(inputs=input, filters=8, kernel_size=(7,7), strides=4, name='conv1', kernel_regularizer=wr)
        net = tf.nn.relu(net)
        net = tf.nn.max_pool2d(net, ksize=2, strides=2, padding='SAME')
        net = tf.layers.conv2d(inputs=net, filters=16, kernel_size=(3, 3), strides=1, name='conv2',
                               kernel_regularizer=wr)
        net = tf.nn.relu(net)
        net = tf.nn.max_pool2d(net, ksize=2, strides=2, padding='SAME')
        net = tf.layers.flatten(net)
        net = tf.layers.dense(net, 400, activation=tf.nn.relu, kernel_regularizer=wr)
        # net = tf.layers.dropout(net, 0.5)
        q_state_action_values = tf.layers.dense(net, self.dim_actions, activation=None, kernel_regularizer=wr)

        return q_state_action_values

    # def check_early_stop(self, reward, totalreward):
    #     return False, 0.0

    def get_random_action(self):
        return np.random.choice(self.dim_actions)

    def get_epsilon(self):
        if not self.do_training:
            return self.playing_epsilon
        elif self.global_counter >= self.epsilon_decay_steps:
            return self.min_epsilon
        else:
            # linear decay
            r = 1.0 - self.global_counter / float(self.epsilon_decay_steps)
            return self.min_epsilon + (self.initial_epsilon - self.min_epsilon) * r

    def train(self):
        batch = self.exp_history.sample_mini_batch(self.batchsize)
        # Feed dict
        fd = {
            self.input_reward: "reward",
            self.input_prev_state: "prev_state",
            self.input_next_state: "next_state",
            self.input_actions: "actions",
            self.input_done_mask: "done_mask"
        }
        fd1 = {ph: batch[k] for ph, k in fd.items()}
        self.session.run([self.train_op], fd1)

    def play_episode(self, render, load_checkpoint):
        eh = (
            self.exp_history if self.do_training
            else self.playing_cache
        )
        total_reward = 0
        total_score = 0
        frames_in_episode = 0

        first_frame = self.env.reset()
        first_frame_pp = processimage.process_image(first_frame)

        eh.start_new_episode(first_frame_pp)

        epsilon = self.get_epsilon()
        while True:
            if np.random.rand() > epsilon and not load_checkpoint:
                action_idx = self.session.run(
                    self.best_action,
                    {self.input_prev_state: eh.current_state()[np.newaxis, ...]}
                )[0]
            elif not load_checkpoint:
                action_idx = self.get_random_action()
            elif load_checkpoint:
                action_idx = self.session.run(
                    self.best_action,
                    {self.input_prev_state: eh.current_state()[np.newaxis, ...]}
                )[0]

            if self.action_map is not None:
                action = self.action_map[action_idx]
            else:
                action = action_idx

            reward = 0
            score = 0
            for _ in range(self.frame_skip):
                observation, r, done, info = self.env.step(action)
                if render:
                    self.env.render()


                score += r
                #Increase rewards on the last frames if reward is positive
                if r > 0:
                    r = r + frames_in_episode*0.2 #in 230 frames late game it adds +- 50 reward to tiles
                reward += r

                if done:
                    break

            early_done, punishment = self.check_early_stop(reward, total_reward, frames_in_episode)
            if early_done:
                reward += punishment

            done = done or early_done

            total_reward += reward
            total_score += score
            frames_in_episode += 1
            observation = processimage.process_image(observation)
            eh.add_experience(observation, action_idx, done, reward)

            if self.do_training:
                self.global_counter += 1
                step = self.session.run(self.increment_global_step_op)
                if self.global_counter % self.target_network_update_freq:
                    self.update_target_network()
                train_cond = (
                    self.exp_history.counter >= self.min_experience_size and
                    self.global_counter % self.train_freq == 0
                )
                if train_cond:
                    self.train()

            if done:
                if self.do_training:
                    self.episode_counter += 1

                return total_score, total_reward, frames_in_episode, epsilon

    def update_target_network(self):
        self.session.run(self.copy_network_ops)
