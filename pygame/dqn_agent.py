from collections import deque
import os

import numpy as np
import tensorflow as tf


class DQNAgent:
    """
    Convolutional Neural Network with Experience Replay
    """

    def __init__(self, enable_actions, environment_name):
        # parameters
        self.name = os.path.splitext(os.path.basename(__file__))[0]
        self.environment_name = environment_name
        self.enable_actions = enable_actions
        self.n_actions = len(self.enable_actions)
        self.minibatch_size = 32            # original: 32
        self.replay_memory_size = 50000     # original: 1,000,000
        self.replay_start_size = 10000      # original: 50,000
        self.target_update_frequency = 1    # original; 10,000
        self.learning_rate = 1e-6           # original: 0.00025
        self.momentum = 0.0                 # original: 0.95
        self.discount_factor = 0.99         # original: 0.99
        self.initial_exploration = 0.1      # original: 0.1
        self.final_exploration = 0.1        # original: 1.0
        self.exploration_size = 1000000     # original: 1,000,000
        self.model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", self.environment_name)
        self.model_name = "{}.ckpt".format(self.environment_name)

        # replay memory
        self.D = deque()

        # model
        self.init_model()

        # variables
        self.current_loss = 0.0

    def init_model(self):
        # input layer (84 x 84 x 4)
        self.x = tf.placeholder(tf.float32, [None, 84, 84, 4])

        # convolution layer 1 (20 x 20 x 32)
        W_conv1 = tf.Variable(tf.truncated_normal([8, 8, 4, 32], stddev=0.01))
        b_conv1 = tf.Variable(tf.zeros([32]))
        h_conv1 = tf.nn.relu(tf.nn.conv2d(self.x, W_conv1, strides=[1, 4, 4, 1], padding="VALID") + b_conv1)

        # convolution layer 2 (9 x 9 x 64)
        W_conv2 = tf.Variable(tf.truncated_normal([4, 4, 32, 64], stddev=0.01))
        b_conv2 = tf.Variable(tf.zeros([64]))
        h_conv2 = tf.nn.relu(tf.nn.conv2d(h_conv1, W_conv2, strides=[1, 2, 2, 1], padding="VALID") + b_conv2)

        # convolution layer 3 (7 x 7 x 64)
        W_conv3 = tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev=0.01))
        b_conv3 = tf.Variable(tf.zeros([64]))
        h_conv3 = tf.nn.relu(tf.nn.conv2d(h_conv2, W_conv3, strides=[1, 1, 1, 1], padding="VALID") + b_conv3)

        # flatten (3136)
        h_conv3_flat = tf.reshape(h_conv3, [-1, 3136])

        # fully connected layer (512)
        W_fc4 = tf.Variable(tf.truncated_normal([3136, 512], stddev=0.01))
        b_fc4 = tf.Variable(tf.zeros([512]))
        h_fc4 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc4) + b_fc4)

        # output layer (n_actions)
        W_out = tf.Variable(tf.truncated_normal([512, self.n_actions], stddev=0.01))
        b_out = tf.Variable(tf.zeros([self.n_actions]))
        self.y = tf.matmul(h_fc4, W_out) + b_out

        # loss function
        self.y_ = tf.placeholder(tf.float32, [None, self.n_actions])
        self.loss = tf.reduce_mean(tf.square(self.y_ - self.y))

        # train operation
        optimizer = tf.train.RMSPropOptimizer(self.learning_rate, momentum=self.momentum)
        self.training = optimizer.minimize(self.loss)

        # saver
        self.saver = tf.train.Saver()

        # session
        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())

    def Q_values(self, state):
        # Q(state, action) of all actions
        return self.sess.run(self.y, feed_dict={self.x: [state]})[0]

    def select_action(self, state, epsilon):
        if np.random.rand() <= epsilon:
            # random
            return np.random.choice(self.enable_actions)
        else:
            # max_action Q(state, action)
            return self.enable_actions[np.argmax(self.Q_values(state))]

    def current_epsilon(self, global_step):
        annealing = (self.initial_exploration - self.final_exploration) / self.exploration_size * global_step
        return max(self.initial_exploration - annealing, self.final_exploration)

    def store_experience(self, state, action, reward, state_1, terminal):
        self.D.append((state, action, reward, state_1, terminal))
        if len(self.D) > self.replay_memory_size:
            self.D.popleft()

    def experience_replay(self):
        state_minibatch = []
        y_minibatch = []

        # sample random minibatch
        minibatch_size = min(len(self.D), self.minibatch_size)
        minibatch_indexes = np.random.randint(0, len(self.D), minibatch_size)

        for j in minibatch_indexes:
            state_j, action_j, reward_j, state_j_1, terminal = self.D[j]
            action_j_index = self.enable_actions.index(action_j)

            y_j = self.Q_values(state_j)

            if terminal:
                y_j[action_j_index] = reward_j
            else:
                # reward_j + gamma * max_action' Q(state', action')
                y_j[action_j_index] = reward_j + self.discount_factor * np.max(self.Q_values(state_j_1))

            state_minibatch.append(state_j)
            y_minibatch.append(y_j)

        # training
        self.sess.run(self.training, feed_dict={self.x: state_minibatch, self.y_: y_minibatch})

        # for log
        self.current_loss = self.sess.run(self.loss, feed_dict={self.x: state_minibatch, self.y_: y_minibatch})

    def load_model(self, model_path=None):
        global_step = 0

        if model_path:
            # load from model_path
            self.saver.restore(self.sess, model_path)
        else:
            # load from checkpoint
            checkpoint = tf.train.get_checkpoint_state(self.model_dir)
            if checkpoint and checkpoint.model_checkpoint_path:
                self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
                global_step = int(checkpoint.model_checkpoint_path.split("-")[-1])

        return global_step

    def save_model(self, global_step=0):
        self.saver.save(self.sess, os.path.join(self.model_dir, self.model_name), global_step=global_step)
