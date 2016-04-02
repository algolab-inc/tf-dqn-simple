import tensorflow as tf
import numpy as np
import os
from collections import deque


class DQNAgent:
    """
    Multi Layer Perceptron with Experience Replay
    """

    def __init__(self, enable_actions, input_size):
        # parameters
        self.name = "dqn"
        self.input_size = input_size
        self.enable_actions = enable_actions
        self.n_actions = len(self.enable_actions)
        self.minibatch_size = 32
        self.replay_memory_size = 1000
        self.learning_rate = 0.001
        self.discount_factor = 0.9
        self.exploration = 0.1
        self.model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

        # replay memory
        self.D = deque()

        # model
        self.init_model()

        # states
        self.current_loss = 0.0

    def init_model(self):
        # input layer (input_n_rows x input_n_cols)
        self.x = tf.placeholder(tf.float32, [None, self.input_size[0], self.input_size[1]])

        # flatten ((input_n_rows * input_n_cols) x 1)
        x_flat = tf.reshape(self.x, [-1, self.input_size[0] * self.input_size[1]])

        # fully connected layer (256 x 1)
        W_fc1 = tf.Variable(tf.truncated_normal([self.input_size[0] * self.input_size[1], 256], stddev=0.01))
        b_fc1 = tf.Variable(tf.truncated_normal([256], stddev=0.01))
        h_fc1 = tf.nn.relu(tf.matmul(x_flat, W_fc1) + b_fc1)

        # output layer (n_actions x 1)
        W_out = tf.Variable(tf.truncated_normal([256, self.n_actions], stddev=0.01))
        b_out = tf.Variable(tf.truncated_normal([self.n_actions], stddev=0.01))
        self.y = tf.matmul(h_fc1, W_out) + b_out

        # loss function
        self.y_ = tf.placeholder(tf.float32, [None, self.n_actions])
        self.loss = tf.reduce_mean(tf.square(self.y_ - self.y))

        # train operation
        optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
        self.training = optimizer.minimize(self.loss)

        # saver
        self.saver = tf.train.Saver()

        # session
        self.sess = tf.InteractiveSession()
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

    def store_experience(self, state, action, reward, state_1, terminal):
        self.D.append([state, action, reward, state_1, terminal])
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

            y_j = self.Q_values(state_j)

            if terminal:
                y_j[action_j] = reward_j
            else:
                # reward_j + gamma * max_action' Q(state', action')
                y_j[action_j] = reward_j + self.discount_factor * np.max(self.Q_values(state_j_1))

            state_minibatch.append(state_j)
            y_minibatch.append(y_j)

        # training
        self.sess.run(self.training, feed_dict={self.x: state_minibatch, self.y_: y_minibatch})

        # for log
        self.current_loss = self.sess.run(self.loss, feed_dict={self.x: state_minibatch, self.y_: y_minibatch})

    def load_model(self, model_path=None):
        if model_path:
            # load from model_path
            self.saver.restore(self.sess, model_path)
        else:
            # load from checkpoint
            checkpoint = tf.train.get_checkpoint_state(self.model_dir)
            if checkpoint and checkpoint.model_checkpoint_path:
                self.saver.restore(self.sess, checkpoint.model_checkpoint_path)

    def save_model(self, model_name="model.ckpt"):
        self.saver.save(self.sess, os.path.join(self.model_dir, model_name))
