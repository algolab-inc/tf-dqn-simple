import tensorflow as tf
import numpy as np
import os
from collections import deque


class DQNAgent:
    """
    Multi Layer Perceptron with Experience Replay
    """
    def __init__(self, enable_actions):
        # variables
        self.enable_actions = enable_actions
        self.n_actions = len(self.enable_actions)
        self.memory_size = 1000
        self.batch_size = 32
        self.learning_rate = 0.001
        self.gamma = 0.9
        self.model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        self.model_name = "model.ckpt"

        # memory
        self.D = deque()

        # network
        self.init_network()

        # states
        self.current_loss = 0.0

    def init_network(self):
        # input layer
        self.x = tf.placeholder(tf.float32, [None, 8, 8])

        # flatten
        x_flat = tf.reshape(self.x, [-1, 64])

        # fully connected layer
        W_fc1 = tf.Variable(tf.truncated_normal([64, 256], stddev=0.01))
        b_fc1 = tf.Variable(tf.truncated_normal([256], stddev=0.01))
        h_fc1 = tf.nn.relu(tf.matmul(x_flat, W_fc1) + b_fc1)

        # output layer
        W_out = tf.Variable(tf.truncated_normal([256, self.n_actions], stddev=0.01))
        b_out = tf.Variable(tf.truncated_normal([self.n_actions], stddev=0.01))
        self.y = tf.matmul(h_fc1, W_out) + b_out

        # loss function
        self.y_ = tf.placeholder(tf.float32, [None, self.n_actions])
        self.loss = tf.reduce_mean(tf.square(self.y_ - self.y))

        # train operation
        optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
        self.train_op = optimizer.minimize(self.loss)

        # saver
        self.saver = tf.train.Saver()

        # session
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.initialize_all_variables())

    def get_Q_sa(self, state):
        # Q(state, action)
        return self.sess.run(self.y, feed_dict={self.x: [state]})[0]

    def select_action(self, state, epsilon):
        if np.random.rand() <= epsilon:
            # random action
            return np.random.choice(self.enable_actions)
        else:
            # max_action Q(state, action)
            Q_sa = self.get_Q_sa(state)
            return self.enable_actions[np.argmax(Q_sa)]

    def store_experience(self, state, action, reward, state_1, terminal):
        self.D.append([state, action, reward, state_1, terminal])
        if len(self.D) > self.memory_size:
            self.D.popleft()

    def experience_replay(self):
        state_batch = []
        y_batch = []

        # sample random minibatch
        batch_size = min(len(self.D), self.batch_size)
        batch_indexes = np.random.randint(0, len(self.D), batch_size)

        for j in batch_indexes:
            state_j, action_j, reward_j, state_j_1, terminal = self.D[j]

            Q_sa_j = self.get_Q_sa(state_j)
            Q_sa_j_1 = self.get_Q_sa(state_j_1)

            y_j = Q_sa_j

            if terminal:
                y_j[action_j] = reward_j
            else:
                # reward_j + gamma * max_action' Q(state', action')
                y_j[action_j] = reward_j + self.gamma * np.max(Q_sa_j_1)

            state_batch.append(state_j)
            y_batch.append(y_j)

        # training
        self.sess.run(self.train_op, feed_dict={self.x: state_batch, self.y_: y_batch})

        # for log
        self.current_loss = self.sess.run(self.loss, feed_dict={self.x: state_batch, self.y_: y_batch})

    def load_model(self, model_path=None):
        if model_path:
            # load from model_path
            self.saver.restore(self.sess, model_path)
        else:
            # load from checkpoint
            checkpoint = tf.train.get_checkpoint_state(self.model_dir)
            if checkpoint and checkpoint.model_checkpoint_path:
                self.saver.restore(self.sess, checkpoint.model_checkpoint_path)

    def save_model(self):
        self.saver.save(self.sess, os.path.join(self.model_dir, self.model_name))
