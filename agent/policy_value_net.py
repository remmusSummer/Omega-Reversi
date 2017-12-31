"""
Deep policy network implemented by Junxiao Song
Original repo: https://github.com/junxiaosong/AlphaZero_Gomoku.git
"""
import tensorflow as tf
import keras
import numpy as np
from __future__ import print_function

class PolicyValueNet():
    """
    policy value net
    """

    def __init__(self, board_width, board_height, net_params = None):
        self.board_width = board_width
        self.board_height = board_height
        self.learning_rate = tf.Variable(name="learning_rate")
        self.l2_const = 1e-4 #coef of l2 penalty
        self.create_policy_value_net()
        self._loss_train_op()
        if net_params:
            #set net_params to all the layer, need to be rewrite using tensorflow

    def  create_policy_value_net(self):
        self.state_input = keras.Input(shape=(4, self.board_width, self.board_height))
        self.winner = keras.Input()
        self.mcts_probs = keras.Input()

        # keras conv layers
        conv1 = keras.layers.convolutional.Conv2D(filters = 32, kernel_size = (3, 3), padding='same')(self.state_input)
        conv2 = keras.layers.convolutional.Conv2D(filters = 64, kernel_size = (3, 3), padding='same')(conv1)
        conv3 = keras.layers.convolutional.Conv2D(filters = 128, kernel_size = (3, 3), padding='same')(conv2)



        # tensorflow conv layers
        # conv1 = tf.layers.conv2d(inputs = state_input, filters = 32, kernel_size = [3, 3], padding='same')
        # conv2 = tf.layers.conv2d(inputs = conv1, filters = 64, kernel_size = [3, 3], padding='same')
        # conv3 = tf.layers.conv2d(inputs = conv2, filters = 128, kernel_size = [3, 3], padding='same')

        #keras policy network
        policy_net = keras.layers.convolutional.Conv2D(filters = 4, kernel_size = [1, 1])(conv3)
        self.policy_net = keras.layers.core.Dense(units = self.board_width * self.board_height, activation='softmax')
        
        # tensorflow action policy layers
        # policy_net = tf.layers.conv2d(conv3, filters = 4, kernel_size = [1, 1])
        # self.policy_net = tf.layers.dense(policy_net, units = self.board_width * self.board_height, activation=tf.nn.softmax)

        #keras state value layers
        value_layer1 = keras.layers.convolutional.Conv2D(filters = 2, kernel_size = [1, 1])(conv3)
        value_layer2 = keras.layers.core.Dense(units = self.board_width*self.board_height)(value_layer1)
        self.value_net = keras.layers.Dense(units = 1, activation='tanh')(value_layer2)


        # #tensorflow state value layers
        # value_layer1 = tf.layers.conv2d(inputs = conv3, filters = 2, kernel_size = [1, 1])
        # value_layer2 = tf.layers.dense(value_layer1, units = self.board_width*self.board_height)
        # self.value_net = tf.layers.dense(value_layer2, units = 1, activation=tf.nn.tanh)

        self.action_probs = self.policy_net.output
        self.value = self.value_net.output
        
        #get action probs and sate score value
        #self.action_probs, self.value = tensorflow sess run

    def policy_value(self, state_input):
        policy_net_model = keras.Model(inputs = state_input, outputs = self.action_probs)
        value_net_model = keras.Model(inputs = state_input, outputs = self.value)        
        return policy_net_model, value_net_model

    def policy_value_fn(self, board):
        """
        input: board
        output: a list of (action, probability) tuples for each available aciton and the score of the board state
        """
        legal_position = board.get_avalible_move()
        current_state = board.get_current_state()
        act_probs, value = self.policy_value(current_state.reshape(-1, 4, self.board_width, self.board_height))
        act_probs = zip(legal_position, act_probs.flatten()[legal_position])
        return act_probs, value[0][0]

    def _loss_train_op(self):
        """
        There are three loss terms:
        loss = (z - v)^2 + pi^T * log(p) + c||theta||^2
        """
        value_loss = keras.losses.mean_squared_error(self.winner, self.value.flatten())
        policy_loss = keras.losses.categorical_crossentropy(self.action_probs, self.mcts_probs)
        l2_penalty = keras.regularizers.l2(self.l2_const)
        self.loss = value_loss + policy_loss

    def train_step(self,state_input, mcts_probs, winner, learning_rate):
        
