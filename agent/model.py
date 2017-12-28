import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np 
import os
import pickle as pk
import random

from env import board


class TFPolicyValueNet():
    "policy value network using tensorflow"

    def __init__(self, board_width, board_height, net_params=None):
        self.board_width = board_width
        self.board_height = board_height
        self.learning_rate = tf.variable(dtype=tf.float32, name="learning_rate")
        self.12_const = 1e-4
        

    def build_value_net(self):
        """build value network"""
        self.state_input = tf.zeros([None, 4, 8, 8], name='state')
        self.winner = tf.variable(dtype=tf.int32)
        
