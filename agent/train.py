"""
Training pipline class
"""

from __future__ import print_function
import random
import numpy as np
import pickle as pickle
from collections import defaultdict, deque
from env.board import Board, Game
from agent.policy_value_net import PolicyValueNet
from agent.mcts import MCTSPlayer

WIDTH = 8
HEIGHT = 8

class TrainPipline():
    def __init__(self):
        # init board and game 
        self.board_width = WIDTH
        self.board_height = HEIGHT
        self.board = Board()
        self.game = Game(self.board)
        # init training parameters
        self.learning_rate = 5e-3
        self.lr_multiplier = 1.0
        self.temp = 1.0 #temporary parameter
        self.n_playout = 400 # number of simulations for each move
        self.c_puct = 5
        self.buffer_size = 10000
        self.batch_size = 512 # number of mini-batch 
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = 1
        self.epochs = 5 #number of train step for each update
        self.kl_targ = 0.025
        self.check_freq = 50
        self.game_batch_num = 1500
        self.best_win_ratio = 0.0
        
        self.policy_value_net = PolicyValueNet(self.board_width, self.board_height)
        self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,c_puct = self.c_puct, n_playout = self.n_playout, is_selfplay = 1)
        