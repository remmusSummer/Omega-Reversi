"""
Deep policy network implemented by Junxiao Song
Original repo: https://github.com/junxiaosong/AlphaZero_Gomoku.git
"""
from __future__ import print_function
import tensorflow as tf
import keras
import numpy as np
import random

import pickle as pickle
from collections import defaultdict, deque
from env.board import Board, Game
from agent.mcts import MCTSPlayer

class PolicyValueNet():
    """
    policy value net
    """

    def __init__(self, board_width, board_height, net_params = None):

        # init network parameters
        self.learning_rate = 5e-3
        self.l2_const = 1e-4 #coef of l2 penalty
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
        self.pure_mcts_playout_num = 1000
        
        # initial env
        self.board = Board()
        self.game = Game(self.board)
        self.board.init_board(self.game)
        self.board_width = board_width
        self.board_height = board_height

        self.create_policy_value_net()
        self._loss_train_op()
        
        #init mcts player
        self.mcts_player = MCTSPlayer(self.policy_value_fn, self.board.get_current_player(), c_puct = self.c_puct, n_playout = self.n_playout, is_selfplay = 1)


    def  create_policy_value_net(self):
        self.state_input = keras.Input(shape=(4, self.board_width, self.board_height))
        self.winner = keras.Input(shape = (1, ))
        self.mcts_probs = keras.Input(shape = (64, ))

        # keras conv layers
        conv1 = keras.layers.convolutional.Conv2D(filters = 32, kernel_size = (3, 3), padding='same', activation='relu')(self.state_input)
        conv2 = keras.layers.convolutional.Conv2D(filters = 64, kernel_size = (3, 3), padding='same', activation='relu')(conv1)
        conv3 = keras.layers.convolutional.Conv2D(filters = 128, kernel_size = (3, 3), padding='same', activation='relu')(conv2)


        # tensorflow conv layers
        # conv1 = tf.layers.conv2d(inputs = state_input, filters = 32, kernel_size = [3, 3], padding='same')
        # conv2 = tf.layers.conv2d(inputs = conv1, filters = 64, kernel_size = [3, 3], padding='same')
        # conv3 = tf.layers.conv2d(inputs = conv2, filters = 128, kernel_size = [3, 3], padding='same')
        
        #regularization teams
        l2_penalty = keras.regularizers.l2(self.l2_const)

        #keras policy network
        policy_net1 = keras.layers.convolutional.Conv2D(filters = 4, kernel_size = [1, 1], activation='relu')(conv3)
        policy_net2 = keras.layers.Flatten()(policy_net1)
        self.policy_net = keras.layers.Dense(units = self.board_width * self.board_height, activation='softmax', activity_regularizer=l2_penalty)(policy_net2)
        
        # tensorflow action policy layers
        # policy_net = tf.layers.conv2d(conv3, filters = 4, kernel_size = [1, 1])
        # self.policy_net = tf.layers.dense(policy_net, units = self.board_width * self.board_height, activation=tf.nn.softmax)

        #keras state value layers
        value_layer1 = keras.layers.convolutional.Conv2D(filters = 2, kernel_size = [1, 1], activation='relu')(conv3)
        value_layer2 = keras.layers.Flatten()(value_layer1)
        value_layer3 = keras.layers.Dense(units = self.board_width*self.board_height, activation='relu')(value_layer2)
        self.value_net = keras.layers.Dense(units = 1, activation='tanh', activity_regularizer=l2_penalty)(value_layer3)

        # #tensorflow state value layers
        # value_layer1 = tf.layers.conv2d(inputs = conv3, filters = 2, kernel_ssize = [1, 1])
        # value_layer2 = tf.layers.dense(value_layer1, units = self.board_width*self.board_height)
        # self.value_net = tf.layers.dense(value_layer2, units = 1, activation=tf.nn.tanh)

        self.model = keras.Model(input = self.state_input, outputs = [self.policy_net, self.value_net])

        #get action probs and sate score value
        #self.action_probs, self.value = tensorflow sess run

    def _loss_train_op(self):
        #There are three loss terms:
        #loss = (z - v)^2 + pi^T * log(p) + c||theta||^2
        value_loss = keras.losses.mean_squared_error(self.winner, self.value_net)
        policy_loss = keras.losses.categorical_crossentropy(self.policy_net, self.mcts_probs)

        self.loss = value_loss + policy_loss


    def policy_value(self, state_input):
        """
        This function predict the action probability and value
        input: state
        output: action probability and value
        """

        act_probs, value = self.model.predict(state_input)
        return act_probs, value

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
               
    def get_extend_data(self, play_data):
        """
        augment the data by rotation and flipping
        play_data:[(state, mcts_prob, winner_z),..., ..., ...]
        """
        extend_data = []
        for state, mcts_prob, winner in play_data:
            for i in [1, 2, 3, 4]:
                # rotation
                equi_state = np.array([np.rot90(s ,i) for s in state])
                equi_mcts_prob = np.rot90(np.flipud(mcts_prob.reshape(self.board_height,self.board_width)),i)
                extend_data.append((equi_state, np.flipud(equi_mcts_prob).flatten(), winner))
                #flip horizontally
                equi_state = np.array([np.fliplr(s) for s in equi_state])
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                extend_data.append((equi_state, np.flipud(equi_mcts_prob).flatten(), winner))
        return extend_data

    def collect_selfplay_data(self, n_games = 1):
        """
        collect self-play data for training
        """
        for i in range(n_games):
            winner, play_data = self.game.start_self_play(self.mcts_player, temp=self.temp)
            self.episode_len = len(play_data)
            #augment the data
            play_data = self.get_extend_data(play_data)
            self.data_buffer.extend(play_data)
    
    #TODO: two model aggregation
    def policy_update(self):
        mini_batch = random.sample(self.data_buffer, self.batch_size)

        state_batch = np.array([data[0] for data in mini_batch]).reshape(-1, 4, 8, 8)
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]
        old_probs, old_v = self.policy_value(state_batch)


        self.winner = winner_batch
        self.mcts_probs = mcts_probs_batch

        optimizer = keras.optimizers.Adam(lr=self.learning_rate * self.lr_multiplier)
        self.model.compile(optimizer=optimizer, loss=self.loss)

        self._loss_train_op()

        for i in range(self.epochs):
            loss = self.model.fit(state_batch, [mcts_probs_batch, winner_batch], epochs=self.epochs).loss
            new_probs, new_v = self.policy_value(state_batch)
            kl = np.mean(np.sum(old_probs * (np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),axis=1))
            if kl > self.kl_targ * 4:
                break            
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        explained_var_old =  1 - np.var(np.array(winner_batch) - old_v.flatten())/np.var(np.array(winner_batch))
        explained_var_new = 1 - np.var(np.array(winner_batch) - new_v.flatten())/np.var(np.array(winner_batch)) 

        print("kl:{:.5f},lr_multiplier:{:.3f},loss:{},explained_var_old:{:.3f},explained_var_new:{:.3f}".format(
                kl, self.lr_multiplier, loss, explained_var_old, explained_var_new))

        return loss


    def policy_evaluate(self, n_games = 0):
        """
        Evaluate the trained policy by playing games against thr pure MCTS player
        Only for monitoring the progress of training
        """
        current_mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,c_puct = self.c_puct, n_playout = self.n_playout)    
    
    def get_policy_param(self):
        """
        return the parameters of both policy and value network
        """
        policy_net_parameters = []
        for layer in self.policy_net_model.layers:
            policy_net_parameters.append(layer.get_weights())
        
        value_net_parameters = []
        for layer in self.value_net_model.layers:
            value_net_parameters.append(layer.get_weights())
        
        return [policy_net_parameters, value_net_parameters]

    def run(self):
        """
        run the training pipline
        """
        try:
            for i in range(self.game_batch_num):
                self.collect_selfplay_data(self.play_batch_size)
                print("batch i:{}, episode_len:{}".format(i+1, self.episode_len))
                if len(self.data_buffer) > self.batch_size:
                    loss, entropy = self.policy_update()
                #check the performance of the current model, and save the model parameters
                if (i+1) % self.check_freq == 0:
                    print("current self-play batch: {}".format(i+1))
                    win_ratio = self.policy_evaluate()
                    net_params = self.get_policy_param() #get model parameters
                    pickle.dump(net_params, open('../model/current_policy.model','wb'),pickle.HIGHEST_PROTOCOL) # save model parameters to file
                    if win_ratio > self.best_win_ratio:
                        print("New best policy get!")
                        self.best_win_ratio = win_ratio
                        pickle.dump(net_params, open('../model/best_policy.model','wb'), pickle.HIGHEST_PROTOCOL) # update the best policy
                        if self.best_win_ratio == 1.0 and self.pure_mcts_playout_num < 5000:
                            self.pure_mcts_playout_num += 1000
                            self.best_win_ratio = 0.0
        except KeyboardInterrupt:
            print('\n\rquit')

if __name__ == '__main__':
    training_pipeline = PolicyValueNet(8, 8)
    training_pipeline.run()
