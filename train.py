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

        self.pure_mcts_playout_num = 1000  
        
        self.policy_value_net = PolicyValueNet(self.board_width, self.board_height)
        self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,c_puct = self.c_puct, n_playout = self.n_playout, is_selfplay = 1)

    def get_extend_data(self, play_data):
        """
        augment the data set by rotation and flipping
        play_data:[(state, mcts_prob, winner_z), ..., ...]
        """
        extend_data = []
        for state, mcts_prob, winner in play_data:
            for i in [1, 2, 3, 4]:
                #rotate counterclockwise
                equi_state = np.array([np.rot90(s ,i) for s in equi_state])
                equi_mcts_prob = np.rot90(np.flipud(mcts_prob.reshape(self.board_height,self.board_width)),i)
                extend_data.append((equi_state, np.flipud(equi_mcts_prob).flatten(), winner))
                #flip horizontally
                equi_state = np.array([np.fliplr(s) for s in equi_state])
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                extend_data.append((equi_state, np.flipud(equi_mcts_prob).flatten(), winner))
        return extend_data

    def collect_selfplay_data(self):
        """
        collect self-play data for training
        """
        for i in range(n_games):
            winner, play_data = self.game.start_self_play(self.mcts_player, temp=self.temp)
            self.episode_len = len(play_data)
            #augment the data
            play_data = self.get_extend_data(play_data)
            self.data_buffer.extend(play_data)

    def policy_update(self):
        """
        update the policy value net
        """
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]
        old_probs, old_v = self.policy_value_net.policy_value(state_batch)
        for i in range(self.epochs):
            loss, entropy = self.policy_value_net.train_step(state_batch, mcts_probs_batch, winner_batch, self.learning_rate*self.lr_multiplier)
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)
            kl = np.mean(np.sum(old_probs*(np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)), axis=1))
            if kl > self.kl_targ * 4: #early stopping of D_KL diverges badly
                break
        # adaptively adjust the learning rate
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        explained_var_old = 1 - np.var(np.array(winner_batch) - old_v.flatten())/np.var(np.array(winner_batch))
        explained_var_new = 1 - np.var(np.array(winner_batch) - new_v.flatten())/np.var(np.array(winner_batch))
        print("kl:{:.5f}, lr_mutiplier:{:.3f}, loss:{}, explained_var_old:{:.3f}, explained_var_new:{:.3f}".format(kl, self.lr_multiplier, loss, entropy, explained_var_old, explained_var_new))
        
        return loss, entropy

    def policy_evaluate(self, n_games = 0):
        """
        Evaluate the trained policy by playing games against thr pure MCTS player
        Only for monitoring the progress of training
        """
        current_mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,c_puct = self.c_puct, n_playout = self.n_playout)


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
                    net_params = self.policy_value_net.get_policy_param() #get model parameters
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
    training_pipeline = TrainPipline()
    training_pipeline.run()
