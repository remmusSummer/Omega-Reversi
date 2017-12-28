"""
A pure implementation of Monte Carlo Tree Search
"""

import numpy as np
import copy
from operator import itemgetter

def rollout_policy_fn(board):
    """
    A coarse, fast version of policy_fn used in the rollout phase.
    """ 
    # rollout randomly
    avalible = board.get_avalible_move()
    action_probs = np.random.rand(len(avalible))
    return zip(avalible, action_probs)

def policy_value_fn(board):
    """
    A function that takes in a state and output a list of (action, probality) tuples
    and a score for the state
    """
    # return uniform probabilities and 0 score for pure MCTS
    avalible = board.get_avalible_move()
    action_probs = np.ones(len(avalible))/len(avalible)
    return zip(avalible, action_probs), 0

class TreeNode(object):
    """
    A node in the MCTS tree. Each node keeps track of its own value Q. prior probability P, 
    and its visit-count-adjusted prior score u.
    """ 

    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {} # a map from action to TreeNode
        self._n_visit = 0
        self._Q = 0
        self._u = 0
        self._p = prior_p

    def expand(self, action_priors):
        """
        Expand tree by creating new childen
        action_priors -- output from policy function - a list of tuples of actions
        and their prior probability according to the policy function.
        """
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self,prob)
    
    def select(self, c_puct):        
    """
    Select action among children that gives maximum action value, Q plus bonus u(P).
    Returns：
    A tuple of (action, next_node)
    """
        return max(self._children.iteritems(), key = lambda act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        """
        update node values from leaf evaluation
        leaf_value: the value of subtree evaluation from the current player's perspective
        """
        #count visit
        self._n_visit += 1
        # update Q, average of values for all visits
        self._Q += 1.0*(leaf_value - self._Q) / self._n_visit
    
    def update_recursive(self, leaf_value):
        """
        apply update recursively to all ancestors.
        """
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct):
        """
        Calculate and return the value for this node: a combanation of leaf evaluations, Q
        and this node's prior adjusted for its visit count, u
        c_puct -- a number in (0, inf) controlling the relative impact of values, Q, 
        and prior probability, P, on this node's score
        """
        self._u = c_puct * self._p * np.sqrt(self._parent._n_visit)/(1 + self._n_visit)
        return self._Q + self._u

    def is_leaf(self):
        """
        Check if it's a leaf node
        """
        return self._children == {}

    def is_root(self):
        return self._parent is None


class MCTS(object):
    """
    Simple implementation of Monte Carlo Tree Search
    """
    def __init__(self, policy_value_fn, c_puct = 5, n_playout = 10000):
        """
        policy_value_fn: a function that takes in a board state and outputs a list of(action, probality)
        tuples and also a score in [-1, 1](the expected value of the end game score from the current player's perspective)
        for the current player.
        c_puct: a number in (0, inf) that controls how quickly exploration converges to the maximum-value policy,
        where a higher value means relying on the prior more
        """
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout

    def _playout(self, state):
        """
        Run a single playout from the root to the leaf, getting a value at the leaf and
        propagating it back through its parents. State is modified in-place, so a copy must be
        provided.
        Arguments:
        state -- a copy of the state.
        """
        node = self._root
        while(True):
            if node.is_leaf():
                break
            action, node = node.select(self._c_puct)
            state.move_chess(action)
        
        action_probs, _ = self._policy(state)
        # check for end of the game
        end, winner = state.game_end()
        if not end:
            node.expand(action_probs)
        # evaluate the leaf node by random rollout
        leaf_value = self._evaluate_rollout(state)
        # update value and visit count of nodes in this traversal
        node.update_recursive(-leaf_value)
        
    def _evaluate_rollout(self, state, limit = 1000):
        """
        Use the rollout policy to play until the end of the game, returning +1 of the currnt player wins,
        -1 if the oppnent wins, and 0 if it is a tie
        """
        player = state.get_current_player()
        for i in range(limit):
            end, winner = state.game_end()
            if end:
                break
            action_probs = rollout_policy_fn(state)
            max_action = max(action_probs, key = itemgetter(1))[0]
            #Todo state.do_move(max_action)
        else:
            print("WARNING: rollout reached move limit")
        if winner == -1:
            return 0
        else:
            return 1 if winner == player else -1

    def get_move(self, state):
        """
        Run all playout sequencially, return the most frequent visited action.
        state: the current state, including both game state and the current player
        """   
        for n in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)
        return max(self._root._children.iteritems(), key=lambda act_node:act_node[1]._n_visit)
    
    def update_with_move(self, last_move):
        """
        Step forward in the tree
        """
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def __str__(self):
        return "MCTS"

class MCTSPlayer(object):
    """
    AI player based on MCTS
    """
    def __init__(self, c_puct = 5, n_playout=2000):
        self.mcts = MCTS(policy_value_fn, c_puct, n_playout)

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self, board):
        avalible = board.get_avalible_move()
        sensible_move = avalible
        if len(sensible_move) > 0:
            move = self.mcts.get_move(board)
            self.mcts.update_with_move(-1)
            return move
        else:
            print("WARNING: the board is full")
    
    def __str__(self):
        return "MCTS {}".format(self.player)
