"""
Human play with Omerga-Reversi
"""

from train import PolicyValueNet
from agent.mcts import MCTSPlayer
from env.board import Game, Board, PIECEHEIGHT, PIECEWIDTH

import pygame

def run():
    board = Board()
    board.init_board()

    game = Game(board)
    game.reset_graph()

    policy_path = './model/best_policy_model.h5'

    policy_value_net = PolicyValueNet(8, 8, policy_path)
    omega_reversi = policy_value_net.mcts_player

    human_turn = 1
    ai_turn = -1

    while True:
        game.show_backgraph()
        for event in pygame.event.get():

            game.draw_chess()
            if event.type == pygame.QUIT:
                game.terminate()

            is_human_move = False

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                #human player move
                board.set_caption("Reversi -- current turn: " + board.readableTurns[str(board.currentTurn)])
                game.update_board()
                location = pygame.mouse.get_pos()
                x = location[0] // PIECEWIDTH
                y = location[1] // PIECEHEIGHT
                human_move = board.location_to_move((x,y))
                if board.currentTurn == human_turn:
                    is_human_move = board.move_chess(human_move)
                    game.draw_chess()
                    game.update_board()

                # AI move
                if is_human_move:
                    board.set_caption("Reversi -- current turn: " + board.readableTurns[str(board.currentTurn)])
                    game.update_board()
                    if board.currentTurn == ai_turn:
                        print("omega-reversi is computing its move")
                        ai_move = omega_reversi.get_action(board, temp=1e-3)
                        board.move_chess(ai_move)
                        game.draw_chess()
                        game.update_board()

                break

if __name__ == '__main__':
    run()
