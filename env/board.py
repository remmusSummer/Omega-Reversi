import pygame, sys, random
from pygame.locals import *
import numpy as np

BACKGROUNDCOLOR = (255, 255, 255)
FPS = 10
ROW = 8
COL = 8
PIECEWIDTH = 75
PIECEHEIGHT = 75
BOARDX = 0
BOARDY = 0


class Board(object):

    #terminal function
    def __init__(self, width = ROW, height = COL):
        # set every position in the borad to 0
        self.map = [[0 for col in range(COL)] for row in range(ROW)]
        self.state = {}
        self.readableTurns = {'1':'black','-1':'white'}
        self.turnsdic = {'black': 1, 'white': -1}
        self.players = set([1, -1])


    def init_board(self):
        self.currentTurn = 1
        self.width = ROW
        self.height = COL
        self.last_location = None

        #set display window
        self.set_caption("Reversi")
        self.reset_board()
    def set_caption(self, caption):
        pygame.display.set_caption(caption)
    def reset_board(self):
        for x in range (ROW):
            for y in range (COL):
                self.map[x][y] = 0
        
        self.map[3][3] = 1
        self.map[3][4] = -1
        self.map[4][3] = -1
        self.map[4][4] = 1  
        self.last_location = None

        self.currentTurn = 1
        self.width = ROW
        self.height = COL
        self.last_location = None
        self.state = {}

    def get_new_board(self):
        self.map = []
        for i in range (ROW):
            self.map.append([0] * 8)
        return self.map

    def _is_on_board(self,x, y):
        return x >=0 and x <= 7 and y >= 0 and y <= 7

    def is_vaild_move(self,chess, xstart, ystart):
        #if the chess is out of board or the position is already have a chess
        if not self._is_on_board(xstart, ystart) or self.map[xstart][ystart] != 0:
            return False
    
        #put the chess temporally
        self.map[xstart][ystart] = chess

        if chess == 1:
            otherChess = -1
        else:
            otherChess = 1
    
        #Reverse
        chessToFlip  = []
        adjecent = [[0, 1], [0, -1], [1, 1], [1, -1], [1, 0], [-1, 0], [-1, 1],[-1, -1]]
        for xdire, ydire in adjecent:
            x, y = xstart, ystart
            x += xdire
            y += ydire
            if self._is_on_board(x, y) and self.map[x][y] == otherChess:
                x += xdire
                y += ydire
                if not self._is_on_board(x, y):
                    continue
                #untile it reach the position where not the opponent's chess
                while self.map[x][y] == otherChess:
                    x += xdire
                    y += ydire
                    if not self._is_on_board(x, y):
                        break
                #out of board
                if not self._is_on_board(x, y):
                    continue
                #own chess
                if self.map[x][y] == chess:
                    while True:
                        x -= xdire
                        y -= ydire
                        #back to the beginning position
                        if x == xstart and y == ystart:
                            break
                        #chess that need to be flip
                        chessToFlip.append([x, y])
        #undo the temporally chess
        self.map[xstart][ystart] = 0
    
        #if no chess to flip, invalid move
        if len(chessToFlip) == 0:
            return False
    
        return chessToFlip
    #get the valid position
    def get_valid_moves(self,chess):
        validMoves = []
        for x in range(ROW):
            for y in range(COL):
                if self.is_vaild_move(chess, x, y) != False:
                    validMoves.append([x, y])
        return validMoves

    # get the avalible move for policy value net
    def get_avalible_move(self):
        locations = self.get_valid_moves(self.currentTurn)
    
        avalibleMove = []
        for location in locations:
            avalibleMove.append(self.location_to_move(location))
        
        return avalibleMove

    #get the score of black and white
    def get_score(self):
        blackScore = 0
        whiteScore = 0
        for x in range(ROW):
            for y in range(COL):
                if self.map[x][y] == 1:
                    blackScore += 1
                elif self.map[x][y] == -1:
                    whiteScore += 1
        return {'black': blackScore, 'white': whiteScore}

    def move_to_location(self, move):
        x = int(move / self.width)
        y = move % self.width
        return [x, y]

    def location_to_move(self, location):
        if(len(location) != 2):
            return -1
        x = location[0]
        y = location[1]

        move = x* self.width + y

        if(move not in range(self.width * self.height)):
            return -1 
        return move

    def make_move(self,chess, xstart, ystart):
        chessToFlip = self.is_vaild_move(chess, xstart, ystart)
        if chessToFlip ==  False:
            return False

        self.map[xstart][ystart] = chess
        for x, y in chessToFlip:
            self.map[x][y] = chess
        return True

    def is_on_corner(x, y):
        return (x == 0 and y == 0) or (x == 7 and y == 7) or (x == 0 and y ==7) or (x ==7 and y == 0)

    #check if the board have empty space
    def is_game_over(self):
        #check that if the board is full
        isFull = True
        for x in range(ROW):
            for y in range(COL):
                if self.map[x][y] == 0:
                    isFull = False
        #check that if there is avalible position for both side
        haveNoPosition = False
        if(len(self.get_valid_moves(-1)) == 0 or len(self.get_valid_moves(1)) == 0):
            haveNoPosition = True

        if isFull or haveNoPosition:
            return True
        else:
            return False

    def judge_winner(self):
        score = self.get_score()
        blackScore = score['black']
        whiteScore = score['white']
        if blackScore > whiteScore:
            return 'black'
        elif whiteScore > blackScore:
            return 'white'
        else:
            return 'tie'

    def set_readable_turns(self, readableTurns):
        self.readableTurns = readableTurns

    def _move(self, col, row):

        nextTurn = [*(self.players - set([self.currentTurn]))][0]
        if self.make_move(self.currentTurn, col, row) == True:
            self.last_location = (col, row)
            if self.get_valid_moves(nextTurn) != []:
                self.currentTurn = nextTurn 

    def move_chess(self, move):
        x, y = self.move_to_location(move)
        self._move(x, y)
        return True
    
    def get_current_player(self):
        return self.currentTurn

    def get_current_state(self):
        """
        return the current state of the board from the perspective of the current player
        shape: 4*width*height
        """
        square_state = np.zeros((4, self.width, self.height))
        for x in range(self.width):
            for y in range(self.height):
                if self.map[x][y] == self.currentTurn:
                    square_state[0][x, y] = 1.0
                elif self.map[x][y] == [*(self.players - set([self.currentTurn]))][0]:
                    square_state[1][x, y] = 1.0
        if self.last_location:
            square_state[2][self.last_location[0], self.last_location[1]]
        if self.currentTurn == 1:  # if currentplayer is black, 
            square_state[3][:,:] = 1.0
        return square_state[:,::-1,:]



    def game_end(self):
        if self.is_game_over():
            winner = self.judge_winner()
            if winner == 'black':
                return True, 1
            elif winner == 'white':
                return True, -1
            else:
                return True, 0  #game tie
        else:
            return False, None


class Game(object):
    """
    game class 
    """
    def __init__(self, board):
        self.board = board

    def reset_graph(self):
        self.game = pygame.init()
        self.mainClock = pygame.time.Clock()

        # set pygame display parameter
        self.boardImage = pygame.image.load('env/images/board.png')
        self.boardRect = self.boardImage.get_rect()
        self.black_image = pygame.image.load('env/images/black.png')
        self.black_rect = self.black_image.get_rect()
        self.white_image = pygame.image.load('env/images/white.png')
        self.white_rect = self.white_image.get_rect()

        self.basicFront = pygame.font.SysFont(None, 48)
        self.gameOverStr = 'Score '

        self.windowSurface = pygame.display.set_mode((self.boardRect.width, self.boardRect.height))

    def show_backgraph(self):
        self.windowSurface.fill(BACKGROUNDCOLOR)
        self.windowSurface.blit(self.boardImage, self.boardRect, self.boardRect)

    def update_board(self):
        pygame.display.update()
        self.mainClock.tick(FPS)

    def draw_chess(self):
        """
        draw chess on the graphic board
        """

        for x in range(self.board.width):
            for y in range(self.board.height):
                if self.board.map[x][y] == 1:
                    rect_dst = pygame.Rect(BOARDX + x*PIECEWIDTH, BOARDY + y*PIECEHEIGHT, PIECEWIDTH, PIECEHEIGHT)
                    self.windowSurface.blit(self.black_image, rect_dst, self.black_rect)
                elif self.board.map[x][y] == -1:
                    rect_dst = pygame.Rect(BOARDX + x*PIECEWIDTH, BOARDY + y*PIECEHEIGHT, PIECEWIDTH, PIECEHEIGHT)
                    self.windowSurface.blit(self.white_image, rect_dst, self.white_rect)
    

    def terminate(self):
        pygame.quit()
        sys.exit()

    def print_result(self):
        blackScore = self.board.get_score()['black']
        whiteScore = self.board.get_score()['white']

        gameOverStr = "Score "

        outputStr = gameOverStr + str(blackScore) + ":" + str(whiteScore) + ", Winner:" + self.board.judge_winner()
        text = self.basicFront.render(outputStr, True, (0, 0, 0), (0, 0, 255))
        textRect = text.get_rect()
        textRect.centerx = self.windowSurface.get_rect().centerx
        textRect.centery = self.windowSurface.get_rect().centery
        self.windowSurface.blit(text, textRect)
    
    def start_play(self, player1, player2, isShow = False):
        
        self.board.reset_board()
        if isShow:
            self.reset_graph()
            self.draw_chess()
            self.update_board()

        while True:
            end, winner = self.board.game_end()
            if not end:
                player1_move = player1.get_action(self.board)
                player2_move = player2.get_action(self.board)

                self.board.move_chess(player1_move)
                self.board.move_chess(player2_move)
                if isShow:
                    self.draw_chess()
                    self.update_board()

                    self.draw_chesss()
                    self.update_board()

            else:
                if winner != 0:
                    print("Game end. Winner is ", self.board.readableTurns[str(winner)])
                else:
                    print("Game end. Tie")

                if isShow:
                    pygame.quit()

                return winner


        #     for event in pygame.event.get():
        #         if event.type == QUIT:
        #             self.board.terminate()
        #
        #         caption = "Reversi -- current turn: " + self.board.readableTurns[str(playBoard.currentTurn)]
        #         self.board.set_caption(caption)
        #
        #         if event.type == MOUSEBUTTONDOWN and event.button == 1:
        #             location = pygame.mouse.get_pos()
        #             #when using AI, we only got action in a one dimention array
        #             self.board.move_chess(self.board.location_to_move(location))
        #
        #         if self.board.is_game_over():
        #             self.board.print_result()
        #
        #     self.board.update_board(self)

    def start_self_play(self, player, is_shown=0, temp = 1e-3):

        self.board.reset_board()
        states, mcts_probs, current_player = [],[],[]
        while(True):
            move, move_probs = player.get_action(self.board,temp = temp, return_prob=1)
            #store the data
            states.append(self.board.get_current_state())
            mcts_probs.append(move_probs)
            current_player.append(self.board.get_current_player())
            #perform a move
            self.board.move_chess(move)
            end, winner = self.board.game_end()
            if end:
                winnwes_z = np.zeros(len(current_player))
                if winner != 0:
                    winnwes_z[np.array(current_player) == winner] = 1.0
                    winnwes_z[np.array(current_player) != winner] = -1.0
                    #reset MCTS root node
                player.reset_player()
                
                return winner, list(zip(states, mcts_probs, winnwes_z))
                

if __name__ == '__main__':
    playBoard = Board()
    game = Game(playBoard)
    playBoard.init_board(game)
    game.start_play()