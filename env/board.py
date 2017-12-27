import pygame, sys, random
from pygame.locals import *
import numpy as np

BACKGROUNDCOLOR = (255, 255, 255)
FPS = 40
ROW = 8
COL = 8
PIECEWIDTH = 75
PIECEHEIGHT = 75
BOARDX = 0
BOARDY = 0


class Board(object):

    #terminal function
    def __init__(self):
        self.game = pygame.init()
        self.mainClock = pygame.time.Clock()
        # set every position in the borad to 0
        self.map = np.zeros(shape=(ROW,COL), dtype=np.int32)
        self.state = {}
        self.readableTurns = {'1':'black','-1':'white'}
        self.turnsdic = {'black': 1, 'white': -1}
        self.players = set([1, -1])
        self.currentTurn = 1
        self.width = ROW
        self.height = COL

         #load images
        boardImage = pygame.image.load('env/images/board.png')
        boardRect = boardImage.get_rect()
        self.blackImage = pygame.image.load('env/images/black.png')
        self.blackRect = self.blackImage.get_rect()
        self.WhiteImage = pygame.image.load('env/images/white.png')
        self.whiteRect = self.WhiteImage.get_rect()

        self.basicFront = pygame.font.SysFont(None, 48)
        self.gameOverStr = 'Score '


        #set display window
        self.windoeSurface = pygame.display.set_mode((boardRect.width, boardRect.height))
        self.set_caption("Reversi")

        self.windoeSurface.fill(BACKGROUNDCOLOR)
        self.windoeSurface.blit(boardImage, boardRect, boardRect)

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
        
        self._move(3,3)
        self._move(3,4)
        self._move(4,4)
        self._move(4,3)

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

        x = int((x - BOARDX)/PIECEWIDTH)
        y = int((y - BOARDY)/PIECEHEIGHT)
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
        for x in range(ROW):
            for y in range(COL):
                if self.map[x][y] == 0:
                    return False
        return True

    def judge_winner(self):
        score = self.get_score()
        blackScore = score['black']
        whiteScore = score['white']
        if blackScore > whiteScore:
            return 'black'
        else: 
            return 'white'

    def set_readable_turns(self, readableTurns):
        self.readableTurns = readableTurns

    def _move(self, col, row):

        nextTurn = [*(self.players - set([self.currentTurn]))][0]
        if self.make_move(self.currentTurn, col, row) == True:
            if self.get_valid_moves(nextTurn) != []:
                self.currentTurn = nextTurn
        
        for x in range(ROW):
            for y in range(COL):
                rectDst = pygame.Rect(BOARDX+x*PIECEWIDTH, BOARDY+y*PIECEHEIGHT, PIECEWIDTH, PIECEHEIGHT)
                if self.map[x][y] == 1:
                    self.windoeSurface.blit(self.blackImage, rectDst, self.blackRect)
                elif self.map[x][y] == -1:
                    self.windoeSurface.blit(self.WhiteImage, rectDst, self.whiteRect)

    def move_chess(self, move):
        x, y = self.move_to_location(move)
        self._move(x, y)

    def print_result(self):
        blackScore = self.get_score()['black']
        whiteScore = self.get_score()['white']

        gameOverStr = "Score "

        outputStr = gameOverStr + str(blackScore)+ ":" + str(whiteScore) +", Winner:" + self.judge_winner()
        text = self.basicFront.render(outputStr, True, (0, 0, 0), (0, 0, 255))
        textRect = text.get_rect()
        textRect.centerx = self.windoeSurface.get_rect().centerx
        textRect.centery = self.windoeSurface.get_rect().centery
        self.windoeSurface.blit(text, textRect)

    def update_board(self):
        pygame.display.update()  
        self.mainClock.tick(FPS)
    
    def get_current_player(self):
        return self.currentTurn

    def game_end(self):
        if self.is_game_over():
            winner = self.judge_winner()
            return True, winner
        else:
            return False, None


class Game(object):
    """
    game class 
    """
    def __init__(self, board):
        self.board = board
        self.game = pygame.init()
        self.mainClock = pygame.time.Clock()
    
    def terminate(self):
        pygame.quit()
        sys.exit()
    
    def start_play(self):
        
        while True:
            for event in pygame.event.get():
                if event.type == QUIT:
                    self.board.terminate()

                caption = "Reversi -- current turn: " + self.board.readableTurns[str(playBoard.currentTurn)]
                self.board.set_caption(caption)

                if event.type == MOUSEBUTTONDOWN and event.button == 1:
                    location = pygame.mouse.get_pos()
                    #when using AI, we only got action in a one dimention array
                    self.board.move_chess(self.board.location_to_move(location))

                if self.board.is_game_over():
                    self.board.print_result()
            
            self.board.update_board()

    
if __name__ == '__main__':
    playBoard = Board()
    game = Game(playBoard)

    game.start_play()