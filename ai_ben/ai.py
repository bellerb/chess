import math
import numpy
import pandas as pd
from copy import deepcopy
#from model import TransformerModel

class Agent:
    def __init__(self):
        self.notation = {1:'p',2:'n',3:'b',4:'r',5:'q',6:'k'} #Map of notation to part number
        self.token_bank = pd.read_csv('ai_ben/token_bank.csv') #All tokens
        self.MCTS = MCTS()

    def choose_action(self,game):
        cur,next = self.MCTS.search(game)
        '''
        state = self.encode_state(deepcopy(game))
        for cur_cord,m_bank in game.possible_board_moves(capture=True).items():
            if len(m_bank) > 0:
                action = self.encode_valid_actions(game,cur_cord,m_bank)
                input = [x for y in state for x in y] + ['SEP'] + [x for y in action for x in y] #Flatened input
                input = [self.token_bank['token'].eq(t).idxmax() for t in input] #Set tokens to numeric representation
                print(input)
                break
        #print(state)
        '''
        return cur,next

    def encode_state(self,game):
        temp_board = game.board
        for y,row in enumerate(temp_board):
            for x,peice in enumerate(row):
                if peice != 0:
                    temp_board[y][x] = f'{self.notation[abs(peice)]}w' if peice > 0 else f'{self.notation[abs(peice)]}b'
                else:
                    temp_board[y][x] = 'MASK'
        return temp_board

    def encode_valid_actions(self,game,cur_pos,m_bank):
        c_cord = game.board_2_array(cur_pos)
        c_peice = game.board[c_cord[1]][c_cord[0]]
        temp_board = self.encode_state(deepcopy(game))
        for m in m_bank:
            temp_board[m[1]][m[0]] = f'{self.notation[abs(c_peice)]}wm' if c_peice > 0 else f'{self.notation[abs(c_peice)]}bm'
        return temp_board

class MCTS:
    def __init__(self):
        self.tree = {}
        self.Cpuct = 2

    def search(self,game):
        parent_hash = game.EPD_hash()
        if parent_hash not in self.tree:
            self.tree[parent_hash] = Node()
        #print(numpy.array(game.board))
        #print(game.EPD_hash())
        b_action = (None,None)
        b_upper = float('-inf')
        for cur,moves in game.possible_board_moves(capture=True).items():
            if len(moves) > 0 and ((cur[0].isupper() and game.p_move == 1) or (cur[0].islower() and game.p_move == -1)):
                for next in moves:
                    imag_game = deepcopy(game)
                    if imag_game.move(cur,f'{imag_game.x[next[0]]}{imag_game.y[next[1]]}') == True:
                        hash = imag_game.EPD_hash()
                        if hash not in self.tree:
                            self.tree[hash] = Node()
                        else:
                            self.tree[hash].N += 1
                        #print(numpy.array(imag_game.board))
                        #print(imag_game.EPD_hash())
                        #print('------')
                        state = imag_game.is_end()
                        #imag_game.p_move = imag_game.p_move * (-1)
                        if sum(state) > 0:
                            if (state == [1,0,0] and game.p_move == 1) or (state == [0,0,1] and game.p_move == -1):
                                print('WIN',state)
                                self.tree[hash].Q = 3 #Win
                            elif state == [0,0,0]:
                                print('TIE')
                                self.tree[hash].Q = 1 #Tie
                            else:
                                print('LOSS')
                                self.tree[hash].Q = 0 #Loss
                            self.tree[hash].P = 1
                        else:
                            self.tree[hash].Q = 1
                            self.tree[hash].P = 1
                        u = self.tree[hash].Q + self.Cpuct * self.tree[hash].P * math.sqrt(self.tree[parent_hash].N)/(1+self.tree[hash].N)
                        if u > b_upper:
                            b_action = (cur,f'{imag_game.x[next[0]]}{imag_game.y[next[1]]}')
                    else:
                        continue

        self.tree[parent_hash].N += 1
        #print(b_action)
        return b_action

class Node:
    def __init__(self):
        self.Q = 0 #Reward
        self.P = 0 #Policy
        self.N = 0 #Visits
