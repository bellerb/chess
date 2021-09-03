import math
import numpy
import random
import pandas as pd
from copy import deepcopy
#from model import TransformerModel

class Agent:
    def __init__(self):
        self.notation = {1:'p',2:'n',3:'b',4:'r',5:'q',6:'k'} #Map of notation to part number
        self.token_bank = pd.read_csv('ai_ben/token_bank.csv') #All tokens
        self.MCTS = MCTS()

    def choose_action(self,game):
        #print(self.MCTS.tree)
        parent_hash = game.EPD_hash()
        if parent_hash not in self.MCTS.tree:
            self.MCTS.tree[parent_hash] = self.MCTS.Node()
        self.MCTS.Player = game.p_move
        b_action = (None,None)
        b_upper = float('-inf')
        u_bank = {}
        for c,moves in game.possible_board_moves(capture=True).items():
            if len(moves) > 0 and ((c[0].isupper() and game.p_move == 1) or (c[0].islower() and game.p_move == -1)):
                for n in moves:
                    imag_game = deepcopy(game)
                    if imag_game.move(c,f'{game.x[n[0]]}{game.y[n[1]]}') == True:
                        imag_game.p_move = imag_game.p_move * (-1)
                        hash = imag_game.EPD_hash()
                        self.MCTS.search(imag_game)
                        if hash in self.MCTS.tree:
                            #print(c,f'{game.x[n[0]]}{game.y[n[1]]}',self.MCTS.tree[hash].Q,self.MCTS.tree[hash].P)
                            u_bank[f'{c}-{game.x[n[0]]}{game.y[n[1]]}'] = self.MCTS.tree[hash].Q + self.MCTS.Cpuct * self.MCTS.tree[hash].P * math.sqrt(self.MCTS.tree[parent_hash].N)/(1+self.MCTS.tree[hash].N)
                            #u = self.MCTS.tree[hash].Q + self.MCTS.Cpuct * self.MCTS.tree[hash].P * math.sqrt(self.MCTS.tree[parent_hash].N)/(1+self.MCTS.tree[hash].N)
                            #if u > b_upper:
                                #cur = c
                                #next = f'{game.x[n[0]]}{game.y[n[1]]}'
                                #b_upper =  u
        #print(cur,next)
        m_bank = [k for k,v in u_bank.items() if v == max(u_bank.values())]
        print(m_bank)
        cur,next = random.choice(m_bank).split('-')
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

    class Node:
        def __init__(self):
            self.Q = 0 #Reward
            self.P = 0 #Policy
            self.N = 0 #Visits
            self.Player = None

    def search(self,game):
        parent_hash = game.EPD_hash()
        if parent_hash not in self.tree:
            self.tree[parent_hash] = self.Node()
        #print(numpy.array(game.board))
        #print(game.EPD_hash())
        b_action = (None,None)
        b_upper = float('-inf')
        for cur,moves in game.possible_board_moves(capture=True).items():
            if len(moves) > 0 and ((cur[0].isupper() and game.p_move == 1) or (cur[0].islower() and game.p_move == -1)):
                for next in moves:
                    imag_game = deepcopy(game)
                    if imag_game.move(cur,f'{game.x[next[0]]}{game.y[next[1]]}') == True:
                        imag_game.p_move = imag_game.p_move * (-1)
                        hash = imag_game.EPD_hash()
                        state = imag_game.is_end()
                        if hash not in self.tree and sum(state) == 0:
                            self.tree[hash] = self.Node()
                            self.tree[hash].Q = 1 #Use NN output value
                            self.tree[hash].P = 1 #Use NN output policy
                            return self.tree[hash].Q, self.tree[hash].P
                        elif sum(state) > 0:
                            if hash not in self.tree:
                                self.tree[hash] = self.Node()
                            if (state == [1,0,0] and self.Player == 1) or (state == [0,0,1] and self.Player == -1):
                                print('WIN',state)
                                self.tree[hash].Q = 3 #Win
                            elif state == [0,0,0]:
                                print('TIE')
                                self.tree[hash].Q = 1 #Tie
                            else:
                                print('LOSS')
                                self.tree[hash].Q = 0 #Loss
                            self.tree[hash].P = 1
                            return self.tree[hash].Q, self.tree[hash].P
                        else:
                            self.tree[hash].Q, self.tree[hash].P = self.search(imag_game)
                        u = self.tree[hash].Q + self.Cpuct * self.tree[hash].P * math.sqrt(self.tree[parent_hash].N)/(1+self.tree[hash].N)
                        if u > b_upper:
                            b_action = (self.tree[hash].Q, self.tree[hash].P)
                            b_upper = u
        self.tree[parent_hash].N += 1
        #print(b_action)
        return b_action
