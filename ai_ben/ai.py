  
import math
import numpy
import torch
import random
import pandas as pd
from copy import deepcopy
from ai_ben.model import TransformerModel

class Agent:
    def __init__(self):
        self.notation = {1:'p',2:'n',3:'b',4:'r',5:'q',6:'k'} #Map of notation to part number
        self.token_bank = pd.read_csv('ai_ben/token_bank.csv') #All tokens
        self.MCTS = MCTS(self)

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
                            u_bank[f'{c}-{game.x[n[0]]}{game.y[n[1]]}'] = self.MCTS.tree[hash].Q + self.MCTS.Cpuct * self.MCTS.tree[hash].P * math.sqrt(self.MCTS.tree[parent_hash].N)/(1+self.MCTS.tree[hash].N)
        m_bank = [k for k,v in u_bank.items() if v == max(u_bank.values())]
        cur,next = random.choice(m_bank).split('-')
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

    def encode_valid_action(self,x,y,game,cur_pos,move):
        c_cord = game.board_2_array(cur_pos)
        c_peice = game.board[c_cord[1]][c_cord[0]]
        temp_board = numpy.array([['MASK']*x for i in range(y)])
        temp_board[move[1]][move[0]] = f'{self.notation[abs(c_peice)]}wm' if c_peice > 0 else f'{self.notation[abs(c_peice)]}bm'
        return temp_board



class MCTS:
    def __init__(self,agent):
        self.tree = {}
        self.Cpuct = 2
        self.Agent = agent

        ntokens = len(self.Agent.token_bank) #The size of vocabulary
        emsize = 200 #Embedding dimension
        nhid = 200 #The dimension of the feedforward network model in nn.TransformerEncoder
        nlayers = 2 #The number of nn.TransformerEncoderLayer in nn.TransformerEncoder
        nhead = 2 #The number of heads in the multiheadattention models
        dropout = 0.2 #The dropout value
        self.Device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #Set divice training will use
        self.Model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(self.Device) #Initialize the transformer model

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
        enc_state = self.Agent.encode_state(deepcopy(game))
        b_action = (None,None)
        b_upper = float('-inf')
        for cur,moves in game.possible_board_moves(capture=True).items():
            if len(moves) > 0 and ((cur[0].isupper() and game.p_move == 1) or (cur[0].islower() and game.p_move == -1)):
                for next in moves:
                    imag_game = deepcopy(game)
                    if imag_game.move(cur,f'{game.x[next[0]]}{game.y[next[1]]}') == True:
                        hash = imag_game.EPD_hash()
                        state = imag_game.check_state(hash)
                        if state == '50M':
                            state = [0,1,0] #Auto tie
                        elif state == '3F':
                            state = [0,1,0] #Auto tie
                        elif state == 'PP':
                            imag_game.pawn_promotion(n_part='Q') #Auto queen
                            hash = imag_game.EPD_hash()
                        if state != [0,1,0]:
                            state = imag_game.is_end()
                        imag_game.p_move = imag_game.p_move * (-1)
                        if hash not in self.tree and sum(state) == 0:
                            self.tree[hash] = self.Node()
                            self.tree[hash].Q = 1 #Use NN output value
                            self.tree[hash].P = 1 #Use NN output policy

                            print(cur)
                            action = self.Agent.encode_valid_action(8,8,game,cur,next)
                            model_input = [x for y in enc_state for x in y] + ['SEP'] + [x for y in action for x in y] #Flatened input
                            print(model_input)
                            model_input = [self.Agent.token_bank['token'].eq(t).idxmax() for t in model_input] #Set tokens to numeric representation
                            print(model_input)
                            print(len(model_input))
                            pred = self.Model.predict(torch.tensor([model_input]),self.Device)
                            print(pred.size())
                            #print(pred)
                            print(self.greedy_prob_2_index(pred))
                            quit()

                            return self.tree[hash].Q, self.tree[hash].P
                        elif sum(state) > 0:
                            if hash not in self.tree:
                                self.tree[hash] = self.Node()
                            if (state == [1,0,0] and self.Player == 1) or (state == [0,0,1] and self.Player == -1):
                                #print('WIN',state)
                                self.tree[hash].Q = 3 #Win
                            elif state == [0,0,0]:
                                p#rint('TIE')
                                self.tree[hash].Q = 1 #Tie
                            else:
                                #print('LOSS')
                                self.tree[hash].Q = -3 #Loss
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

    def greedy_prob_2_index(self,data):
        #print(torch.argmax(data,dim=0).reshape(1))
        result = []
        '''
        for y in data:
            if len(y) > 0:
                result.append([torch.argmax(x) for x in y if len(x.size()) > 0])
        '''
        return torch.argmax(data,dim=0)
