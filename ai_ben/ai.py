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
        if self.MCTS.Player == None:
            self.MCTS.Player = game.p_move
        parent_hash = game.EPD_hash()
        if parent_hash not in self.MCTS.tree:
            self.MCTS.tree[parent_hash] = self.MCTS.Node()
        #for x in range(3):
        while True:
            self.MCTS.search(game,count=0)
            u_bank = {}
            for c,moves in game.possible_board_moves(capture=True).items():
                if len(moves) > 0 and ((c[0].isupper() and game.p_move == 1) or (c[0].islower() and game.p_move == -1)):
                    for n in moves:
                        imag_game = deepcopy(game)
                        if imag_game.move(c,f'{game.x[n[0]]}{game.y[n[1]]}') == True:
                            imag_game.p_move = imag_game.p_move * (-1)
                            hash = imag_game.EPD_hash()
                            if hash in self.MCTS.tree:
                                u_bank[f'{c}-{game.x[n[0]]}{game.y[n[1]]}'] = self.MCTS.tree[hash].Q + self.MCTS.Cpuct * self.MCTS.tree[hash].P * math.sqrt(self.MCTS.tree[parent_hash].N)/(1+self.MCTS.tree[hash].N)
            m_bank = [k for k,v in u_bank.items() if v == max(u_bank.values())]
            if len(m_bank) > 0:
                cur,next = random.choice(m_bank).split('-')
                break
            else:
                print('SMALL')
        return cur,next

    def encode_state(self,game):
        temp_board = deepcopy(game.board)
        #print(temp_board)
        for y,row in enumerate(temp_board):
            for x,peice in enumerate(row):
                if peice != 0:
                    temp_board[y][x] = f'{self.notation[abs(peice)]}w' if peice > 0 else f'{self.notation[abs(peice)]}b'
                else:
                    temp_board[y][x] = 'PAD'
        if len(temp_board) > 0:
            flat = [x for y in temp_board for x in y]
            result = [self.token_bank['token'].eq(t).idxmax() for t in flat]
        else:
            result = []
        return result

class MCTS:
    def __init__(self,agent):
        self.tree = {}
        self.Cpuct = 2
        self.Agent = agent
        self.Player = None

        sinp = 64 #Size of input layer
        ntokens = len(self.Agent.token_bank) #The size of vocabulary
        emsize = 200 #Embedding dimension
        nhid = 200 #The dimension of the feedforward network model in nn.TransformerEncoder
        nlayers = 2 #The number of nn.TransformerEncoderLayer in nn.TransformerEncoder
        nhead = 2 #The number of heads in the multiheadattention models
        dropout = 0.2 #The dropout value
        self.Device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #Set divice training will use
        self.Model = TransformerModel(sinp, ntokens, emsize, nhead, nhid, nlayers, dropout).to(self.Device) #Initialize the transformer model

    class Node:
        def __init__(self):
            self.Q = 0 #Reward
            self.P = 0 #Policy
            self.N = 0 #Visits

    def search(self,game,lim=10,count=0,):
        parent_hash = game.EPD_hash()
        if parent_hash not in self.tree:
            self.tree[parent_hash] = self.Node()
        else:
            self.tree[parent_hash].N += 1
        b_action = (None,None)
        b_upper = float('-inf')
        for cur,moves in game.possible_board_moves(capture=True).items():
            if len(moves) > 0 and ((cur[0].isupper() and game.p_move == 1) or (cur[0].islower() and game.p_move == -1)):
                for next in moves:
                    imag_game = deepcopy(game)
                    if imag_game.move(cur,f'{game.x[next[0]]}{game.y[next[1]]}') == True:
                        if count >= lim:
                            break
                        hash = imag_game.EPD_hash()
                        state = imag_game.check_state(hash)
                        if state == '50M':
                            state = [0,1,0] #Auto tie
                        elif state == '3F':
                            state = [0,1,0] #Auto tie
                        elif state == 'PP':
                            imag_game.pawn_promotion(n_part='Q') #Auto queen
                        imag_game.p_move = imag_game.p_move * (-1)
                        hash = imag_game.EPD_hash()
                        if state != [0,1,0]:
                            state = imag_game.is_end()
                        if hash not in self.tree and sum(state) == 0:
                            #Use NN
                            self.tree[hash] = self.Node()
                            enc_state = self.Agent.encode_state(imag_game)
                            v,p = self.Model(torch.tensor([enc_state]))
                            state[torch.argmax(v).item()] = 1
                            if (state == [1,0,0] and self.Player == 1) or (state == [0,0,1] and self.Player == -1):
                                self.tree[hash].Q = 3 #Win
                                self.tree[hash].P = p.item()
                                return self.tree[hash].Q, self.tree[hash].P
                            elif state == [0,0,0]:
                                self.tree[hash].Q = 1 #Tie
                            else:
                                self.tree[hash].Q = -3 #Loss
                            self.tree[hash].P = p.item()
                            #return self.tree[hash].Q, self.tree[hash].P
                        elif sum(state) > 0:
                            #End state found [leaf node]
                            if hash not in self.tree:
                                self.tree[hash] = self.Node()
                            if (state == [1,0,0] and self.Player == 1) or (state == [0,0,1] and self.Player == -1):
                                self.tree[hash].Q = 3 #Win
                            elif state == [0,0,0]:
                                self.tree[hash].Q = 1 #Tie
                            else:
                                self.tree[hash].Q = -3 #Loss
                            self.tree[hash].P = 1
                            #return self.tree[hash].Q, self.tree[hash].P
                        else:
                            v,p = self.search(imag_game,count=count)
                            print(v,p)
                            self.tree[hash].Q = v
                            self.tree[hash].P = p
                        u = self.tree[hash].Q + self.Cpuct * self.tree[hash].P * math.sqrt(self.tree[parent_hash].N)/(1+self.tree[hash].N)
                        if u > b_upper:
                            b_action = (self.tree[hash].Q, self.tree[hash].P)
                            b_upper = u
                        count += 1
                    if count >= lim:
                        break
        return b_action