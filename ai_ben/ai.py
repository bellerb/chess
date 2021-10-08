import os
import math
import json
import numpy
import torch
import random
import pandas as pd
from copy import deepcopy
from ai_ben.model import TransformerModel

"""
Class used to create game playing AI
"""
class Agent:
    """
    Input: max_depth - integer representing the max depth in the graph the angent is alowed (Default=5) [OPTIONAL]
           search_amount - integer representing the amount of searches you want to run (Default=50) [OPTIONAL]
    Description: AI initail variables
    Output: None
    """
    def __init__(self,max_depth=5,search_amount=50,train=False):
        self.log = [] #Log of moves used in training
        self.train = train #Control for if the agent is being trained
        self.search_amount = search_amount #Amount of searches run when choosing move
        self.MCTS = deepcopy(MCTS(max_depth=max_depth,train=train)) #MCTS instance for the agent

    """
    Input: game - object containing the game current state
    Description: Main entrance point for AI to make moves from [This is the function called when playing games]
    Output: tuple of strings representing the curent and next moves for the AI to make
    """
    def choose_action(self,game):
        self.MCTS.Player = game.p_move
        for n in self.MCTS.tree:
            self.MCTS.tree[n].max_depth = False
        parent_hash = game.EPD_hash()
        #Search game actions
        for x in range(self.search_amount):
            self.MCTS.depth = 0
            self.MCTS.search(game)
            if self.train == True:
                if sum(self.MCTS.state) > 0:
                    game_train_data = pd.DataFrame(self.MCTS.log)
                    for i,x in enumerate(self.MCTS.state):
                        game_train_data[f'value{i}'] = [x]*len(self.MCTS.log)
                        game_train_data[f'value{i}'] = game_train_data[f'value{i}'].astype(float)
                    game_train_data = game_train_data.to_dict('records')
                    if len(game_train_data) > 0:
                        self.log += game_train_data
                    self.MCTS.state = [0,0,0] #Reset search state
                self.MCTS.log = [] #Reset search log
        #Check actions for best move
        u_bank = {}
        for c,moves in game.possible_board_moves(capture=True).items():
            if len(moves) > 0 and ((c[0].isupper() and game.p_move == 1) or (c[0].islower() and game.p_move == -1)):
                for n in moves:
                    imag_game = deepcopy(game)
                    if imag_game.move(c,f'{game.x[n[0]]}{game.y[n[1]]}') == True:
                        imag_game.p_move = imag_game.p_move * (-1)
                        hash = imag_game.EPD_hash()
                        if hash in self.MCTS.tree:
                            if self.MCTS.tree[hash].leaf == True and self.MCTS.tree[hash].Q == 6:
                                return c,f'{game.x[n[0]]}{game.y[n[1]]}'
                            else:
                                #print(f'{c}-{game.x[n[0]]}{game.y[n[1]]}',self.MCTS.tree[hash].Q,self.MCTS.tree[hash].P,math.log(self.MCTS.tree[parent_hash].N),1+self.MCTS.tree[hash].N,math.sqrt(math.log(self.MCTS.tree[parent_hash].N)/(1+self.MCTS.tree[hash].N)),self.MCTS.tree[hash].Q + (self.MCTS.Cpuct * self.MCTS.tree[hash].P * (math.sqrt(math.log(self.MCTS.tree[parent_hash].N)/(1+self.MCTS.tree[hash].N)))))
                                u_bank[f'{c}-{game.x[n[0]]}{game.y[n[1]]}'] = self.MCTS.tree[hash].Q + (self.MCTS.Cpuct * self.MCTS.tree[hash].P * (math.sqrt(math.log(self.MCTS.tree[parent_hash].N)/(1+self.MCTS.tree[hash].N))))
        m_bank = [k for k,v in u_bank.items() if v == max(u_bank.values())]
        if len(m_bank) > 0:
            cur,next = random.choice(m_bank).split('-')
        else:
            cur,next = ''
        return cur,next

"""
Monte Carlo Tree Search algorithm used to search game tree for the best move
"""
class MCTS:
    """
    Input: max_depth - integer representing the max depth in the graph the angent is alowed (Default=5) [OPTIONAL]
           folder - string representing the location of the folder storing the model parameters (Default='ai_ben/data') [OPTIONAL]
           filename - string representing the model name (Default='model.pth.tar') [OPTIONAL]
    Description: MCTS initail variables
    Output: None
    """
    def __init__(self,max_depth=5,train=False,folder='ai_ben/data',filename = 'model.pth.tar'):
        self.train = train #Control for if in training mode
        self.tree = {} #Game tree
        self.Cpuct = 0.77 #Exploration hyper parameter [0-1]
        self.Player = None #What player is searching
        self.depth = 0 #Curent node depth
        self.max_depth = max_depth #Max allowable depth
        self.log = [] #Each search log
        self.state = [0,0,0] #Used to know if leaf node was found and who won
        self.notation = {1:'p',2:'n',3:'b',4:'r',5:'q',6:'k'} #Map of notation to part number
        self.token_bank = pd.read_csv(f'{folder}/token_bank.csv') #All tokens
        #Model Parameters
        with open(os.path.join(folder,'model_param.json')) as f:
            m_param = json.load(f)
        sinp = m_param['input_size'] #Size of input layer 8x8 board
        ntokens = m_param['ntokens'] #The size of vocabulary
        emsize = m_param['emsize'] #Embedding dimension
        nhid = m_param['nhid'] #The dimension of the feedforward network model in nn.TransformerEncoder
        nlayers = m_param['nlayers'] #The number of nn.TransformerEncoderLayer in nn.TransformerEncoder
        nhead = m_param['nhead'] #The number of heads in the multiheadattention models
        dropout = m_param['dropout'] #The dropout value
        self.Device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #Set divice training will use
        self.Model = TransformerModel(sinp, ntokens, emsize, nhead, nhid, nlayers, dropout).to(self.Device) #Initialize the transformer model
        #Load Saved Model
        filepath = os.path.join(folder, filename)
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=self.Device)
            self.Model.load_state_dict(checkpoint['state_dict'])

    """
    Node for each state in the game tree
    """
    class Node:
        """
        Input: None
        Description: MCTS initail variables
        Output: None
        """
        def __init__(self):
            self.Q = 0 #Reward
            self.P = 0 #Policy
            self.N = 0 #Visits
            self.leaf = False #Leaf control
            self.max_depth = False #Max depth control

    """
    Input: game - object containing the game current state
    Description: Search the game tree using upper confidence value
    Output: tuple of integers representing node reward and policy values
    """
    def search(self,game):
        self.depth += 1
        parent_hash = game.EPD_hash()
        if parent_hash not in self.tree:
            self.tree[parent_hash] = self.Node()
        self.tree[parent_hash].N += 1
        state = game.check_state(parent_hash)
        if state == '50M':
            state = [0,1,0] #Auto tie
        elif state == '3F':
            state = [0,1,0] #Auto tie
        elif state == 'PP':
            game.pawn_promotion(n_part='Q') #Auto queen
        if state != [0,1,0]:
            state = game.is_end()
        if sum(state) > 0:
            #End state found [leaf node]
            if self.train == True:
                self.state = state
            self.tree[parent_hash].leaf = True
            if (state == [1,0,0] and self.Player == 1) or (state == [0,0,1] and self.Player == -1):
                self.tree[parent_hash].Q = 6 #Win
                #print(f'FOUND WIN {self.Player}')
            elif state == [0,1,0]:
                self.tree[parent_hash].Q = 1 #Tie
                #print(f'FOUND TIE {self.Player}')
            else:
                self.tree[parent_hash].Q = -6 #Loss
                #print(f'FOUND LOSS {self.Player}')
            return self.tree[parent_hash].Q, self.tree[parent_hash].P
        if self.tree[parent_hash].Q == 0:
            #Use NN
            enc_state = self.encode_state(game)
            v,p = self.Model(enc_state)
            state[torch.argmax(v).item()] = 1
            if (state == [1,0,0] and self.Player == 1) or (state == [0,0,1] and self.Player == -1):
                self.tree[parent_hash].Q = 3 #Win
            elif state == [0,1,0]:
                self.tree[parent_hash].Q = 1 #Tie
            else:
                self.tree[parent_hash].Q = -3 #Loss
            p = p.reshape(64,8,8)
            for cur,moves in game.possible_board_moves(capture=True).items():
                if len(moves) > 0 and ((cur[0].isupper() and game.p_move == 1) or (cur[0].islower() and game.p_move == -1)):
                    for next in moves:
                        imag_game = deepcopy(game)
                        if imag_game.move(cur,f'{game.x[next[0]]}{game.y[next[1]]}') == True:
                            imag_game.p_move = imag_game.p_move * (-1)
                            hash = imag_game.EPD_hash()
                            if hash not in self.tree:
                                self.tree[hash] = self.Node()
                            cur_pos = game.board_2_array(cur)
                            self.tree[hash].P = p[cur_pos[0]+(cur_pos[1]*8)][next[1]][next[0]].item()
            return self.tree[parent_hash].Q, self.tree[parent_hash].P
        else:
            if self.depth == self.max_depth:
                return self.tree[parent_hash].Q, self.tree[parent_hash].P
            b_cur = None
            b_next = None
            b_action = None
            w_check = False
            b_upper = float('-inf')
            for cur,moves in game.possible_board_moves(capture=True).items():
                if len(moves) > 0 and ((cur[0].isupper() and game.p_move == 1) or (cur[0].islower() and game.p_move == -1)):
                    for next in moves:
                        imag_game = deepcopy(game)
                        if imag_game.move(cur,f'{game.x[next[0]]}{game.y[next[1]]}') == True:
                            state = imag_game.check_state(parent_hash)
                            if state == '50M':
                                state = [0,1,0] #Auto tie
                            elif state == '3F':
                                state = [0,1,0] #Auto tie
                            elif state == 'PP':
                                imag_game.pawn_promotion(n_part='Q') #Auto queen
                            if state != [0,1,0]:
                                state = imag_game.is_end()
                            if (state == [1,0,0] and imag_game.p_move == 1) or (state == [0,0,1] and imag_game.p_move == -1):
                                imag_game.p_move = imag_game.p_move * (-1)
                                b_action = deepcopy(imag_game)
                                w_check = True
                                break
                            imag_game.p_move = imag_game.p_move * (-1)
                            hash = imag_game.EPD_hash()
                            if hash in self.tree and self.tree[hash].max_depth == False:
                                #print(next,self.tree[hash].Q, self.Cpuct, self.tree[hash].P, math.sqrt(math.log(1+self.tree[parent_hash].N)/(self.tree[hash].N)))
                                u = self.tree[hash].Q + (self.Cpuct * self.tree[hash].P * (math.sqrt(math.log(self.tree[parent_hash].N)/(1+self.tree[hash].N))))
                                if u > b_upper:
                                    b_action = deepcopy(imag_game)
                                    b_cur = deepcopy(game.board_2_array(cur))
                                    b_next = deepcopy(next)
                                    b_upper = u
                    if w_check == True:
                        break
            if b_action != None and b_cur != None and b_next != None:
                #print('SEARCH')
                if self.train == True:
                    self.log.append({**{f'state{i}':float(s) for i,s in enumerate(self.encode_state(b_action)[0])},
                                     **{f'action{x}':1 if x == ((b_cur[0]+(b_cur[1]*8))*64)+(b_next[0]+(b_next[1]*8)) else 0 for x in range(4096)}})
                v,p = self.search(b_action)
                hash = b_action.EPD_hash()
                if hash in self.tree:
                    if self.depth == self.max_depth:
                        self.tree[hash].max_depth = True
                    self.tree[hash].Q = v
                    self.tree[hash].P = p
                    return self.tree[hash].Q,self.tree[hash].P
            return self.tree[parent_hash].Q, self.tree[parent_hash].P

    """
    Input: game - object containing the game current state
    Description: encode the game board as tokens for the NN
    Output: list containing integers representing a tokenized game board
    """
    def encode_state(self,game):
        temp_board = deepcopy(game.board)
        for y,row in enumerate(temp_board):
            for x,peice in enumerate(row):
                if peice != 0:
                    temp_board[y][x] = f'{self.notation[abs(peice)]}w' if peice > 0 else f'{self.notation[abs(peice)]}b'
                else:
                    temp_board[y][x] = 'PAD'
        if len(temp_board) > 0:
            flat = [x for y in temp_board for x in y]
            result = [self.token_bank['token'].eq(t).idxmax() for t in flat]
            result.insert(0,1) if game.p_move == 1 else result.insert(0,2)
        else:
            result = []
        return torch.tensor([result])
