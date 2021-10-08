import os
import time
import math
import json
import torch
import random
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor,as_completed

import sys
sys.path.insert(0,os.getcwd())

from chess import Chess
from ai_ben.ai import Agent,MCTS
from ai_ben.model import TransformerModel

def play_game(game_name):
    bot = deepcopy(Agent(search_amount=random.choice([50,75,100]),max_depth=random.choice([5,6,7,8,9,10])))
    log = []
    chess_game = deepcopy(Chess())
    while True:
        cur,next = bot.choose_action(chess_game)
        #cur = input('What piece do you want to move?\n')
        #next = input('Where do you want to move the piece to?\n')
        valid = False
        if chess_game.move(cur,next) == False:
            print('Invalid move')
        else:
            valid = True
            cur_pos = chess_game.board_2_array(cur)
            next_pos = chess_game.board_2_array(next)
            log.append({**{f'state{i}':float(s) for i,s in enumerate(MCTS().encode_state(chess_game)[0])},
                        **{f'action{x}':1 if x == ((cur_pos[0]+(cur_pos[1]*8))*64)+(next_pos[0]+(next_pos[1]*8)) else 0 for x in range(4096)}})
        print(f'GAME-{game_name} MOVE:{len(log)} w {cur.lower()}-->{next.lower()}') if chess_game.p_move > 0 else print(f'GAME-{game_name} MOVE:{len(log)} b {cur.lower()}-->{next.lower()}')
        chess_game.display()
        if chess_game.check_state(chess_game.EPD_hash()) == 'PP':
            chess_game.pawn_promotion(n_part='Q') #Auto queen
        state = chess_game.is_end()
        if sum(state) > 0:
            print(f'GAME-{game_name}-FINISHED MOVE:{len(log)} STATE:{state}')
            game_train_data = pd.DataFrame(log)
            for i,x in enumerate(state):
                game_train_data[f'value{i}'] = [x]*len(log)
                game_train_data[f'value{i}'] = game_train_data[f'value{i}'].astype(float)
            break
        if valid == True:
            chess_game.p_move = chess_game.p_move * (-1)
    return state,game_train_data

def multi_process(func,workers=None):
    data = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        future_func = {ex.submit(f['func'],f['args']):f['name'] for f in func}
        for future in as_completed(future_func):
            data[future_func[future]] = future.result()
        return data

if __name__ == '__main__':
    GAMES = 1
    THREADS = 3

    folder = 'ai_ben/data'
    filename = 'model.pth.tar'

    #Model parameters
    with open(os.path.join(folder,'model_param.json')) as f:
        m_param = json.load(f)
    sinp = m_param['input_size'] #Size of input layer 8x8 board
    ntokens = m_param['ntokens'] #The size of vocabulary
    emsize = m_param['emsize'] #Embedding dimension
    nhid = m_param['nhid'] #The dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = m_param['nlayers'] #The number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = m_param['nhead'] #The number of heads in the multiheadattention models
    dropout = m_param['dropout'] #The dropout value
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #Set divice training will use

    #Training parameters
    bsz = 20 #Batch size
    lr = 0.0005 #Learning rate
    total_loss = 0.0 #Initalize total loss

    train_data = pd.DataFrame()
    game_results = {'black':0,'white':0,'tie':0}
    for epoch in range(GAMES):
        print('STARTING GAMES')
        func = [{'name':f'game-{x}','func':play_game,'args':(x)} for x in range(THREADS)]
        games = multi_process(func,workers=THREADS)
        print(games)
        for g in games:
            state,game_train_data = games[g]
            if state == [0,0,1]:
                print('BLACK WINS\n')
                game_results['black'] += 1
            elif state == [1,0,0]:
                print('WHITE WINS\n')
                game_results['white'] += 1
            else:
                print('TIE GAME\n')
                game_results['tie'] += 1
            print(game_train_data)
            train_data = train_data.append(game_train_data,ignore_index=True)
        if (epoch * THREADS) % 2 == 0 or epoch == GAMES-1:
            train_data = train_data.sample(frac=1).reset_index(drop=True)
            print(train_data)
            model = TransformerModel(sinp, ntokens, emsize, nhead, nhid, nlayers, dropout).to(device) #Initialize the transformer model
            filepath = os.path.join(folder, filename)
            if os.path.exists(filepath):
                checkpoint = torch.load(filepath, map_location=device)
                model.load_state_dict(checkpoint['state_dict'])

            criterion = torch.nn.BCELoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=lr) #Optimization algorithm using stochastic gradient descent
            model.train() #Turn on the train mode
            start_time = time.time() #Get time of starting process
            train_data = torch.tensor(train_data.values)
            for batch, i in enumerate(range(0, train_data.size(0) - 1, bsz)):
                data, v_targets, p_targets = TransformerModel.get_batch(train_data,i,bsz) #Get batch data with the selected targets being masked
                output = model(data) #Make prediction using the model
                v_loss = criterion(output[0], v_targets) #Apply loss function to results
                p_loss = criterion(output[1], p_targets) #Apply loss function to results
                loss = v_loss + p_loss
                loss.backward() #Backpropegate through model
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                total_loss += loss.item() #Increment total loss
            filepath = os.path.join(folder, filename)
            if not os.path.exists(folder):
                os.mkdir(folder)
            torch.save({
                'state_dict': model.state_dict(),
            }, filepath)
            train_data = pd.DataFrame()
        print(epoch,game_results)
