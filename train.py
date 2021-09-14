import os
import time
import math
import torch
import random
import pandas as pd
from chess import Chess
from ai_ben.ai import Agent
from ai_ben.model import TransformerModel

GAMES = 500

folder = 'ai_ben/data'
filename = 'model.pth.tar'

sinp = 64 #Size of input layer 8x8 board
ntokens = 33 #The size of vocabulary
emsize = 200 #Embedding dimension
nhid = 200 #The dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 2 #The number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 2 #The number of heads in the multiheadattention models
dropout = 0.2 #The dropout value
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #Set divice training will use

game_results = {'black':0,'white':0,'tie':0}
for epoch in range(GAMES):
    log = []
    #w_bot = Agent(max_depth=random.choice([100,200,500,800,1000])) #Current AI
    #b_bot = Agent(max_depth=random.choice([100,200,500,800,1000])) #Current AI
    w_bot = Agent(max_depth=1000) #Current AI
    b_bot = Agent(max_depth=1000) #Current AI
    chess_game = Chess()
    while True:
        #chess_game.display()
        if chess_game.p_move == 1:
            cur,next = w_bot.choose_action(chess_game)
        else:
            cur,next = b_bot.choose_action(chess_game)
        print(f'w {cur.lower()}-->{next.lower()}') if chess_game.p_move > 0 else print(f'b {cur.lower()}-->{next.lower()}')
        valid = False
        if chess_game.move(cur,next) == False:
            print('Invalid move')
        else:
            valid = True
            log.append({f'state{i}':float(s) for i,s in enumerate(Agent().encode_state(chess_game))})
        if chess_game.check_state(chess_game.EPD_hash()) == 'PP':
            chess_game.pawn_promotion(n_part='Q') #Auto queen
        state = chess_game.is_end()
        if sum(state) > 0:
            print('\n*********************\n      GAME OVER\n*********************\n')
            print('\nGame Result:\n------------\n')
            if state == [0,0,1]:
                print('BLACK WINS\n')
                game_results['black'] += 1
            elif state == [1,0,0]:
                print('WHITE WINS\n')
                game_results['white'] += 1
            else:
                print('TIE GAME\n')
                game_results['tie'] += 1
            train_data = pd.DataFrame(log)
            for i,x in enumerate(state):
                train_data[f'value{i}'] = [x]*len(log)
                train_data[f'value{i}'] = train_data[f'value{i}'].astype(float)
            train_data['probability'] = [1]*len(log)
            train_data['probability'] = train_data['probability'].astype(float)
            model = TransformerModel(sinp, ntokens, emsize, nhead, nhid, nlayers, dropout).to(device) #Initialize the transformer model

            filepath = os.path.join(folder, filename)
            if os.path.exists(filepath):
                checkpoint = torch.load(filepath, map_location=device)
                model.load_state_dict(checkpoint['state_dict'])

            lr = 5.0 #Learning rate
            criterion = torch.nn.BCELoss()
            #criterion = torch.nn.NLLLoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=lr) #Optimization algorithm using stochastic gradient descent
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95) #Adjust the learn rate through epochs

            model.train() #Turn on the train mode
            total_loss = 0. #Initalize total loss
            start_time = time.time() #Get time of starting process
            train_data = torch.tensor(train_data.values)
            bsz = 10
            for batch, i in enumerate(range(0, train_data.size(0) - 1, bsz)):
                data, v_targets, p_targets = TransformerModel.get_batch(train_data,i,bsz) #Get batch data with the selected targets being masked
                output = model(data) #Make prediction using the model
                v_loss = criterion(output[0], v_targets) #Apply loss function to results
                #print(output[0], v_targets, v_loss)
                p_loss = criterion(output[1], p_targets) #Apply loss function to results
                #print(output[1], p_targets, p_loss)
                loss = v_loss + p_loss
                #print(loss.item(),v_loss.item(),p_loss.item())
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
            print(epoch,game_results)
            break
        if valid == True:
            chess_game.p_move = chess_game.p_move * (-1)


cur_loss = total_loss
elapsed = time.time() - start_time
print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | loss {:5.2f}'.format(
            epoch,
            batch,
            len(train_data),
            scheduler.get_lr()[0],
            elapsed * 1000,
            cur_loss
        )
     )
