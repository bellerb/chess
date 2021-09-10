import pandas as pd
from chess import Chess
from ai_ben.ai import Agent
from copy import deepcopy

white = 'ai' #Values ['human','ai']
black = 'ai' #Values ['human','ai']
chess_game = Chess()
w_bot = Agent()
b_bot = Agent()

log = []
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
        #log.append({'state':chess_game.EPD_hash()})
        #print(w_bot.encode_state(chess_game))
        log.append({'state':Agent().encode_state(chess_game),'hash':chess_game.EPD_hash()})
    if chess_game.check_state(chess_game.EPD_hash()) == 'PP':
        chess_game.pawn_promotion(n_part='Q') #Auto queen
    state = chess_game.is_end()
    if sum(state) > 0:
        print('\n*********************\n      GAME OVER\n*********************\n')
        #chess_game.display()
        print('Game Log:\n---------\n')
        print(f'INITIAL POSITION = {chess_game.init_pos}')
        print(f'MOVES = {chess_game.log}')
        print('\nGame Result:\n------------\n')
        if state == [0,0,1]:
            print('BLACK WINS\n')
        elif state == [1,0,0]:
            print('WHITE WINS\n')
        else:
            print('TIE GAME\n')
        log = pd.DataFrame(log)
        log['value'] = [state]*len(log)
        print(log)
        break
    if valid == True:
        chess_game.p_move = chess_game.p_move * (-1)
