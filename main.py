from chess import Chess
from ai_ben.ai import Agent as w_agent
from ai_random.ai import Agent as b_agent

print('''****************************
  Welcome to Console Chess
****************************
White = Upper Case
Black = Lower Case
P,p = Pawn
N,n = Knight
B,b = Bishop
R,r = Rook
Q,q = Queen
K,k = King
When asked where you want to moves please use the following cordinate system:
a8 b8 c8 d8 e8 f8 g8 h8
a7 b7 c7 d7 e7 f7 g7 h7
a6 b6 c6 d6 e6 f6 g6 h6
a5 b5 c5 d5 e5 f5 g5 h5
a4 b4 c4 d4 e4 f4 g4 h4
a3 b3 c3 d3 e3 f3 g3 h3
a2 b2 c2 d2 e2 f2 g2 h2
a1 b1 c1 d1 e1 f1 g1 h1''')

white = 'ai' #Values ['human','ai']
black = 'ai' #Values ['human','ai']
chess_game = Chess()

p_type = [0,0]
if white == 'ai':
    p_type[0] = 1
    #w_bot = w_agent(max_depth=100) #Initailize white bot
    w_bot = b_agent() #Initailize black bot
else:
    p_type[0] = 0
if black == 'ai':
    p_type[1] = 1
    #b_bot = b_agent() #Initailize black bot
    b_bot = w_agent(max_depth=50) #Initailize white bot
else:
    p_type[1] = 0

while True:
    if chess_game.p_move == 1:
        print('\nWhites Turn [UPPER CASE]\n')
    else:
        print('\nBlacks Turn [LOWER CASE]\n')
    chess_game.display()
    if (chess_game.p_move == 1 and p_type[0] == 0) or (chess_game.p_move == -1 and p_type[1] == 0):
        cur = input('What piece do you want to move?\n')
        next = input('Where do you want to move the piece to?\n')
    else:
        if chess_game.p_move == 1:
            cur,next = w_bot.choose_action(chess_game)
        else:
            cur,next = b_bot.choose_action(chess_game)

        print('What piece do you want to move?\n')
        print(cur.lower())
        print('\nWhere do you want to move the piece to?\n')
        print(next.lower())
    valid = False
    if chess_game.move(cur,next) == False:
        print('Invalid move')
    else:
        valid = True
    if chess_game.check_state(chess_game.EPD_hash()) == 'PP':
        if (chess_game.p_move == 1 and p_type[0] == 1) or (chess_game.p_move == -1 and p_type[1] == 1):
            chess_game.pawn_promotion(n_part='Q') #Auto queen
        else:
            chess_game.pawn_promotion() #Pawn promotion found
    state = chess_game.is_end()
    if sum(state) > 0:
        print('\n*********************\n      GAME OVER\n*********************\n')
        chess_game.display()
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
        break
    if valid == True:
        chess_game.p_move = chess_game.p_move * (-1)
