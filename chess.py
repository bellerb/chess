class Chess:
    def __init__(self):
        self.x = ['a','b','c','d','e','f','g','h'] #Board x representation
        self.y = ['8','7','6','5','4','3','2','1'] #Board y representation
        self.parts = {1:'Pawn',2:'Knight',3:'Bishop',4:'Rook',5:'Queen',6:'King'} #Map of number to part
        self.reset() #Reset game board and state

    def reset(self):
        self.log = [] #Game log
        self.p_move = 1 #Current players move white = 1 black = -1
        self.board = np.zeros((8,8)) #Generate empty chess board
        king = self.King()
        queen = self.Queen()
        rook = self.Rook()
        bishop = self.Bishop()
        knight = self.Knight()
        pawn = self.Pawn()
        self.board[0][0] = -rook.value
        self.board[0][7] = -rook.value
        self.board[7][0] = rook.value
        self.board[7][7] = rook.value
        self.board[0][1] = -knight.value
        self.board[0][6] = -knight.value
        self.board[7][1] = knight.value
        self.board[7][6] = knight.value
        self.board[0][2] = -bishop.value
        self.board[0][5] = -bishop.value
        self.board[7][2] = bishop.value
        self.board[7][5] = bishop.value
        self.board[0][3] = -queen.value
        self.board[7][3] = queen.value
        self.board[0][4] = -king.value
        self.board[7][4] = king.value
        for i in range(8):
            self.board[1][i] = -pawn.value
            self.board[6][i] = pawn.value

    def display(self):
        result = '  a b c d e f g h  \n  ----------------\n'
        for c,y in enumerate(self.board):
            result += f'{8-c}|'
            for x in y:
                if x != 0:
                    n = getattr(Chess,self.parts[int(x) if x > 0 else int(x)*(-1)])().notation.upper() if x < 0 else getattr(Chess,self.parts[int(x) if x > 0 else int(x)*(-1)])().notation.lower()
                    if n == '':
                        n = 'P' if x < 0 else 'p'
                    result += n
                else:
                    result += '.'
                result += ' '
            result += f'|{8-c}\n'
        result += '  ----------------\n  a b c d e f g h\n'
        print(result)

    def move(self,cur_pos,next_pos):
        cp = self.board_2_array(cur_pos)
        np = self.board_2_array(next_pos)
        if self.valid_move(cp,np) == True:
            part = self.board[cp[0]][cp[1]]
            self.log_move(part,cur_pos,next_pos,np)
            self.board[cp[0]][cp[1]] = 0
            self.board[np[0]][np[1]] = part
            self.p_move = self.p_move * (-1)
            return True
        return False

    def valid_move(self,cur_pos,next_pos):
        part = self.board[cur_pos[0]][cur_pos[1]]
        if part * self.p_move > 0 and part != 0:
            p_name = self.parts[int(part) if part > 0 else int(part)*(-1)] #Get name of part
            if next_pos in getattr(Chess,p_name).movement(self.board,self.p_move,cur_pos):
                return True
        return False

    def board_2_array(self,cord):
        cord = list(cord)
        if len(cord) == 2 and str(cord[0]).lower() in self.x and str(cord[1]) in self.y:
            return self.y.index(str(cord[1])), self.x.index(str(cord[0]).lower())
        else:
            return None

    def log_move(self,part,cur_cord,next_cord,next_pos):
        #castling with rook on h side o-o casteling with rook on a o-o-o
        #pawn promotion location=piece ex a8=Q
        #to remove ambiguity where multiple pieces could make the move add starting identifier after piece notation ex Rab8
        p_name = self.parts[int(part) if part > 0 else int(part)*(-1)] #Get name of part
        move = getattr(Chess,p_name)().notation #Get part notation
        if self.board[next_pos[0]][next_pos[1]] != 0: #Check if there is a capture
            move += 'x' if move != '' else str(cur_cord)[0] + 'x'
        move += next_cord
        if self.is_checkmate() == True:
            move += '#'
        elif self.is_check() == True:
            move += '+'
        self.log.append(move)

    def is_check(self):
        return False

    def is_checkmate(self):
        return False

    def is_dead_position(self):
        #King against King
        #King against King and bishop
        #King against king and knight
        #King and bishop against king and bishop with both bishops on squares of the same colour
        #No available moves
        return False

    def is_draw(self):
        #Fifty move rule
        #Five fold repetition rule
        if self.is_dead_position() == True:
            return True
        return False

    def is_end(self):
        moves = {}
        for y,row in enumerate(self.board):
            for x,part in enumerate(row):
                if part != 0:
                    p_name = self.parts[int(part) if part > 0 else int(part)*(-1)] #Get name of part
                    p_colour = 1 if part > 0 else -1
                    moves[f'{self.x[x]}{self.y[y]}'] = getattr(Chess,p_name).movement(self.board,p_colour,[x,y])
        #print(moves)
        print({k:[f'{self.x[x[0]]}{self.y[x[1]]}' for x in v] for k,v in moves.items()})
        return False

    class King:
        def __init__(self):
            self.value = 6 #Numerical value of piece
            self.notation = 'K' #Chess notation

        def movement(board,player,pos):
            result = []
            if pos[1]+1 >= 0 and pos[1]+1 <= 7 and pos[0] >= 0 and pos[0] <= 7 and (board[pos[1]+1][pos[0]]*player < 0 or board[pos[1]+1][pos[0]] == 0):
                result.append((pos[1]+1,pos[0]))
            if pos[1]-1 >= 0 and pos[1]-1 <= 7 and pos[0] >= 0 and pos[0] <= 7 and (board[pos[1]-1][pos[0]]*player < 0 or board[pos[1]-1][pos[0]] == 0):
                result.append((pos[1]-1,pos[0]))
            if pos[1] >= 0 and pos[1] <= 7 and pos[0]+1 >= 0 and pos[0]+1 <= 7 and (board[pos[1]][pos[0]+1]*player < 0 or board[pos[1]][pos[0]+1] == 0):
                result.append((pos[1],pos[0]+1))
            if pos[1] >= 0 and pos[1] <= 7 and pos[0]-1 >= 0 and pos[0]-1 <= 7 and (board[pos[1]][pos[0]-1]*player < 0 or board[pos[1]][pos[0]-1] == 0):
                result.append((pos[1],pos[0]-1))
            if pos[1]+1 >= 0 and pos[1]+1 <= 7 and pos[0]+1 >= 0 and pos[0]+1 <= 7 and (board[pos[1]+1][pos[0]+1]*player < 0 or board[pos[1]+1][pos[0]+1] == 0):
                result.append((pos[1]+1,pos[0]+1))
            if pos[1]+1 >= 0 and pos[1]+1 <= 7 and pos[0]-1 >= 0 and pos[0]-1 <= 7 and (board[pos[1]+1][pos[0]-1]*player < 0 or board[pos[1]+1][pos[0]-1] == 0):
                result.append((pos[1]+1,pos[0]-1))
            if pos[1]-1 >= 0 and pos[1]-1 <= 7 and pos[0]+1 >= 0 and pos[0]+1 <= 7 and (board[pos[1]-1][pos[0]+1]*player < 0 or board[pos[1]-1][pos[0]+1] == 0):
                result.append((pos[1]-1,pos[0]+1))
            if pos[1]-1 >= 0 and pos[1]-1 <= 7 and pos[0]-1 >= 0 and pos[0]-1 <= 7 and (board[pos[1]-1][pos[0]-1]*player < 0 or board[pos[1]-1][pos[0]-1] == 0):
                result.append((pos[1]-1,pos[0]-1))
            return result

    class Queen:
        def __init__(self):
            self.value = 5 #Numerical value of piece
            self.notation = 'Q' #Chess notation

        def movement(board,player,pos):
            result = []
            check = [True,True,True,True,True,True,True,True]
            for c in range(1,8,1):
                if pos[1]+c >= 0 and pos[1]+c <= 7 and pos[0] >= 0 and pos[0] <= 7 and (board[pos[1]+c][pos[0]]*player < 0 or board[pos[1]+c][pos[0]] == 0) and check[0] == True:
                    result.append((pos[1]+c,pos[0]))
                else:
                    check[0] = False
                if pos[1]-c >= 0 and pos[1]-c <= 7 and pos[0] >= 0 and pos[0] <= 7 and (board[pos[1]-c][pos[0]]*player < 0 or board[pos[1]-c][pos[0]] == 0) and check[1] == True:
                    result.append((pos[1]-c,pos[0]))
                else:
                    check[1] = False
                if pos[1] >= 0 and pos[1] <= 7 and pos[0]+c >= 0 and pos[0]+c <= 7 and (board[pos[1]][pos[0]+c]*player < 0 or board[pos[1]][pos[0]+c] == 0) and check[2] == True:
                    result.append((pos[1],pos[0]+c))
                else:
                    check[2] = False
                if pos[1] >= 0 and pos[1] <= 7 and pos[0]-c >= 0 and pos[0]-c <= 7 and (board[pos[1]][pos[0]-c]*player < 0 or board[pos[1]][pos[0]-c] == 0) and check[3] == True:
                    result.append((pos[1],pos[0]-c))
                else:
                    check[3] = False
                if pos[1]+c >= 0 and pos[1]+c <= 7 and pos[0]+c >= 0 and pos[0]+c <= 7 and (board[pos[1]+c][pos[0]+c]*player < 0 or board[pos[1]+c][pos[0]+c] == 0) and check[4] == True:
                    result.append((pos[1]+c,pos[0]+c))
                else:
                    check[4] = False
                if pos[1]+c >= 0 and pos[1]+c <= 7 and pos[0]-c >= 0 and pos[0]-c <= 7 and (board[pos[1]+c][pos[0]-c]*player < 0 or board[pos[1]+c][pos[0]-c] == 0) and check[5] == True:
                    result.append((pos[1]+c,pos[0]-c))
                else:
                    check[5] = False
                if pos[1]-c >= 0 and pos[1]-c <= 7 and pos[0]+c >= 0 and pos[0]+c <= 7 and (board[pos[1]-c][pos[0]+c]*player < 0 or board[pos[1]-c][pos[0]+c] == 0) and check[6] == True:
                    result.append((pos[1]-c,pos[0]+c))
                else:
                    check[6] = False
                if pos[1]-c >= 0 and pos[1]-c <= 7 and pos[0]-c >= 0 and pos[0]-c <= 7 and (board[pos[1]-c][pos[0]-c]*player < 0 or board[pos[1]-c][pos[0]-c] == 0) and check[7] == True:
                    result.append((pos[1]-c,pos[0]-c))
                else:
                    check[7] = False
                if True not in check:
                    break
            return result

    class Rook:
        def __init__(self):
            self.value = 4 #Numerical value of piece
            self.notation = 'R' #Chess notation

        def movement(board,player,pos):
            result = []
            check = [True,True,True,True]
            for c in range(1,8,1):
                if pos[1]+c >= 0 and pos[1]+c <= 7 and pos[0] >= 0 and pos[0] <= 7 and (board[pos[1]+c][pos[0]]*player < 0 or board[pos[1]+c][pos[0]] == 0) and check[0] == True:
                    result.append((pos[1]+c,pos[0]))
                else:
                    check[0] = False
                if pos[1]-c >= 0 and pos[1]-c <= 7 and pos[0] >= 0 and pos[0] <= 7 and (board[pos[1]-c][pos[0]]*player < 0 or board[pos[1]-c][pos[0]] == 0) and check[1] == True:
                    result.append((pos[1]-c,pos[0]))
                else:
                    check[1] = False
                if pos[1] >= 0 and pos[1] <= 7 and pos[0]+c >= 0 and pos[0]+c <= 7 and (board[pos[1]][pos[0]+c]*player < 0 or board[pos[1]][pos[0]+c] == 0) and check[2] == True:
                    result.append((pos[1],pos[0]+c))
                else:
                    check[2] = False
                if pos[1] >= 0 and pos[1] <= 7 and pos[0]-c >= 0 and pos[0]-c <= 7 and (board[pos[1]][pos[0]-c]*player < 0 or board[pos[1]][pos[0]-c] == 0) and check[3] == True:
                    result.append((pos[1],pos[0]-c))
                else:
                    check[3] = False
                if True not in check:
                    break
            return result

    class Bishop:
        def __init__(self):
            self.value = 3 #Numerical value of piece
            self.notation = 'B' #Chess notation

        def movement(board,player,pos):
            result = []
            check = [True,True,True,True]
            for c in range(1,8,1):
                if pos[1]+c >= 0 and pos[1]+c <= 7 and pos[0]+c >= 0 and pos[0]+c <= 7 and (board[pos[1]+c][pos[0]+c]*player < 0 or board[pos[1]+c][pos[0]+c] == 0) and check[0] == True:
                    result.append((pos[1]+c,pos[0]+c))
                else:
                    check[0] = False
                if pos[1]+c >= 0 and pos[1]+c <= 7 and pos[0]-c >= 0 and pos[0]-c <= 7 and (board[pos[1]+c][pos[0]-c]*player < 0 or board[pos[1]+c][pos[0]-c] == 0) and check[1] == True:
                    result.append((pos[1]+c,pos[0]-c))
                else:
                    check[1] = False
                if pos[1]-c >= 0 and pos[1]-c <= 7 and pos[0]+c >= 0 and pos[0]+c <= 7 and (board[pos[1]-c][pos[0]+c]*player < 0 or board[pos[1]-c][pos[0]+c] == 0) and check[2] == True:
                    result.append((pos[1]-c,pos[0]+c))
                else:
                    check[2] = False
                if pos[1]-c >= 0 and pos[1]-c <= 7 and pos[0]-c >= 0 and pos[0]-c <= 7 and (board[pos[1]-c][pos[0]-c]*player < 0 or board[pos[1]-c][pos[0]-c] == 0) and check[3] == True:
                    result.append((pos[1]-c,pos[0]-c))
                else:
                    check[3] = False
                if True not in check:
                    break
            return result

    class Knight:
        def __init__(self):
            self.value = 2 #Numerical value of piece
            self.notation = 'N' #Chess notation

        def movement(board,player,pos):
            result = []
            for i in [-1,1]:
                if pos[0]-i >= 0 and pos[0]-i <= 7 and pos[1]-(2*i) >= 0 and pos[1]-(2*i) <= 7 and (board[pos[1]-(2*i)][pos[0]-i]*player < 0 or board[pos[1]-(2*i)][pos[0]-i] == 0):
                    result.append((pos[0]-i,pos[1]-(2*i)))
                if pos[0]+i >= 0 and pos[0]+i <= 7 and pos[1]-(2*i) >= 0 and pos[1]-(2*i) <= 7 and (board[pos[1]-(2*i)][pos[0]+i]*player < 0 or board[pos[1]-(2*i)][pos[0]+i] == 0):
                    result.append((pos[0]+i,pos[1]-(2*i)))
                if pos[0]-(2*i) >= 0 and pos[0]-(2*i) <= 7 and pos[1]-i >= 0 and pos[1]-i <= 7 and (board[pos[1]-i][pos[0]-(2*i)]*player < 0 or board[pos[1]-i][pos[0]-(2*i)] == 0):
                    result.append((pos[0]-(2*i),pos[1]-i))
                if pos[0]-(2*i) >= 0 and pos[0]-(2*i) <= 7 and pos[1]+i >= 0 and pos[1]+i <= 7 and (board[pos[1]+i][pos[0]-(2*i)]*player < 0 or board[pos[1]+i][pos[0]-(2*i)] == 0):
                    result.append((pos[0]-(2*i),pos[1]+i))
            return result

    class Pawn:
        def __init__(self):
            self.value = 1 #Numerical value of piece
            self.notation = '' #Chess notation

        def movement(board,player,pos):
            result = []
            init = 1 if player < 0 else 6
            amt = 1 if pos[0] != init else 2
            print(pos)
            for i in range(amt):
                if pos[0]-((i+1)*player) >= 0 and pos[0]-((i+1)*player) <= 7 and board[pos[0]-((i+1)*player)][pos[1]] == 0:
                    print((pos[0]-((i+1)*player),pos[1]),pos[0]-((i+1)*player) >= 0,pos[0]-((i+1)*player) <= 7,board[pos[0]-((i+1)*player)][pos[1]] == 0)
                    result.append((pos[0]-((i+1)*player),pos[1]))
                else:
                    break
            if pos[0]-player <= 7 and pos[0]-player >= 0 and pos[1]+1 <= 7 and pos[1]+1 >= 0 and board[pos[0]-player][pos[1]+1]*player < 0:
                result.append((pos[0]-player,pos[1]+1))
            if pos[0]-player >= 0 and pos[0]-player <= 7 and pos[1]-1 >= 0 and pos[1]-1 <= 7 and board[pos[0]-player][pos[1]-1]*player < 0:
                result.append((pos[0]-player,pos[1]-1))
            return result

print('''
****************************
  Welcome to Console Chess
****************************
White = Lower Case
Black = Upper Case

P,p = Pawn
N,n = Knight
B,b = Bishop
R,r = Rook
Q,q = Queen
K,k = King

When asked where you want to moves plase use the following cordinate system:

a8 b8 c8 d8 e8 f8 g8 h8
a7 b7 c7 d7 e7 f7 g7 h7
a6 b6 c6 d6 e6 f6 g6 h6
a5 b5 c5 d5 e5 f5 g5 h5
a4 b4 c4 d4 e4 f4 g4 h4
a3 b3 c3 d3 e3 f3 g3 h3
a2 b2 c2 d2 e2 f2 g2 h2
a1 b1 c1 d1 e1 f1 g1 h1
''')
chess_game = Chess()
while True:
    if chess_game.p_move == 1:
        print('Whites Turn [LOWER CASE]\n')
    else:
        print('Blacks Turn [UPPER CASE]\n')
    chess_game.display()
    #print(chess_game.board)
    cur = input('What piece do you want to move?\n')
    next = input('Where do you want to move the piece to?\n')
    if chess_game.move(cur,next) == False:
        print('Invalid move')
    if chess_game.is_end() == True:
        break
