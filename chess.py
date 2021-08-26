class Chess:
    def __init__(self,EPD='rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -'):
        self.x = ['a','b','c','d','e','f','g','h'] #Board x representation
        self.y = ['8','7','6','5','4','3','2','1'] #Board y representation
        self.parts = {1:'Pawn',2:'Knight',3:'Bishop',4:'Rook',5:'Queen',6:'King'} #Map of number to part
        self.notation = {'p':1,'n':2,'b':3,'r':4,'q':5,'k':6} #Map of notation to part number
        self.reset(EPD=EPD) #Reset game board and state

    def reset(self,EPD='rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -'):
        self.log = [] #Game log
        self.EPD_table = {} #EPD hashtable
        self.p_move = 1 #Current players move white = 1 black = -1
        self.castling = [1,1,1,1] #Castling control
        self.en_passant = None #En passant control
        self.board = [[0,0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,0]] #Generate empty chess board
        self.load_EPD(EPD) #Load in game starting position

    def display(self):
        result = '  a b c d e f g h  \n  ----------------\n'
        for c,y in enumerate(self.board):
            result += f'{8-c}|'
            for x in y:
                if x != 0:
                    n = getattr(Chess,self.parts[int(x) if x > 0 else int(x)*(-1)])().notation.lower() if x < 0 else getattr(Chess,self.parts[int(x) if x > 0 else int(x)*(-1)])().notation.upper()
                    if n == '':
                        n = 'p' if x < 0 else 'P'
                    result += n
                else:
                    result += '.'
                result += ' '
            result += f'|{8-c}\n'
        result += '  ----------------\n  a b c d e f g h\n'
        print(result)

    def board_2_array(self,cord):
        cord = list(cord)
        if len(cord) == 2 and str(cord[0]).lower() in self.x and str(cord[1]) in self.y:
            return self.x.index(str(cord[0]).lower()), self.y.index(str(cord[1]))
        else:
            return None

    def EPD_hash(self):
        result = ''
        for i,rank in enumerate(self.board):
            e_count = 0
            for square in rank:
                if square == 0:
                    e_count += 1
                else:
                    if e_count > 0:
                        result += str(e_count)
                    e_count = 0
                    p_name = self.parts[int(square) if square > 0 else int(square)*(-1)] #Get name of part
                    p_notation = getattr(Chess,p_name)().notation
                    if p_notation == '':
                        p_notation = 'p'
                    if square < 0:
                        p_notation = str(p_notation).lower()
                    else:
                        p_notation = str(p_notation).upper()
                    result += p_notation
            if e_count > 0:
                result += str(e_count)
            if i < 7:
                result += '/'
        if self.p_move == -1:
            result += ' w'
        else:
            result += ' b'
        result += ' '
        if sum(self.castling) == 0:
            result += '-'
        else:
            if self.castling[0] == 1:
                result += 'K'
            if self.castling[1] == 1:
                result += 'Q'
            if self.castling[2] == 1:
                result += 'k'
            if self.castling[3] == 1:
                result += 'q'
        result += ' '
        if self.en_passant == None:
            result += '-'
        else:
            result += f'{self.x[self.en_passant[0]]}{self.y[self.en_passant[1]]}'
        return result

    def load_EPD(self,EPD):
        data = EPD.split(' ')
        if len(data) == 4:
            for x,rank in enumerate(data[0].split('/')):
                for y,p in enumerate(rank):
                    if p.isdigit():
                        for i in range(int(p)):
                            self.board[x][y+i] = 0
                    else:
                        self.board[x][y] = self.notation[str(p).lower()]*(-1) if str(p).islower() else self.notation[str(p).lower()]
            self.p_move = 1 if data[1] == 'w' else -1
            for c in data[2]:
                if c == 'K':
                    self.castling[0] = 1
                elif c == 'Q':
                    self.castling[1] = 1
                elif c == 'k':
                    self.castling[2] = 1
                elif c == 'q':
                    self.castling[3] = 1
            self.en_passant = None if data[3] == '-' else self.board_2_array(data[3])
            return True
        else:
            return False

    def log_move(self,part,cur_cord,next_cord,next_pos):
        #castling with rook on h side o-o casteling with rook on a o-o-o
        #pawn promotion location=piece ex a8=Q
        #to remove ambiguity where multiple pieces could make the move add starting identifier after piece notation ex Rab8
        p_name = self.parts[int(part) if part > 0 else int(part)*(-1)] #Get name of part
        move = str(getattr(Chess,p_name)().notation).upper() #Get part notation
        if self.board[next_pos[0]][next_pos[1]] != 0 or (next_pos == self.en_passant and (part == 1 or part == -1)): #Check if there is a capture
            move += 'x' if move != '' else str(cur_cord)[0] + 'x'
        move += str(next_cord).lower()
        moves = self.possible_board_moves()
        if sum(self.is_checkmate(moves)) > 0:
            move += '#'
        elif self.is_check(moves) == True:
            move += '+'
        self.log.append(move)

    def move(self,cur_pos,next_pos):
        cp = self.board_2_array(cur_pos)
        np = self.board_2_array(next_pos)
        if self.valid_move(cp,np) == True:
            part = self.board[cp[1]][cp[0]]
            if np == self.en_passant and (part == 1 or part == -1):
                self.board[self.en_passant[1]-(self.p_move*(-1))][self.en_passant[0]] = 0
            self.log_move(part,cur_pos,next_pos,np)
            if (part == 1 and np[1] == 4) or (part == -1 and np[1] == 3):
                self.en_passant = (np[0],np[1]+1) if part == 1 else (np[0],np[1]-1)
            else:
                self.en_passant = None
            self.board[cp[1]][cp[0]] = 0
            self.board[np[1]][np[0]] = part
            return True
        return False

    def valid_move(self,cur_pos,next_pos):
        part = self.board[cur_pos[1]][cur_pos[0]]
        if part * self.p_move > 0 and part != 0:
            p_name = self.parts[int(part) if part > 0 else int(part)*(-1)] #Get name of part
            if next_pos in getattr(Chess,p_name).movement(self,self.p_move,cur_pos):
                return True
        return False

    def possible_board_moves(self):
        moves = {}
        for y,row in enumerate(self.board):
            for x,part in enumerate(row):
                if part != 0:
                    p_name = self.parts[int(part) if part > 0 else int(part)*(-1)] #Get name of part
                    p_colour = 1 if part > 0 else -1
                    moves[f'{str(self.x[x]).upper() if p_colour > 0 else str(self.x[x]).lower()}{self.y[y]}'] = getattr(Chess,p_name).movement(self,p_colour,[x,y])
        #print({k:[f'{self.x[x[0]]}{self.y[x[1]]}' for x in v] for k,v in moves.items()})
        return moves

    def is_check(self,moves):
        k_pos = ()
        o_moves = []
        for p,a in moves.items():
            if (str(p[0]).isupper() and self.p_move == -1) or  (str(p[0]).islower() and self.p_move == 1):
                pos = self.board_2_array(p)
                if self.board[pos[1]][pos[0]] == self.King().value * (self.p_move*(-1)):
                    k_pos = pos
            else:
                for m in a:
                    if m not in o_moves:
                        o_moves.append(m)
        if len(k_pos) == 0:
            for y,row in enumerate(self.board):
                for x,part in enumerate(row):
                    if part == self.King().value*(self.p_move*(-1)):
                        k_pos = (x,y)
                    if len(k_pos) > 0:
                        break
                if len(k_pos) > 0:
                    break
        if len(k_pos) > 0 and k_pos in o_moves:
            return True
        return False

    def is_checkmate(self,moves):
        k_pos = ()
        o_moves = []
        for p,a in moves.items():
            if (str(p[0]).isupper() and self.p_move == 1) or (str(p[0]).islower() and self.p_move == -1):
                pos = self.board_2_array(p)
                if self.board[pos[1]][pos[0]] == self.King().value * (self.p_move*-1):
                    k_pos = (pos,a)
            else:
                for m in a:
                    if m not in o_moves:
                        o_moves.append(m)
        if len(k_pos) > 0 and k_pos[0] not in o_moves:
            return [0,0,0]
        elif len(k_pos) > 0 and k_pos[0] in o_moves:
            if False in [True if m in o_moves else False for m in k_pos[1]]:
                return [0,0,0]
        elif len(k_pos) == 0:
            k_count = [0,0]
            for y in self.board:
                for x in y:
                    if x == self.King().value:
                        k_count[0] = 1
                    elif x == self.King().value*(-1):
                        k_count[1] = 1
                    if sum(k_count) == 2:
                        break
                if sum(k_count) == 2:
                    break
            if k_count[0] == 0:
                return [0,0,1] #Black wins
            elif k_count[1] == 0:
                return [1,0,0] #White wins
            else:
                return [0,0,0]
        if self.p_move == -1:
            return [0,0,1] #Black wins
        else:
            return [1,0,0] #White wins

    def fifty_move_rule(self,moves):
        if len(self.log) > 100:
            for m in self.log[-100:]:
                if 'x' in m or m[0].islower():
                    return False
        else:
            return False
        while True:
            choice = input('Fifty move rule - do you want to claim a draw? [Y/N]')
            if choice.lower() == 'y' or choice.lower() == 'yes' or choice.lower() == '1':
                return True
            elif choice.lower() == 'n' or choice.lower() == 'no' or choice.lower() == '0':
                return False
            print('Unsupported answer')

    def seventy_five_move_rule(self,moves):
        if len(self.log) > 150:
            for m in self.log[-150:]:
                if 'x' in m or m[0].islower():
                    return False
        else:
            return False
        return True

    def three_fold_rule(self,hash):
        if hash in self.EPD_table:
            if self.EPD_table[hash] == 3:
                while True:
                    choice = input('Three fold rule - do you want to claim a draw? [Y/N]')
                    if choice.lower() == 'y' or choice.lower() == 'yes' or choice.lower() == '1':
                        return True
                    elif choice.lower() == 'n' or choice.lower() == 'no' or choice.lower() == '0':
                        return False
                    print('Unsupported answer')
        return False

    def five_fold_rule(self,hash):
        if hash in self.EPD_table:
            if self.EPD_table[hash] == 5:
                return True
        return False

    def is_dead_position(self,moves):
        #King and bishop against king and bishop with both bishops on squares of the same colour
        a_pieces = []
        for y in self.board:
            for x in y:
                if x != 0:
                    a_pieces.append(x)
                if len(a_pieces) > 4:
                    return False
        if len(a_pieces) == 2 and -6 in a_pieces and 6 in a_pieces:
            return True
        elif len(a_pieces) == 3 and ((-6 in a_pieces and 3 in a_pieces and 6 in a_pieces) or (-6 in a_pieces and -3 in a_pieces and 6 in a_pieces)):
            return True
        elif len(a_pieces) == 3 and ((-6 in a_pieces and 2 in a_pieces and 6 in a_pieces) or (-6 in a_pieces and -2 in a_pieces and 6 in a_pieces)):
            return True
        return False

    def is_stalemate(self,moves):
        if False not in [False for p,a in moves.items() if len(a) > 0 and ((self.p_move == 1 and str(p[0]).isupper()) or (self.p_move == -1 and str(p[0]).islower()))]:
            return True
        return False

    def is_draw(self,moves,hash):
        if self.is_stalemate(moves) == True:
            return True
        elif self.is_dead_position(moves) == True:
            return True
        elif self.seventy_five_move_rule(moves) == True:
            return True
        elif self.five_fold_rule(hash) == True:
            return True
        elif self.fifty_move_rule(moves) == True:
            return True
        elif self.three_fold_rule(hash) == True:
            return True
        return False

    def is_end(self):
        moves = self.possible_board_moves()
        check_mate = self.is_checkmate(moves)
        print(check_mate)
        hash = self.EPD_hash()
        if hash in self.EPD_table:
            self.EPD_table[hash] += 1
        else:
            self.EPD_table[hash] = 1
        if sum(check_mate) > 0:
            return check_mate
        elif self.is_draw(moves,hash) == True:
            return [0,1,0]
        return [0,0,0]

    class King:
        def __init__(self):
            self.value = 6 #Numerical value of piece
            self.notation = 'K' #Chess notation

        def movement(game,player,pos):
            result = []
            if pos[1]+1 >= 0 and pos[1]+1 <= 7 and pos[0] >= 0 and pos[0] <= 7 and (game.board[pos[1]+1][pos[0]]*player < 0 or game.board[pos[1]+1][pos[0]] == 0):
                result.append((pos[0],pos[1]+1))
            if pos[1]-1 >= 0 and pos[1]-1 <= 7 and pos[0] >= 0 and pos[0] <= 7 and (game.board[pos[1]-1][pos[0]]*player < 0 or game.board[pos[1]-1][pos[0]] == 0):
                result.append((pos[0],pos[1]-1))
            if pos[1] >= 0 and pos[1] <= 7 and pos[0]+1 >= 0 and pos[0]+1 <= 7 and (game.board[pos[1]][pos[0]+1]*player < 0 or game.board[pos[1]][pos[0]+1] == 0):
                result.append((pos[0]+1,pos[1]))
            if pos[1] >= 0 and pos[1] <= 7 and pos[0]-1 >= 0 and pos[0]-1 <= 7 and (game.board[pos[1]][pos[0]-1]*player < 0 or game.board[pos[1]][pos[0]-1] == 0):
                result.append((pos[0]-1,pos[1]))
            if pos[1]+1 >= 0 and pos[1]+1 <= 7 and pos[0]+1 >= 0 and pos[0]+1 <= 7 and (game.board[pos[1]+1][pos[0]+1]*player < 0 or game.board[pos[1]+1][pos[0]+1] == 0):
                result.append((pos[0]+1,pos[1]+1))
            if pos[1]+1 >= 0 and pos[1]+1 <= 7 and pos[0]-1 >= 0 and pos[0]-1 <= 7 and (game.board[pos[1]+1][pos[0]-1]*player < 0 or game.board[pos[1]+1][pos[0]-1] == 0):
                result.append((pos[0]-1,pos[1]+1))
            if pos[1]-1 >= 0 and pos[1]-1 <= 7 and pos[0]+1 >= 0 and pos[0]+1 <= 7 and (game.board[pos[1]-1][pos[0]+1]*player < 0 or game.board[pos[1]-1][pos[0]+1] == 0):
                result.append((pos[0]+1,pos[1]-1))
            if pos[1]-1 >= 0 and pos[1]-1 <= 7 and pos[0]-1 >= 0 and pos[0]-1 <= 7 and (game.board[pos[1]-1][pos[0]-1]*player < 0 or game.board[pos[1]-1][pos[0]-1] == 0):
                result.append((pos[0]-1,pos[1]-1))
            return result

    class Queen:
        def __init__(self):
            self.value = 5 #Numerical value of piece
            self.notation = 'Q' #Chess notation

        def movement(game,player,pos):
            result = []
            check = [True,True,True,True,True,True,True,True]
            for c in range(1,8,1):
                if pos[1]+c >= 0 and pos[1]+c <= 7 and pos[0] >= 0 and pos[0] <= 7 and (game.board[pos[1]+c][pos[0]]*player < 0 or game.board[pos[1]+c][pos[0]] == 0) and check[0] == True:
                    result.append((pos[0],pos[1]+c))
                    if game.board[pos[1]+c][pos[0]]*player < 0:
                        check[0] = False
                else:
                    check[0] = False
                if pos[1]-c >= 0 and pos[1]-c <= 7 and pos[0] >= 0 and pos[0] <= 7 and (game.board[pos[1]-c][pos[0]]*player < 0 or game.board[pos[1]-c][pos[0]] == 0) and check[1] == True:
                    result.append((pos[0],pos[1]-c))
                    if game.board[pos[1]-c][pos[0]]*player < 0:
                        check[1] = False
                else:
                    check[1] = False
                if pos[1] >= 0 and pos[1] <= 7 and pos[0]+c >= 0 and pos[0]+c <= 7 and (game.board[pos[1]][pos[0]+c]*player < 0 or game.board[pos[1]][pos[0]+c] == 0) and check[2] == True:
                    result.append((pos[0]+c,pos[1]))
                    if game.board[pos[1]][pos[0]+c]*player < 0:
                        check[2] = False
                else:
                    check[2] = False
                if pos[1] >= 0 and pos[1] <= 7 and pos[0]-c >= 0 and pos[0]-c <= 7 and (game.board[pos[1]][pos[0]-c]*player < 0 or game.board[pos[1]][pos[0]-c] == 0) and check[3] == True:
                    result.append((pos[0]-c,pos[1]))
                    if game.board[pos[1]][pos[0]-c]*player < 0:
                        check[3] = False
                else:
                    check[3] = False
                if pos[1]+c >= 0 and pos[1]+c <= 7 and pos[0]+c >= 0 and pos[0]+c <= 7 and (game.board[pos[1]+c][pos[0]+c]*player < 0 or game.board[pos[1]+c][pos[0]+c] == 0) and check[4] == True:
                    result.append((pos[0]+c,pos[1]+c))
                    if game.board[pos[1]+c][pos[0]+c]*player < 0:
                        check[4] = False
                else:
                    check[4] = False
                if pos[1]+c >= 0 and pos[1]+c <= 7 and pos[0]-c >= 0 and pos[0]-c <= 7 and (game.board[pos[1]+c][pos[0]-c]*player < 0 or game.board[pos[1]+c][pos[0]-c] == 0) and check[5] == True:
                    result.append((pos[0]-c,pos[1]+c))
                    if game.board[pos[1]+c][pos[0]-c]*player < 0:
                        check[5] = False
                else:
                    check[5] = False
                if pos[1]-c >= 0 and pos[1]-c <= 7 and pos[0]+c >= 0 and pos[0]+c <= 7 and (game.board[pos[1]-c][pos[0]+c]*player < 0 or game.board[pos[1]-c][pos[0]+c] == 0) and check[6] == True:
                    result.append((pos[0]+c,pos[1]-c))
                    if game.board[pos[1]-c][pos[0]+c]*player < 0:
                        check[6] = False
                else:
                    check[6] = False
                if pos[1]-c >= 0 and pos[1]-c <= 7 and pos[0]-c >= 0 and pos[0]-c <= 7 and (game.board[pos[1]-c][pos[0]-c]*player < 0 or game.board[pos[1]-c][pos[0]-c] == 0) and check[7] == True:
                    result.append((pos[0]-c,pos[1]-c))
                    if game.board[pos[1]-c][pos[0]-c]*player < 0:
                        check[7] = False
                else:
                    check[7] = False
                if True not in check:
                    break
            return result

    class Rook:
        def __init__(self):
            self.value = 4 #Numerical value of piece
            self.notation = 'R' #Chess notation

        def movement(game,player,pos):
            #Add in castling support
            #self.castling
            result = []
            check = [True,True,True,True]
            for c in range(1,8,1):
                if pos[1]+c >= 0 and pos[1]+c <= 7 and pos[0] >= 0 and pos[0] <= 7 and (game.board[pos[1]+c][pos[0]]*player < 0 or game.board[pos[1]+c][pos[0]] == 0) and check[0] == True:
                    result.append((pos[0],pos[1]+c))
                    if game.board[pos[1]+c][pos[0]]*player < 0:
                        check[0] = False
                else:
                    check[0] = False
                if pos[1]-c >= 0 and pos[1]-c <= 7 and pos[0] >= 0 and pos[0] <= 7 and (game.board[pos[1]-c][pos[0]]*player < 0 or game.board[pos[1]-c][pos[0]] == 0) and check[1] == True:
                    result.append((pos[0],pos[1]-c))
                    if game.board[pos[1]-c][pos[0]]*player < 0:
                        check[1] = False
                else:
                    check[1] = False
                if pos[1] >= 0 and pos[1] <= 7 and pos[0]+c >= 0 and pos[0]+c <= 7 and (game.board[pos[1]][pos[0]+c]*player < 0 or game.board[pos[1]][pos[0]+c] == 0) and check[2] == True:
                    result.append((pos[0]+c,pos[1]))
                    if game.board[pos[1]][pos[0]+c]*player < 0:
                        check[2] = False
                else:
                    check[2] = False
                if pos[1] >= 0 and pos[1] <= 7 and pos[0]-c >= 0 and pos[0]-c <= 7 and (game.board[pos[1]][pos[0]-c]*player < 0 or game.board[pos[1]][pos[0]-c] == 0) and check[3] == True:
                    result.append((pos[0]-c,pos[1]))
                    if game.board[pos[1]][pos[0]-c]*player < 0:
                        check[3] = False
                else:
                    check[3] = False
                if True not in check:
                    break
            return result

    class Bishop:
        def __init__(self):
            self.value = 3 #Numerical value of piece
            self.notation = 'B' #Chess notation

        def movement(game,player,pos):
            result = []
            check = [True,True,True,True]
            for c in range(1,8,1):
                if pos[1]+c >= 0 and pos[1]+c <= 7 and pos[0]+c >= 0 and pos[0]+c <= 7 and (game.board[pos[1]+c][pos[0]+c]*player < 0 or game.board[pos[1]+c][pos[0]+c] == 0) and check[0] == True:
                    result.append((pos[0]+c,pos[1]+c))
                    if game.board[pos[1]+c][pos[0]+c]*player < 0:
                        check[0] = False
                else:
                    check[0] = False
                if pos[1]+c >= 0 and pos[1]+c <= 7 and pos[0]-c >= 0 and pos[0]-c <= 7 and (game.board[pos[1]+c][pos[0]-c]*player < 0 or game.board[pos[1]+c][pos[0]-c] == 0) and check[1] == True:
                    result.append((pos[0]-c,pos[1]+c))
                    if game.board[pos[1]+c][pos[0]-c]*player < 0:
                        check[1] = False
                else:
                    check[1] = False
                if pos[1]-c >= 0 and pos[1]-c <= 7 and pos[0]+c >= 0 and pos[0]+c <= 7 and (game.board[pos[1]-c][pos[0]+c]*player < 0 or game.board[pos[1]-c][pos[0]+c] == 0) and check[2] == True:
                    result.append((pos[0]+c,pos[1]-c))
                    if game.board[pos[1]-c][pos[0]+c]*player < 0:
                        check[2] = False
                else:
                    check[2] = False
                if pos[1]-c >= 0 and pos[1]-c <= 7 and pos[0]-c >= 0 and pos[0]-c <= 7 and (game.board[pos[1]-c][pos[0]-c]*player < 0 or game.board[pos[1]-c][pos[0]-c] == 0) and check[3] == True:
                    result.append((pos[0]-c,pos[1]-c))
                    if game.board[pos[1]-c][pos[0]-c]*player < 0:
                        check[3] = False
                else:
                    check[3] = False
                if True not in check:
                    break
            return result

    class Knight:
        def __init__(self):
            self.value = 2 #Numerical value of piece
            self.notation = 'N' #Chess notation

        def movement(game,player,pos):
            result = []
            for i in [-1,1]:
                if pos[0]-i >= 0 and pos[0]-i <= 7 and pos[1]-(2*i) >= 0 and pos[1]-(2*i) <= 7 and (game.board[pos[1]-(2*i)][pos[0]-i]*player < 0 or game.board[pos[1]-(2*i)][pos[0]-i] == 0):
                    result.append((pos[0]-i,pos[1]-(2*i)))
                if pos[0]+i >= 0 and pos[0]+i <= 7 and pos[1]-(2*i) >= 0 and pos[1]-(2*i) <= 7 and (game.board[pos[1]-(2*i)][pos[0]+i]*player < 0 or game.board[pos[1]-(2*i)][pos[0]+i] == 0):
                    result.append((pos[0]+i,pos[1]-(2*i)))
                if pos[0]-(2*i) >= 0 and pos[0]-(2*i) <= 7 and pos[1]-i >= 0 and pos[1]-i <= 7 and (game.board[pos[1]-i][pos[0]-(2*i)]*player < 0 or game.board[pos[1]-i][pos[0]-(2*i)] == 0):
                    result.append((pos[0]-(2*i),pos[1]-i))
                if pos[0]-(2*i) >= 0 and pos[0]-(2*i) <= 7 and pos[1]+i >= 0 and pos[1]+i <= 7 and (game.board[pos[1]+i][pos[0]-(2*i)]*player < 0 or game.board[pos[1]+i][pos[0]-(2*i)] == 0):
                    result.append((pos[0]-(2*i),pos[1]+i))
            return result

    class Pawn:
        def __init__(self):
            self.value = 1 #Numerical value of piece
            self.notation = '' #Chess notation

        def movement(game,player,pos):
            result = []
            init = 1 if player < 0 else 6
            amt = 1 if pos[1] != init else 2
            for i in range(amt):
                if pos[1]-((i+1)*player) >= 0 and pos[1]-((i+1)*player) <= 7 and game.board[pos[1]-((i+1)*player)][pos[0]] == 0:
                    result.append((pos[0],pos[1]-((i+1)*player)))
                else:
                    break
            if pos[1]-player <= 7 and pos[1]-player >= 0 and pos[0]+1 <= 7 and pos[0]+1 >= 0 and game.board[pos[1]-player][pos[0]+1]*player < 0:
                result.append((pos[0]+1,pos[1]-player))
            if pos[1]-player >= 0 and pos[1]-player <= 7 and pos[0]-1 >= 0 and pos[0]-1 <= 7 and game.board[pos[1]-player][pos[0]-1]*player < 0:
                result.append((pos[0]-1,pos[1]-player))
            if pos[1]-player <= 7 and pos[1]-player >= 0 and pos[0]+1 <= 7 and pos[0]+1 >= 0 and (pos[0]+1,pos[1]-player) == game.en_passant:
                result.append((pos[0]+1,pos[1]-player))
            if pos[1]-player >= 0 and pos[1]-player <= 7 and pos[0]-1 >= 0 and pos[0]-1 <= 7 and (pos[0]-1,pos[1]-player) == game.en_passant:
                result.append((pos[0]-1,pos[1]-player))
            return result

if __name__ == '__main__':
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
    chess_game = Chess()
    while True:
        if chess_game.p_move == 1:
            print('\nWhites Turn [UPPER CASE]\n')
        else:
            print('\nBlacks Turn [LOWER CASE]\n')
        chess_game.display()
        cur = input('What piece do you want to move?\n')
        next = input('Where do you want to move the piece to?\n')
        valid = False
        if chess_game.move(cur,next) == False:
            print('Invalid move')
        else:
            valid = True
        state = chess_game.is_end()
        if sum(state) > 0:
            print('\n*********************\n      GAME OVER\n*********************\n')
            chess_game.display()
            print('Game Log:')
            print(chess_game.log)
            print('\nGame Result:\n')
            if state == [0,0,1]:
                print('BLACK WINS')
            elif state == [1,0,0]:
                print('WHITE WINS')
            else:
                print('TIE GAME')
            break
        if valid == True:
            chess_game.p_move = chess_game.p_move * (-1)