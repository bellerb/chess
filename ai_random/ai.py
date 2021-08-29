from random import choice as rand_choice

class Agent:
    def __init__(self):
        pass

    def choose_action(self,game):
        p_moves = {k:v for k,v in game.possible_board_moves(capture=True).items() if len(v) > 0}
        cur = rand_choice(list(p_moves.keys()))
        next = rand_choice(p_moves[cur])
        return cur,f'{game.x[next[0]]}{game.y[next[1]]}'
