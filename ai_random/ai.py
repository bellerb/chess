from random import choice as rand_choice

"""
AI agent that makes random moves
"""
class Agent:
    """
    Input: None
    Description: AI initail variables
    Output: None
    """
    def __init__(self):
        pass

    """
    Input: game - object containing the game current state
    Description: Main entrance point for AI to make moves from [This is the function called when playing games]
    Output: tuple of strings representing the curent and next moves for the AI to make
    """
    def choose_action(self,game):
        p_moves = {k:v for k,v in game.possible_board_moves(capture=True).items() if len(v) > 0 and ((k[0].isupper() and game.p_move == 1) or (k[0].islower() and game.p_move == -1))}
        cur = rand_choice(list(p_moves.keys()))
        next = rand_choice(p_moves[cur])
        return cur,f'{game.x[next[0]]}{game.y[next[1]]}'
