"""
Class used to create game playing AI
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
    def choose_action(self, game):
        cur = 'a2'
        next = 'a4'
        return cur, next
