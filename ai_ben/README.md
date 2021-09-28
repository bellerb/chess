# Chess AI

### Description
The following is a reinforcement learning AI program that plays the game of chess. This AI is based off of Deepminds AlphaGo algorithm where it uses a Neural Network in combination with the Monte Carlo Tree Search Algorithm. However the Neural Network (NN) used in this AI differs from AlphaGo since in their papers they use a convolutional neural network (CNN) where with this AI a decoder only transformer is used. Because of this difference the game state has to be encoded differently than in the AlphaGo paper. In this algorithm each piece on the board is represented with an embedding allowing the model to learn unique representations for each piece. With these piece embeddings the model is able to assess the board using a self attention mechanism to predict the outcome of the game along with probability of taking each action.

# Training Instructions
step 1: open ai_ben/train.py and adjust the amount of games you wish to train the model for using self play;

```python
GAMES = 10 #Amount of games you wish to play
```

step 2: cd [app directory] <br>
step 3: python3 ai_ben/tain.py