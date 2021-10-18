# Chess AI

### Description
The following is a reinforcement learning AI program that plays the game of chess. This AI is trained using a combination of reinforcement learning and evolutionary learning. This AI is based off of [DeepMind's](https://deepmind.com/) AlphaGo algorithm where it uses a Neural Network in combination with the Monte Carlo Tree Search Algorithm. However the Neural Network (NN) used in this AI differs from AlphaGo since in their papers they use a convolutional neural network (CNN) where with this AI a decoder only transformer is used. Because of this difference the game state has to be encoded differently than in the AlphaGo paper. In this algorithm each piece on the board is represented with an embedding allowing the model to learn unique representations for each piece. With these piece embeddings the model is able to assess the board using a [self-attention](https://arxiv.org/abs/1706.03762) mechanism to predict the outcome of the game along with probability of taking each action.

# Modify Model
step 1: open ai_ben/data/model_param.json <br>
step 2: update variable amounts and save

```
{
  "input_size":64,
  "ntokens":33,
  "emsize":200,
  "nhid":200,
  "nlayers":2,
  "nhead":2,
  "dropout":0.2
}
```

# Training Instructions
step 1: open ai_ben/train.py and adjust the amount of games you wish to train the model for using self play;

```python
GAMES = 10 #Amount of games you wish to play
```

step 2: adjust the amount of boards you want to play at once;

```python
BOARDS = 1 #Amount of boards to play on at a time
```

step 3: adjust the best of you want to use when playing a round robin training series (Odd number);

```python
BEST_OF = 5 #Amount of games played when evaluating the models
```

step 4: adjust the type of players you want to train with;

```python
white = 'ai' #Values ['human','ai']
black = 'ai' #Values ['human','ai']
```

step 5: adjust the file parameters for locations of model files;

```python
folder = 'ai_ben/data' #Folder name where data is saved
parameters = 'model_param.json' #Model parameters filename
active_weights = 'model-active.pth.tar' #Active model saved weights filename
new_weights = 'model-new.pth.tar' #Active model saved weights filename
```

step 6: cd [app directory] <br>
step 7: python3 ai_ben/tain.py

# Resources
* https://deepmind.com/blog/article/alphazero-shedding-new-light-grand-games-chess-shogi-and-go
* https://web.stanford.edu/~surag/posts/alphazero.html
* https://arxiv.org/abs/1706.03762
* https://jalammar.github.io/illustrated-transformer/
