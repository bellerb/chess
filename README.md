# Chess

### Description
The following is a program for playing chess in the console written in python3. The program has been built to use the computer chess standard EPD (extended position description) for loading and exporting game positions. The program also makes use of standard chess notation (English) as you would in an actual chess tournament so that the games can be universally understood and later analysed.

### Board Display

#### Pieces
* White = Upper Case (Positive)
* Black = Lower Case (Negative)
* P,p = Pawn (1,-1)
* N,n = Knight (2,-2)
* B,b = Bishop (3,-3)
* R,r = Rook (4,-4)
* Q,q = Queen (5,-5)
* K,k = King (6,-6)

#### Board
| a8 | b8 | c8 | d8 | e8 | f8 | g8 | h8 |
| -- | -- | -- | -- | -- | -- | -- | -- |
| a7 | b7 | c7 | d7 | e7 | f7 | g7 | h7 |
| a6 | b6 | c6 | d6 | e6 | f6 | g6 | h6 |
| a5 | b5 | c5 | d5 | e5 | f5 | g5 | h5 |
| a4 | b4 | c4 | d4 | e4 | f4 | g4 | h4 |
| a3 | b3 | c3 | d3 | e3 | f3 | g3 | h3 |
| a2 | b2 | c2 | d2 | e2 | f2 | g2 | h2 |
| a1 | b1 | c1 | d1 | e1 | f1 | g1 | h1 |

# Launch Instructions
**(PVP)** <br>
step 1: open main.py and make sure the "white" & "black" global variable are as follows;

```python
white = 'human' #Values ['human','ai']
black = 'human' #Values ['human','ai']
```
step 2: open your console
step 3: type the following command "cd [app directory]" <br>
step 4: type the following command "python3 main.py"

**(PVAI)** <br>
step 1: open main.py and make sure the "white" & "black" global variable are as follows;

```python
#Play as white
white = 'human' #Values ['human','ai']
black = 'ai' #Values ['human','ai']
```
or
```python
#Play as black
white = 'ai' #Values ['human','ai']
black = 'human' #Values ['human','ai']
```

step 2: makes sure you have imported the ai you wish to play

```python
from ai_ben.ai import Agent
```

step 3: make sure your ai is properly initialized

```python
#Play as white
b_bot = Agent(max_depth=100) #Initailize white bot
```
or
```python
#Play as black
w_bot = Agent(max_depth=100) #Initailize white bot
```
step 4: open your console <br>
step 5: type the following command "cd [app directory]" <br>
step 6: type the following command "python3 main.py"
