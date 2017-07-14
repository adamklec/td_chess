import numpy as np
import pickle
from game import Chess
import chess
from sklearn.model_selection import train_test_split

Xs = []
ys = []

with open('simple_positions.fen', 'r') as f:
    for i, l in enumerate(f):
        print(i)
        fen, score = l.split(',')
        board = chess.Board(fen)
        fv = Chess.make_feature_vector(board)
        Xs.append(fv)
        ys.append(score)

X_train, X_test, y_train, y_test = train_test_split(np.vstack(Xs), np.vstack(ys), test_size=.2)

pickle.dump([X_train, y_train, X_test, y_test], open('simple_data.pkl', 'wb'))
