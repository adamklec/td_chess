import numpy as np
from copy import deepcopy
import chess
from chess.pgn import read_game
from random import choice
from anytree import Node, RenderTree


class Chess(object):
    def __init__(self, board=None, random_position=False, load_pgn=False):
        self.random_position = random_position
        self.load_pgn = load_pgn

        if self.load_pgn:
            pgn = open("millionbase-2.22.pgn")
            self.board_generator = board_generator(pgn)
        else:
            self.board_generator = None

        if board is None:
            if not random_position:
                self.board = chess.Board()
            else:

                self.board = chess.Board()
        else:
            self.board = board

    def reset(self, board=None):
        if board is None:
            if self.random_position:
                self.board = self.board_generator.__next__()
            else:
                self.board = chess.Board()
        else:
            self.board = board

    def get_reward(self, board=None):
        if board is None:
            board = self.board
        result = board.result()
        if result == '1-0':
            return 1.0
        elif result == '0-1':
            return -1.0
        elif result == '1/2-1/2':
            return 0.0
        elif result == '*':
            return None
        else:
            raise ValueError('Invalid reward encountered')

    def make_move(self, move):
        if move in self.board.legal_moves:
            self.board.push(move)
        else:
            raise ValueError('Illegal move encountered')

    def get_legal_moves(self, board=None):
        if board is None:
            board = self.board
        legal_moves = list(board.legal_moves)
        return legal_moves

    def clone(self):
        return Chess(board=deepcopy(self.board))

    @staticmethod
    def simple_value_function(board):
        values = [1, 3, 3, 5, 9]
        s = 0
        for i, v in enumerate(values):
            s += Chess.pad_bitmask(board.pieces_mask(i + 1, 1)).sum() * v
            s -= Chess.pad_bitmask(board.pieces_mask(i + 1, 0)).sum() * v
        return np.tanh(s / 5)

    @staticmethod
    def make_feature_vector(board):
        piece_matrix = np.zeros((64, len(chess.PIECE_TYPES) + 1, len(chess.COLORS)))

        # piece positions
        for piece in chess.PIECE_TYPES:
            for color in chess.COLORS:
                piece_matrix[:, piece, int(color)] = Chess.pad_bitmask(board.pieces_mask(piece, color))

        # en passant target squares
        if board.ep_square:
            piece_matrix[board.ep_square, -1, int(board.turn)] = 1

        reshaped_piece_matrix = piece_matrix.reshape((64, (len(chess.PIECE_TYPES) + 1) * len(chess.COLORS)))
        feature_array = np.zeros((64, (len(chess.PIECE_TYPES) + 1) * len(chess.COLORS) + 2))
        feature_array[:, :-2] = reshaped_piece_matrix

        # empty squares
        empty_squares = (reshaped_piece_matrix.sum(axis=1) == 0)
        feature_array[empty_squares, :-2] = 1

        # castling rights
        feature_array[:, -1] = Chess.pad_bitmask(board.castling_rights)

        feature_vector = np.zeros((1, 1025))
        feature_vector[0, :-1] = np.reshape(feature_array, (1024,))
        feature_vector[0, -1] = board.turn

        return feature_vector

    @staticmethod
    def pad_bitmask(mask):
        mask = [int(s) for s in list(bin(mask)[2:])]
        while len(mask) < 64:
            mask.insert(0, 0)
        return np.array(mask)

    @staticmethod
    def get_simple_value_weights():
        W_1 = np.zeros((1025, 1))
        pieces = ['p', 'n', 'b', 'r', 'q', 'P', 'N', 'B', 'R', 'Q']
        values = [-1, -3, -3, -5, -9, 1, 3, 3, 5, 9]
        for piece, value in zip(pieces, values):
            fen = '/'.join([8 * piece for _ in range(8)]) + ' b -- - 0 1'
            board = chess.Board(fen)
            W_1[Chess.make_feature_vector(board)[0] == 1, 0] = value
        return W_1

def board_generator(pgn):
    while True:
        game = read_game(pgn)
        if game and len(list(game.main_line())) > 0:
            node = game
            move_number = np.random.randint(0, high=len(
                list(game.main_line())) - 1)  # don't take the last move
            while 2 * (node.board().fullmove_number - 1) + int(
                    not node.board().turn) < move_number:
                next_node = node.variation(0)
                node = next_node
            yield node.board()
        else:
            pgn.seek(0)

def make_random_move(board):
    random_move = choice(list(board.legal_moves))
    board.push(random_move)
    return board

def make_tree(board, depth):
    legal_moves = board.legal_moves

