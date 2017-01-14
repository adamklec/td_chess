import numpy as np
from copy import deepcopy
import chess
from chess.pgn import read_game
import random


class Chess(object):
    def __init__(self, board=None):
        if board is None:
            self.board = chess.Board()
        else:
            self.board = board

    def reset(self):
        self.board = chess.Board()

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

    def get_candidate_states(self):
        legal_moves = self.board.legal_moves
        candidate_states = []
        for legal_move in legal_moves:
            candidate_state = self.clone()
            candidate_state.board.push(legal_move)
            candidate_states.append(candidate_state)
        return candidate_states

    def play(self, players, verbose=False):
        while self.get_reward() is None:
            if verbose:
                print(self.board)
                print('\n')
            player = players[int(not self.board.turn)]
            move = player.get_move(self)
            self.make_move(move)

        reward = self.get_reward()
        if verbose:
            print(self.board)
            if reward == 1:
                print("X won!")
            elif reward == -1:
                print("O won!")
            else:
                print("draw")
        return self.get_reward()

    def clone(self):
        return Chess(board=deepcopy(self.board))

    def get_feature_vector(self, board=None):
        if board is None:
            board = self.board

        piece_matrix = np.zeros((64, len(chess.PIECE_TYPES) + 1, len(chess.COLORS)))

        # piece positions
        for piece in chess.PIECE_TYPES:
            for color in chess.COLORS:
                piece_matrix[:, piece, int(not color)] = pad_bitmask(board.pieces_mask(piece, color))

        # en passant taget squares
        if board.ep_square:
            piece_matrix[board.ep_square, len(chess.PIECE_TYPES), int(not board.turn)] = 1

        reshaped_piece_matrix = piece_matrix.reshape((64, (len(chess.PIECE_TYPES) + 1) * len(chess.COLORS)))
        feature_vector = np.zeros((64, (len(chess.PIECE_TYPES) + 1) * len(chess.COLORS) + 2))
        feature_vector[:, :-2] = reshaped_piece_matrix

        # empty squares
        empty_squares = (reshaped_piece_matrix.sum(axis=1) == 0)
        feature_vector[empty_squares, :-2] = 1

        # castling rights
        feature_vector[:, -1] = pad_bitmask(board.castling_rights)

        return feature_vector

def pad_bitmask(mask):
    mask = [int(s) for s in list(bin(mask)[2:])]
    while len(mask) < 64:
        mask.insert(0, 0)
    return np.array(mask)


def bitmask_to_array(mask):
    array = pad_bitmask(mask).reshape((8, 8))
    return array


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


