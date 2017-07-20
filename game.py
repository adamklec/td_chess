import numpy as np
import chess
from chess.pgn import read_game
from random import choice, randint


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
            if random_position:
                for _ in range(randint(1, 100)):  # non-optimal pseudo-random generator.
                    self.board = self.board_generator.__next__()
            else:
                self.board = chess.Board()
        else:
            self.board = board

    def reset(self, board=None):
        if board is None:
            if self.random_position:
                for _ in range(randint(1, 100)):  # non-optimal pseudo-random generator.
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


    @staticmethod
    def make_feature_vector(board):
        # feature_vector = np.zeros(((len(chess.PIECE_TYPES) + 1) * len(chess.COLORS) + 3, 64), dtype='float32')

        # 6 piece type maps + en passant square map for each color + 4 castling rights bit + 1 turn bit

        feature_vector = np.zeros((1, ((len(chess.PIECE_TYPES) + 1) * len(chess.COLORS)) * 64 + 5), dtype='float32')

        masks = [board.pieces_mask(piece, color) for color in chess.COLORS for piece in chess.PIECE_TYPES]
        padded_masks = Chess.pad_bitmasks(masks)
        feature_vector[0, :-(128 + 5)] = np.hstack(padded_masks)
        ep_square = board.ep_square
        if ep_square:
            feature_vector[0, -64 * (board.turn + 1) - 5 + board.ep_square] = 1

        feature_vector[0, -5] = board.has_kingside_castling_rights(0)
        feature_vector[0, -4] = board.has_queenside_castling_rights(0)
        feature_vector[0, -3] = board.has_kingside_castling_rights(1)
        feature_vector[0, -2] = board.has_queenside_castling_rights(1)
        feature_vector[0, -1] = board.turn

        return feature_vector

    @staticmethod
    def pad_bitmasks(masks):
        padded_masks = np.zeros((len(masks), 64))
        for i, mask in enumerate(masks):
            for j, bit in enumerate(bin(mask)[:1:-1]):
                padded_masks[i, -int(j)] = bit
        return padded_masks

    @staticmethod
    def get_simple_value_weights():
        W_1 = np.zeros((901, 1))
        pieces = ['p', 'n', 'b', 'r', 'q', 'P', 'N', 'B', 'R', 'Q']
        values = [-1, -3, -3, -5, -9, 1, 3, 3, 5, 9]
        for piece, value in zip(pieces, values):
            fen = '/'.join([8 * piece for _ in range(8)]) + ' b - - 0 1'
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


def simple_value_from_fen(fen):
    fen = fen.split()[0]
    value = 0

    value += fen.count('P') * 1
    value += fen.count('N') * 3
    value += fen.count('B') * 3
    value += fen.count('R') * 5
    value += fen.count('Q') * 9

    value -= fen.count('p') * 1
    value -= fen.count('n') * 3
    value -= fen.count('b') * 3
    value -= fen.count('r') * 5
    value -= fen.count('q') * 9

    return value

def simple_value_from_board(board):
    value = simple_value_from_fen(board.fen())
    return np.array([[value]])

if __name__ == '__main__':
    env = Chess(random_position=True, load_pgn=True)
    import time

    t0 = time.time()
    fv1 = env.make_feature_vector(env.board)
    print(time.time() - t0)
    t0 = time.time()
    fv2 = env.make_feature_vector2(env.board)
    print(time.time() - t0)

    W1 = env.get_simple_value_weights()
    W2 = env.get_simple_value_weights2()

    print((W1==1).sum())
    print((W2==1).sum())
