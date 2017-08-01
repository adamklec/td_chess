import numpy as np
import chess
from chess.pgn import read_game
from random import choice, randint
from .board_game_base import BoardGameEnvBase
from os import listdir
import pandas as pd


class ChessEnv(BoardGameEnvBase):

    def __init__(self, board=None, random_position=False, load_pgn=False):
        self.random_position = random_position
        self.load_pgn = load_pgn
        self.board = board

        if self.load_pgn:
            pgn = open("./data/millionbase-2.22.pgn")
            self.board_generator = random_board_generator(pgn)
        else:
            self.board_generator = None

        self.reset(board)

    def get_board(self):
        return self.board

    def get_null_move(self):
        return chess.Move.null()

    def get_move_stack(self):
        return self.board.move_stack

    def reset(self, board=None):
        if board is None:
            # skip through a random number of boards to avoid different threads training on the same board.
            if self.random_position:
                for _ in range(randint(1, 100)):
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

    def make_feature_vector(self, board=None):
        if board is None:
            board = self.board

        # 6 piece type maps + en passant square map for each color + 4 castling rights bit + 1 turn bit

        feature_vector = np.zeros((1, ((len(chess.PIECE_TYPES) + 1) * len(chess.COLORS)) * 64 + 5), dtype='float32')

        for piece in range(1, 6):
            for color in range(2):
                squares = board.pieces(piece, color)
                for square in squares:
                    feature_vector[0, (piece-1) + 6 * color + 12 * square] = 1

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
    def is_quiet(board):
        parent_board = board.copy()
        move = parent_board.pop()

        is_check = board.is_check()
        parent_is_check = parent_board.is_check()

        if parent_board.is_capture(move):
            capturing_piece_type = parent_board.piece_type_at(move.from_square)
            captured_piece_type = parent_board.piece_type_at(move.to_square)
            is_winning_capture = (capturing_piece_type < captured_piece_type)
        else:
            is_winning_capture = False

        is_promotion = move.promotion is not None

        return not (is_check or parent_is_check or is_winning_capture or is_promotion)

    @staticmethod
    def move_order_key(board, ttable):
        parent_board = board.copy()
        move = parent_board.pop()

        tt_row = ttable.get(board.zobrist_hash())
        if tt_row is not None:
            if tt_row['flag'] == 'EXACT':
                return 0
            if tt_row['flag'] == 'LOWERBOUND':
                return 1
        if parent_board.is_capture(move):
            return 2
        else:
            return 3

    def test(self, get_move_function, test_idx):
        tests = []
        path = "/Users/adam/Documents/projects/td_chess/STS[1-14]/"
        for filename in listdir(path):
            tests.append((parse_tests(path + filename), filename))

        df, name = tests[test_idx]
        result = 0
        print('running test suite:', name)
        # for fen, c0 in zip(df.fen[:1], df.c0[:1]):
        for fen, c0 in zip(df.fen, df.c0):
            board = chess.Board(fen=fen)
            self.reset(board=board)
            move = get_move_function(self)
            reward = c0.get(board.san(move), 0)
            result += reward
        return result

    def get_feature_vector_size(self):
        return self.make_feature_vector().shape[1]

    def get_simple_value_weights(self):
        fv_size = self.get_feature_vector_size()
        W_1 = np.zeros((fv_size, 1))
        pieces = ['p', 'n', 'b', 'r', 'q', 'P', 'N', 'B', 'R', 'Q']
        values = [-1, -3, -3, -5, -9, 1, 3, 3, 5, 9]
        for piece, value in zip(pieces, values):
            fen = '/'.join([8 * piece for _ in range(8)]) + ' w - - 0 1'
            board = chess.Board(fen)
            W_1[self.make_feature_vector(board)[0] == 1, 0] = value
            W_1[-5:] = 0
        return W_1


def random_board_generator(pgn):
    while True:
        game = read_game(pgn)
        if game and len(list(game.main_line())) > 0:
            move_number = np.random.randint(0, high=len(list(game.main_line())) - 1)  # don't take the last move
            while 2 * (game.board().fullmove_number - 1) + int(not game.board().turn) < move_number:
                game = game.variation(0)
            yield game.board()
        else:
            pgn.seek(0)


def board_generator(pgn):
    while True:
        game = read_game(pgn)
        if not game:
            pgn.seek()
        while not game.board().is_game_over():
            game = game.variation(0)
        yield game.board()


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


def parse_tests(filename):
    with open(filename, "r") as f:
        tests = f.readlines()

    dicts = []
    data = [[s for s in t.split('; ')] for t in tests]
    for row in data:
        d = dict()
        d['fen'] = row[0].split(' bm ')[0] + " 0 0"
        d['bm'] = row[0].split(' bm ')[1]

        for c in row[1:]:
            c = c.replace('"', '')
            c = c.replace(';\n', '')
            item = c.split(maxsplit=1, sep=" ")
            d[item[0]] = item[1]
        dicts.append(d)

    for d in dicts:
        move_rewards = {}
        answers = d['c0'].split(',')
        for answer in answers:
            move_reward = answer.split('=')
            move_rewards[move_reward[0].strip()] = int(move_reward[1])
        d['c0'] = move_rewards
    df = pd.DataFrame.from_dict(dicts)
    df = df.set_index('id')
    return df