import numpy as np
import chess
from chess.polyglot import zobrist_hash
from chess.pgn import read_game
from random import choice, randint
from .game_env_base import GameEnvBase
import pandas as pd


class ChessEnv(GameEnvBase):

    def __init__(self):
        self.board = chess.Board()

        pgn = open("./data/millionbase-2.22.pgn")
        self.board_generator = self.random_board_generator(pgn)

        self.tests = []
        # path = "./old_chess_tests/"
        # filenames = ['STS%s.epd' % str(i).zfill(2) for i in range(1, 15)]
        # for filename in filenames:
        #     self.tests.append((parse_tests(path + filename), filename))

    def get_null_move(self):
        return chess.Move.null()

    def get_move_stack(self):
        return self.board.move_stack

    def reset(self):
        self.board = chess.Board()

    def set_board(self, board):
        self.board = board

    def make_board(self, fen):
        self.board = chess.Board(fen)

    def random_position(self, episode_count=None):
        self.episode_count_ = episode_count
        for _ in range(randint(1, 100)):
            self.board = self.board_generator.__next__()

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

    @classmethod
    def make_feature_vector(cls, board):
        # 6 piece type each color + 1 turn bit
        fv_size = cls.get_feature_vector_size()

        feature_vector = np.zeros((1, fv_size), dtype='float32')

        for piece in range(6):
            for color in range(2):
                squares = board.pieces(piece + 1, color)
                for square in squares:
                    feature_vector[0, piece + 6 * color + 12 * square] = 1
        feature_vector[0, -1] = board.turn

        return feature_vector

    @staticmethod
    def is_quiet(board, depth):
        parent_board = board.copy()
        move = parent_board.pop()

        # if depth > -2:
        #     is_check = board.is_check()
        #     parent_is_check = parent_board.is_check()
        # else:
        #     is_check = False
        #     parent_is_check = False

        is_check = board.is_check()

        if parent_board.is_capture(move) and not parent_board.is_en_passant(move):
            capturing_piece_type = parent_board.piece_type_at(move.from_square)
            captured_piece_type = parent_board.piece_type_at(move.to_square)
            is_losing_capture = (capturing_piece_type > captured_piece_type)
        else:
            is_losing_capture = False

        is_promotion = move.promotion is not None

        return not (is_losing_capture or is_promotion or is_check) # or parent_is_check)

    def sort_children(self, parent, children, ttable, killers):
        hashed_nodes = []
        in_killers = []
        captures = []
        others = []

        tt_row = ttable.get(parent.board.fen())
        for child in children:
            if tt_row is not None:
                hashed_nodes.append(child.move)
            elif child.move in killers:
                in_killers.append(child)
            elif parent.board.is_capture(child.move):
                captures.append(child)

            else:
                others.append(child)

        captures = sorted(captures, key=lambda node: self.mvv_lva(node.parent.board, node.move))
        hashed_nodes = sorted(hashed_nodes, key=lambda node: ttable[self.zobrist_hash(node.board)]['value'], reverse=not parent.board.turn)
        return hashed_nodes + in_killers + captures + others

    @staticmethod
    def mvv_lva(board, move):
        if board.is_en_passant(move):
            aggressor = chess.PAWN
            victim = chess.PAWN
        else:
            aggressor = board.piece_type_at(move.from_square)
            victim = board.piece_type_at(move.to_square)
        return int(str(6 - victim) + str(aggressor - 1))

    def get_test(self, test_idx):
        df, _ = self.tests[test_idx]
        return df

    @staticmethod
    def get_feature_vector_size():
        # return (len(chess.PIECE_TYPES) + 1) * len(chess.COLORS) * 64 + 5
        return 171

    # @classmethod
    # def get_simple_value_weights(cls):
    #     return np.array([[-1, -3, -3, -5, -9, -15, 1, 3, 3, 5, 9, 15] * 64 + [0]]).T

    # @classmethod
    # def get_material_value_weights(cls):
    #     values = np.array([1] * 8 + [3] * 4 + [5] * 2 + [9] + [15] + [-1] * 8 + [-3] * 4 + [-5] * 2 + [-9] + [-15])
    #     weights = np.zeros((193, 1))
    #     weights[2:192:6, 0] = values
    #     return weights

    @classmethod
    def get_material_value_weights(cls):
        w = np.zeros((1, 171))
        w[0, -10:] = [8, 6, 6, 10, 9, -8, -6, -6, -10, -9]
        return w.T

    def zobrist_hash(self, board):
        return zobrist_hash(board)

    def random_board_generator(self, pgn, decay=5000):
        while True:
            game = read_game(pgn)
            main_line_length = len(list(game.main_line()))
            if self.episode_count_ is None:
                min_move_number = 0
            else:
                min_move_number = int(np.e**(-self.episode_count_/decay) * (main_line_length - 1))

            if game and len(list(game.main_line())) > 0:
                move_number = np.random.randint(min_move_number, high=main_line_length - 1)  # don't take the last move
                for _ in range(move_number):
                    game = game.variation(0)
                yield game.board()
            else:
                pgn.seek(0)

    @staticmethod
    def make_feature_vector2(board):
        fv = np.zeros((1, 171))
        from_squares = [move.from_square for move in board.legal_moves]
        white_pawn_features, white_pawn_material = pawn_features(board, 1, from_squares)
        white_pair_piece_features, white_pair_material = pair_piece_features(board, 1, from_squares)
        white_queen_king_features, white_queen_king_material = queen_king_features(board, 1, from_squares)
        black_pawn_features, black_pawn_material = pawn_features(board, 0, from_squares)
        black_pair_piece_features, black_pair_material = pair_piece_features(board, 0, from_squares)
        black_queen_king_features, black_queen_king_material = queen_king_features(board, 0, from_squares)
        features = white_pawn_features + white_pair_piece_features + white_queen_king_features + black_pawn_features + black_pair_piece_features + black_queen_king_features
        material = white_pawn_material + white_pair_material + white_queen_king_material[:1] + black_pawn_material + black_pair_material + black_queen_king_material[:1]
        for idx, feature in enumerate(features):
            fv[0, 5 * idx:5 * (idx + 1)] = feature
        fv[0, -11] = board.turn

        fv[0, -10:] = material
        return fv


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


def material_value_from_fen(fen):
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


def material_value_from_board(board):
    value = material_value_from_fen(board.fen())
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

max_mobilities = {1: 3.0, 2: 8.0, 3: 13.0, 4: 14.0, 5: 27.0, 6: 8.0}


def pawn_features(board, side, from_squares):
    features = [[0, 0, 0, 0, 0]] * 8
    material = []
    empty_slots = set(range(8))
    unplaced_squares = set()
    squares = set(board.pieces(1, side))
    material.append(len(squares)/8.0)
    for i, square in enumerate(squares):
        file = square % 8
        if file in empty_slots:
            rank = int(square / 8)
            file = square % 8
            slot = file
            empty_slots.remove(slot)
            min_attacker = min_attacker_value(board, square, not side)
            min_defender = min_attacker_value(board, square, side)
            mobility = sum([square == from_square for from_square in from_squares]) / 27.0
            features[slot] = [file/8, rank/8, min_defender, min_attacker, mobility]
        else:
            unplaced_squares.add(square)

    empty_slots = list(empty_slots)
    for i, square in enumerate(unplaced_squares):
        rank = int(square / 8)
        file = square % 8
        dists = [(slot - file)**2 for slot in empty_slots]
        slot = empty_slots[np.argmin(dists)]
        empty_slots.remove(slot)

        min_attacker = min_attacker_value(board, square, not side)
        min_defender = min_attacker_value(board, square, side)
        mobility = sum([square == from_square for from_square in from_squares]) / max_mobilities[1]
        features[slot] = [file/8, rank/8, min_defender, min_attacker, mobility]
    return features, material


def pair_piece_features(board, side, from_squares):
    features = [[0, 0, 0, 0, 0]] * 6
    material = []
    for piece in range(2, 5):
        squares = list(board.pieces(piece, side))[:2]
        material.append(len(squares)/2.0)
        for i, square in enumerate(squares):
            rank = int(square / 8)
            file = square % 8

            if piece == 3:  # select bishop slot based on square color
                slot = (file + rank % 2) % 2 + (piece - 2) * 2
            else:
                slot = i + (piece - 2) * 2
            min_attacker = min_attacker_value(board, square, not side)
            min_defender = min_attacker_value(board, square, side)
            mobility = sum([square == from_square for from_square in from_squares]) / max_mobilities[piece]
            features[slot] = [file/8, rank/8, min_defender, min_attacker, mobility]
    return features, material


def queen_king_features(board, side, from_squares):
    features = [[0, 0, 0, 0, 0]] * 2
    material = []
    for piece in range(5, 7):
        slot = piece - 5
        squares = list(board.pieces(piece, side))
        material.append(len(squares))
        for square in squares[:1]:
            rank = int(square / 8)
            file = square % 8
            min_attacker = min_attacker_value(board, square, not side)
            min_defender = min_attacker_value(board, square, side)
            mobility = sum([square == from_square for from_square in from_squares]) / max_mobilities[piece]
            features[slot] = [file/8, rank/8, min_defender, min_attacker, mobility]
    return features, material


def min_attacker_value(board, square, side):
    piece_type_to_value = {0: 0, 1: 1, 2: 3, 3: 3, 4: 5, 5: 9, 6: 15}
    attacker_squares = board.attackers(side, square)
    attacker_piece_types = []
    for attacker_square in attacker_squares:
        attacker_piece = board.piece_at(attacker_square)
        attacker_piece_types.append(attacker_piece.piece_type)
    if attacker_piece_types:
        min_attacker_type = min(attacker_piece_types)
    else:
        min_attacker_type = 0
    if side:
        return piece_type_to_value[min_attacker_type] / 15.0
    else:
        return -piece_type_to_value[min_attacker_type] / 15.0

