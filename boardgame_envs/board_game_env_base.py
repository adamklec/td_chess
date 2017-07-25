from abc import ABCMeta, abstractmethod
import chess


class BoardGameEnvBase(metaclass=ABCMeta):

    @abstractmethod
    def get_board(self):
        return NotImplemented

    @abstractmethod
    def get_null_move(self):
        return NotImplemented

    @abstractmethod
    def get_move_stack(self):
        return NotImplemented

    @abstractmethod
    def reset(self, board=None):
        return NotImplemented

    @abstractmethod
    def get_reward(self, board=None):
        return NotImplemented

    @abstractmethod
    def make_move(self, move):
        return NotImplemented

    @abstractmethod
    def get_legal_moves(self, board=None):
        return NotImplemented

    @staticmethod
    @abstractmethod
    def is_quiet(board):
        return NotImplemented

    @staticmethod
    @abstractmethod
    def make_feature_vector(board):
        return NotImplemented

    @staticmethod
    def move_order_key(board, ttable):
        return NotImplemented

    @abstractmethod
    def test(self, get_move_function, test_idx):
        return NotImplemented


class BoardBase(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, fen=None):
        pass

    @property
    @abstractmethod
    def turn(self):
        pass

    @turn.setter
    def turn(self, value):
        self.board = value

    @abstractmethod
    def fen(self):
        return NotImplemented

    @abstractmethod
    def copy(self):
        return NotImplemented

    @abstractmethod
    def zobrist_hash(self):
        return NotImplemented

BoardBase.register(chess.Board)
