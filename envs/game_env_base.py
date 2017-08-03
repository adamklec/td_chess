from abc import ABCMeta, abstractmethod

class GameEnvBase(metaclass=ABCMeta):

    @abstractmethod
    def get_board(self):
        return NotImplemented

    @abstractmethod
    def get_null_move(self):
        return NotImplemented

    @abstractmethod
    def get_move_stack(self):
        return NotImplemented

    @staticmethod
    @abstractmethod
    def get_feature_vector_size():
        return NotImplemented

    @abstractmethod
    def reset(self):
        return NotImplemented

    @abstractmethod
    def set_board(self, board):
        return NotImplemented

    @abstractmethod
    def random_position(self):
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

    @classmethod
    @abstractmethod
    def make_feature_vector(cls, board):
        return NotImplemented

    @staticmethod
    @abstractmethod
    def is_quiet(board):
        return NotImplemented

    @staticmethod
    def move_order_key(board, ttable):
        return NotImplemented

    @abstractmethod
    def test(self, get_move_function, test_idx):
        return NotImplemented
