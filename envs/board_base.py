from abc import ABCMeta, abstractmethod
from chess import Board


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

BoardBase.register(Board)
