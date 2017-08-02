from abc import ABCMeta, abstractmethod


class AgentBase(metaclass=ABCMeta):

    @abstractmethod
    def get_move(self, env):
        return NotImplemented
