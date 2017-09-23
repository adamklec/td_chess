import tensorflow as tf
import numpy as np
from agents.agent_base import AgentBase


class SLAgent(AgentBase):

    def __init__(self):
        pass
    def train(self):
        pass
    def enqueue(self):
        pass

    def get_example(self, g):
        def f():
            board = g.__next__()
            fv = self.env.make_feature_vector(board)[0]
            value = self.env.board_value(board).astype('float32')[0]
            return fv, np.tanh(value / 5.0)

        return f