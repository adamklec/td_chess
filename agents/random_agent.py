from agents.agent_base import AgentBase
import random


class RandomAgent(AgentBase):
    def get_move(self, env):
        legal_moves = env.get_legal_moves()
        move = random.choice(legal_moves)
        return move

    def get_move_function(self, env):
        return self.get_move
