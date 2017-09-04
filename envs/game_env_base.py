from abc import ABCMeta, abstractmethod
from agents.random_agent import RandomAgent


class GameEnvBase(metaclass=ABCMeta):

    def getboard(self):
        return self.__board

    def setboard(self, value):
        self.__board = value

    board = property(getboard, setboard)

    @abstractmethod
    def get_null_move(self):
        return NotImplemented

    @abstractmethod
    def make_board(self, fen):
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
    def is_quiet(board, depth):
        return NotImplemented

    @abstractmethod
    def get_test(self, test_idx):
        return NotImplemented

    @abstractmethod
    def zobrist_hash(self, board):
        return NotImplemented

    @abstractmethod
    def sort_children(self, parent, children, ttable, killers):
        return NotImplemented

    def play_random(self, get_move_function, side):

        self.reset()
        random_agent = RandomAgent('random_agent_0', None, self)
        if side:
            move_functions = [random_agent.get_move, get_move_function]  # True == 1 == 'X'
        else:
            move_functions = [get_move_function, random_agent.get_move]

        while self.get_reward() is None:
            move_function = move_functions[int(self.board.turn)]
            move = move_function(self)
            self.make_move(move)

        reward = self.get_reward()

        return reward

    def play_self(self, get_move_function):
        self.reset()
        while self.get_reward() is None:
            move = get_move_function(self)
            self.make_move(move)

        reward = self.get_reward()

        return reward

    # def random_agent_test(self, get_move_function):
    #     x_counter = Counter()
    #     for _ in range(100):
    #         self.reset()
    #         reward = self.play_random(get_move_function, True)
    #         x_counter.update([reward])
    #
    #     o_counter = Counter()
    #     for _ in range(100):
    #         self.reset()
    #         reward = self.play_random(get_move_function, False)
    #         o_counter.update([reward])
    #
    #     return [x_counter[1], x_counter[0], x_counter[-1],
    #             o_counter[1], o_counter[0], o_counter[-1]]
