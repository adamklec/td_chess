import unittest
from anytree import Node, RenderTree
from anytree.render import AsciiStyle
from agents.td_leaf_agent import TDLeafAgent
from envs.chess import ChessEnv
from value_model import ValueModel


class TestMinimax(unittest.TestCase):
    def test_render(self):

        class TestChessEnv(ChessEnv):
            @staticmethod
            def is_quiet(board, depth):
                return True

        env = TestChessEnv(load_tests=False, load_pgn=False)

        env2 = TestChessEnv(load_tests=False, load_pgn=False)
        env2.make_move(list(env2.get_legal_moves(env2.board))[0])

        env3 = TestChessEnv(load_tests=False, load_pgn=False)
        env3.make_move(list(env3.get_legal_moves(env3.board))[0])
        env3.make_move(list(env3.get_legal_moves(env3.board))[0])

        env4 = TestChessEnv(load_tests=False, load_pgn=False)
        env4.make_move(list(env4.get_legal_moves(env4.board))[0])
        env4.make_move(list(env4.get_legal_moves(env4.board))[0])
        env4.make_move(list(env4.get_legal_moves(env4.board))[0])

        model = ValueModel()
        agent = TDLeafAgent('agent', model, env)

        a = Node('a', board=env.board, value=6)

        b1 = Node('b1', parent=a, board=env2.board, value=3)
        b2 = Node('b2', parent=a, board=env2.board, value=6)
        b3 = Node('b3', parent=a, board=env2.board, value=5)

        c11 = Node('c11', parent=b1, board=env3.board, value=5)
        c12 = Node('c12', parent=b1, board=env3.board, value=3)

        c21 = Node('c21', parent=b2, board=env3.board, value=6)
        c22 = Node('c22', parent=b2, board=env3.board, value=7)

        c31 = Node('c31', parent=b3, board=env3.board, value=5)
        c32 = Node('c32', parent=b3, board=env3.board, value=8)

        d111 = Node('d111', parent=c11, board=env4.board, value=5)
        d112 = Node('d112', parent=c11, board=env4.board, value=4)

        d121 = Node('d121', parent=c12, board=env4.board, value=3)

        d211 = Node('d211', parent=c21, board=env4.board, value=6)
        d212 = Node('d212', parent=c21, board=env4.board, value=6)

        d221 = Node('d221', parent=c22, board=env4.board, value=7)

        d311 = Node('d311', parent=c31, board=env4.board, value=5)

        d321 = Node('d321', parent=c32, board=env4.board, value=8)
        d322 = Node('d322', parent=c32, board=env4.board, value=6)

        e1111 = Node('e1111', parent=d111, board=env4.board, value=5)
        e1112 = Node('e1112', parent=d111, board=env4.board, value=6)

        e1121 = Node('e1121', parent=d112, board=env4.board, value=7)
        e1122 = Node('e1122', parent=d112, board=env4.board, value=4)
        e1123 = Node('e1123', parent=d112, board=env4.board, value=5)

        e1211 = Node('e1211', parent=d121, board=env4.board, value=3)

        e2111 = Node('e2111', parent=d211, board=env4.board, value=6)

        e2121 = Node('e2121', parent=d212, board=env4.board, value=6)
        e2122 = Node('e2122', parent=d212, board=env4.board, value=9)

        e2211 = Node('e2211', parent=d221, board=env4.board, value=7)

        e3111 = Node('e3111', parent=d311, board=env4.board, value=5)

        e3211 = Node('e3211', parent=d321, board=env4.board, value=9)
        e3212 = Node('e3212', parent=d321, board=env4.board, value=8)

        e3221 = Node('e3221', parent=d322, board=env4.board, value=6)

        # print(RenderTree(a, style=AsciiStyle()))

        print('\n')
        v, n = agent.minimax(a, 4, -100000, 100000, lambda node: node.value, False)

        print('\n\n', v, n)

        assert(True)
