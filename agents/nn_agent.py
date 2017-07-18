import tensorflow as tf
import numpy as np
import pandas as pd
from random import choice
from chess import Board, Move
from os import listdir
from anytree import Node
from game import Chess, simple_value_from_board


class NeuralNetworkAgent(object):
    def __init__(self,
                 name,
                 network,
                 global_episode_count=None,
                 checkpoint=None,
                 verbose=False,
                 model_path="/Users/adam/Documents/projects/td_chess/model",
                 load_tests=False,
                 load_pgn=True):

        self.name = name

        self.checkpoint = checkpoint
        self.verbose = verbose
        self.model_path = model_path
        self.load_tests = load_tests
        self.load_pgn = load_pgn
        self.ttable = dict()

        if self.load_tests:
            self.tests = []
            path = "/Users/adam/Documents/projects/td_chess/STS[1-14]/"
            for filename in listdir(path):
                self.tests.append((NeuralNetworkAgent.parse_tests(path + filename), filename))

        if self.load_pgn:
            self.env = Chess(load_pgn=True, random_position=True)

        else:
            self.env = Chess()

        self.game_turn_count = tf.Variable(0, name='game_turn_count')
        self.game_turn_count_ = tf.placeholder(tf.int32, name='game_turn_count_')
        self.set_game_turn_count_op = tf.assign(self.game_turn_count, self.game_turn_count_, name='set_game_turn_count')
        tf.summary.scalar("game_turn_count", self.game_turn_count)

        self.test_score_ = tf.placeholder(tf.float32, name='test_score_')
        self.test_results = tf.Variable(tf.zeros([14]), name="test_results")
        self.test_idx_ = tf.placeholder(tf.int32, name='test_idx_')
        self.test_result_ = tf.placeholder(tf.float32, name='test_result_')

        self.update_test_results = tf.scatter_update(self.test_results, self.test_idx_, self.test_result_)
        test_total = tf.reduce_sum(self.test_results)
        tf.summary.scalar("test_total", test_total)

        self.neural_network = network
        self.global_episode_count = global_episode_count
        for tvar in self.neural_network.trainable_variables:
            tf.summary.histogram(tvar.op.name, tvar)

        self.traces = []
        set_trace_tensor_ops = []
        self.trace_tensor_placeholders = []

        self.trainer = tf.train.AdamOptimizer()
        if self.global_episode_count is not None:
            with tf.variable_scope('turn_count'):
                self.increment_global_episode_count_op = self.global_episode_count.assign_add(1)
        with tf.variable_scope('traces'):
            for var in self.neural_network.trainable_variables:
                trace = np.zeros(var.get_shape())
                self.traces.append(trace)

                trace_tensor = tf.Variable(initial_value=trace, dtype=tf.float32, trainable=False, name=var.op.name+'_trace')

                trace_tensor_ = tf.placeholder(tf.float32, shape=var.get_shape(),  name=var.op.name+'_trace_')
                self.trace_tensor_placeholders.append(trace_tensor_)

                set_trace_tensor_op = trace_tensor.assign(trace_tensor_)
                set_trace_tensor_ops.append(set_trace_tensor_op)

                tf.summary.histogram(var.op.name+'_trace', trace_tensor)

        self.set_trace_tensors_op = tf.group(*set_trace_tensor_ops, name='set_trace_tensors_op')

        self.lamda = .7

        self.grad_vars = self.trainer.compute_gradients(self.neural_network.value, self.neural_network.trainable_variables)

        self.delta_trace_placeholders = [tf.placeholder(tf.float32, shape=var.get_shape(), name=var.op.name+'_PLACEHOLDER') for var in self.neural_network.trainable_variables]
        self.apply_grads = self.trainer.apply_gradients(zip(self.delta_trace_placeholders, self.neural_network.trainable_variables), name='apply_grads')

    def update_traces(self, grads):
        for idx in range(len(grads)):
            self.traces[idx] = self.lamda * self.traces[idx] + grads[idx]

    def reset_traces(self):
        traces = []
        for trace in self.traces:
            traces.append(np.zeros_like(trace))
        self.traces = traces

    def train(self, sess, depth=1):
        turn_count = 0
        self.reset_traces()
        self.env.reset()  # creates random board if env.random_position
        starting_position_move_str = ','.join([str(m) for m in self.env.board.move_stack])
        selected_moves = []
        while self.env.get_reward() is None and turn_count < 10:
            move, leaf_value = self.get_move(sess, self.env, depth)
            selected_moves.append(move)

            feature_vector = Chess.make_feature_vector(self.env.board)
            value, learned_value, simple_value, grad_vars = sess.run([self.neural_network.value,
                                         self.neural_network.learned_value,
                                         self.neural_network.simple_value,
                                         self.grad_vars],
                                        feed_dict={self.neural_network.feature_vector_: feature_vector})
            grads, _ = zip(*grad_vars)

            self.update_traces(grads)
            sess.run(self.set_trace_tensors_op,
                     feed_dict={trace_tensor_: trace
                                for trace_tensor_, trace in zip(self.trace_tensor_placeholders, self.traces)})
            delta = (leaf_value - value)[0][0]

            sess.run(self.apply_grads,
                     feed_dict={delta_trace_: -delta * trace
                                for delta_trace_, trace in
                                zip(self.delta_trace_placeholders, self.traces)}
                     )

            # push the move onto the environment
            self.env.make_move(move)
            turn_count += 1

        if self.env.board.is_game_over():
            # update traces with final state and reward
            reward = self.env.get_reward()
            feature_vector = Chess.make_feature_vector(self.env.board)
            value, grad_vars = sess.run([self.neural_network.value,
                                         self.grad_vars],
                                        feed_dict={
                                            self.neural_network.feature_vector_: feature_vector}
                                        )
            grads, _ = zip(*grad_vars)
            self.update_traces(grads)
            sess.run(self.set_trace_tensors_op,
                     feed_dict={trace_tensor_: trace
                                for trace_tensor_, trace in zip(self.trace_tensor_placeholders, self.traces)})

            delta = (reward - value)[0][0]
            sess.run(self.apply_grads,
                     feed_dict={delta_trace_: -delta * trace
                                for delta_trace_, trace in
                                zip(self.delta_trace_placeholders, self.traces)
                                }
                     )
        if self.verbose:
            print("global episode:", sess.run(self.global_episode_count),
                  self.name,
                  "turn count:", turn_count,
                  'reward:', self.env.get_reward())

        sess.run([self.set_game_turn_count_op, self.increment_global_episode_count_op],
                 feed_dict={self.game_turn_count_: turn_count})

        selected_moves_string = ','.join([str(m) for m in selected_moves])

        with open("move_log.txt", "a") as move_log:
            move_log.write(starting_position_move_str + '/' + selected_moves_string + '\n')

    @staticmethod
    def parse_tests(fn):
        with open(fn, "r") as f:
            tests = f.readlines()

        dicts = []
        data = [[s for s in test.split('; ')] for test in tests]
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

    def test(self, sess, test_idxs=None, depth=1):
        if test_idxs is None:
            test_idxs = range(14)

        for test_idx in test_idxs:
            df, name = self.tests[test_idx]
            tot = 0
            print('running test suite:', name)
            # for fen, c0 in zip(df.fen[:1], df.c0[:1]):
            for fen, c0 in zip(df.fen, df.c0):

                board = Board(fen=fen)
                self.env.reset(board=board)
                move, _ = self.get_move(sess, self.env, depth)
                # move = choice(list(board.legal_moves))
                reward = c0.get(board.san(move), 0)
                tot += reward

            sess.run(self.update_test_results, feed_dict={self.test_idx_: test_idx,
                                                          self.test_result_: tot})
            global_episode_count = sess.run(self.global_episode_count)
            print("EPISODE", global_episode_count,
                  "TEST_IDX", test_idx,
                  "TEST TOTAL:", tot)
            print(sess.run(self.test_results))

    def get_move(self, sess, env, depth):
        self.ttable = dict()
        node = Node('root', board=env.board, move=Move.null())
        leaf_value, leaf_node = self.negascout(node, depth, -1, 1, self.neural_network.value_function(sess))
        # leaf_value, leaf_node = self.negamax(node, depth, -1, 1, self.neural_network.value_function(sess))
        # leaf_value, leaf_node = self.alpha_beta(node, depth, -1, 1, self.neural_network.value_function(sess))
        # leaf_value, leaf_node = self.alpha_beta(node, depth, -1, 1, simple_value_from_board)
        move = leaf_node.path[1].move
        return move, leaf_value[0, 0]

    def alpha_beta(self, node, depth, alpha, beta, value_function):
        if (depth <= 0 and NeuralNetworkAgent.is_quiet(node)) or node.board.is_game_over():

            hash_key = node.board.zobrist_hash()
            hash_value = self.ttable.get(hash_key)
            if hash_value is None:
                value = value_function(node.board)
                self.ttable[hash_key] = value
            else:
                value = hash_value

            return value, node

        legal_moves = list(node.board.legal_moves)
        child_boards = [node.board.copy() for _ in legal_moves]
        children = []
        for idx in range(len(node.board.legal_moves)):
            child_boards[idx].push(legal_moves[idx])
            child = Node(str(legal_moves[idx]), parent=node, board=child_boards[idx], move=legal_moves[idx])
            children.append(child)
        n = node
        if node.board.turn:
            v = -100000
            for child in children:
                vv, nn = self.alpha_beta(child, depth - 1, alpha, beta, value_function)
                if vv > v:
                    v = vv
                    n = nn
                alpha = max(alpha, v)
                if beta <= alpha:
                    break  # (* β cut-off *)
            return v, n
        else:
            v = 100000
            for child in children:
                vv, nn = self.alpha_beta(child, depth - 1, alpha, beta, value_function)
                if vv < v:
                    v = vv
                    n = nn
                beta = min(beta, v)
                if beta <= alpha:
                    break  # (* α cut-off *)
            return v, n

    def negamax(self, node, depth, alpha, beta, value_function):
        alpha_orig = alpha

        hash_key = node.board.zobrist_hash()
        tt_row = self.ttable.get(hash_key)
        if tt_row is not None:# and tt_row['depth'] >= depth:
            if tt_row['flag'] == 'EXACT':
                return tt_row['value'], node
            elif tt_row['flag'] == 'LOWERBOUND':
                alpha = max(alpha, tt_row['value'])
            elif tt_row['flag'] == 'UPPERBOUND':
                beta = min(beta, tt_row['value'])
            if alpha >= beta:
                return tt_row['value'], node

        if (depth <= 0 and NeuralNetworkAgent.is_quiet(node)) or node.board.is_game_over():
            if node.board.turn:
                return value_function(node.board), node
            else:
                return -value_function(node.board), node

        legal_moves = list(node.board.legal_moves)
        children = []
        child_boards = [node.board.copy() for _ in legal_moves]
        for idx in range(len(node.board.legal_moves)):
            child_boards[idx].push(legal_moves[idx])
            child = Node(str(legal_moves[idx]), parent=node, board=child_boards[idx], move=legal_moves[idx])
            children.append(child)

        v = -100000
        n = node
        for child in children:
            vv, nn = self.negamax(child, depth - 1, -beta, -alpha, value_function)
            vv = -vv
            if vv > v:
                v = vv
                n = nn
            alpha = max(alpha, vv)
            if alpha >= beta:
                break

        if tt_row is None:
            tt_row = dict()
        tt_row['value'] = v
        if v <= alpha_orig:
            tt_row['flag'] = 'UPPERBOUND'
        elif v >= beta:
            tt_row['flag'] = 'LOWERBOUND'
        else:
            tt_row['flag'] = 'EXACT'

        tt_row['depth'] = depth
        self.ttable[hash_key] = tt_row
        return v, n

    def negascout(self, node, depth, alpha, beta, value_function):

        alpha_orig = alpha

        hash_key = node.board.zobrist_hash()
        tt_row = self.ttable.get(hash_key)
        if tt_row is not None and tt_row['depth'] >= depth:
            if tt_row['flag'] == 'EXACT':
                return tt_row['value'], node
            elif tt_row['flag'] == 'LOWERBOUND':
                alpha = max(alpha, tt_row['value'])
            elif tt_row['flag'] == 'UPPERBOUND':
                beta = min(beta, tt_row['value'])
            if alpha >= beta:
                return tt_row['value'], node

        if depth == 0 or node.board.is_game_over():
            if node.board.turn:
                return value_function(node.board), node
            else:
                return -value_function(node.board), node

        children = []
        for move in node.board.legal_moves:
            child_board = node.board.copy()
            child_board.push(move)
            child = Node(str(move), parent=node, board=child_board, move=move)
            children.append(child)

        children = sorted(children, key=self.move_order_key)
        n = node
        for idx, child in enumerate(children):
            if idx == 0:
                score, nn = self.negascout(child, depth - 1, -beta, -alpha, value_function)
                score = -score
            else:
                score, nn = self.negascout(child, depth - 1, -alpha - 1, -alpha, value_function)
                score = -score
                if alpha < score < beta:
                    score, nn = self.negascout(child, depth - 1, -beta, -score, value_function)
                    score = -score
            if score > alpha:
                alpha = score
                n = nn
            if alpha >= beta:
                break

        if tt_row is None:
            tt_row = dict()
        tt_row['value'] = score
        if score <= alpha_orig:
            tt_row['flag'] = 'UPPERBOUND'
        elif score >= beta:
            tt_row['flag'] = 'LOWERBOUND'
        else:
            tt_row['flag'] = 'EXACT'

        tt_row['depth'] = depth
        self.ttable[hash_key] = tt_row

        return alpha, n

    def move_order_key(self, node):
        if self.ttable.get(node.board.zobrist_hash()) is not None:
            return 0
        elif node.parent.board.is_capture(node.move):
            return 1
        else:
            return 2

    @staticmethod
    def is_quiet(node):

        is_check = node.board.is_check()
        capturing_piece_type = node.parent.board.piece_type_at(node.move.from_square)
        captured_piece_type = node.parent.board.piece_type_at(node.move.to_square)

        if captured_piece_type is None:
            is_winning_capture = False
        else:
            is_winning_capture = capturing_piece_type > captured_piece_type

        is_promotion = node.move.promotion is not None

        # is_capture = node.parent.board.is_capture(node.move)
        return not (is_check or is_winning_capture or is_promotion)
