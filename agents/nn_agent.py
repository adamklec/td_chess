import tensorflow as tf
import numpy as np
import pandas as pd
from random import choice
from chess import Board
from os import listdir
from anytree import Node
from game import Chess


class NeuralNetworkAgent(object):
    def __init__(self, name, network, global_episode_count=None, checkpoint=None, verbose=False,
                 model_path="/Users/adam/Documents/projects/td_chess/model", load_tests=False, create_trainer=True):
        self.verbose = verbose

        self.name = name
        self.checkpoint = checkpoint
        self.model_path = model_path

        self.load_tests = load_tests

        if self.load_tests:
            self.tests = []
            path = "/Users/adam/Documents/projects/td_chess/STS[1-13]/"
            for filename in listdir(path):
                self.tests.append((NeuralNetworkAgent.parse_tests(path + filename), filename))

        self.game_turn_count = tf.Variable(0, name='game_turn_count')
        self.game_turn_count_ = tf.placeholder(tf.int32, name='game_turn_count_')
        self.set_game_turn_count_op = tf.assign(self.game_turn_count, self.game_turn_count_, name='set_game_turn_count')
        tf.summary.scalar("game_turn_count", self.game_turn_count)

        self.test_score_ = tf.placeholder(tf.int32, name='test_score_')
        self.test_score = tf.Variable(0, name='test_total')
        self.set_test_score_op = self.test_score.assign(self.test_score_)
        tf.summary.scalar("test_score", self.test_score)

        self.neural_network = network

        if create_trainer:
            self.traces = []
            set_trace_tensor_ops = []
            self.trace_tensor_placeholders = []

            self.trainer = tf.train.AdamOptimizer()
            self.global_episode_count = global_episode_count
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
        for idx in range(len(self.traces)):
            self.traces[idx][:] = 0

    def train(self, sess, env, depth=1):

        turn_count = 0
        self.reset_traces()

        while env.get_reward() is None:
            move, next_value = self.get_move(sess, env, depth)
            feature_vector = Chess.make_feature_vector(env.board)
            value, grad_vars = sess.run([self.neural_network.value,
                                         self.grad_vars],
                                        feed_dict={
                                            self.neural_network.feature_vector_: feature_vector,
                                            self.neural_network.target_value_: next_value}
                                        )
            grads, _ = zip(*grad_vars)
            self.update_traces(grads)
            delta = (next_value - value)[0][0]
            sess.run(self.apply_grads,
                     feed_dict={delta_trace_: -delta * trace
                                for delta_trace_, trace in
                                zip(self.delta_trace_placeholders, self.traces)}
                     )

            # push the move onto the environment
            env.make_move(move)
            turn_count += 1

        # update traces with final state and reward
        reward = env.get_reward()
        feature_vector = Chess.make_feature_vector(env.board)
        value, grad_vars = sess.run([self.neural_network.value,
                                     self.grad_vars],
                                    feed_dict={
                                        self.neural_network.feature_vector_: feature_vector,
                                        self.neural_network.target_value_: reward}
                                    )
        grads, _ = zip(*grad_vars)
        self.update_traces(grads)
        delta = (reward - value)[0][0]
        sess.run(self.apply_grads,
                 feed_dict={delta_trace_: -delta * trace
                            for delta_trace_, trace in
                            zip(self.delta_trace_placeholders, self.traces)}
                 )
        if self.verbose:
            print("global episode:", sess.run(self.global_episode_count),
                  self.name,
                  "turn count:", turn_count,
                  'reward:', reward)

        sess.run([self.set_game_turn_count_op,
                  self.increment_global_episode_count_op],
                 feed_dict={self.game_turn_count_: turn_count})

        sess.run(self.set_trace_tensors_op, feed_dict={trace_tensor_: trace
                                                       for trace_tensor_, trace in zip(self.trace_tensor_placeholders, self.traces)})

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

    def test(self, sess, env, depth=1):
        tot = 0
        test_count = 0
        correct_count = 0
        for (df, name) in self.tests:
            print('running test suite:', name)
            for idx, (fen, c0) in enumerate(zip(df.fen, df.c0)):
                board = Board(fen=fen)
                env.reset(board=board)
                move, _ = self.get_move(sess, env, depth)
                reward = c0.get(board.san(move), 0)
                test_count += 1
                if reward > 0:
                    correct_count += 1
                tot += reward

        sess.run(self.set_test_score_op, feed_dict={self.test_score_: tot})
        global_episode_count = sess.run(self.global_episode_count)
        print("EPISODE", global_episode_count, "TEST TOTAL:", tot)
        return tot

    def get_move(self, sess, env, depth):
        node = Node('root', board=env.board)
        leaf_value, leaf_node = self.alpha_beta(node, depth, -1, 1, self.neural_network.value_function(sess))
        move = leaf_node.path[1].move
        return move, leaf_value[0, 0]

    @staticmethod
    def alpha_beta(node, depth, alpha, beta, value_function):
        if depth == 0 or node.board.is_game_over():
            return value_function(node.board), node

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
                vv, nn = NeuralNetworkAgent.alpha_beta(child, depth - 1, alpha, beta, value_function)
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
                vv, nn = NeuralNetworkAgent.alpha_beta(child, depth - 1, alpha, beta, value_function)
                if vv < v:
                    v = vv
                    n = nn
                beta = min(beta, v)
                if beta <= alpha:
                    break  # (* α cut-off *)
            return v, n
